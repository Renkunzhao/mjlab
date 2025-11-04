from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.utils import resolve_param
from mjlab.third_party.isaaclab.isaaclab.utils.string import resolve_matching_names
from mjlab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass
class ActuatorCommand:
  """Input data for actuator computation."""

  position_target: torch.Tensor
  """Target positions for controlled joints (num_envs, num_targets)."""
  velocity_target: torch.Tensor
  """Target velocities for controlled joints (num_envs, num_targets)."""
  effort_target: torch.Tensor
  """Target efforts/torques for controlled joints (num_envs, num_targets)."""
  joint_pos: torch.Tensor
  """Current joint positions (num_envs, num_targets)."""
  joint_vel: torch.Tensor
  """Current joint velocities (num_envs, num_targets)."""


@dataclass(kw_only=True)
class ActuatorCfg(ABC):
  """Base configuration for an actuator."""

  target_names_expr: list[str]
  """Regular expressions to match joint names."""
  effort_limit: float | dict[str, float]
  """Maximum force/torque limit."""
  armature: float | dict[str, float] = 0.0
  """Reflected rotor inertia. Defaults to 0."""
  frictionloss: float | dict[str, float] = 0.0
  """Static friction loss. Defaults to 0."""
  min_delay: int = 0
  """Minimum delay in timesteps (inclusive)."""
  max_delay: int = 0
  """Maximum delay in timesteps (inclusive)."""

  @abstractmethod
  def build(self, entity: Entity) -> Actuator:
    """Build actuator instance from this config."""
    raise NotImplementedError


class Actuator(ABC):
  """Base actuator interface."""

  def __init__(self, cfg: ActuatorCfg, entity: Entity) -> None:
    self.cfg = cfg
    self.entity = entity
    self._targets_to_actuate: list[mujoco.MjsJoint] = []
    self._target_indices: torch.Tensor | None = None
    self._ctrl_ids: torch.Tensor | None = None
    self._delay_buffer: DelayBuffer | None = None

  @property
  def target_indices(self) -> torch.Tensor:
    assert self._target_indices is not None
    return self._target_indices

  @property
  def ctrl_ids(self) -> torch.Tensor:
    assert self._ctrl_ids is not None
    return self._ctrl_ids

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    """Identify which joints this actuator controls.

    This is called during entity construction, before the model is compiled.

    Args:
      spec: The entity's MjSpec to edit.
    """
    target_names = [j.name.split("/")[-1] for j in spec.joints]
    target_indices, _ = resolve_matching_names(
      self.cfg.target_names_expr, target_names, preserve_order=True
    )
    self._targets_to_actuate = [spec.joints[i] for i in target_indices]

  def create_actuator_for_target(
    self, spec: mujoco.MjSpec, target: mujoco.MjsJoint
  ) -> None:
    """Create MuJoCo actuator element for a target joint.

    Called once per joint in the spec. Sets common joint parameters
    (armature, frictionloss) and delegates actuator element creation to subclass.

    Args:
      spec: The entity's MjSpec.
      target: A joint from spec.joints.
    """
    if target not in self._targets_to_actuate:
      return

    target.armature = resolve_param(self.cfg.armature, target.name)
    target.frictionloss = resolve_param(self.cfg.frictionloss, target.name)

    self._create_mj_actuator(spec, target)

  @abstractmethod
  def _create_mj_actuator(self, spec: mujoco.MjSpec, target: mujoco.MjsJoint) -> None:
    """Create the MuJoCo actuator element for this target.

    Subclasses implement this to configure actuator-specific parameters.

    Args:
      spec: The entity's MjSpec.
      target: The target joint for this actuator.
    """
    raise NotImplementedError

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    """Initialize the actuator after model compilation.

    This is called after the MjSpec is compiled into an MjModel and the simulation
    is ready to run.

    Args:
      mj_model: The compiled MuJoCo model.
      model: The compiled mjwarp model.
      data: The mjwarp data arrays.
      device: Device for tensor operations (e.g., "cuda", "cpu").
    """
    del mj_model, model  # Unused.

    joint_indices, target_names = self.entity.find_joints(
      self.cfg.target_names_expr, preserve_order=True
    )
    self._target_indices = torch.tensor(joint_indices, dtype=torch.long, device=device)
    self._target_names = target_names

    ctrl_ids_list = []
    for target_name in target_names:
      for local_idx, act in enumerate(self.entity.spec.actuators):
        if act.name.split("/")[-1] == target_name:
          ctrl_ids_list.append(local_idx)
          break
      else:
        raise RuntimeError(
          f"Actuator for joint '{target_name}' not found in spec. "
          f"This should not happen - was create_actuator_for_target() called?"
        )
    self._ctrl_ids = torch.tensor(ctrl_ids_list, dtype=torch.long, device=device)

    if self.cfg.max_delay > 0:
      self._delay_buffer = DelayBuffer(
        min_lag=self.cfg.min_delay,
        max_lag=self.cfg.max_delay,
        batch_size=data.nworld,
        device=device,
      )

  @abstractmethod
  def compute(self, command: ActuatorCommand) -> torch.Tensor:
    """Compute actuator outputs based on targets and current joint state.

    Args:
      command: Actuator command data containing targets and current state.

    Returns:
      Actuator output values of shape (num_envs, num_targets).
    """
    raise NotImplementedError

  def _maybe_apply_delay(self, value: torch.Tensor) -> torch.Tensor:
    """Apply delay to a value if delay buffer is configured.

    Args:
      value: Input tensor to delay (shape: [num_envs, num_targets]).

    Returns:
      Delayed value if delay is configured, otherwise the input value.
    """
    if self._delay_buffer is not None:
      self._delay_buffer.append(value)
      return self._delay_buffer.compute()
    return value

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset actuator state for specified environments.

    Resets delay buffers. Override in actuators that maintain additional internal state.

    Args:
      env_ids: Environment indices to reset. If None, reset all environments.
    """
    if self._delay_buffer is not None:
      buffer_env_ids = None if isinstance(env_ids, slice) else env_ids
      self._delay_buffer.reset(buffer_env_ids)

  def update(self, dt: float) -> None:
    """Update actuator state after a simulation step.

    Base implementation does nothing. Override in actuators that need
    per-step updates.

    Args:
      dt: Time step in seconds.
    """
    del dt  # Unused.

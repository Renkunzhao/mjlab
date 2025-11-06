from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch

from mjlab.actuator.actuator import (
  Actuator,
  ActuatorCfg,
  ActuatorCmd,
  resolve_param_to_list,
)
from mjlab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(kw_only=True)
class BuiltinPdActuatorCfg(ActuatorCfg):
  """Configuration for MuJoCo built-in PD actuator.

  All parameters can be specified as a single float (broadcast to all joints)
  or a dict mapping joint names/patterns to values.
  """

  stiffness: float | dict[str, float]
  """PD proportional gain."""
  damping: float | dict[str, float]
  """PD derivative gain."""
  armature: float | dict[str, float] = 0.0
  """Reflected rotor inertia."""
  stiction: float | dict[str, float] = 0.0
  """Joint friction loss."""
  effort_limit: float | dict[str, float] | None = None
  """Maximum actuator force/torque. If None, no limit is applied."""

  def build(
    self, entity: Entity, joint_ids: list[int], joint_names: list[str]
  ) -> BuiltinPdActuator:
    return BuiltinPdActuator(self, entity, joint_ids, joint_names)


class BuiltinPdActuator(Actuator):
  """MuJoCo built-in PD actuator."""

  def __init__(
    self,
    cfg: BuiltinPdActuatorCfg,
    entity: Entity,
    joint_ids: list[int],
    joint_names: list[str],
  ) -> None:
    super().__init__(entity, joint_ids, joint_names)
    self.cfg = cfg

  def edit_spec(self, spec: mujoco.MjSpec, joint_names: list[str]) -> None:
    # Resolve parameters to per-joint lists.
    stiffness = resolve_param_to_list(self.cfg.stiffness, joint_names)
    damping = resolve_param_to_list(self.cfg.damping, joint_names)
    armature = resolve_param_to_list(self.cfg.armature, joint_names)
    stiction = resolve_param_to_list(self.cfg.stiction, joint_names)
    if self.cfg.effort_limit is not None:
      effort_limit = resolve_param_to_list(self.cfg.effort_limit, joint_names)
    else:
      effort_limit = None

    # Add <position> actuator to spec, one per joint.
    for i, joint_name in enumerate(joint_names):
      actuator = spec.add_actuator()
      actuator.name = joint_name
      actuator.target = joint_name
      actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
      actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
      actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
      actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
      actuator.inheritrange = 1.0
      actuator.gainprm[0] = stiffness[i]
      actuator.biasprm[1] = -stiffness[i]
      actuator.biasprm[2] = -damping[i]
      if effort_limit is not None:
        actuator.forcerange[:] = np.array([-effort_limit[i], effort_limit[i]])
      spec.joint(joint_name).armature = armature[i]
      spec.joint(joint_name).frictionloss = stiction[i]
      self._actuator_specs.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.position_target


@dataclass(kw_only=True)
class DelayedBuiltinPdActuatorCfg(BuiltinPdActuatorCfg):
  """Built-in PD actuator config with action delays."""

  delay_min_lag: int = 0
  """Minimum delay lag in timesteps."""
  delay_max_lag: int = 0
  """Maximum delay lag in timesteps."""
  delay_hold_prob: float = 0.0
  """Probability of keeping previous lag when updating."""
  delay_update_period: int = 0
  """Period for updating delays (0 = every step)."""
  delay_per_env_phase: bool = True
  """Whether each environment has a different phase offset."""

  def build(
    self, entity: Entity, joint_ids: list[int], joint_names: list[str]
  ) -> DelayedBuiltinPdActuator:
    return DelayedBuiltinPdActuator(self, entity, joint_ids, joint_names)


class DelayedBuiltinPdActuator(BuiltinPdActuator):
  """Built-in PD actuator with action delays.

  Delays position targets before passing them to the built-in PD controller.
  """

  def __init__(
    self,
    cfg: DelayedBuiltinPdActuatorCfg,
    entity: Entity,
    joint_ids: list[int],
    joint_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, joint_ids, joint_names)
    self._delay_buffer: DelayBuffer | None = None

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)

    cfg = self.cfg
    assert isinstance(cfg, DelayedBuiltinPdActuatorCfg)

    self._delay_buffer = DelayBuffer(
      min_lag=cfg.delay_min_lag,
      max_lag=cfg.delay_max_lag,
      batch_size=data.nworld,
      device=device,
      hold_prob=cfg.delay_hold_prob,
      update_period=cfg.delay_update_period,
      per_env_phase=cfg.delay_per_env_phase,
    )

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self._delay_buffer is not None
    self._delay_buffer.append(cmd.position_target)
    delayed_position_target = self._delay_buffer.compute()
    return delayed_position_target

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._delay_buffer is not None:
      self._delay_buffer.reset(env_ids)


@dataclass(kw_only=True)
class BuiltinTorqueActuatorCfg(ActuatorCfg):
  """Configuration for MuJoCo built-in torque actuator.

  All parameters can be specified as a single float (broadcast to all joints)
  or a dict mapping joint names/patterns to values.
  """

  effort_limit: float | dict[str, float] = float("inf")
  """Maximum actuator effort."""
  gear: float | dict[str, float] = 1.0
  """Actuator gear ratio."""
  armature: float | dict[str, float] = 0.0
  """Reflected rotor inertia."""
  stiction: float | dict[str, float] = 0.0
  """Joint friction loss."""

  def build(
    self, entity: Entity, joint_ids: list[int], joint_names: list[str]
  ) -> BuiltinTorqueActuator:
    return BuiltinTorqueActuator(self, entity, joint_ids, joint_names)


class BuiltinTorqueActuator(Actuator):
  """MuJoCo built-in torque actuator.

  Direct torque/force pass-through with no control law. The control signal
  is directly applied as joint torque/force, clamped to effort_limit.
  """

  def __init__(
    self,
    cfg: BuiltinTorqueActuatorCfg,
    entity: Entity,
    joint_ids: list[int],
    joint_names: list[str],
  ) -> None:
    super().__init__(entity, joint_ids, joint_names)
    self.cfg = cfg

  def edit_spec(self, spec: mujoco.MjSpec, joint_names: list[str]) -> None:
    # Resolve parameters to per-joint lists.
    effort_limit = resolve_param_to_list(self.cfg.effort_limit, joint_names)
    armature = resolve_param_to_list(self.cfg.armature, joint_names)
    stiction = resolve_param_to_list(self.cfg.stiction, joint_names)
    gear = resolve_param_to_list(self.cfg.gear, joint_names)

    # Add <motor> actuator to spec, one per joint.
    for i, joint_name in enumerate(joint_names):
      actuator = spec.add_actuator()
      actuator.name = joint_name
      actuator.target = joint_name
      actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
      actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
      actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
      actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
      actuator.gainprm[0] = 1.0
      actuator.gear[0] = gear[i]
      actuator.forcerange[:] = np.array([-effort_limit[i], effort_limit[i]])
      spec.joint(joint_name).armature = armature[i]
      spec.joint(joint_name).frictionloss = stiction[i]
      self._actuator_specs.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.effort_target


@dataclass(kw_only=True)
class DelayedBuiltinTorqueActuatorCfg(BuiltinTorqueActuatorCfg):
  """Built-in torque actuator config with action delays."""

  delay_min_lag: int = 0
  """Minimum delay lag in timesteps."""
  delay_max_lag: int = 0
  """Maximum delay lag in timesteps."""
  delay_hold_prob: float = 0.0
  """Probability of keeping previous lag when updating."""
  delay_update_period: int = 0
  """Period for updating delays (0 = every step)."""
  delay_per_env_phase: bool = True
  """Whether each environment has a different phase offset."""

  def build(
    self, entity: Entity, joint_ids: list[int], joint_names: list[str]
  ) -> DelayedBuiltinTorqueActuator:
    return DelayedBuiltinTorqueActuator(self, entity, joint_ids, joint_names)


class DelayedBuiltinTorqueActuator(BuiltinTorqueActuator):
  """Built-in torque actuator with action delays.

  Delays effort targets before passing them to the motor.
  """

  def __init__(
    self,
    cfg: DelayedBuiltinTorqueActuatorCfg,
    entity: Entity,
    joint_ids: list[int],
    joint_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, joint_ids, joint_names)
    self._delay_buffer: DelayBuffer | None = None

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)

    cfg = self.cfg
    assert isinstance(cfg, DelayedBuiltinTorqueActuatorCfg)

    self._delay_buffer = DelayBuffer(
      min_lag=cfg.delay_min_lag,
      max_lag=cfg.delay_max_lag,
      batch_size=data.nworld,
      device=device,
      hold_prob=cfg.delay_hold_prob,
      update_period=cfg.delay_update_period,
      per_env_phase=cfg.delay_per_env_phase,
    )

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self._delay_buffer is not None
    self._delay_buffer.append(cmd.effort_target)
    delayed_effort_target = self._delay_buffer.compute()
    return delayed_effort_target

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._delay_buffer is not None:
      self._delay_buffer.reset(env_ids)

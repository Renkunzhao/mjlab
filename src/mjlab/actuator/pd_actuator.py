"""Explicit PD-control actuator."""

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
class IdealPdActuatorCfg(ActuatorCfg):
  """Configuration for ideal PD control actuator."""

  stiffness: float | dict[str, float]
  """PD stiffness (proportional gain)."""
  damping: float | dict[str, float]
  """PD damping (derivative gain)."""
  effort_limit: float | dict[str, float] = float("inf")
  """Maximum force/torque limit."""
  armature: float | dict[str, float] = 0.0
  """Reflected rotor inertia."""
  stiction: float | dict[str, float] = 0.0
  """Joint friction loss."""

  def build(
    self, entity: Entity, joint_ids: list[int], joint_names: list[str]
  ) -> IdealPdActuator:
    return IdealPdActuator(self, entity, joint_ids, joint_names)


class IdealPdActuator(Actuator):
  """Ideal PD control actuator.

  Unlike the builtin MuJoCo PD actuator, this class explicitly computes the torques
  using the PD control law and forwards them to a torque pass-through motor actuator.
  """

  cfg: IdealPdActuatorCfg

  def __init__(
    self,
    cfg: IdealPdActuatorCfg,
    entity: Entity,
    joint_ids: list[int],
    joint_names: list[str],
  ) -> None:
    super().__init__(entity, joint_ids, joint_names)
    self.cfg = cfg
    self.stiffness: torch.Tensor | None = None
    self.damping: torch.Tensor | None = None
    self.force_limit: torch.Tensor | None = None

  def edit_spec(self, spec: mujoco.MjSpec, joint_names: list[str]) -> None:
    # Resolve parameters to per-joint lists.
    armature = resolve_param_to_list(self.cfg.armature, joint_names)
    stiction = resolve_param_to_list(self.cfg.stiction, joint_names)
    effort_limit = resolve_param_to_list(self.cfg.effort_limit, joint_names)

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
      actuator.forcerange[:] = np.array([-effort_limit[i], effort_limit[i]])
      spec.joint(joint_name).armature = armature[i]
      spec.joint(joint_name).frictionloss = stiction[i]
      self._actuator_specs.append(actuator)

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)

    stiffness_list = resolve_param_to_list(self.cfg.stiffness, self._joint_names)
    damping_list = resolve_param_to_list(self.cfg.damping, self._joint_names)
    force_limit_list = resolve_param_to_list(self.cfg.effort_limit, self._joint_names)

    self.stiffness = torch.tensor(stiffness_list, dtype=torch.float, device=device)
    self.damping = torch.tensor(damping_list, dtype=torch.float, device=device)
    self.force_limit = torch.tensor(force_limit_list, dtype=torch.float, device=device)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self.stiffness is not None
    assert self.damping is not None
    assert self.force_limit is not None
    pos_error = cmd.position_target - cmd.joint_pos
    vel_error = cmd.velocity_target - cmd.joint_vel
    computed_torques = self.stiffness * pos_error
    computed_torques += self.damping * vel_error
    computed_torques += cmd.effort_target
    return torch.clamp(computed_torques, -self.force_limit, self.force_limit)


@dataclass(kw_only=True)
class DelayedIdealPdActuatorCfg(IdealPdActuatorCfg):
  """Ideal PD actuator config with action delays."""

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
  ) -> DelayedIdealPdActuator:
    return DelayedIdealPdActuator(self, entity, joint_ids, joint_names)


class DelayedIdealPdActuator(IdealPdActuator):
  """Ideal PD actuator with action delays.

  Delays position targets before computing PD torques.
  """

  def __init__(
    self,
    cfg: DelayedIdealPdActuatorCfg,
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
    assert isinstance(cfg, DelayedIdealPdActuatorCfg)

    num_envs = data.qpos.shape[0]
    self._delay_buffer = DelayBuffer(
      min_lag=cfg.delay_min_lag,
      max_lag=cfg.delay_max_lag,
      batch_size=num_envs,
      device=device,
      hold_prob=cfg.delay_hold_prob,
      update_period=cfg.delay_update_period,
      per_env_phase=cfg.delay_per_env_phase,
    )

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert (
      self.stiffness is not None
      and self.damping is not None
      and self.force_limit is not None
    )
    assert self._delay_buffer is not None

    self._delay_buffer.append(cmd.position_target)
    delayed_position_target = self._delay_buffer.compute()

    pos_error = delayed_position_target - cmd.joint_pos
    vel_error = cmd.velocity_target - cmd.joint_vel
    computed_torques = (
      self.stiffness * pos_error + self.damping * vel_error + cmd.effort_target
    )
    computed_torques = torch.clamp(
      computed_torques, -self.force_limit, self.force_limit
    )
    return computed_torques

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._delay_buffer is not None:
      self._delay_buffer.reset(env_ids)

"""Explicit PD-control actuator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCommand
from mjlab.actuator.utils import resolve_param, resolve_param_tensor
from mjlab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass
class IdealPDActuatorCfg(ActuatorCfg):
  """Configuration for ideal PD control actuator."""

  kp: float | dict[str, float]
  """Proportional gain."""

  kd: float | dict[str, float]
  """Derivative gain."""

  def build(self, entity: Entity) -> IdealPDActuator:
    return IdealPDActuator(self, entity)


class IdealPDActuator(Actuator):
  """Ideal PD control actuator.

  Unlike the built-in MuJoCo PD actuator, this class explicitly computes the torques
  using the PD control law and forwards them to a torque pass-through motor actuator.
  This allows for greater flexibility, such as adding delays or modeling more complex
  actuator dynamics.
  """

  cfg: IdealPDActuatorCfg

  def __init__(self, cfg: IdealPDActuatorCfg, entity: Entity):
    super().__init__(cfg, entity)
    self.kp: torch.Tensor
    self.kd: torch.Tensor
    self.force_limit: torch.Tensor

    self._velocity_delay_buffer: DelayBuffer | None = None
    self._effort_delay_buffer: DelayBuffer | None = None

  def _create_mj_actuator(self, spec: mujoco.MjSpec, target: mujoco.MjsJoint) -> None:
    actuator = spec.add_actuator(name=target.name)
    actuator.target = target.name
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
    actuator.gainprm[0] = 1.0
    actuator.inheritrange = 1.0

    effort = resolve_param(self.cfg.effort_limit, target.name)
    actuator.forcerange[0] = -effort
    actuator.forcerange[1] = effort

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)

    self.kp = resolve_param_tensor(
      self.cfg.kp, self._target_names, default=0.0, device=device
    )
    self.kd = resolve_param_tensor(
      self.cfg.kd, self._target_names, default=0.0, device=device
    )
    self.force_limit = resolve_param_tensor(
      self.cfg.effort_limit, self._target_names, default=float("inf"), device=device
    )

    if self.cfg.max_delay > 0:
      self._velocity_delay_buffer = DelayBuffer(
        min_lag=self.cfg.min_delay,
        max_lag=self.cfg.max_delay,
        batch_size=data.nworld,
        device=device,
      )
      self._effort_delay_buffer = DelayBuffer(
        min_lag=self.cfg.min_delay,
        max_lag=self.cfg.max_delay,
        batch_size=data.nworld,
        device=device,
      )

  def compute(self, command: ActuatorCommand) -> torch.Tensor:
    assert self.kp is not None
    assert self.kd is not None

    pos_target = self._maybe_apply_delay(command.position_target)
    vel_target = command.velocity_target
    eff_target = command.effort_target

    if self._velocity_delay_buffer is not None:
      self._velocity_delay_buffer.append(vel_target)
      vel_target = self._velocity_delay_buffer.compute()
    if self._effort_delay_buffer is not None:
      self._effort_delay_buffer.append(eff_target)
      eff_target = self._effort_delay_buffer.compute()

    kp = self.kp.unsqueeze(0)
    kd = self.kd.unsqueeze(0)

    pos_error = pos_target - command.joint_pos
    vel_error = vel_target - command.joint_vel
    computed_torques = kp * pos_error + kd * vel_error + eff_target

    force_limit = self.force_limit.unsqueeze(0)
    computed_torques = torch.clamp(computed_torques, -force_limit, force_limit)
    return computed_torques

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    buffer_env_ids = None if isinstance(env_ids, slice) else env_ids
    if self._velocity_delay_buffer is not None:
      self._velocity_delay_buffer.reset(buffer_env_ids)
    if self._effort_delay_buffer is not None:
      self._effort_delay_buffer.reset(buffer_env_ids)

"""Generic delayed actuator wrapper.

This module provides a wrapper that adds delay functionality to any actuator.
Delays commands (position, velocity, or effort) by a specified number of physics
timesteps, useful for modeling actuator latency and communication delays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(kw_only=True)
class DelayedActuatorCfg(ActuatorCfg):
  """Configuration for delayed actuator wrapper.

  Wraps any actuator config to add delay functionality. Delays are quantized
  to physics timesteps (not control timesteps). For example, with 500Hz physics
  and 50Hz control (decimation=10), a lag of 2 represents a 4ms delay (2 physics
  steps).
  """

  base_cfg: ActuatorCfg
  """Configuration for the underlying actuator."""

  delay_target: Literal["position", "velocity", "effort"] = "position"
  """Which command target to delay: 'position', 'velocity', or 'effort'."""

  delay_min_lag: int = 0
  """Minimum delay lag in physics timesteps."""

  delay_max_lag: int = 0
  """Maximum delay lag in physics timesteps."""

  delay_hold_prob: float = 0.0
  """Probability of keeping previous lag when updating."""

  delay_update_period: int = 0
  """Period for updating delays in physics timesteps (0 = every step)."""

  delay_per_env_phase: bool = True
  """Whether each environment has a different phase offset."""

  def build(
    self, entity: Entity, joint_ids: list[int], joint_names: list[str]
  ) -> DelayedActuator:
    base_actuator = self.base_cfg.build(entity, joint_ids, joint_names)
    return DelayedActuator(self, base_actuator)


class DelayedActuator(Actuator):
  """Generic wrapper that adds delay to any actuator.

  Delays the specified command target (position, velocity, or effort)
  before passing it to the underlying actuator's compute method.
  """

  def __init__(self, cfg: DelayedActuatorCfg, base_actuator: Actuator) -> None:
    super().__init__(
      base_actuator.entity,
      base_actuator._joint_ids_list,
      base_actuator._joint_names,
    )
    self.cfg = cfg
    self._base_actuator = base_actuator
    self._delay_buffer: DelayBuffer | None = None

  def edit_spec(self, spec: mujoco.MjSpec, joint_names: list[str]) -> None:
    self._base_actuator.edit_spec(spec, joint_names)
    self._mjs_actuators = self._base_actuator._mjs_actuators

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    self._base_actuator.initialize(mj_model, model, data, device)

    self._joint_ids = self._base_actuator._joint_ids
    self._ctrl_ids = self._base_actuator._ctrl_ids

    self._delay_buffer = DelayBuffer(
      min_lag=self.cfg.delay_min_lag,
      max_lag=self.cfg.delay_max_lag,
      batch_size=data.nworld,
      device=device,
      hold_prob=self.cfg.delay_hold_prob,
      update_period=self.cfg.delay_update_period,
      per_env_phase=self.cfg.delay_per_env_phase,
    )

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self._delay_buffer is not None

    if self.cfg.delay_target == "position":
      self._delay_buffer.append(cmd.position_target)
      delayed_target = self._delay_buffer.compute()
      cmd = ActuatorCmd(
        position_target=delayed_target,
        velocity_target=cmd.velocity_target,
        effort_target=cmd.effort_target,
        joint_pos=cmd.joint_pos,
        joint_vel=cmd.joint_vel,
      )
    elif self.cfg.delay_target == "velocity":
      self._delay_buffer.append(cmd.velocity_target)
      delayed_target = self._delay_buffer.compute()
      cmd = ActuatorCmd(
        position_target=cmd.position_target,
        velocity_target=delayed_target,
        effort_target=cmd.effort_target,
        joint_pos=cmd.joint_pos,
        joint_vel=cmd.joint_vel,
      )
    elif self.cfg.delay_target == "effort":
      self._delay_buffer.append(cmd.effort_target)
      delayed_target = self._delay_buffer.compute()
      cmd = ActuatorCmd(
        position_target=cmd.position_target,
        velocity_target=cmd.velocity_target,
        effort_target=delayed_target,
        joint_pos=cmd.joint_pos,
        joint_vel=cmd.joint_vel,
      )

    return self._base_actuator.compute(cmd)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._delay_buffer is not None:
      self._delay_buffer.reset(env_ids)
    self._base_actuator.reset(env_ids)

  def update(self, dt: float) -> None:
    self._base_actuator.update(dt)

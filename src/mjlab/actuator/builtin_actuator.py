"""Actuators that wrap MuJoCo builtin actuators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCommand
from mjlab.actuator.utils import resolve_param

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(kw_only=True)
class BuiltinMotorActuatorCfg(ActuatorCfg):
  """Configuration for torque control actuator."""

  gear: float | dict[str, float] = 1.0
  """Gear ratio for the actuator."""

  def build(self, entity: Entity) -> BuiltinMotorActuator:
    return BuiltinMotorActuator(self, entity)


@dataclass(kw_only=True)
class BuiltinPositionActuatorCfg(ActuatorCfg):
  """Configuration for position control actuator (PD control via MuJoCo)."""

  kp: float | dict[str, float]
  """Proportional gain."""
  kv: float | dict[str, float]
  """Derivative gain."""

  def build(self, entity: Entity) -> BuiltinPositionActuator:
    return BuiltinPositionActuator(self, entity)


@dataclass(kw_only=True)
class BuiltinVelocityActuatorCfg(ActuatorCfg):
  """Configuration for velocity control actuator."""

  kv: float | dict[str, float]
  """Derivative gain."""

  def build(self, entity: Entity) -> BuiltinVelocityActuator:
    return BuiltinVelocityActuator(self, entity)


class BuiltinMotorActuator(Actuator):
  """Motor actuator that passes through effort commands."""

  cfg: BuiltinMotorActuatorCfg

  def _create_mj_actuator(self, spec: mujoco.MjSpec, target: mujoco.MjsJoint) -> None:
    actuator = spec.add_actuator(name=target.name)
    actuator.target = target.name
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
    actuator.gear[0] = resolve_param(self.cfg.gear, target.name)
    actuator.inheritrange = 1.0

    effort = resolve_param(self.cfg.effort_limit, target.name)
    actuator.forcerange[0] = -effort
    actuator.forcerange[1] = effort

  def compute(self, command: ActuatorCommand) -> torch.Tensor:
    return self._maybe_apply_delay(command.effort_target)


class BuiltinPositionActuator(Actuator):
  """Position actuator that uses PD control via MuJoCo builtin."""

  cfg: BuiltinPositionActuatorCfg

  def _create_mj_actuator(self, spec: mujoco.MjSpec, target: mujoco.MjsJoint) -> None:
    actuator = spec.add_actuator(name=target.name)
    actuator.target = target.name
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    actuator.inheritrange = 1.0

    kp = resolve_param(self.cfg.kp, target.name)
    kv = resolve_param(self.cfg.kv, target.name)
    actuator.gainprm[0] = kp
    actuator.biasprm[1] = -kp
    actuator.biasprm[2] = -kv

    effort = resolve_param(self.cfg.effort_limit, target.name)
    actuator.forcerange[0] = -effort
    actuator.forcerange[1] = effort

  def compute(self, command: ActuatorCommand) -> torch.Tensor:
    return self._maybe_apply_delay(command.position_target)


class BuiltinVelocityActuator(Actuator):
  """Velocity actuator that uses velocity control via MuJoCo builtin."""

  cfg: BuiltinVelocityActuatorCfg

  def _create_mj_actuator(self, spec: mujoco.MjSpec, target: mujoco.MjsJoint) -> None:
    actuator = spec.add_actuator(name=target.name)
    actuator.target = target.name
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    actuator.inheritrange = 1.0

    kv = resolve_param(self.cfg.kv, target.name)
    actuator.gainprm[0] = kv
    actuator.biasprm[2] = -kv

    effort = resolve_param(self.cfg.effort_limit, target.name)
    actuator.forcerange[0] = -effort
    actuator.forcerange[1] = effort

  def compute(self, command: ActuatorCommand) -> torch.Tensor:
    return self._maybe_apply_delay(command.velocity_target)

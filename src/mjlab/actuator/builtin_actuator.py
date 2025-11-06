from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.actuator import (
  Actuator,
  ActuatorCfg,
  ActuatorCmd,
  resolve_param_to_list,
)
from mjlab.utils.spec import create_motor_actuator, create_position_actuator

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
      effort_limit = [None] * len(joint_names)

    # Add <position> actuator to spec, one per joint.
    for i, joint_name in enumerate(joint_names):
      actuator = create_position_actuator(
        spec,
        joint_name,
        stiffness=stiffness[i],
        damping=damping[i],
        effort_limit=effort_limit[i],
        armature=armature[i],
        stiction=stiction[i],
      )
      self._actuator_specs.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.position_target


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
      actuator = create_motor_actuator(
        spec,
        joint_name,
        effort_limit=effort_limit[i],
        gear=gear[i],
        armature=armature[i],
        stiction=stiction[i],
      )
      self._actuator_specs.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.effort_target

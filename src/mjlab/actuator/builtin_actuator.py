"""MuJoCo built-in actuators.

This module provides actuators that use MuJoCo's native actuator implementations,
created programmatically via the MjSpec API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.utils.spec import (
  create_motor_actuator,
  create_position_actuator,
  create_velocity_actuator,
)
from mjlab.utils.string import resolve_param_to_list

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(kw_only=True)
class BuiltinPdActuatorCfg(ActuatorCfg):
  """Configuration for MuJoCo built-in PD actuator.

  All parameters can be specified as a single float (broadcast to all joints)
  or a dict mapping joint names/regex patterns to values. When using a dict,
  all patterns must match at least one joint, and each joint must match exactly
  one pattern, or a ValueError will be raised.

  Under the hood, this creates a <position> actuator for each joint and sets
  the stiffness, damping and effort limits accordingly. It also modifies the
  actuated joint's properties, namely armature and frictionloss.
  """

  stiffness: float | dict[str, float]
  """PD proportional gain."""
  damping: float | dict[str, float]
  """PD derivative gain."""
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
    frictionloss = resolve_param_to_list(self.cfg.frictionloss, joint_names)
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
        frictionloss=frictionloss[i],
      )
      self._mjs_actuators.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.position_target


@dataclass(kw_only=True)
class BuiltinTorqueActuatorCfg(ActuatorCfg):
  """Configuration for MuJoCo built-in torque actuator.

  All parameters can be specified as a single float (broadcast to all joints)
  or a dict mapping joint names/regex patterns to values. When using a dict,
  all patterns must match at least one joint, and each joint must match exactly
  one pattern, or a ValueError will be raised.

  Under the hood, this creates a <motor> actuator for each joint and sets
  its effort limit and gear ratio accordingly. It also modifies the actuated
  joint's properties, namely armature and frictionloss.
  """

  effort_limit: float | dict[str, float]
  """Maximum actuator effort."""
  gear: float | dict[str, float] = 1.0
  """Actuator gear ratio."""

  def build(
    self, entity: Entity, joint_ids: list[int], joint_names: list[str]
  ) -> BuiltinTorqueActuator:
    return BuiltinTorqueActuator(self, entity, joint_ids, joint_names)


class BuiltinTorqueActuator(Actuator):
  """MuJoCo built-in torque actuator."""

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
    frictionloss = resolve_param_to_list(self.cfg.frictionloss, joint_names)
    gear = resolve_param_to_list(self.cfg.gear, joint_names)

    # Add <motor> actuator to spec, one per joint.
    for i, joint_name in enumerate(joint_names):
      actuator = create_motor_actuator(
        spec,
        joint_name,
        effort_limit=effort_limit[i],
        gear=gear[i],
        armature=armature[i],
        frictionloss=frictionloss[i],
      )
      self._mjs_actuators.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.effort_target


@dataclass(kw_only=True)
class BuiltinVelocityActuatorCfg(ActuatorCfg):
  """Configuration for MuJoCo built-in velocity actuator.

  All parameters can be specified as a single float (broadcast to all joints)
  or a dict mapping joint names/regex patterns to values. When using a dict,
  all patterns must match at least one joint, and each joint must match exactly
  one pattern, or a ValueError will be raised.

  Under the hood, this creates a <velocity> actuator for each joint and sets
  the damping gain. It also modifies the actuated joint's properties, namely
  armature and frictionloss.
  """

  damping: float | dict[str, float]
  """Damping gain."""
  effort_limit: float | dict[str, float] | None = None
  """Maximum actuator force/torque. If None, no limit is applied."""

  def build(
    self, entity: Entity, joint_ids: list[int], joint_names: list[str]
  ) -> BuiltinVelocityActuator:
    return BuiltinVelocityActuator(self, entity, joint_ids, joint_names)


class BuiltinVelocityActuator(Actuator):
  """MuJoCo built-in velocity actuator."""

  def __init__(
    self,
    cfg: BuiltinVelocityActuatorCfg,
    entity: Entity,
    joint_ids: list[int],
    joint_names: list[str],
  ) -> None:
    super().__init__(entity, joint_ids, joint_names)
    self.cfg = cfg

  def edit_spec(self, spec: mujoco.MjSpec, joint_names: list[str]) -> None:
    # Resolve parameters to per-joint lists.
    damping = resolve_param_to_list(self.cfg.damping, joint_names)
    armature = resolve_param_to_list(self.cfg.armature, joint_names)
    frictionloss = resolve_param_to_list(self.cfg.frictionloss, joint_names)
    if self.cfg.effort_limit is not None:
      effort_limit = resolve_param_to_list(self.cfg.effort_limit, joint_names)
    else:
      effort_limit = [None] * len(joint_names)

    # Add <velocity> actuator to spec, one per joint.
    for i, joint_name in enumerate(joint_names):
      actuator = create_velocity_actuator(
        spec,
        joint_name,
        damping=damping[i],
        effort_limit=effort_limit[i],
        armature=armature[i],
        frictionloss=frictionloss[i],
      )
      self._mjs_actuators.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.velocity_target

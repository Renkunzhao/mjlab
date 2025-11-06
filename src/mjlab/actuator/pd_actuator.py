"""Explicit PD-control actuator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import (
  Actuator,
  ActuatorCfg,
  ActuatorCmd,
  resolve_param_to_list,
)
from mjlab.utils.spec import create_motor_actuator

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
      actuator = create_motor_actuator(
        spec,
        joint_name,
        effort_limit=effort_limit[i],
        armature=armature[i],
        stiction=stiction[i],
      )
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

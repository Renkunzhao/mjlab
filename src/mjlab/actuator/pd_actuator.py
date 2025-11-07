"""Ideal torque-controlled actuator model.

This module provides a PD controller that computes torques explicitly
using the standard PD control law: τ = Kp*(q_target - q) + Kd*(v_target - v).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.utils.spec import create_motor_actuator
from mjlab.utils.string import resolve_param_to_list

if TYPE_CHECKING:
  from mjlab.entity import Entity

IdealPdCfgT = TypeVar("IdealPdCfgT", bound="IdealPdActuatorCfg")


@dataclass(kw_only=True)
class IdealPdActuatorCfg(ActuatorCfg):
  """Configuration for ideal PD actuator.

  All parameters can be specified as a single float (broadcast to all joints)
  or a dict mapping joint names/regex patterns to values. When using a dict,
  all patterns must match at least one joint, and each joint must match exactly
  one pattern, or a ValueError will be raised.
  """

  stiffness: float | dict[str, float]
  """PD stiffness (proportional gain)."""
  damping: float | dict[str, float]
  """PD damping (derivative gain)."""
  effort_limit: float | dict[str, float] = float("inf")
  """Maximum force/torque limit."""
  armature: float | dict[str, float] = 0.0
  """Reflected rotor inertia."""
  frictionloss: float | dict[str, float] = 0.0
  """Friction loss force limit.

  Applies a constant friction force opposing motion, independent of load or velocity.
  Acts as a constraint that opposes motion before it starts, rather than a velocity-
  dependent damping force. Also known as dry friction or load-independent friction.
  """

  def build(
    self, entity: Entity, joint_ids: list[int], joint_names: list[str]
  ) -> IdealPdActuator:
    return IdealPdActuator(self, entity, joint_ids, joint_names)


class IdealPdActuator(Actuator, Generic[IdealPdCfgT]):
  """Ideal PD control actuator."""

  def __init__(
    self,
    cfg: IdealPdCfgT,
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
    frictionloss = resolve_param_to_list(self.cfg.frictionloss, joint_names)
    effort_limit = resolve_param_to_list(self.cfg.effort_limit, joint_names)

    # Add <motor> actuator to spec, one per joint.
    for i, joint_name in enumerate(joint_names):
      actuator = create_motor_actuator(
        spec,
        joint_name,
        effort_limit=effort_limit[i],
        armature=armature[i],
        frictionloss=frictionloss[i],
      )
      self._mjs_actuators.append(actuator)

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

    num_envs = data.nworld
    self.stiffness = (
      torch.tensor(stiffness_list, dtype=torch.float, device=device)
      .unsqueeze(0)
      .expand(num_envs, -1)
      .clone()
    )
    self.damping = (
      torch.tensor(damping_list, dtype=torch.float, device=device)
      .unsqueeze(0)
      .expand(num_envs, -1)
      .clone()
    )
    self.force_limit = (
      torch.tensor(force_limit_list, dtype=torch.float, device=device)
      .unsqueeze(0)
      .expand(num_envs, -1)
      .clone()
    )

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self.stiffness is not None
    assert self.damping is not None

    pos_error = cmd.position_target - cmd.joint_pos
    vel_error = cmd.velocity_target - cmd.joint_vel

    computed_torques = self.stiffness * pos_error
    computed_torques += self.damping * vel_error
    computed_torques += cmd.effort_target

    return self._clip_effort(computed_torques)

  def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
    assert self.force_limit is not None
    return torch.clamp(effort, -self.force_limit, self.force_limit)

  def set_gains(
    self,
    env_ids: torch.Tensor | slice,
    kp: torch.Tensor | None = None,
    kd: torch.Tensor | None = None,
  ) -> None:
    """Set PD gains for specified environments.

    Args:
      env_ids: Environment indices to update.
      kp: New proportional gains. Shape: (num_envs, num_actuators) or (num_envs,).
      kd: New derivative gains. Shape: (num_envs, num_actuators) or (num_envs,).
    """
    assert self.stiffness is not None
    assert self.damping is not None

    if kp is not None:
      if kp.ndim == 1:
        kp = kp.unsqueeze(-1)
      self.stiffness[env_ids] = kp

    if kd is not None:
      if kd.ndim == 1:
        kd = kd.unsqueeze(-1)
      self.damping[env_ids] = kd

  def set_effort_limit(
    self, env_ids: torch.Tensor | slice, effort_limit: torch.Tensor
  ) -> None:
    """Set effort limits for specified environments.

    Args:
      env_ids: Environment indices to update.
      effort_limit: New effort limits. Shape: (num_envs, num_actuators) or (num_envs,).
    """
    assert self.force_limit is not None

    if effort_limit.ndim == 1:
      effort_limit = effort_limit.unsqueeze(-1)
    self.force_limit[env_ids] = effort_limit

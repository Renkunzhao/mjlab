from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class HoppingCommand(CommandTerm):
  cfg: HoppingCommandCfg

  def __init__(self, cfg: HoppingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.asset_name]

    self.height = torch.zeros(self.num_envs, device=self.device)

    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.height.unsqueeze(-1)
  
  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_height"] += (
      torch.abs(self.height - self.robot.data.root_link_pos_w[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.height[env_ids] = r.uniform_(*self.cfg.ranges.height)

  def _update_command(self) -> None:
    # Height command stays constant between resamplings
    pass

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows.

    Note: Only visualizes the selected environment (visualizer.env_idx).
    """
    batch = visualizer.env_idx

    if batch >= self.num_envs:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()

    base_pos_w = base_pos_ws[batch]
    cmd_height = cmds[batch]

    # Skip if robot appears uninitialized (at origin).
    if np.linalg.norm(base_pos_w) < 1e-6:
      return

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    # Command height arrow (green).
    cmd_h_from = base_pos_w.copy()
    cmd_h_from[2] += z_offset
    cmd_h_to = cmd_h_from.copy()
    cmd_h_to[2] += cmd_height
    visualizer.add_arrow(
        cmd_h_from, cmd_h_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
    )

    # Actual height arrow (blue).
    act_h_from = cmd_h_from.copy()
    act_h_to = cmd_h_from.copy()
    act_h_to[2] += base_pos_w[2]  # 当前高度
    visualizer.add_arrow(
        act_h_from, act_h_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
    )

@dataclass(kw_only=True)
class HoppingCommandCfg(CommandTermCfg):
  asset_name: str
  class_type: type[CommandTerm] = HoppingCommand

  @dataclass
  class Ranges:
    height: tuple[float, float]

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

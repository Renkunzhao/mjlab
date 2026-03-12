"""Convert MimicKit Go2 .pkl motion files to .npz format for mjlab tracking."""

import pickle
from typing import Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.go2.env_cfgs import (
  unitree_go2_flat_tracking_env_cfg,
)
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_from_angle_axis,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

# MimicKit Go2 joint order (depth-first from go2.xml).
GO2_JOINT_NAMES = (
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
)


def _exp_map_to_quat(exp_map: torch.Tensor) -> torch.Tensor:
  """Convert 3D exponential map to quaternion (wxyz).

  Args:
    exp_map: Shape (N, 3). Vector direction = axis, magnitude = angle.

  Returns:
    Quaternion (w, x, y, z). Shape (N, 4).
  """
  angle = torch.norm(exp_map, dim=-1)  # (N,)
  eps = 1e-8
  axis = exp_map / (angle.unsqueeze(-1) + eps)  # (N, 3)
  return quat_from_angle_axis(angle, axis)


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    output_fps: int,
    device: torch.device | str,
  ):
    self.motion_file = motion_file
    self.output_fps = output_fps
    self.output_dt = 1.0 / output_fps
    self.current_idx = 0
    self.device = device
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    with open(self.motion_file, "rb") as f:
      data = pickle.load(f)

    self.input_fps: int = data["fps"]
    self.input_dt = 1.0 / self.input_fps
    frames: np.ndarray = data["frames"]  # (T, 18)
    self.input_frames = frames.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

    frames_t = torch.from_numpy(frames).to(torch.float32).to(self.device)
    self.motion_base_poss_input = frames_t[:, :3]  # (T, 3)
    exp_map = frames_t[:, 3:6]  # (T, 3)
    self.motion_base_rots_input = _exp_map_to_quat(exp_map)  # (T, 4) wxyz
    self.motion_dof_poss_input = frames_t[:, 6:]  # (T, 12)

  def _interpolate_motion(self):
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_times = times
    self.output_frames = times.shape[0]
    idx0, idx1, blend = self._compute_frame_blend(times)

    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[idx0],
      self.motion_base_poss_input[idx1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[idx0],
      self.motion_base_rots_input[idx1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[idx0],
      self.motion_dof_poss_input[idx1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion loaded: {self.input_frames} frames @ {self.input_fps} fps → "
      f"{self.output_frames} frames @ {self.output_fps} fps "
      f"({self.duration:.2f}s)"
    )

  def _lerp(self, a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return a * (1 - t) + b * t

  def _slerp(self, a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(a)
    for i in range(a.shape[0]):
      out[i] = quat_slerp(a[i], b[i], float(t[i]))
    return out

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = times / self.duration
    idx0 = (phase * (self.input_frames - 1)).floor().long()
    idx1 = torch.minimum(idx0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - idx0
    return idx0, idx1, blend

  def _compute_velocities(self):
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    q_prev = self.motion_base_rots[:-2]
    q_next = self.motion_base_rots[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * self.output_dt)
    self.motion_base_ang_vels = torch.cat([omega[:1], omega, omega[-1:]], dim=0)

  def get_next_state(self):
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset = self.current_idx >= self.output_frames
    if reset:
      self.current_idx = 0
    return state, reset


def run_sim(
  sim: Simulation,
  scene: Scene,
  input_file: str,
  output_fps: int,
  output_name: str,
  render: bool,
  renderer: OffscreenRenderer | None,
):
  motion = MotionLoader(
    motion_file=input_file,
    output_fps=output_fps,
    device=sim.device,
  )

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(GO2_JOINT_NAMES, preserve_order=True)[0]

  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }

  frames = []
  scene.reset()
  file_saved = False

  pbar = tqdm(
    total=motion.output_frames,
    desc="Processing frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  while not file_saved:
    (
      (
        base_pos,
        base_rot,
        base_lin_vel,
        base_ang_vel,
        dof_pos,
        dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = base_rot
    root_states[:, 7:10] = base_lin_vel
    root_states[:, 10:] = base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = dof_pos
    joint_vel[:, robot_joint_indexes] = dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())

    frame_count += 1
    pbar.update(1)
    if frame_count % 100 == 0:
      pbar.set_description(f"Processing frames (t={frame_count / output_fps:.1f}s)")

    if reset_flag:
      file_saved = True
      pbar.close()

      print("\nStacking arrays...")
      for k in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      ):
        log[k] = np.stack(log[k], axis=0)

      print("Saving to /tmp/motion.npz...")
      np.savez("/tmp/motion.npz", **log)

      print("Uploading to Weights & Biases...")
      import wandb

      run = wandb.init(project="pkl_to_npz_go2", name=output_name)
      print(f"[INFO]: Logging motion to wandb: {output_name}")
      REGISTRY = "motions"
      artifact = run.log_artifact(
        artifact_or_path="/tmp/motion.npz",
        name=output_name,
        type=REGISTRY,
      )
      run.link_artifact(
        artifact=artifact,
        target_path=f"wandb-registry-{REGISTRY}/{output_name}",
      )
      print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{output_name}")

      if render and frames:
        import mediapy as media

        print("Creating video...")
        media.write_video("./motion.mp4", frames, fps=output_fps)
        print("Logging video to wandb...")
        wandb.log({"motion_video": wandb.Video("./motion.mp4", format="mp4")})

      wandb.finish()


def main(
  input_file: str,
  output_name: str,
  output_fps: int = 50,
  device: str = "cuda:0",
  render: bool = False,
):
  """Convert a MimicKit Go2 .pkl motion file to .npz and upload to wandb.

  Args:
    input_file: Path to the input .pkl file.
    output_name: Artifact name for wandb registry.
    output_fps: Desired output frame rate (default 50 Hz).
    device: Compute device.
    render: Whether to render a video and upload to wandb.
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA unavailable. Falling back to CPU.")
    device = "cpu"

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(unitree_go2_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if render:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      distance=2.0,
      elevation=-5.0,
      azimuth=20,
    )
    renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
    renderer.initialize()

  run_sim(
    sim=sim,
    scene=scene,
    input_file=input_file,
    output_fps=output_fps,
    output_name=output_name,
    render=render,
    renderer=renderer,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)

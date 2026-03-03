Context

Create a new hopping task for Unitree Go2 in mjlab, closely following the go2_jump implementation
from IsaacGym (/home/rkz/code/Isaacgym/src/My_unitree_go2_gym/legged_gym/envs/Go2_MoB/GO2_JUMP/).
The task trains the robot to jump/hop using a gait-phase-synchronized reward, placed in
/home/rkz/code/mjlab/src/mjlab/tasks/hopping/.

---
File Structure to Create

src/mjlab/tasks/hopping/
├── __init__.py                  (empty, triggers auto-import)
├── hopping_env_cfg.py           (make_hopping_env_cfg() factory)
├── mdp/
│   ├── __init__.py              (re-exports like velocity/mdp/__init__.py)
│   ├── observations.py          (gait_phase_encoding, euler_xyz, gait_stance_mask, joint_pos_abs)
│   └── rewards.py               (7 new reward functions)
└── config/
    └── go2/
        ├── __init__.py          (task registration)
        ├── env_cfgs.py          (unitree_go2_hopping_flat_env_cfg)
        └── rl_cfg.py            (unitree_go2_hopping_ppo_runner_cfg)

---
1. hopping/mdp/observations.py — New Functions

def gait_phase_encoding(env, cycle_time: float) -> torch.Tensor:
    # Returns [B, 2]: [sin(2π·phase), cos(2π·phase)]
    # phase = (episode_length_buf * step_dt / cycle_time) % 1.0
    phase = (env.episode_length_buf * env.step_dt / cycle_time) % 1.0
    return torch.stack([torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)], dim=1)

def euler_xyz(env, asset_cfg) -> torch.Tensor:
    # Returns [B, 3]: roll, pitch, yaw from root quaternion
    # Computed via quat_to_euler_xyz from mjlab.utils.lab_api.math (if available)
    # or via manual quaternion decomposition

def gait_stance_mask(env, cycle_time: float) -> torch.Tensor:
    # Returns [B, 2]: stance_mask[:, 0] = phase < 0.6, stance_mask[:, 1] = phase > 0.6
    phase = (env.episode_length_buf * env.step_dt / cycle_time) % 1.0
    return torch.stack([(phase < 0.6).float(), (phase > 0.6).float()], dim=1)

def joint_pos_abs(env, asset_cfg) -> torch.Tensor:
    # Returns [B, N]: raw joint positions (without default subtraction)
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]

2. hopping/mdp/rewards.py — New Functions

Matching go2_jump reward logic:

┌─────────────────────────────────────────────────────────────────┬─────────────────────────────┬───────┐
│                            Function                             │     go2_jump equivalent     │ Scale │
├─────────────────────────────────────────────────────────────────┼─────────────────────────────┼───────┤
│ jump_gait_sync(env, sensor_name, command_name, cycle_time)      │ _reward_jump                │ 2.0   │
├─────────────────────────────────────────────────────────────────┼─────────────────────────────┼───────┤
│ default_hip_pos(env, asset_cfg)                                 │ _reward_default_hip_pos     │ 0.3   │
├─────────────────────────────────────────────────────────────────┼─────────────────────────────┼───────┤
│ jump_feet_clearance(env, asset_cfg, cycle_time, command_name)   │ _reward_feet_clearance      │ 0.5   │
├─────────────────────────────────────────────────────────────────┼─────────────────────────────┼───────┤
│ lin_vel_z_reward(env, asset_cfg)                                │ _reward_lin_vel_z           │ 0.05  │
├─────────────────────────────────────────────────────────────────┼─────────────────────────────┼───────┤
│ ang_vel_xy_reward(env, asset_cfg)                               │ _reward_ang_vel_xy          │ 0.2   │
├─────────────────────────────────────────────────────────────────┼─────────────────────────────┼───────┤
│ base_height_reward(env, target_height, command_name, asset_cfg) │ _reward_base_height         │ 1.0   │
├─────────────────────────────────────────────────────────────────┼─────────────────────────────┼───────┤
│ default_pos_penalty(env, asset_cfg)                             │ _reward_default_pos         │ -0.1  │
├─────────────────────────────────────────────────────────────────┼─────────────────────────────┼───────┤
│ foot_contact_force_penalty(env, sensor_name, max_force)         │ _reward_feet_contact_forces │ -0.01 │
└─────────────────────────────────────────────────────────────────┴─────────────────────────────┴───────┘

Key implementations:
- jump_gait_sync: contact[:,0]==contact[:,1]==contact[:,2]==contact[:,3]==stance_mask[:,0], active when cmd_norm>0.2
- default_hip_pos: hip joints = regex ".*_hip_joint.*", exp(-4 * sum(|hip_pos|))
- jump_feet_clearance: swing_mask = (phase>=0.6), clamp(feet_z-0.02, 0, 0.05) * swing_mask, active when cmd_norm>0.2
- base_height_reward: exp(-10*|z-0.3|) when cmd_norm<0.2

3. hopping/hopping_env_cfg.py — Factory Function

make_hopping_env_cfg() builds observation/reward/command config:

Actor observations (matches go2_jump's 47-dim, minus frame stack handled by runner):
- gait_phase: gait_phase_encoding(cycle_time=1.5) → 2
- command: generated_commands("twist") → 3
- base_ang_vel: builtin_sensor("robot/imu_ang_vel") → 3
- euler_xyz: new euler_xyz() → 3  (replaces projected_gravity from velocity)
- joint_pos: joint_pos_rel → 12
- joint_vel: joint_vel_rel → 12
- actions: last_action → 12

Critic observations (actor + privileged):
- All actor terms +
- base_lin_vel: builtin_sensor("robot/imu_lin_vel") → 3
- gait_stance_mask: gait_stance_mask(cycle_time=1.5) → 2
- foot_contact: foot_contact("feet_ground_contact") → 4
- foot_contact_forces: foot_contact_forces("feet_ground_contact") → 12
- joint_pos_abs: joint_pos_abs → 12

Rewards (go2_jump mapping):

┌────────────────────────┬────────────────────────────────────────────────┬─────────┐
│       Reward key       │                Function source                 │ Weight  │
├────────────────────────┼────────────────────────────────────────────────┼─────────┤
│ track_linear_velocity  │ velocity.mdp.rewards                           │ 2.0     │
├────────────────────────┼────────────────────────────────────────────────┼─────────┤
│ track_angular_velocity │ velocity.mdp.rewards                           │ 2.0     │
├────────────────────────┼────────────────────────────────────────────────┼─────────┤
│ lin_vel_z              │ hopping.mdp.rewards.lin_vel_z_reward           │ 0.05    │
├────────────────────────┼────────────────────────────────────────────────┼─────────┤
│ ang_vel_xy             │ hopping.mdp.rewards.ang_vel_xy_reward          │ 0.2     │
├────────────────────────┼────────────────────────────────────────────────┼─────────┤
│ orientation            │ velocity.mdp.rewards.flat_orientation          │ 0.6     │
├────────────────────────┼────────────────────────────────────────────────┼─────────┤
│ torques                │ envs.mdp.rewards.joint_torques_l2              │ -0.0002 │
├────────────────────────┼────────────────────────────────────────────────┼─────────┤
│ dof_acc                │ envs.mdp.rewards.joint_acc_l2                  │ -5.5e-4 │
├────────────────────────┼────────────────────────────────────────────────┼─────────┤
│ base_height            │ hopping.mdp.rewards.base_height_reward         │ 1.0     │
├────────────────────────┼────────────────────────────────────────────────┼─────────┤
│ air_time               │ velocity.mdp.rewards.feet_air_time             │ 1.0     │
├────────────────────────┼────────────────────────────────────────────────┼─────────┤
│ action_rate            │ envs.mdp.rewards.action_rate_l2                │ -0.01   │
├────────────────────────┼────────────────────────────────────────────────┼─────────┤
│ default_pos            │ hopping.mdp.rewards.default_pos_penalty        │ -0.1    │
├────────────────────────┼────────────────────────────────────────────────┼─────────┤
│ default_hip_pos        │ hopping.mdp.rewards.default_hip_pos            │ 0.3     │
├────────────────────────┼────────────────────────────────────────────────┼─────────┤
│ foot_contact_forces    │ hopping.mdp.rewards.foot_contact_force_penalty │ -0.01   │
├────────────────────────┼────────────────────────────────────────────────┼─────────┤
│ jump                   │ hopping.mdp.rewards.jump_gait_sync             │ 2.0     │
├────────────────────────┼────────────────────────────────────────────────┼─────────┤
│ feet_clearance         │ hopping.mdp.rewards.jump_feet_clearance        │ 0.5     │
├────────────────────────┼────────────────────────────────────────────────┼─────────┤
│ dof_pos_limits         │ envs.mdp.rewards.joint_pos_limits              │ -1.0    │
└────────────────────────┴────────────────────────────────────────────────┴─────────┘

Commands: UniformVelocityCommandCfg with heading_command=False, ranges:
- lin_vel_x: (-1.0, 1.0), lin_vel_y: (-1.0, 1.0), ang_vel_z: (-1.0, 1.0)
- resampling_time_range: (5.0, 5.0) (fixed 5s like go2_jump)

Terminations: time_out + illegal_contact (trunk touches ground)

Flat terrain (no terrain generator, no rayscanner).

4. hopping/config/go2/env_cfgs.py

unitree_go2_hopping_flat_env_cfg(play=False):
- Start from make_hopping_env_cfg()
- Add robot: get_go2_robot_cfg()
- Add feet_ground_contact ContactSensor (same geoms as velocity: FR_foot_collision etc.)
- Set illegal_contact termination using trunk_ground_contact
- Set Go2 joint names for hip reward: ".*_hip_joint.*"
- Set site_names for foot observations: ("FR", "FL", "RR", "RL")
- Action scale: GO2_ACTION_SCALE (0.25 * effort/stiffness per joint)
- Viewer: trunk, distance=1.5, elevation=-10

5. hopping/config/go2/rl_cfg.py

unitree_go2_hopping_ppo_runner_cfg():
- Same network architecture as velocity: [512, 256, 128] hidden dims, elu activation
- learning_rate = 1e-4 (go2_jump uses 1e-4 vs velocity's 1e-3)
- max_iterations = 15_000
- num_steps_per_env = 24
- save_interval = 100
- experiment_name = "go2_hopping"
- Uses default VelocityOnPolicyRunner (ONNX export)

6. hopping/config/go2/__init__.py

register_mjlab_task(
    task_id="Mjlab-Hopping-Flat-Unitree-Go2",
    env_cfg=unitree_go2_hopping_flat_env_cfg(),
    play_env_cfg=unitree_go2_hopping_flat_env_cfg(play=True),
    rl_cfg=unitree_go2_hopping_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

---
Critical Files to Read During Implementation

- /home/rkz/code/mjlab/src/mjlab/tasks/velocity/config/go2/env_cfgs.py — reference for Go2-specific setup
- /home/rkz/code/mjlab/src/mjlab/tasks/velocity/mdp/rewards.py — reuse track_linear_velocity etc.
- /home/rkz/code/mjlab/src/mjlab/envs/mdp/rewards.py — reuse joint_torques_l2, joint_acc_l2, etc.
- /home/rkz/code/mjlab/src/mjlab/tasks/velocity/mdp/observations.py — reuse foot_contact, foot_height
- /home/rkz/code/mjlab/src/mjlab/tasks/velocity/config/go2/rl_cfg.py — reference for RL runner config
- /home/rkz/code/Isaacgym/src/My_unitree_go2_gym/legged_gym/envs/Go2_MoB/GO2_JUMP/go2_jump_env.py — all reward functions

---
Verification

# Type check
uv run ty check src/mjlab/tasks/hopping/

# Verify task is registered
uv run python -c "import mjlab.tasks; from mjlab.tasks.registry import list_tasks; print([t for t in list_tasks() if 'Hopping' in t])"

# Check env config loads without error
uv run python -c "from mjlab.tasks.registry import load_env_cfg; cfg = load_env_cfg('Mjlab-Hopping-Flat-Unitree-Go2'); print('obs:', list(cfg.observations['actor'].terms.keys()))"

Summary

Created Mjlab-Hopping-Flat-Unitree-Go2 task at /home/rkz/code/mjlab/src/mjlab/tasks/hopping/:

New files:
hopping/
├── __init__.py
├── hopping_env_cfg.py          — make_hopping_env_cfg() factory
├── mdp/
│   ├── __init__.py
│   ├── observations.py         — gait_phase_encoding, euler_xyz, gait_stance_mask, joint_pos_abs
│   └── rewards.py              — 8 new reward functions
└── config/go2/
    ├── __init__.py             — task registration
    ├── env_cfgs.py             — unitree_go2_hopping_flat_env_cfg()
    └── rl_cfg.py               — PPO config (lr=1e-4, 15k iterations)

Key design aligned with go2_jump:
- Gait phase: (episode_length_buf * step_dt / 1.5s) % 1.0 — phase<0.6 = stance, phase≥0.6 = swing
- Jump reward: All 4 feet must synchronize contact state matching the gait phase
- Actor obs (47 dims): [sin/cos_phase(2), cmd(3), ang_vel(3), euler_xyz(3), joint_pos(12), joint_vel(12), actions(12)]
- Learning rate: 1e-4 (matching go2_jump, lower than velocity's 1e-3)
- Episode length: 24s, Cycle time: 1.5s

Reused from velocity task: track_linear_velocity, track_angular_velocity, flat_orientation, feet_air_time, contact sensors setup, domain randomization events.
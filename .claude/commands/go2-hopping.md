Create or update the `Mjlab-Hopping-Flat-Unitree-Go2` task in `/home/rkz/code/mjlab/src/mjlab/tasks/hopping/`.

The task trains Unitree Go2 to hop/jump using a gait-phase-synchronized reward, closely following
the go2_jump implementation at
`/home/rkz/code/Isaacgym/src/My_unitree_go2_gym/legged_gym/envs/Go2_MoB/GO2_JUMP/`.

## File structure

```
src/mjlab/tasks/hopping/
├── __init__.py
├── hopping_env_cfg.py           make_hopping_env_cfg() factory
├── mdp/
│   ├── __init__.py
│   ├── observations.py          gait_phase_encoding, euler_xyz, gait_stance_mask, joint_pos_abs
│   └── rewards.py               jump_gait_sync, default_hip_pos, jump_feet_clearance,
│                                lin_vel_z_reward, ang_vel_xy_reward, base_height_reward,
│                                default_pos_penalty, foot_contact_force_penalty,
│                                jump_air_time (stateful), body_contact_penalty
└── config/go2/
    ├── __init__.py              registers Mjlab-Hopping-Flat-Unitree-Go2
    ├── env_cfgs.py              unitree_go2_hopping_flat_env_cfg(play=False)
    └── rl_cfg.py                unitree_go2_hopping_ppo_runner_cfg()
```

## Key design decisions

**Gait phase** (`cycle_time = 1.5s`):
- `phase = (episode_length_buf * step_dt / 1.5) % 1.0`
- `phase < 0.6` → stance (all 4 feet on ground)
- `phase >= 0.6` → swing (all 4 feet in air)

**Actor observations** (47 dims × 10 frames = 470 dims, matches go2_jump frame_stack=10):
`gait_phase(2) + command(3) + base_ang_vel(3) + euler_xyz(3) + joint_pos(12) + joint_vel(12) + actions(12)`
Set via `ObservationGroupCfg(history_length=10)`.

**Critic adds** (3-frame history, matches go2_jump c_frame_stack=3):
`base_lin_vel(3) + gait_stance_mask(2) + foot_contact(4) + foot_contact_forces(12) + joint_pos_abs(12)`
Set via `ObservationGroupCfg(history_length=3)`.

**Reward weights** (from go2_jump config):

| key | weight |
|---|---|
| track_linear_velocity | 2.0 |
| track_angular_velocity | 2.0 |
| jump (jump_gait_sync) | 2.0 |
| base_height | 1.0 |
| air_time | 1.0 |
| orientation | 0.6 |
| ang_vel_xy | 0.2 |
| default_hip_pos | 0.3 |
| feet_clearance | 0.5 |
| lin_vel_z | 0.05 |
| torques | -0.0002 |
| dof_acc | -5.5e-4 |
| action_rate | -0.01 |
| foot_contact_forces | -0.01 |
| default_pos | -0.1 |
| dof_pos_limits | -1.0 |
| collision (thigh/calf) | -1.0 |

**RL config**: lr=1e-4, max_iterations=15000, num_steps_per_env=24, experiment_name="go2_hopping"

## Known gotchas

- **Termination vs penalty**: `trunk` contact → termination (`illegal_contact`). Thigh/calf
  contact → penalty (`collision` reward, weight=-1.0, sensor `thigh_calf_contact`). The thigh/calf
  sensor uses `ContactMatch(mode="body", pattern=r"(FL|FR|RL|RR)_(thigh|calf)")`.
  Do NOT terminate on thigh/calf contact — that's excessive.

- **`air_time` reward**: uses `mdp.jump_air_time` (stateful class), NOT `vel_mdp.feet_air_time`.
  Triggers once per landing event; reward = `(air_time - 0.5)` per foot. Negative for short hops
  (<0.5s), positive for genuine jumps (>0.5s). Matches go2_jump's `_reward_feet_air_time`.
  IMPORTANT: must reset `prev_air_time` at episode start (`episode_length_buf <= 1`), otherwise
  the stale air time from the previous episode causes a spurious ~-2 penalty every episode reset.

- **ContactSensorCfg reduce**: valid values are `'none', 'mindist', 'maxforce', 'netforce'`.
  `"any"` is not valid — use `"none"` for the trunk sensor (`illegal_contact` uses `torch.any()` internally).

- **`dof_acc` must NOT use `joint_acc_l2`**: mjlab's `joint_acc_l2` reads MuJoCo's raw `qacc`
  (forward dynamics), which can be NaN or extremely large during contact impacts in early training.
  This causes NaN advantages → NaN gradients → `std < 0` crash in RSL-RL.
  Use `hopping.mdp.rewards.joint_vel_diff_l2` instead — it computes `sum((vel_t - vel_{t-1})²)`,
  matching go2_jump's actual `_reward_dof_acc` implementation.

- **`euler_xyz` NaN on CUDA (Blackwell / sm_120)**: `euler_xyz_from_quat` uses `torch.where` +
  `torch.asin`. On CUDA, both branches of `torch.where` are evaluated before selection; if the
  quaternion has floating-point drift (`|sin_pitch| > 1`), `asin` returns NaN which poisons the
  output even when `torch.where` should select the other branch. Fix already applied in
  `hopping/mdp/observations.py`: normalize the quaternion with `F.normalize` before conversion,
  then wrap with `torch.nan_to_num(..., nan=0.0)` as a safety net.

- **Commands**: `heading_command=False` (go2_jump has no heading command).

- **Action scale**: use `GO2_ACTION_SCALE` (per-joint `0.25 * effort_limit / stiffness`), same as velocity task.

## Verification

```sh
uv run python -c "
import mjlab.tasks
from mjlab.tasks.registry import list_tasks, load_env_cfg
print([t for t in list_tasks() if 'Hopping' in t])
cfg = load_env_cfg('Mjlab-Hopping-Flat-Unitree-Go2')
print('actor:', list(cfg.observations['actor'].terms.keys()))
print('rewards:', list(cfg.rewards.keys()))
"
uv run ty check src/mjlab/tasks/hopping/
```

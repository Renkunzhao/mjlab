# Actuators

Actuators convert high-level commands (position, velocity, effort) into
low-level control signals that drive joints. Implementations use either
built-in actuators (physics engine computes torques and integrates damping
forces implicitly) or explicit actuators (user computes torques explicitly,
integrator cannot account for their velocity derivatives).

## Choosing an Actuator Type

**Built-in actuators** (`BuiltinPositionActuator`, `BuiltinVelocityActuator`): Use
MuJoCo's native implementations. The physics engine computes torques and
integrates damping forces implicitly, providing the best numerical stability. 

**Explicit actuators** (`IdealPdActuator`, `DcMotorActuator`): User computes
torques explicitly so the simulator cannot account for velocity derivatives.
Use when you need custom control laws or actuator dynamics that
can't be expressed with built-in types (e.g., velocity-dependent torque
limits, learned actuator networks).

**XML actuators** (`XmlPositionActuator`, `XmlMotorActuator`,
`XmlVelocityActuator`): Wrap actuators already defined in your robot's XML
file.

**Delayed actuators** (`DelayedActuator`): Generic wrapper that adds command
delays to any actuator type. Use for modeling communication latency.

## TL;DR

**Basic PD control:**

```python
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg

robot_cfg = EntityCfg(
  spec_fn=lambda: load_robot_spec(),
  articulation=EntityArticulationInfoCfg(
    actuators=(
      BuiltinPositionActuatorCfg(
        joint_names_expr=[".*_hip_.*", ".*_knee_.*"],
        stiffness=80.0,
        damping=10.0,
        effort_limit=100.0,
      ),
    ),
  ),
)
```

**Different parameters for different joint groups:**

```python
# Use separate actuator configs for different joint groups
actuators=(
  BuiltinPositionActuatorCfg(
    joint_names_expr=[".*_hip_.*"],
    stiffness=120.0,
    damping=10.0,
  ),
  BuiltinPositionActuatorCfg(
    joint_names_expr=[".*_knee_.*"],
    stiffness=80.0,
    damping=10.0,
  ),
  BuiltinPositionActuatorCfg(
    joint_names_expr=[".*_ankle_.*"],
    stiffness=60.0,
    damping=10.0,
  ),
)
```

**Add delays:**

```python
from mjlab.actuator import DelayedActuatorCfg, BuiltinPositionActuatorCfg

DelayedActuatorCfg(
  joint_names_expr=[".*"],
  base_cfg=BuiltinPositionActuatorCfg(
    joint_names_expr=[".*"],
    stiffness=80.0,
    damping=10.0,
  ),
  delay_target="position",
  delay_min_lag=2,  # Minimum 2 physics steps
  delay_max_lag=5,  # Maximum 5 physics steps
)
```

## Actuator Interface

All actuators implement a unified `compute()` interface that receives an
`ActuatorCmd` (containing position, velocity, and effort targets) and returns
control signals. The abstraction provides lifecycle hooks for model
modification, initialization, reset, and runtime updates.

**Core interface:**

```python
def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
  """Convert high-level commands to control signals.

  Args:
    cmd: Command containing position_target, velocity_target, effort_target
         (each is a [num_envs, num_joints] tensor or None)

  Returns:
    Control signals for this actuator ([num_envs, num_joints] tensor)
  """
```

**Lifecycle hooks:**

- **`edit_spec`**: Modify MjSpec before compilation (add actuators, set gains)
- **`initialize`**: Post-compilation setup (resolve indices, allocate buffers)
- **`reset`**: Per-environment reset logic
- **`update`**: Pre-step updates
- **`compute`**: Convert commands to control signals

## Actuator Types

### Built-in Actuators

Use MuJoCo's native actuator types via the MjSpec API. The physics engine
computes the control law and integrates velocity-dependent damping forces
implicitly, providing best numerical stability.

**BuiltinPositionActuator**: Creates `<position>` actuators for PD control.

**BuiltinVelocityActuator**: Creates `<velocity>` actuators for velocity control.

**BuiltinMotorActuator**: Creates `<motor>` actuators for direct torque control.

```python
from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg

# Mobile manipulator: PD for arm joints, velocity control for wheels.
actuators = (
  BuiltinPositionActuatorCfg(
    joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
    stiffness=100.0,
    damping=10.0,
    effort_limit=150.0,
  ),
  BuiltinVelocityActuatorCfg(
    joint_names_expr=[".*_wheel_.*"],
    damping=20.0,
    effort_limit=50.0,
  ),
)
```

### Explicit Actuators

User computes torques explicitly. This enables custom control laws and actuator
dynamics that can't be expressed with built-in types (e.g., velocity-dependent torque
limits, learned actuator networks).

> **⚠️ Stability warning**: Explicit actuators may be less numerically stable
> than built-in actuators because the integrator cannot account for the
> velocity derivatives of the control forces, especially with high damping
> gains. See [Numerical Stability: Built-in vs.
> Explicit](#numerical-stability-built-in-vs-explicit) for details.

**IdealPdActuator**: Base class that implements an ideal PD controller.

**DcMotorActuator**: Example of a more realistic actuator model built on top
of `IdealPdActuator`. Adds velocity-dependent torque saturation to model DC
motor torque-speed curves (back-EMF effects). It implements a linear
torque-speed curve: maximum torque at zero velocity, zero torque at maximum
velocity.

```python
from mjlab.actuator import IdealPdActuatorCfg, DcMotorActuatorCfg

# Ideal PD for hips, DC motor model with torque-speed curve for knees.
actuators = (
  IdealPdActuatorCfg(
    joint_names_expr=[".*_hip_.*"],
    stiffness=80.0,
    damping=10.0,
    effort_limit=100.0,
  ),
  DcMotorActuatorCfg(
    joint_names_expr=[".*_knee_.*"],
    stiffness=80.0,
    damping=10.0,
    effort_limit=25.0,       # Continuous torque limit
    saturation_effort=50.0,  # Peak torque at stall
    velocity_limit=30.0,     # No-load speed (rad/s)
  ),
)
```

**DcMotorActuator parameters:**

- **`saturation_effort`**: Peak motor torque at zero velocity (stall torque)
- **`velocity_limit`**: Maximum motor velocity (no-load speed, *rad/s*)
- **`effort_limit`**: Continuous torque limit (from base class)

The actuator computes torque limits based on current joint velocity. At zero
velocity, it can produce full `saturation_effort`. At `velocity_limit`, it
produces zero torque. Between these points, torque varies linearly. The
`effort_limit` further constrains output below the torque-speed curve.

### XML Actuators

Wrap actuators already defined in your robot's XML file. The config finds
existing actuators by matching their `target` joint name against the
`joint_names_expr` patterns. Each joint must have exactly one matching
actuator.

**XmlPositionActuator**: Wraps existing `<position>` actuators

**XmlVelocityActuator**: Wraps existing `<velocity>` actuators

**XmlMotorActuator**: Wraps existing `<motor>` actuators

```python
from mjlab.actuator import XmlPositionActuatorCfg

# Robot XML already has:
# <actuator>
#   <position name="hip_actuator" joint="hip_joint" kp="100"/>
# </actuator>

# Wrap existing XML actuators by joint name.
actuators = (
  XmlPositionActuatorCfg(
    joint_names_expr=["hip_joint", "knee_joint"],
  ),
)
```

### Delayed Actuator

Generic wrapper that adds command delays to any actuator. Useful for modeling
actuator latency and communication delays. The delay operates on command
targets before they reach the actuator's control law.

```python
from mjlab.actuator import DelayedActuatorCfg, IdealPdActuatorCfg

# Add 2-5 step delay to position commands.
actuators = (
  DelayedActuatorCfg(
    joint_names_expr=[".*"],
    base_cfg=IdealPdActuatorCfg(
      joint_names_expr=[".*"],
      stiffness=80.0,
      damping=10.0,
    ),
    delay_target="position",     # Delay position commands
    delay_min_lag=2,
    delay_max_lag=5,
    delay_hold_prob=0.3,         # 30% chance to keep previous lag
    delay_update_period=10,      # Update lag every 10 steps
  ),
)
```

**Multi-target delays:**

For actuators that use multiple command targets (like `IdealPdActuator`), you
can delay all targets together:

```python
DelayedActuatorCfg(
  joint_names_expr=[".*"],
  base_cfg=IdealPdActuatorCfg(...),
  delay_target=("position", "velocity", "effort"),
  delay_min_lag=2,
  delay_max_lag=5,
)
```

Delays are quantized to physics timesteps. For example, with 500Hz physics
(2ms/step), `delay_min_lag=2` represents a 4ms minimum delay.

> **Note**: Each target gets an independent delay buffer with its own lag
> schedule. This provides maximum flexibility for modeling different latency
> characteristics for position, velocity, and effort commands.

---

## Built-in (Implicit) vs. Explicit PD Control

**BuiltinPositionActuator** uses MuJoCo's internal PD implementation:

- Creates `<position>` actuators in the MjSpec
- Physics engine computes the PD law and integrates the velocity-dependent
  damping force (−Kd·v) implicitly

**IdealPdActuator** implements PD control explicitly:

- Creates `<motor>` actuators in the MjSpec
- Computes torques explicitly: `τ = Kp·pos_error + Kd·vel_error` and writes
  them to `data.ctrl`
- The integrator cannot account for the velocity derivatives of these forces

They match closely in the linear, unconstrained regime and small time steps.
However, built-in PD is more numerically robust and as such can be used with
larger gains and larger timesteps.

---

## Configuration

### Multiple Actuator Configs for Different Joint Groups

Since actuator parameters are uniform within each config, use separate
actuator configs for joints that need different parameters:

```python
from mjlab.actuator import BuiltinPositionActuatorCfg

# G1 humanoid with different gains per joint group.
G1_ACTUATORS = (
  BuiltinPositionActuatorCfg(
    joint_names_expr=[".*_hip_.*", "waist_yaw_joint"],
    stiffness=180.0,
    damping=18.0,
    effort_limit=88.0,
    armature=0.0015,
  ),
  BuiltinPositionActuatorCfg(
    joint_names_expr=["left_hip_pitch_joint", "right_hip_pitch_joint"],
    stiffness=200.0,
    damping=20.0,
    effort_limit=88.0,
    armature=0.0015,
  ),
  BuiltinPositionActuatorCfg(
    joint_names_expr=[".*_knee_joint"],
    stiffness=150.0,
    damping=15.0,
    effort_limit=139.0,
    armature=0.0025,
  ),
  BuiltinPositionActuatorCfg(
    joint_names_expr=[".*_ankle_.*"],
    stiffness=40.0,
    damping=5.0,
    effort_limit=25.0,
    armature=0.0008,
  ),
)
```

---

## Usage

### Action Terms

Actuators are typically controlled via action terms in the action manager:

```python
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import ActionTermCfg

# In your environment config:
ActionTermCfg(
  cls=JointPositionActionCfg,
  asset_name="robot",
  actuator_names=[".*"],  # Regex patterns for joint selection
  scale=1.0,
  offset=0.0,
)
```

**Available action terms:**

- **`JointPositionAction`**: Sets position targets (for PD actuators)
- **`JointVelocityAction`**: Sets velocity targets (for velocity actuators)
- **`JointEffortAction`**: Sets effort/torque targets (for torque actuators)

The action manager calls `entity.set_joint_position_target()`,
`set_joint_velocity_target()`, or `set_joint_effort_target()` under the hood,
which populate the `ActuatorCmd` passed to each actuator's `compute()` method.

### Domain Randomization

```python
from mjlab.envs.mdp import events
from mjlab.managers.manager_term_config import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

# In your environment config:
EventTermCfg(
  func=events.randomize_pd_gains,
  mode="reset",
  params={
    "asset_cfg": SceneEntityCfg("robot", actuator_names=[".*"]),
    "kp_range": (0.8, 1.2),  # Scale existing gains by 0.8x to 1.2x
    "kd_range": (0.8, 1.2),
    "distribution": "uniform",
    "operation": "scale",  # or "abs" for absolute values
  },
)

EventTermCfg(
  func=events.randomize_effort_limits,
  mode="reset",
  params={
    "asset_cfg": SceneEntityCfg("robot", actuator_names=[".*_leg_.*"]),
    "effort_limit_range": (0.7, 1.0),  # Reduce effort by 0-30%
    "operation": "scale",
  },
)
```

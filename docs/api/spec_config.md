# Spec Configuration Classes

## Overview

The spec configuration system provides a declarative, pattern-based interface
for configuring MuJoCo models. Define configurations once and apply them to
multiple joints or geoms using regex patterns, rather than writing loops to
modify individual elements.

Configuration classes are applied during `Entity` initialization, after your
custom spec function (if provided) returns the base spec.

## Quick Start

```python
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg, Entity
from mjlab.utils.spec_config import CollisionCfg

robot_cfg = EntityCfg(
  spec_fn=get_robot_spec,
  collisions=(
    CollisionCfg(
      geom_names_expr=[".*_collision"],
      contype=1,
      conaffinity=1
    ),
  )
)

# Configuration is automatically applied during entity creation.
robot = Entity(robot_cfg)
```

---

## Configuration Classes

### CollisionCfg

Controls collision properties for geoms using pattern matching. Uses regex
patterns to select geoms, then applies collision attributes. For dict-based
parameters, the first matching pattern wins for each geom.

#### Real-World Examples

**Go1 Quadruped - Feet Only Collision:**
```python
# From go1_constants.py
_foot_regex = "^[FR][LR]_foot_collision$"

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=[_foot_regex],  # Only match foot collision geoms.
  contype=0, conaffinity=1,       # Disable self-collisions.
  condim=3,
  priority=1,                     # Higher priority for foot contacts.
  friction=(0.6,),                # Sliding friction coefficient.
  solimp=(0.9, 0.95, 0.023),      # Solver impedance parameters.
)
```

**G1 Humanoid - Full Collision with special foot configuration:**
```python
# From g1_constants.py
FULL_COLLISION = CollisionCfg(
  geom_names_expr=[".*_collision"],  # Match ALL collision geoms.
  condim={
    r"^(left|right)_foot[1-7]_collision$": 3,
    ".*_collision": 1,
  },
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={
    r"^(left|right)_foot[1-7]_collision$": (0.6,)  # Custom friction for feet.
  }
)
```

**G1 Humanoid - No Self-Collision:**
```python
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=[".*_collision"],
  contype=0,
  conaffinity=1,
  condim={
    r"^(left|right)_foot[1-7]_collision$": 3,
    ".*_collision": 1,
  },
  priority={
    r"^(left|right)_foot[1-7]_collision$": 1
  },
  friction={
    r"^(left|right)_foot[1-7]_collision$": (0.6,)
  }
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `geom_names_expr` | `list[str]` | *required* | Regex patterns to match geom names |
| `contype` | `int` or `dict[str, int]` | `1` | Collision type bitmask |
| `conaffinity` | `int` or `dict[str, int]` | `1` | Collision affinity bitmask |
| `condim` | `int` or `dict[str, int]` | `3` | Contact dimensions: `1`, `3`, `4`, `6` |
| `priority` | `int` or `dict[str, int]` | `0` | Collision priority |
| `friction` | `tuple` or `dict[str, tuple]` | `None` | Friction coefficients (sliding, torsional, rolling) |
| `solref` | `tuple` or `dict[str, tuple]` | `None` | Solver reference parameters |
| `solimp` | `tuple` or `dict[str, tuple]` | `None` | Solver impedance parameters |
| `disable_other_geoms` | `bool` | `True` | Disables collisions for all non-matching geoms |

For a detailed explanation of the above collision parameters,
see the [MuJoCo Contact Documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html#contact).

---

### SensorCfg

Adds sensors to measure physical quantities.

#### Examples

**IMU sensor:**
```python
SensorCfg(
  name="base_imu",
  sensor_type="accelerometer",
  objtype="body",
  objname="base_link"
)
```

**Gyroscope:**
```python
SensorCfg(
  name="base_gyro",
  sensor_type="gyro",
  objtype="body",
  objname="base_link"
)
```

#### Available Sensor Types

| Type | Measures | Dimensions |
|------|----------|------------|
| `accelerometer` | Linear acceleration | 3 |
| `gyro` | Angular velocity | 3 |
| `velocimeter` | Linear velocity | 3 |
| `framequat` | Orientation (quaternion) | 4 |
| `framepos` | Position | 3 |
| `framelinvel` | Linear velocity | 3 |
| `frameangvel` | Angular velocity | 3 |
| `upvector` | Up vector | 3 |
| `framezaxis` | Up vector | 3 |
| `subtreeangmom` | Angular momentum | 3 |

---

### ContactSensorCfg

Detects and measures contact forces between objects.

#### Examples

**Any contact with a body:**
```python
ContactSensorCfg(
  name="hand_contacts",
  body1="hand"
)
```

**Contact between two specific geoms:**
```python
ContactSensorCfg(
  name="gripper_object",
  geom1="gripper_left",
  geom2="object"
)
```

**Self-collisions within a subtree:**
```python
ContactSensorCfg(
  name="arm_self_collision",
  subtree1="left_arm",
  subtree2="left_arm"
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Sensor name |
| **Primary object (exactly one):** ||||
| `geom1` | `str` | `None` | Primary geom name |
| `body1` | `str` | `None` | Primary body name |
| `subtree1` | `str` | `None` | Primary subtree name |
| `site` | `str` | `None` | Site volume (requires secondary object) |
| **Secondary object (optional, at most one):** ||||
| `geom2` | `str` | `None` | Secondary geom name |
| `body2` | `str` | `None` | Secondary body name |
| `subtree2` | `str` | `None` | Secondary subtree name |
| **Options:** ||||
| `num` | `int` | `1` | Max contacts to track |
| `data` | `tuple` | `("found",)` | Data to extract: `"found"`, `"force"`, `"torque"`, `"dist"`, `"pos"`, `"normal"`, `"tangent"` |
| `reduce` | `str` | `"none"` | Combine method: `"none"`, `"mindist"`, `"maxforce"`, `"netforce"` |

---

### Visual Elements

#### TextureCfg

Creates textures for materials.

```python
TextureCfg(
  name="checker",
  type="2d",
  builtin="checker",
  rgb1=(0.2, 0.3, 0.4),
  rgb2=(0.8, 0.8, 0.8),
  width=256,
  height=256
)
```

#### MaterialCfg

Defines surface materials with optional textures.

```python
MaterialCfg(
  name="rubber",
  texuniform=True,
  texrepeat=(2, 2),
  reflectance=0.5,
  texture="checker"  # References texture by name.
)
```

#### LightCfg

Adds lighting to the scene.

```python
LightCfg(
  name="spotlight",
  body="world",
  type="spot",
  pos=(2.0, 2.0, 3.0),
  dir=(0.0, 0.0, -1.0),
  cutoff=45.0
)
```

#### CameraCfg

Adds a camera.

```python
CameraCfg(
  name="front_cam",
  # Adds the camera to the world body.
  body="world",
  fovy=60.0,
  pos=(3.0, 0.0, 1.5),
  quat=(0.924, 0.383, 0.0, 0.0)
)
```

---

## Custom Spec Functions

For modifications beyond what the configuration classes support, provide a
custom `spec_fn` that returns a modified `MjSpec`. Configuration classes are
applied after your custom function runs.

### Basic Custom Spec Function

```python
def get_custom_spec() -> mujoco.MjSpec:
  """Create a spec with custom modifications."""
  spec = mujoco.MjSpec.from_file("robot.xml")

  # Custom modifications.
  for geom in spec.geoms:
    if "foot" in geom.name:
      geom.friction = (0.8, 0.1, 0.005)
      geom.condim = 4

  return spec

robot_cfg = EntityCfg(
  spec_fn=get_custom_spec,
  collisions=(CollisionCfg(...),),  # Applied after custom function.
)
```

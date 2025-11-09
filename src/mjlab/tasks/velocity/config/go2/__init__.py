import gymnasium as gym

from .flat_env_cfg import (
  UNITREE_GO2_FLAT_ENV_CFG as UNITREE_GO2_FLAT_ENV_CFG,
)
from .flat_env_cfg import (
  UNITREE_GO2_FLAT_ENV_CFG_PLAY as UNITREE_GO2_FLAT_ENV_CFG_PLAY,
)
from .rough_env_cfg import (
  UNITREE_GO2_ROUGH_ENV_CFG as UNITREE_GO2_ROUGH_ENV_CFG,
)
from .rough_env_cfg import (
  UNITREE_GO2_ROUGH_ENV_CFG_PLAY as UNITREE_GO2_ROUGH_ENV_CFG_PLAY,
)

gym.register(
  id="Mjlab-Velocity-Rough-Unitree-Go2",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": UNITREE_GO2_ROUGH_ENV_CFG,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo2PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Rough-Unitree-Go2-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": UNITREE_GO2_ROUGH_ENV_CFG_PLAY,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo2PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Unitree-Go2",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": UNITREE_GO2_FLAT_ENV_CFG,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo2PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Unitree-Go2-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": UNITREE_GO2_FLAT_ENV_CFG_PLAY,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo2PPORunnerCfg",
  },
)

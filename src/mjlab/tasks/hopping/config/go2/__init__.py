import gymnasium as gym

from .env_cfgs import UNITREE_GO2_FLAT_ENV_CFG

gym.register(
  id="Mjlab-Hopping-Flat-Unitree-Go2",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": UNITREE_GO2_FLAT_ENV_CFG,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo2PPORunnerCfg",
  },
)

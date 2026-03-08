import utils

import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Record ALOHA data")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
from isaaclab.envs import ManagerBasedEnvCfg, ManagerBasedEnv
from aloha_env_cfg import AlohaMiniActionManagerCfg, AlohaMiniSceneCfg, AlohaMiniObservationManagerCfg
from isaaclab.utils import configclass

@configclass
class AlohaMiniEnvCfg(ManagerBasedEnvCfg):
    """The master configuration for the AlohaMini Environment."""
    scene: AlohaMiniSceneCfg = AlohaMiniSceneCfg(num_envs=1, env_spacing=2.0)
    actions: AlohaMiniActionManagerCfg = AlohaMiniActionManagerCfg() 
    observations: AlohaMiniObservationManagerCfg = AlohaMiniObservationManagerCfg()
    
    # Define execution physics steps
    decimation = 2 # Control frequency vs Simulation frequency


def main():
    # 1. Initialize environment
    env_cfg = AlohaMiniEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    # 2. Reset the environment
    obs, _ = env.reset()

    # 3. The Simulation Loop
    while simulation_app.is_running():
        # --> If using Isaac Lab Teleop, your device manager generates this tensor
        dummy_actions = torch.zeros((env.num_envs, 14), device=env.device)
        
        # Pass actions to the environment
        obs, _, _, _, _ = env.step(dummy_actions)

if __name__ == "__main__":
    main()
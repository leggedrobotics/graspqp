# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from tensordict import TensorDict  # noqa
from isaaclab.app import AppLauncher
from graspqp_isaaclab.parser_utils import add_default_args, parse_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser = add_default_args(parser)

parser.set_defaults(assets=[""])
parser.add_argument("--show", type=bool, default=True, help="Show the environment.")
parser.add_argument("--min_evals", type=int, default=2, help="Minimum number of evaluations.")
parser.add_argument("--step", type=int, default=-1, help="Minimum number of evaluations.")
parser.add_argument("--force_value", type=float, default=0, help="Force value for the grasps.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parse_args(parser)
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# fix all seeds
import random
import torch
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

"""Rest everything follows."""
from graspqp_isaaclab.env_registry import get_env_cfg

import gymnasium as gym

from graspqp_isaaclab.agents.static import StaticShowGraspAgent

import os

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from graspqp_isaaclab.utils.data import resolve_assets

from graspqp_isaaclab.agents.multi_agent import MultiAgentWrapper


import os
import torch
import numpy as np

import omni.kit.commands

print("========================================")
print("Running Grasp Environment for the Following Configurations:")
print("========================================")

if len(args_cli.assets) == 0:
    raise ValueError("No assets provided.")

for idx, asset in enumerate(args_cli.assets):
    print(f"Asset {idx}     : {asset}")
    print(f"Prediction {idx}: {args_cli.prediction_files[idx]}")
    print(f"USD {idx}       : {args_cli.usd_files[idx]}")


def main():
    """Random actions agent with Isaac Lab environment."""

    # create environment
    env_cfg, asset_mapping = get_env_cfg(args_cli, collapse_grippers=True)
    env_cfg.scene.env_spacing = 0.45

    if args_cli.show:
        env_cfg.sim.physx.gpu_collision_stack_size = 2**29

    if args_cli.force_value is not None:
        print("Modifying force value to:", args_cli.force_value)
        events = env_cfg.events
        for attr in vars(events):
            value = getattr(events, attr)
            try:
                if "update_cfg" in value.params:
                    value.params["update_cfg"].modifier.params["max_force"] = args_cli.force_value
                    print("Updated:", attr)
            except AttributeError:
                print("Skipping:", attr)

    print("Creating environment for task:", args_cli.task)
    env = gym.make(args_cli.task, cfg=env_cfg)

    print("========================================")
    print("Num Grasps per Env:", args_cli.n_grasps_per_env)

    asset_mapping = torch.tensor(asset_mapping, device=env.unwrapped.device)

    # resolve assets
    (poses, joint_positions, closing_vel, energies), _ = resolve_assets(
        env.unwrapped.scene["robot_0"], args_cli.prediction_files, num_grasps=args_cli.n_grasps_per_env, use_fps=True
    )
    agents = []
    for idx, (pose, joint_position, vel, energy) in enumerate(zip(poses, joint_positions, closing_vel, energies)):
        env_ids = slice(idx, idx + 1)
        if args_cli.static_show:
            vel *= 0.0
            pass

        agents.append(
            StaticShowGraspAgent(
                env.unwrapped,
                pose.to(env.unwrapped.device),
                joint_position.to(env.unwrapped.device),
                [env.unwrapped.scene[f"robot_{i}"] for i in range(args_cli.n_grasps_per_env)],
                env_ids=env_ids,
                closing_vel=vel.to(env.unwrapped.device),
            )
        )
    agent = MultiAgentWrapper(agents, asset_mapping)

    env = RslRlVecEnvWrapper(env)

    # reset environment
    agent.reset()
    env.unwrapped.reset()

    while simulation_app.is_running() and not agent.finished():
        with torch.no_grad():
            observations, rewards, dones, infos = env.step(agent.get_actions())

        o = infos["observations"] if "observations" in infos else observations
        terminated = infos["time_outs"]
        agent.update_envs(o, rewards.unsqueeze(-1))

        if dones.any():
            dones = dones > 0
            finished = terminated[dones]
            agent.reset_envs(torch.where(dones)[0], finished)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

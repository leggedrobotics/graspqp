# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher
import os
import wandb

from tensordict import TensorDict
from graspqp_isaaclab.parser_utils import add_default_args, parse_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser = add_default_args(parser)

parser.set_defaults(assets=[""])
parser.add_argument("--show", action="store_true", help="Show the environment.")
parser.add_argument("--min_evals", type=int, default=2, help="Minimum number of evaluations.")
parser.add_argument("--step", type=int, default=-1, help="Minimum number of evaluations.")

parser.add_argument("--log_to_wandb", action="store_true", help="Log to wandb.")
parser.add_argument("--wandb_project", type=str, default="grasp_eval", help="Wandb project name.")
parser.add_argument("--wandb_name", type=str, help="Wandb run name.", default=None)
parser.add_argument("--force_value", type=float, default=None, help="Force value for the grasps.")


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

if args_cli.log_to_wandb:
    print("logging to wandb")
    wandb.init(project=args_cli.wandb_project, name=args_cli.wandb_name, config=args_cli)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


"""Rest everything follows."""
from graspqp_isaaclab.env_registry import get_env_cfg

import gymnasium as gym
import isaaclab.sim as sim_utils

# from isaaclab.ui.components.component import Component


import os

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from graspqp_isaaclab.utils.data import resolve_assets
from graspqp_isaaclab.agents.static import StaticGraspAgent
from graspqp_isaaclab.agents.eval import AgentEvalWrapper
from graspqp_isaaclab.agents.multi_agent import MultiAgentWrapper


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
    env_cfg, asset_mapping = get_env_cfg(args_cli)

    if args_cli.show:
        env_cfg.sim.physx.gpu_collision_stack_size = 2**29

    if args_cli.force_value is not None:
        print("Modifying force value to:", args_cli.force_value)
        events = env_cfg.events
        for attr in vars(events):
            value = getattr(events, attr)
            try:
                if "max_force" in value.params and "reset" not in attr:
                    value.params["max_force"] = args_cli.force_value
                    continue
            except AttributeError:
                print("Skipping:", attr)

    env = gym.make(args_cli.task, cfg=env_cfg)

    suffix = ""
    if args_cli.force_value is not None:
        suffix = f"_force_{args_cli.force_value}"

    print("========================================")
    print("Num Grasps per Env:", args_cli.n_grasps_per_env)

    asset_mapping = torch.tensor(asset_mapping, device=env.unwrapped.device)

    # resolve assets
    (poses, joint_positions, closing_vel, energies), _ = resolve_assets(
        env.unwrapped.scene["robot"], args_cli.prediction_files, num_grasps=args_cli.n_grasps_per_env
    )

    agents = []
    for idx, (pose, joint_position, vel, energy) in enumerate(zip(poses, joint_positions, closing_vel, energies)):
        env_ids = slice(idx * args_cli.n_grasps_per_env, (idx + 1) * args_cli.n_grasps_per_env)
        agents.append(
            StaticGraspAgent(
                env.unwrapped,
                pose.to(env.unwrapped.device),
                joint_position.to(env.unwrapped.device),
                env.unwrapped.scene["robot"],
                env_ids=env_ids,
                closing_vel=vel.to(env.unwrapped.device),
                energy=energy.to(env.unwrapped.device),
            )
        )
    agent = MultiAgentWrapper(agents, asset_mapping)

    # Spawn the agent
    agent = AgentEvalWrapper(
        agent,
        asset_mapping=asset_mapping,
        min_evals=args_cli.min_evals if not args_cli.show else 1e9,
        output_folders=[os.path.dirname(f) for f in args_cli.prediction_files],
    )

    env = RslRlVecEnvWrapper(env)

    # reset environment
    agent.reset()
    env.unwrapped.reset()

    with torch.no_grad():
        observations, rewards, dones, infos = env.step(agent.get_actions())

    step = 0
    while simulation_app.is_running() and not agent.finished(
        suffix if args_cli.step == -1 else f"_{suffix}_step_{args_cli.step}"
    ):
        with torch.no_grad():
            observations, rewards, dones, infos = env.step(agent.get_actions())
            # This is for older isaac sim version
            # TODO
            if "observations" in infos:
                observations = {"object_pos": infos["observations"]}
            # observations = {"object_pos": infos["observations"]}

        step += 1
        terminated = infos["time_outs"]
        agent.update_envs(observations["object_pos"], rewards.unsqueeze(-1).clone())

        if dones.any():
            dones = dones > 0
            finished = terminated[dones].clone()
            agent.reset_envs(torch.where(dones)[0], finished)

    df = agent.get_statistics()
    df["total_grasps"] = df["Trials"] * 0 + args_cli.n_grasps_per_env
    number_fields_mask = df.map(lambda x: isinstance(x, (int, float))).all()

    if args_cli.log_to_wandb:
        print("logging to wandb")
        for idx, column in enumerate(df.columns[number_fields_mask]):
            wandb.log({f"eval_statistics/{column}": float(df[column].values[-1])}, commit=True)

        wandb.log({"eval_statistics": wandb.Table(dataframe=df)}, commit=True)

    # mean value
    print("Mean values:")
    print(df.iloc[-1])
    print("waiting for 4 seconds to sync with wandb")
    print("closing the environment")
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

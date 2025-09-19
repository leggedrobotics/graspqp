# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from tensordict import TensorDict
from isaaclab.app import AppLauncher
import os
import wandb
from graspqp_isaaclab.parser_utils import add_default_args, parse_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser = add_default_args(parser)

parser.set_defaults(assets=[""])
parser.add_argument("--capture", action="store_true", help="Capture an image.")
parser.add_argument("--capture_folder", type=str, default=None, help="Capture folder.")
parser.add_argument("--capture_name", type=str, default=None, help="Capture name.")
parser.add_argument("--show", type=bool, default=True, help="Show the environment.")
parser.add_argument("--min_evals", type=int, default=2, help="Minimum number of evaluations.")
parser.add_argument("--step", type=int, default=-1, help="Minimum number of evaluations.")

parser.add_argument("--log_to_wandb", action="store_true", help="Log to wandb.")
parser.add_argument("--wandb_project", type=str, default="grasp_eval", help="Wandb project name.")
parser.add_argument("--wandb_name", type=str, help="Wandb run name.", default=None)
parser.add_argument("--force_value", type=float, default=0, help="Force value for the grasps.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parse_args(parser)
if args_cli.capture_folder is None:
    args_cli.capture_folder = os.path.join(args_cli.data_path, "captures")
if args_cli.capture_name is None:
    args_cli.capture_name = f"{args_cli.grasp_iteration}_{args_cli.energy_type}.png"

# launch omniverse app
app_launcher = (
    AppLauncher(args_cli)
    if not args_cli.capture
    else AppLauncher(args_cli, renderer="PathTracing", samples_per_pixel_per_frame=16)
)
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

from graspqp_isaaclab.agents.static import StaticShowGraspAgent

import os

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from graspqp_isaaclab.utils.data import resolve_assets

from graspqp_isaaclab.agents.multi_agent import MultiAgentWrapper


import os
import re
from datetime import datetime
import itertools
import torch
import numpy as np
from tqdm import tqdm

import omni.kit.commands
import omni.usd
import omni.physics.tensors.impl.api as physx
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdPhysics, Gf, Sdf

# from omni.isaac.core.utils.extensions import enable_extension

# from omni.isaac.lab.assets import Articulation
# from omni.isaac.lab.utils.assets import check_file_path


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
    # print("Creating environment with the following configuration:", env_cfg)
    print("Creating environment for task:", args_cli.task)
    env = gym.make(args_cli.task, cfg=env_cfg)
    # grasp_tracker = env.unwrapped.scene["grasp_tracker"]
    # grasp_tracker.register_assets(env.unwrapped.scene["robot"], env.unwrapped.scene["obj"])

    # debug_vis_component: Component = env.unwrapped.scene["eval_visualizer"]
    suffix = ""
    if args_cli.force_value is not None:
        suffix = f"_force_{args_cli.force_value}"
    if args_cli.mined:
        suffix += "_mined"

    # print("========================================")
    # print("Environment Configuration:")
    # print("========================================")
    # print(env_cfg)
    print("========================================")
    print("Num Grasps per Env:", args_cli.n_grasps_per_env)

    asset_mapping = torch.tensor(asset_mapping, device=env.unwrapped.device)

    # resolve assets
    # resolve assets
    (poses, joint_positions, closing_vel, energies), _ = resolve_assets(
        env.unwrapped.scene["robot_0"], args_cli.prediction_files, num_grasps=args_cli.n_grasps_per_env, use_fps=True
    )
    agents = []
    for idx, (pose, joint_position, vel, energy) in enumerate(zip(poses, joint_positions, closing_vel, energies)):
        env_ids = slice(idx, idx + 1)
        print("Creating agent for env_ids:", env_ids)
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

    # # Spawn the agent
    # agent = AgentEvalWrapper(
    #     agent,
    #     asset_mapping=asset_mapping,
    #     min_evals=args_cli.min_evals if not args_cli.show else 100000,
    #     output_folders=[os.path.dirname(f) for f in args_cli.prediction_files],
    # )

    # debug_vis_component.set_debug_vis_callback(lambda x: agent.debug_vis_callback())
    # debug_vis_component.set_debug_vis_toggle_callback(lambda x: agent.set_debug_vis(x))

    env = RslRlVecEnvWrapper(env)

    # reset environment
    agent.reset()
    env.unwrapped.reset()

    if args_cli.capture:
        capture_helper = sim_utils.ScreenCaptureHelper()

    while simulation_app.is_running() and not agent.finished():
        with torch.no_grad():
            observations, rewards, dones, infos = env.step(agent.get_actions())

        o = infos["observations"] if "observations" in infos else observations
        terminated = infos["time_outs"]
        agent.update_envs(o, rewards.unsqueeze(-1))

        if args_cli.capture:
            if capture_helper.completed():
                print("Saved capture to: ", os.path.join(args_cli.capture_folder, args_cli.capture_name))
                exit()
            else:
                if not capture_helper.processing():
                    if not os.path.exists(args_cli.capture_folder):
                        os.makedirs(args_cli.capture_folder)
                    capture_helper.capture(os.path.join(args_cli.capture_folder, args_cli.capture_name))

        if dones.any():
            dones = dones > 0
            finished = terminated[dones]
            agent.reset_envs(torch.where(dones)[0], finished)

        # recorder.record_state()
        # recorder.save(animation_frequency=int(1))
        # exit()

    # while simulation_app.is_running() and not agent.finished(
    #     suffix if args_cli.step == -1 else f"_{suffix}_step_{args_cli.step}"
    # ):
    #     with torch.no_grad():
    #         observations, rewards, dones, infos = env.step(agent.get_actions())

    #     o = infos["observations"]
    #     terminated = infos["time_outs"]
    #     agent.update_envs(o, rewards.unsqueeze(-1).clone())

    #     if args_cli.capture:
    #         if capture_helper.completed():
    #             print("Saved capture to: ", os.path.join(args_cli.capture_folder, args_cli.capture_name))
    #             exit()
    #         else:
    #             if not capture_helper.processing():
    #                 if not os.path.exists(args_cli.capture_folder):
    #                     os.makedirs(args_cli.capture_folder)
    #                 capture_helper.capture(os.path.join(args_cli.capture_folder, args_cli.capture_name))

    #     if dones.any():
    #         dones = dones > 0
    #         finished = terminated[dones]
    #         agent.reset_envs(torch.where(dones)[0], finished)

    # df = agent.get_statistics()
    # df["total_grasps"] = df["Trials"] * 0 + args_cli.n_grasps_per_env
    # number_fields_mask = df.map(lambda x: isinstance(x, (int, float))).all()
    # if args_cli.log_to_wandb:
    #     print("logging to wandb")
    #     for idx, column in enumerate(df.columns[number_fields_mask]):
    #         wandb.log({f"eval_statistics/{column}": float(df[column].values[-1])}, commit=True)

    #     wandb.log({"eval_statistics": wandb.Table(dataframe=df)}, commit=True)
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

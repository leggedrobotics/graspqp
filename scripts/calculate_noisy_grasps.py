# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""
Based on Dexgraspnet: https://pku-epic.github.io/DexGraspNet/
"""

import os

from matplotlib.pyplot import step
import numpy as np
import torch
import plotly.graph_objects as go
from graspqp.hands import get_hand_model, AVAILABLE_HANDS
from graspqp.core import ObjectModel

import argparse
import glob
import trimesh
from graspqp.utils.transforms import robust_compute_rotation_matrix_from_ortho6d

import roma

torch.manual_seed(1)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def _show_dir(
    dir, args, device, noise_levels=50, max_noise_translation=0.1, max_noise_rotation=0.2, max_noise_joints=0.5
):
    data_path = os.path.join(dir, args.hand_name)

    glob_pattern = os.path.join(data_path, args.num_contacts, args.energy, args.grasp_type, args.filter_expr)
    print(f"Loading from {glob_pattern}")
    if len(glob.glob(glob_pattern, recursive=True)) == 0:
        print(f"No files found for pattern {glob_pattern}")
        return None
    checkpoint_path = sorted(glob.glob(glob_pattern, recursive=True), key=os.path.getmtime)[-1]
    # print in green color
    print(f"\033[92mLoading {checkpoint_path}\033[0m")

    # print(f"Loading Files from {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path)

    hand_model = get_hand_model(
        args.hand_name, args.device, use_collision_if_possible=False, grasp_type=checkpoint_data.get("grasp_type", None)
    )
    params = checkpoint_data["parameters"]
    joint_states = []

    grasp_velocities = []
    for joint_name in hand_model._actuated_joints_names:
        joint_states.append(params[joint_name])
        grasp_velocities.append(checkpoint_data["grasp_velocities"][joint_name])

    grasp_velocities = torch.stack(grasp_velocities, dim=-1).to(device)

    joint_states = torch.stack(joint_states, dim=-1).to(device)
    root_pose = params["root_pose"].to(device)

    # root_pose = root_pose[:2]
    # joint_states = joint_states[:2]
    # contact_idxs = checkpoint_data["contact_idx"].to(device)#[:2]

    energies = checkpoint_data["values"]
    # sort by energy
    energies, indices = torch.sort(energies)
    # contact_idxs = contact_idxs[indices][:args.max_grasps]
    root_pose = root_pose[indices][: args.max_grasps]
    joint_states = joint_states[indices][: args.max_grasps]
    grasp_velocities = grasp_velocities[indices][: args.max_grasps]

    root_orientation = roma.unitquat_to_rotmat(root_pose[..., [4, 5, 6, 3]]).mT.flatten(1, 2)
    hand_params = torch.cat([root_pose[..., :3], root_orientation[..., :6], joint_states], dim=-1).to(device)
    if len(hand_params) == 0:
        print(f"No grasps found for {checkpoint_path}")
        return None

    hand_model.set_parameters(hand_params, contact_point_indices="all")

    batch_size = len(hand_params)
    asset_path = os.path.dirname(dir)
    root_path = os.path.dirname(asset_path)
    print(f"creating object model for {asset_path} with root path {root_path}")
    object_model = ObjectModel(
        data_root_path=root_path,
        batch_size_each=batch_size,
        num_samples=1500,
        device=device,
    )
    object_model.initialize([asset_path])

    def cosine_beta_schedule(timesteps, s=0.001):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps + s) / (1 + s)) * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999), alphas_cumprod

    betas, alphas_cumprod = cosine_beta_schedule(noise_levels, s=1e-3)
    init_hand_params = hand_params.clone()
    device = init_hand_params.device

    noisy_poses, noisy_joints, noisy_energies = [], [], []

    def get_noisy_poses(hand_params, step):
        alpha_bar = alphas_cumprod[step].clamp(max=0.998)
        sqrt_alpha_bar = alpha_bar.sqrt()
        sqrt_one_minus_alpha_bar = (1 - alpha_bar).sqrt()
        noise = torch.randn_like(hand_params)
        noise[:, :3] *= max_noise_translation
        noise[:, 3:9] *= max_noise_rotation
        noise[:, 9:] *= max_noise_joints  # Apply joint noise
        noisy_sample = sqrt_alpha_bar * hand_params + sqrt_one_minus_alpha_bar * noise
        noisy_sample[:, 9:].clamp_(hand_model.joints_lower, hand_model.joints_upper)  # Clamp joint angles to limits
        return noisy_sample

    def calculate_energy(hand_model, object_model, use_e_wall=False):
        distances = hand_model.cal_distance(object_model.surface_points_tensor)
        # show
        distances[distances <= 5e-3] = 0
        E_pen = distances.max(dim=-1).values
        E_spen = (hand_model.self_penetration() - 5e-3).clamp(min=0)
        E_wall = (hand_model.get_surface_points()[..., -1].clamp(max=-2.5e-3) + 2.5e-3).abs().sum(-1) * 0
        print(f"E_pen: {E_pen}, E_spen: {E_spen}, E_wall: {E_wall}")

        return E_pen, E_spen, E_wall

    # ---- 3. Forward Diffusion (20 steps) ----
    for i in range(noise_levels):
        noisy_sample = get_noisy_poses(init_hand_params.clone(), i * 0)
        # t = torch.tensor(i, device=device)

        # alpha_bar = alphas_cumprod[t]
        # sqrt_alpha_bar = alpha_bar.sqrt()
        # sqrt_one_minus_alpha_bar = (1 - alpha_bar).sqrt()
        # noise = torch.randn_like(init_hand_params)
        # noise[:, :3] *= max_noise_translation  # Apply translation noise
        # noise[:, 3:9] *= max_noise_rotation  # Apply rotation noise
        # noise[:, 9:] *= max_noise_joints  # Apply joint noise

        # # clip joints to limits

        # noisy_sample = sqrt_alpha_bar * init_hand_params + sqrt_one_minus_alpha_bar * noise
        # noisy_sample[:, 9:].clamp_(hand_model.joints_lower, hand_model.joints_upper)  # Clamp joint angles to limits
        hand_model.set_parameters(noisy_sample, contact_point_indices="all")

        E_pen = sum(calculate_energy(hand_model, object_model)) + 1
        max_tries = 0

        if i > 0.75 * noise_levels:
            print("Enable strong corruptions for last quarter steps")
            noisy_sample = get_noisy_poses(init_hand_params.clone(), i)

        while (E_pen > 0).any():
            # print("Colliding envs detced, resampling...")
            resampling_envs = E_pen > 0
            # noisy_sample[resampling_envs] = get_noisy_poses(init_hand_params[resampling_envs].clone(), i)

            with torch.enable_grad():
                distance_th = 3e-3

                hand_pose = torch.nn.Parameter(noisy_sample[resampling_envs, :9].clone().detach())
                joint_angles = torch.nn.Parameter(noisy_sample[resampling_envs, 9:].clone().detach())

                optimizers = [
                    torch.optim.AdamW([joint_angles], lr=5e-3),
                    torch.optim.AdamW([hand_pose], lr=5e-4 * (i + 5) / 5.0),
                ]

                for _ in range((i + 1) * 10):
                    for optimizer in optimizers:
                        optimizer.zero_grad()

                    hand_model.set_parameters(torch.cat([hand_pose, joint_angles], dim=-1), contact_point_indices="all")
                    distances = hand_model.cal_distance(object_model.surface_points_tensor[resampling_envs])
                    self_penetration = hand_model.self_penetration()

                    loss = (
                        (1.0 * (distances - distance_th).clamp(min=0).sum())
                        + 3.0 * self_penetration.sum()
                        + 0.05 * (distances).clamp(max=0).sum()
                    )

                    # loss = 3.0 * self_penetration.sum()
                    loss.backward()

                    # optimizer step
                    for optimizer in optimizers:
                        optimizer.step()

                    if loss == 0:
                        print("Successful resampling")
                        break

                noisy_sample[resampling_envs] = torch.cat([hand_pose, joint_angles], dim=-1).detach()

            hand_model.set_parameters(noisy_sample, contact_point_indices="all")
            E_pen = sum(calculate_energy(hand_model, object_model))

            # if i > 0.75 * noise_levels:
            #     # show first five hands
            #     for i in range(5):
            #         plot_data = object_model.get_plotly_data(i, simplify=False)
            #         hand_model.show(idx=i, others=plot_data)
            #     import pdb

            #     pdb.set_trace()

            if max_tries <= 0:
                print(
                    "Max tries reached, stopping resampling. Num wrong envs:", E_pen.nonzero(as_tuple=True)[0].shape[0]
                )
                break

            max_tries -= 1

        #     pdb.set_trace()
        print(f"Colliding envs # {torch.sum(E_pen > 0)} after {100 - max_tries} resampling tries")

        full_hand_poses = hand_model.hand_pose.detach().cpu()
        hand_poses = robust_compute_rotation_matrix_from_ortho6d(full_hand_poses[:, 3:9])
        hand_qxyzw = roma.rotmat_to_unitquat(hand_poses)
        hand_qwxyz = hand_qxyzw[:, [3, 0, 1, 2]]
        hand_poses = torch.cat([full_hand_poses[:, :3], hand_qwxyz], dim=1)
        joint_positions = full_hand_poses[:, 9:]
        parameters = {}
        for idx in range(joint_positions.shape[1]):
            parameters[hand_model._actuated_joints_names[idx]] = joint_positions[:, idx].detach().cpu()
        parameters["root_pose"] = hand_poses.detach().cpu()
        data = {
            "parameters": parameters,
            "contact_idx": E_pen.detach().cpu(),
            "step": i,
            "values": E_pen.detach().cpu(),
        }
        target_dir = os.path.join(dir, args.hand_name, args.num_contacts, args.energy + "_diffused", args.grasp_type)
        os.makedirs(target_dir, exist_ok=True)
        torch.save(data, os.path.join(target_dir, f"step_{i}.dexgrasp.pt"))
        # show first five hands
        # for i in range(5):
        #     plot_data = object_model.get_plotly_data(i, simplify=False)
        #     hand_model.show(idx=i, others=plot_data)
        # import pdb

        # pdb.set_trace()
        print(f"Saved step {i} to {os.path.join(target_dir, f'step_{i}.dexgrasp.pt')}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Visualize hand model")
    arg_parser.add_argument("--device", type=str, default="cuda", help="device to run the model")
    arg_parser.add_argument(
        "--hand_name",
        type=str,
        default="ability_hand",
        help="name of the hand model",
        choices=AVAILABLE_HANDS + ["all"],
    )
    arg_parser.add_argument("--show_jacobian", action="store_true", help="show jacobian")
    arg_parser.add_argument("--show_joint_axes", action="store_true", help="show joint axes")
    arg_parser.add_argument("--show_penetration_points", action="store_true", help="show penetration points")
    arg_parser.add_argument("--show_occupancy_grid", action="store_true", help="show occupancy grid")
    arg_parser.add_argument("--randomize_joints", action="store_true", help="randomize joint angles")
    arg_parser.add_argument("--spacing", type=float, default=0.25, help="spacing for visualization")
    arg_parser.add_argument(
        "--dir",
        type=str,
        default="/data/DexGraspNet/tiny/core-camera-5265ff657b9db80cafae29a76344a143/grasp_predictions",
        help="directory to save the images",
    )

    arg_parser.add_argument("--dataset", type=str, default=None, help="dataset to visualize")

    arg_parser.add_argument("--num_contacts", type=str, default="12_contacts", help="number of contacts")
    arg_parser.add_argument("--energy", type=str, default="graspqp", help="energy")
    arg_parser.add_argument("--max_grasps", type=int, default=-1, help="maximum number of grasps to visualize")
    arg_parser.add_argument("--calc_energy", action="store_true", help="calculate energy")
    arg_parser.add_argument(
        "--vis_dir",
        type=str,
        default="/home/zrene/git/DexGraspNet/grasp_optimization/_vis",
        help="directory to save visualization",
    )
    arg_parser.add_argument("--headless", action="store_true", help="run in headless mode")
    arg_parser.add_argument("--overwrite", action="store_true", help="overwrite existing files")
    arg_parser.add_argument("--num_assets", type=int, default=-1, help="number of assets to visualize")
    arg_parser.add_argument("--grasp_type", type=str, default="default", help="grasp type")

    arg_parser.add_argument("--e_wall", action="store_true", help="use energy wall")
    arg_parser.add_argument("--filter_expr", type=str, default="*.dexgrasp.pt")

    args = arg_parser.parse_args()
    if args.dataset is not None:
        print(f"Visualizing dataset {args.dataset}")
        print(f"Ignoring dir argument")
        # find all grasp predictions in the dataset
        data = sorted(glob.glob(f"{args.dataset}/**/grasp_predictions", recursive=True))
        if len(data) == 0:
            print(f"No grasp predictions found for path {args.dataset} and pattern {args.num_contacts}/{args.energy}")
            exit()
        print(f"Found {len(data)} grasp predictions")
        args.dir = data
    else:
        if isinstance(args.dir, str):
            args.dir = glob.glob(args.dir + "/*/grasp_predictions", recursive=True)
            # args.dir = [f for f in args.dir if "045" in f]
    print(f"Visualizing for:")
    print("\n - ".join(args.dir))
    device = args.device

    data = []

    def _get_origin(idx, n):
        loc_x = idx % np.sqrt(n)
        loc_y = idx // np.sqrt(n)
        spacing = 0.75
        return (loc_x * spacing, loc_y * spacing)

    idx = 0
    import tqdm
    import glob

    for directory in tqdm.tqdm(args.dir):
        res = _show_dir(directory, args, device, noise_levels=10)

    # output_dir = os.path.join(args.vis_dir, "hands", args.hand_name, args.num_contacts, args.energy, args.grasp_type)
    # os.makedirs(output_dir, exist_ok=True)

    # if (not os.path.exists(os.path.join(output_dir, f'grasp_predictions.html')) or args.overwrite) or not args.headless:
    #     fig = go.Figure(_flatten(data))
    #     fig.update_layout(scene_aspectmode='data')
    #     fig.update_layout(height = 980)
    #     if not args.headless:
    #         fig.show()
    #     # save to html
    #     fig.write_html(os.path.join(output_dir, f'grasp_predictions.html'))

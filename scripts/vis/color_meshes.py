"""
Colors the mesh vertices based on interaction frequency with the hand model.
"""

import os

import numpy as np
import torch
import plotly.graph_objects as go
from graspqp.hands import get_hand_model, AVAILABLE_HANDS
from graspqp.core import ObjectModel
from graspqp.core.energy import cal_energy

import argparse
import glob
import trimesh

import roma

torch.manual_seed(1)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def _draw_line_plotly(start, end, color="red", width=5):
    return go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color=color, width=width))


def _flatten(arr):
    data = []
    for a in arr:
        data.extend(a)
    return data


def _show_dir(dir, args, device, origin=(0, 0)):
    data_path = os.path.join(dir, args.hand_name)

    glob_pattern = os.path.join(data_path, args.num_contacts, args.energy, args.grasp_type, "*.dexgrasp.pt")
    print(f"Loading from {glob_pattern}")
    if len(glob.glob(glob_pattern, recursive=True)) == 0:
        print(f"No files found for pattern {glob_pattern}")
        return None
    checkpoint_path = sorted(glob.glob(glob_pattern, recursive=True), key=os.path.getmtime)[-1]
    # print in green color
    print(f"\033[92mLoading {checkpoint_path}\033[0m")

    # print(f"Loading Files from {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path)

    hand_model = get_hand_model(args.hand_name, args.device, use_collision_if_possible=False, grasp_type=checkpoint_data.get("grasp_type", None))
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
    object_model = ObjectModel(
        data_root_path=root_path,
        batch_size_each=batch_size,
        num_samples=1500,
        device=device,
    )
    object_model.initialize([asset_path])

    assert len(object_model.object_mesh_list) == 1

    mesh = object_model.object_mesh_list[0]
    # subdivide the mesh
    verts, faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=0.01)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # if args.show_occupancy_grid:
    # # sample points in 5cm cube, evenly grid
    # n = 250
    # scale = 0.15
    # x = torch.linspace(-scale, scale, n) + 0.01
    # y = torch.linspace(-scale, scale, n) + 0.01
    # z = torch.linspace(-scale, scale, n)
    # x, y, z = torch.meshgrid(x, y, z)
    # pts = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1).to(device)
    # # calcu distances
    # pts_distances = hand_model.cal_distance(pts).cpu()
    # # pts = pts[torch.logical_and(pts_distances[0] > 0 , pts_distances[0] < 0.05)]
    # pts = pts[pts_distances[0] > 0]
    # pts_distances[pts_distances < 0] = -0.5
    # pts_distances[pts_distances > 0] = 0.5
    # colors = ((pts_distances - pts_distances.min()) / (pts_distances.max() - pts_distances.min() + 1e-6)*100).squeeze()
    # pts = pts.detach().cpu().numpy()
    # data += [go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='markers', marker=dict(size=5, color=colors, colorscale='RdBu', opacity=1.0))]

    distances = hand_model.cal_distance(torch.from_numpy(mesh.vertices).to(device).float())
    MODE = "exp"
    if MODE == "exp":
        frequency = (-10 * distances.abs()).exp().sum(0).cpu().detach().numpy()
    elif MODE == "th":
        frequency = (distances.abs() < 0.01).sum(0).cpu().numpy()
    frequency = frequency - frequency.min()
    norm_freq = frequency / frequency.max()
    import matplotlib.cm

    cmap = matplotlib.cm.get_cmap("viridis")
    colors = cmap(norm_freq)
    colors = (colors * 255).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=colors)
    # save it to the directory
    output_folder = os.path.join(args.vis_dir, "interaction_meshes", os.path.basename(os.path.dirname(dir)), args.hand_name, args.num_contacts, args.energy, args.grasp_type)
    os.makedirs(output_folder, exist_ok=True)
    mesh.export(os.path.join(output_folder, "mesh_colored.obj"))
    path = os.path.join(output_folder, "mesh_colored.obj")
    # visualize the mesh
    # from instant_texture import Converter
    # Converter().convert(path)

    print(f"Saved colored mesh to {output_folder}")
    # others = []
    # others += [go.Scatter3d(x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2], mode='markers', marker=dict(size=5, color=colors, colorscale='RdBu', opacity=1.0))]
    # hand_model.show(others = others)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Visualize hand model")
    arg_parser.add_argument("--device", type=str, default="cuda", help="device to run the model")
    arg_parser.add_argument("--hand_name", type=str, default="ability_hand", help="name of the hand model", choices=AVAILABLE_HANDS + ["all"])
    arg_parser.add_argument("--show_jacobian", action="store_true", help="show jacobian")
    arg_parser.add_argument("--show_joint_axes", action="store_true", help="show joint axes")
    arg_parser.add_argument("--show_penetration_points", action="store_true", help="show penetration points")
    arg_parser.add_argument("--show_occupancy_grid", action="store_true", help="show occupancy grid")
    arg_parser.add_argument("--randomize_joints", action="store_true", help="randomize joint angles")
    arg_parser.add_argument("--spacing", type=float, default=0.25, help="spacing for visualization")
    arg_parser.add_argument("--dir", type=str, default="/data/DexGraspNet/tiny/core-camera-5265ff657b9db80cafae29a76344a143/grasp_predictions", help="directory to save the images")

    arg_parser.add_argument("--dataset", type=str, default=None, help="dataset to visualize")

    arg_parser.add_argument("--num_contacts", type=str, default="12_contacts", help="number of contacts")
    arg_parser.add_argument("--energy", type=str, default="dexgrasp", help="energy")
    arg_parser.add_argument("--max_grasps", type=int, default=-1, help="maximum number of grasps to visualize")
    arg_parser.add_argument("--calc_energy", action="store_true", help="calculate energy")
    arg_parser.add_argument("--vis_dir", type=str, default="/home/zrene/git/DexGraspNet/graspqp/_vis", help="directory to save visualization")
    arg_parser.add_argument("--headless", action="store_true", help="run in headless mode")
    arg_parser.add_argument("--overwrite", action="store_true", help="overwrite existing files")
    arg_parser.add_argument("--num_assets", type=int, default=-1, help="number of assets to visualize")
    arg_parser.add_argument("--grasp_type", type=str, default="default", help="grasp type")

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
            args.dir = [args.dir]
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

    for directory in tqdm.tqdm(args.dir):
        res = _show_dir(directory, args, device, origin=_get_origin(idx, len(args.dir)))
        if res is not None:
            data += res
            idx += 1

        if idx >= args.num_assets and args.num_assets > 0:
            break

    output_dir = os.path.join(args.vis_dir, "hands", args.hand_name, args.num_contacts, args.energy, args.grasp_type)
    os.makedirs(output_dir, exist_ok=True)

    if (not os.path.exists(os.path.join(output_dir, f"grasp_predictions.html")) or args.overwrite) or not args.headless:
        fig = go.Figure(_flatten(data))
        fig.update_layout(scene_aspectmode="data")
        fig.update_layout(height=980)
        if not args.headless:
            fig.show()
        # save to html
        fig.write_html(os.path.join(output_dir, f"grasp_predictions.html"))

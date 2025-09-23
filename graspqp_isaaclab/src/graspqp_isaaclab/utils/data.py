"""
Data loading and processing utilities for grasp datasets.

This module provides functions for loading, processing, and resolving grasp data
from various sources including DexGraspNet datasets and saved grasp files.
It handles pose transformations, energy filtering, and asset configuration
for robotic grasping simulations.

Key functionality:
- Loading grasp poses from DexGraspNet and saved torch files
- Energy-based filtering and sorting of grasp candidates
- Asset configuration resolution with visualization options
- Coordinate frame transformations between different representations
"""

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.utils.math import quat_from_matrix
from matplotlib import cm
from transforms3d.euler import euler2mat


def load_dexgrasp_poses(
    articulation: Articulation,
    path: str = "/home/zrene/git/DexGraspNet/data/experiments/exp_2/results/2024-07-23_17-34-11/00316047.npy",
    e_fc_threshold=1e9,
    translation_names=["WRJTx", "WRJTy", "WRJTz"],
    rot_names=["WRJRx", "WRJRy", "WRJRz"],
):
    """
    Load grasp poses from DexGraspNet dataset format.

    Processes DexGraspNet numpy files containing grasp data, filtering by energy
    thresholds and converting coordinate representations to Isaac Lab format.

    Args:
        articulation (Articulation): Target articulated hand for joint mapping
        path (str): Path to DexGraspNet .npy file containing grasp data
        e_fc_threshold (float): Energy threshold for filtering grasps (higher = more permissive)
        translation_names (list): Names of translation parameters in the dataset
        rot_names (list): Names of rotation parameters (Euler angles) in the dataset

    Returns:
        tuple: A tuple containing:
            - poses (torch.Tensor): Hand poses [N, 7] as (x,y,z, qw,qx,qy,qz)
            - joint_positions (torch.Tensor): Joint configurations [N, num_joints]
            - scales (torch.Tensor): Object scaling factors [N]

    Note:
        Poses are sorted by energy (best grasps first). Euler angles are converted
        to quaternions using the Isaac Lab convention.
    """
    # Load and validate DexGraspNet data file
    grasp_data = np.load(path, allow_pickle=True)
    print(f"Found {len(grasp_data)} poses.")
    poses = []
    joint_positions = []
    scales = []
    i = 0

    # Process each grasp entry with energy filtering
    energies = []
    for entry in grasp_data:
        E_fc = entry["E_fc"]
        if E_fc > e_fc_threshold:
            print("SKipping due to high energy", E_fc)
            continue

        energies.append(entry["energy"])
        qpos = entry["qpos"]

        # Convert Euler angles to rotation matrix, then to quaternion
        rot = torch.from_numpy(np.array(euler2mat(*[qpos[name] for name in rot_names])))
        rot_quat = quat_from_matrix(rot)
        position = torch.tensor([qpos[name] for name in translation_names], dtype=torch.float, device="cpu").unsqueeze(0)

        # Combine position and orientation into 7DOF pose
        pose = torch.cat([position, rot_quat.unsqueeze(0)], dim=-1)

        joint_pos = torch.tensor(
            [qpos[name] for name in articulation.cfg.actuated_joints_expr],
            dtype=torch.float,
            device="cpu",
        ).unsqueeze(0)

        i += 1
        poses.append(pose)
        joint_positions.append(joint_pos)
        scales.append(entry["scale"])

    sorted_idxs = np.argsort(energies)
    poses = torch.cat(poses, dim=0)
    joint_positions = torch.cat(joint_positions, dim=0)
    energies = torch.tensor(energies)
    _, sorted_idxs = torch.sort(energies)

    poses = poses[sorted_idxs]
    joint_positions = joint_positions[sorted_idxs]
    scales = torch.tensor(scales)[sorted_idxs]
    return poses, joint_positions, scales


def get_saved_poses(file, articulation: Articulation, num_grasps=3, energy_th=-1e3):
    """
    Load grasp poses from saved PyTorch files.

    Processes saved grasp data files containing optimized grasp parameters,
    joint configurations, and energy values. Supports multiple velocity
    encoding formats and applies energy-based filtering.

    Args:
        file (str): Path to saved .pt file containing grasp data
        articulation (Articulation): Target articulated hand for joint mapping
        num_grasps (int): Maximum number of grasps to return
        energy_th (float): Energy threshold for filtering (grasps above threshold are kept)

    Returns:
        tuple: A tuple containing:
            - hand_poses (torch.Tensor): Hand poses [num_grasps, 7]
            - joint_positions (torch.Tensor): Joint configurations [num_grasps, num_joints]
            - velocities (torch.Tensor): Closing velocities [num_grasps, num_joints]
            - energies (torch.Tensor): Grasp quality energies [num_grasps]

    Note:
        Results are sorted by energy (lowest/best first) and limited to num_grasps.
        Supports legacy velocity formats with optional offset corrections.
    """

    data = torch.load(file, weights_only=False)
    parameters = data["parameters"]  # n_grasps x (3 + 4 + 6)

    values = []
    for joint_name in articulation.cfg.actuated_joints_expr:
        values.append(parameters[joint_name].cpu())
    values = torch.stack(values, dim=-1)
    parameters = torch.cat([parameters["root_pose"].cpu(), values], dim=-1)

    if "grasp_velocities" in data:
        velocities = []
        for joint_name in articulation.cfg.actuated_joints_expr:
            if "grasp_velocities_off" in data:
                velocities.append(
                    data["grasp_velocities_off"][joint_name].cpu() + 0.1 * data["grasp_velocities"][joint_name].cpu()
                )
            else:
                velocities.append(data["grasp_velocities"][joint_name].cpu())
        velocities = torch.stack(velocities, dim=-1)
    else:
        velocities = torch.zeros_like(parameters[..., 7:])

    energies = data["values"]  # n_grasps x 1
    mask = energies > energy_th
    parameters = parameters[mask]
    energies = energies[mask]
    _, sorted_indices = torch.sort(energies, descending=False)
    parameters = parameters[sorted_indices]
    energies = energies[sorted_indices]

    hand_poses = parameters[:, :7]
    joint_positions = parameters[:, 7:]

    num_grasps = num_grasps if num_grasps > 0 else len(energies)
    return (
        hand_poses[:num_grasps],
        joint_positions[:num_grasps],
        velocities[:num_grasps],
        energies[:num_grasps],
    )


def load_grasp_poses(path, num_grasps, articulation: Articulation):
    """
    Universal grasp pose loader supporting multiple file formats.

    Args:
        path (str): Path to grasp data file
        num_grasps (int): Number of grasps to load
        articulation (Articulation): Target articulated hand

    Returns:
        tuple: Hand poses, joint positions, velocities, and energies

    Raises:
        NotImplementedError: If file format is not supported

    Note:
        Currently supports .pt (PyTorch) files. Additional formats
        can be added by extending the format detection logic.
    """
    if path.endswith(".pt"):
        return get_saved_poses(path, num_grasps=num_grasps, articulation=articulation)
    else:
        raise NotImplementedError("Only .pt files are supported for now.")


def resolve_assets(
    articulation: Articulation = None,
    grasp_paths: list = [],
    usd_paths: list = [],
    default_object_cfg: any = None,
    num_grasps: int = -1,
    max_grasps: int = -1,
    use_fps=False,
    collapse_grippers=False,
    colorize=True,
):
    """
    Resolve assets and grasp data for simulation setup.

    Combines USD object files with corresponding grasp data files to create
    complete asset configurations for Isaac Lab simulations. Handles visualization
    options, grasp sampling strategies, and data preprocessing.

    Args:
        articulation (Articulation): Hand articulation for joint mapping
        grasp_paths (list): List of paths to grasp data files (.pt format)
        usd_paths (list): List of paths to USD object files
        default_object_cfg: Base object configuration to clone for each asset
        num_grasps (int): Target number of grasps per object (-1 for all)
        max_grasps (int): Maximum grasps to consider before sampling (-1 for no limit)
        use_fps (bool): Use farthest point sampling to diversify grasp selection
        collapse_grippers (bool): Enable multi-gripper visualization in single env
        colorize (bool): Apply distinct colors to objects for better visualization

    Returns:
        tuple: A tuple containing:
            - grasp_data (tuple): Lists of poses, joint_positions, velocities, energies
            - asset_cfgs (list): Configured asset spawn configurations

    Note:
        When use_fps=True, requires PyTorch3D for farthest point sampling.
        Grasp data is automatically repeated if insufficient grasps are available.
        Colors are assigned using matplotlib's viridis colormap for visual distinction.
    """

    asset_cfgs = []
    for idx, usd_path in enumerate(usd_paths):
        asset_cfg = default_object_cfg.copy()
        asset_cfg.spawn.usd_path = usd_path

        if colorize:
            # Color each object differently for better visualization
            cmap = cm.get_cmap("viridis")
            asset_cfg.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple([c * 1 for c in cmap(idx / (len(usd_paths) * 2))[:3]]),
                opacity=1.0,
            )

        if collapse_grippers:
            # Multiple grippers in one env.
            # Make them visually distinct.
            cmap = cm.get_cmap("viridis")

            def get_material(color):
                MODIFY_SURFACE = False
                if MODIFY_SURFACE:
                    return sim_utils.ModifySurfaceCfg(emissive_color=tuple([c * 1 for c in color[:3]]))
                else:
                    return sim_utils.PreviewSurfaceCfg(diffuse_color=tuple([c * 1 for c in color[:3]]), opacity=1.0)

            asset_cfg.spawn.visual_material = get_material(cmap(idx / (len(usd_paths) * 2)))

        asset_cfgs.append(asset_cfg.spawn.copy())

    poses, joint_positions, energies, velocities = [], [], [], []

    for grasp_path in grasp_paths:
        pose, joints, vel, energy = load_grasp_poses(
            grasp_path,
            num_grasps=min(max_grasps, num_grasps),
            articulation=articulation,
        )

        if use_fps and num_grasps > 1:
            from pytorch3d.ops.sample_farthest_points import \
                sample_farthest_points

            # Reduce to num_grasps using fps
            top_poses = pose[: 3 * num_grasps]
            best_idx = sample_farthest_points(top_poses[:, :3].unsqueeze(0), K=num_grasps)[1].squeeze()
            pose = pose[best_idx]
            energy = energy[best_idx]
            joints = joints[best_idx]

        if len(pose) < num_grasps:
            print(f"Not enough grasps found. Found {len(pose)} grasps, but requested {num_grasps}.")
            print(f"Will repeat the current grasps {num_grasps // len(pose)} times.")
            n_repeat = (num_grasps // len(pose)) + 1
            pose = torch.cat([pose] * n_repeat, dim=0)
            joints = torch.cat([joints] * n_repeat, dim=0)
            vel = torch.cat([vel] * n_repeat, dim=0)
            energy = torch.cat([energy] * n_repeat, dim=0)

        poses.append(pose[:num_grasps])
        joint_positions.append(joints[:num_grasps])
        energies.append(energy[:num_grasps])
        velocities.append(vel[:num_grasps])

    return (poses, joint_positions, velocities, energies), asset_cfgs

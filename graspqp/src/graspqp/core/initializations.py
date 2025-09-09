"""
Last modified date: 2023.02.23
Author: Jialiang Zhang, Ruicheng Wang
Description: initializations
"""

import torch
import transforms3d
import pytorch3d.structures
import pytorch3d.ops
import trimesh as tm
import torch.nn.functional


def initialize_convex_hull(hand_model, object_model, args, env_mask=None, energy_checker=None, init_contacts=True):
    """
    Initialize grasp translation, rotation, joint angles, and contact point indices

    Parameters
    ----------
    hand_model: hand_model.HandModel
    object_model: object_model.ObjectModel
    args: Namespace
    """

    device = hand_model.device
    n_objects = len(object_model.object_mesh_list)
    batch_size_each = object_model.batch_size_each
    total_batch_size = n_objects * batch_size_each

    # initialize translation and rotation

    translation = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)
    rotation = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)

    for i in range(n_objects):

        # get inflated convex hull

        mesh_origin = object_model.object_mesh_list[i].convex_hull
        vertices = mesh_origin.vertices.copy()
        faces = mesh_origin.faces
        vertices *= object_model.object_scale_tensor[i].max().item()
        mesh_origin = tm.Trimesh(vertices, faces)
        mesh_origin.faces = mesh_origin.faces[mesh_origin.remove_degenerate_faces()]

        # vertices += 0.05 * vertices / np.linalg.norm(vertices, axis=1, keepdims=True) # 5cm inflation
        mesh = tm.Trimesh(vertices=vertices, faces=faces)  # .convex_hull
        # vertices = torch.tensor(mesh.vertices, dtype=torch.float, device=device)
        # faces = torch.tensor(mesh.faces, dtype=torch.float, device=device)

        success = False
        tries = 0
        while not success:
            try:
                tries += 1
                points, faces = tm.sample.sample_surface_even(mesh, 100 * batch_size_each)
                # inflate points
                points = points + mesh.face_normals[faces] * 0.01
                # mesh_pytorch3d = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))
                # sample points
                dense_point_cloud = torch.from_numpy(points).to(device).unsqueeze(0)

                p = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=batch_size_each, random_start_point=not init_contacts)[0][0]
                closest_points, _, _ = mesh_origin.nearest.on_surface(p.detach().cpu().numpy())
                success = True
            except FloatingPointError as e:
                print(f"Error in sampling surface: {e}")
                # Retry sampling if an error occurs
                if tries > 3:
                    closest_points = torch.zeros_like(p, dtype=torch.float, device=device)
                    success = True
                    print("Failed to sample surface after 3 tries. Using zeros for closest points.")
                    break

        closest_points = torch.tensor(closest_points, dtype=torch.float, device=device)
        n = (closest_points - p) / (closest_points - p).norm(dim=1).unsqueeze(1)

        # by default, align hands +z axis with the normal of the inflated convex hull
        rotation_global = torch.zeros([batch_size_each, 3, 3], dtype=torch.float, device=device)

        def look_at(camera_positions, target_positions, forward_vector=torch.tensor([1, 0, 0]), up_vector=torch.tensor([0, 0, 1], dtype=torch.float)):
            base_up_vector = up_vector.to(camera_positions.device).float()
            up_vector = base_up_vector.clone().float()
            forward_vector = forward_vector.to(camera_positions.device).float()
            if up_vector.ndim == 1:
                up_vector = up_vector.unsqueeze(0).repeat(camera_positions.shape[0], 1)

            # 1. Compute the forward vectors (camera to target)
            forward = camera_positions - target_positions
            forward = forward / torch.norm(forward, dim=1, keepdim=True)  # Normalize along the batch dimension

            # 2. Compute the right vectors (perpendicular to forward and up vectors)
            # inner product up and forward vectors
            prod = torch.sum(up_vector * forward, dim=1, keepdim=True)
            up_vector = torch.where(prod.abs() < 0.95, up_vector, torch.tensor([0, 1, 0], dtype=torch.float, device=up_vector.device))
            right = torch.cross(up_vector, forward, dim=1)
            right = right / torch.norm(right, dim=1, keepdim=True)  # Normalize along the batch dimension

            # 3. Compute the up vectors (perpendicular to forward and right vectors)
            up = torch.cross(forward, right, dim=1)
            # 4. Construct the batched 3x3 orientation matrices
            # Stack the right, up, and forward vectors into 3x3 matrices
            orientation_matrices = torch.stack([forward, up, right], dim=-1)
            basis = torch.stack([forward_vector, -torch.cross(forward_vector, base_up_vector, dim=-1), base_up_vector], dim=-1)

            return orientation_matrices @ basis

        for j in range(batch_size_each):
            rotation_global[j] = torch.eye(3, dtype=torch.float, device=device)
        rotation_global = look_at(p, p + n, up_vector=hand_model.up_axis, forward_vector=hand_model.forward_axis)

        distance = args.distance_lower + (args.distance_upper - args.distance_lower) * torch.rand([batch_size_each], dtype=torch.float, device=device)

        # Rotation around the normal vector. In the plane of the contact point
        rotate_theta = args.rotate_lower + (args.rotate_upper - args.rotate_lower) * torch.rand([batch_size_each], dtype=torch.float, device=device)
        pitch_theta = args.pitch_lower + (args.pitch_upper - args.pitch_lower) * torch.rand([batch_size_each], dtype=torch.float, device=device)
        tilt_theta = args.tilt_lower + (args.tilt_upper - args.tilt_lower) * torch.rand([batch_size_each], dtype=torch.float, device=device)

        rotation_local = torch.zeros([batch_size_each, 3, 3], dtype=torch.float, device=device)

        for j in range(batch_size_each):
            rotation_local[j] = torch.tensor(transforms3d.euler.euler2mat(tilt_theta[j], pitch_theta[j], rotate_theta[j], axes="rxyz"), dtype=torch.float, device=device)

        translation[i * batch_size_each : (i + 1) * batch_size_each] = p - distance.unsqueeze(1) * n  # * (rotation_global @ rotation_local @ torch.tensor([0, 0, 1], dtype=torch.float, device=device).reshape(1, -1, 1)).squeeze(2)
        rotation[i * batch_size_each : (i + 1) * batch_size_each] = rotation_global @ rotation_local  # @ rotation_hand

    joint_angles_mu = hand_model.default_state
    joint_angles_mu.clamp_(hand_model.joints_lower, hand_model.joints_upper)
    joint_angles_sigma = args.jitter_strength * (hand_model.joints_upper - hand_model.joints_lower)

    joint_angles = torch.zeros([total_batch_size, hand_model.n_dofs], dtype=torch.float, device=device)
    for i in range(hand_model.n_dofs):
        torch.nn.init.trunc_normal_(joint_angles[:, i], joint_angles_mu[i], joint_angles_sigma[i], hand_model.joints_lower[i] - 1e-6, hand_model.joints_upper[i] + 1e-6)

    HANDLE_MODE = "handles" in object_model.data_root_path
    if HANDLE_MODE:
        # clip z component of translation
        translation[:, 2] = torch.clamp(translation[:, 2], min=0.025)

    if not init_contacts:
        return translation, rotation.transpose(1, 2)[:, :2].reshape(-1, 6), joint_angles

    hand_pose = torch.cat([translation, rotation.transpose(1, 2)[:, :2].reshape(-1, 6), joint_angles], dim=1)

    # initialize contact point indices
    hand_pose.requires_grad_()
    contact_point_indices = torch.randint(hand_model.n_contact_candidates, size=[total_batch_size, args.n_contact], device=device)
    hand_model.set_parameters(hand_pose, contact_point_indices, env_mask=env_mask)

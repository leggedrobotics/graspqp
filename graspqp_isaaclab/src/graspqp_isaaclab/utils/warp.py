# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Collision handling functions and kernels.
"""

import torch

import warp as wp

from isaaclab.utils.math import convert_quat
import numpy as np


@wp.kernel
def mesh_sdf_field(
    # inputs
    mesh: wp.uint64,
    points: wp.array1d(dtype=wp.vec3),
    max_dist: float,
    # outputs
    sdf: wp.array1d(dtype=wp.float32),
    normal: wp.array1d(dtype=wp.vec3),
):
    """Computes the signed distance field (SDF) for the given mesh at the given points.
    Args:
        mesh: The input mesh.
        points: The input points. Shape is (N, 3).
        sdf: The output SDF values. Shape is (N).
        max_dist: The maximum distance to consider. Defaults to 1e6.
    """
    # get the thread id
    tid_point = wp.tid()

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    res = wp.mesh_query_point_sign_normal(mesh, points[tid_point], max_dist, sign, face_index, face_u, face_v)

    if res:
        closest = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
        distance_vector = points[tid_point] - closest
        dist = wp.length(distance_vector)
        normal[tid_point] = wp.mesh_eval_face_normal(mesh, face_index)
        sdf[tid_point] = dist * sign
    else:
        sdf[tid_point] = max_dist


def calc_sdf_field(
    points_wp: wp.array1d(dtype=wp.vec3),
    mesh_id: int,
    max_dist: float = 1e6,
):
    """Performs ray-casting against a mesh.
    Note that the :attr:`ray_starts` and :attr:`ray_directions`, and :attr:`ray_hits` should have compatible shapes
    and data types to ensure proper execution. Additionally, they all must be in the same frame.
    Args:
        ray_starts: The starting position of the rays. Shape (B, N, 3).
        ray_directions: The ray directions for each ray. Shape (B, N, 3).
        mesh_id: The warp mesh id to ray-cast against.
        max_dist: The maximum distance to ray-cast. Defaults to 1e6.
        return_distance: Whether to return the distance of the ray until it hits the mesh. Defaults to False.
        return_normal: Whether to return the normal of the mesh face the ray hits. Defaults to False.
        return_face_id: Whether to return the face id of the mesh face the ray hits. Defaults to False.
    Returns:
        The ray hit position. Shape (B, N, 3).
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit distance. Shape (B, N,).
            Will only return if :attr:`return_distance` is True, else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit normal. Shape (B, N, 3).
            Will only return if :attr:`return_normal` is True else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit face id. Shape (B, N,).
            Will only return if :attr:`return_face_id` is True else returns None.
            The returned tensor contains :obj:`int(-1)` for missed hits.
    """

    distance_wp = wp.ones([len(points_wp)], dtype=wp.float32)
    normal_wp = wp.ones([len(points_wp)], dtype=wp.vec3)

    wp.launch(
        kernel=mesh_sdf_field,
        dim=[len(points_wp)],
        inputs=[
            mesh_id,
            points_wp,
            max_dist,
            distance_wp,
            normal_wp,
        ],
        device=points_wp.device,
    )
    return distance_wp, normal_wp


@wp.kernel
def calc_sdf_field_batched(
    object_meshes: wp.array2d(dtype=wp.uint64),  # Shape n_envs x n_objects
    object_positions: wp.array2d(dtype=wp.vec3),  # Shape n_envs x n_objects
    object_rotations: wp.array2d(dtype=wp.quat),  # Shape n_envs x n_objects
    lookup_points: wp.array2d(dtype=wp.vec3),  # Shape n_envs x n_points x 3
    env_ids_wp: wp.array1d(dtype=wp.uint64),  # Shape n_envs
    distances: wp.array3d(dtype=wp.float32),  # Shape n_envs x n_objects x n_points
    normals: wp.array3d(dtype=wp.vec3),  # Shape n_envs x n_objects x n_points
    max_dist: float = 1e6,
):
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    # get the thread id
    tid_env, tid_obj_mesh_id, tid_point = wp.tid()
    env_idx = int(env_ids_wp[tid_env])

    mesh_pose = wp.transform(
        object_positions[tid_env, tid_obj_mesh_id],
        object_rotations[tid_env, tid_obj_mesh_id],
    )
    mesh_pose_inv = wp.transform_inverse(mesh_pose)
    pos_xyz = wp.transform_point(mesh_pose_inv, lookup_points[tid_env, tid_point])

    res = wp.mesh_query_point_sign_normal(
        object_meshes[env_idx, tid_obj_mesh_id],
        pos_xyz,
        max_dist,
        sign,
        face_index,
        face_u,
        face_v,
    )
    if res:
        closest = wp.mesh_eval_position(object_meshes[env_idx, tid_obj_mesh_id], face_index, face_u, face_v)
        distance_vector = pos_xyz - closest
        dist = wp.length(distance_vector)
        # normals[tid_env, tid_obj_mesh_id, tid_point] = wp.mesh_eval_face_normal(
        #     object_meshes[env_idx, tid_obj_mesh_id], face_index
        # )
        normals[tid_env, tid_obj_mesh_id, tid_point] = distance_vector / dist
        normals[tid_env, tid_obj_mesh_id, tid_point] = wp.transform_vector(
            mesh_pose, normals[tid_env, tid_obj_mesh_id, tid_point]
        )
        distances[tid_env, tid_obj_mesh_id, tid_point] = dist * sign
    else:
        distances[tid_env, tid_obj_mesh_id, tid_point] = max_dist


def calc_obj_distances(
    object_meshes: wp.array2d(dtype=wp.uint64),  # Shape n_envs x n_objects
    object_positions: torch.Tensor,  # Shape n_envs x n_objects x 3
    object_rotations: torch.Tensor,  # Shape n_envs x n_objects x 4
    lookup_points: torch.Tensor,  # Shape n_envs x n_points x 3
    max_dist: float = 1e6,
    env_ids: torch.Tensor | None = None,
):

    lookup_points_wp = wp.from_torch(lookup_points, dtype=wp.vec3)
    distances_wp = wp.ones(
        [object_positions.shape[0], object_positions.shape[1], lookup_points.shape[1]],
        dtype=wp.float32,
        device=str(lookup_points.device),
    )
    normals_wp = wp.ones(
        [object_positions.shape[0], object_positions.shape[1], lookup_points.shape[1]],
        dtype=wp.vec3,
        device=str(lookup_points.device),
    )
    object_positions_wp = wp.from_torch(object_positions, dtype=wp.vec3)

    object_rotations = convert_quat(
        object_rotations.to(dtype=torch.float32, device=lookup_points.device), "xyzw"
    ).contiguous()
    object_rotations_wp = wp.from_torch(object_rotations, dtype=wp.quat)

    if env_ids is None:
        env_ids = torch.arange(object_positions.shape[0], device=lookup_points.device)

    env_ids_wp = wp.from_torch(env_ids, dtype=wp.uint64)

    wp.launch(
        kernel=calc_sdf_field_batched,
        dim=[
            object_positions.shape[0],
            object_positions.shape[1],
            lookup_points.shape[1],
        ],
        inputs=[
            object_meshes,
            object_positions_wp,
            object_rotations_wp,
            lookup_points_wp,
            env_ids_wp,
            distances_wp,
            normals_wp,
            max_dist,
        ],
        device=lookup_points_wp.device,
    )
    # convert back to torch
    distances = wp.to_torch(distances_wp)
    normals = wp.to_torch(normals_wp)
    return distances, normals


def convert_to_warp_mesh(points: np.ndarray, indices: np.ndarray, device: str) -> wp.Mesh:
    """Create a warp mesh object with a mesh defined from vertices and triangles.

    Args:
        points: The vertices of the mesh. Shape is (N, 3), where N is the number of vertices.
        indices: The triangles of the mesh as references to vertices for each triangle.
            Shape is (M, 3), where M is the number of triangles / faces.
        device: The device to use for the mesh.

    Returns:
        The warp mesh object.
    """
    return wp.Mesh(
        points=wp.array(points.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(indices.astype(np.int32).flatten(), dtype=wp.int32, device=device),
    )

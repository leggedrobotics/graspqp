# Copyright (c) 2025 ETH Zurich, René Zurbrügg
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Signed-distance-field (SDF) collision backend built on NVIDIA Warp.

This module is the object/mesh SDF backend used by the grasp energies. It provides Warp
kernels that, for a batch of query points, return the signed distance to a triangle
mesh (exterior positive, interior negative), the closest-face outward normal, and the
exact analytical spatial gradient ``d(sdf)/dx = sign * (x - closest) / |x - closest|``.

Two sign conventions are available: pseudo-normal (:func:`mesh_sdf_field`) for clean
watertight meshes, and generalized winding number (:func:`mesh_sdf_field_wn`) for
imperfectly wound meshes. The ``torch.autograd.Function`` wrappers
(:class:`CalcSdfField`, :class:`CalcSdfFieldBatched`, :class:`CalcObjDistances`) bridge
Warp and PyTorch and supply the analytical gradient in ``backward`` -- this is
deliberate: Warp's tape adjoint of ``wp.length`` produces NaN for points lying exactly
on the surface, which would otherwise freeze the optimizer.
"""

import torch
import warp as wp


@wp.kernel
def mesh_sdf_field(
    # inputs
    mesh: wp.uint64,
    points: wp.array1d(dtype=wp.vec3),
    max_dist: float,
    # outputs
    sdf: wp.array1d(dtype=wp.float32),
    normal: wp.array1d(dtype=wp.vec3),
    grad_dir: wp.array1d(dtype=wp.vec3),
):
    """Computes the signed distance field (SDF) for the given mesh at the given points.
    Args:
        mesh: The input mesh.
        points: The input points. Shape is (N, 3).
        sdf: The output SDF values. Shape is (N).
        normal: The outward FACE normal at the closest face. Shape is (N).
        grad_dir: The exact SDF spatial gradient  d(sdf)/dx = sign * (x - closest)/|x - closest|.
            Unlike ``normal`` (piecewise-constant per face, only == grad on face interiors) this is
            correct everywhere including edges/vertices, and is what the differentiable backward uses.
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
        if dist > 1.0e-8:
            grad_dir[tid_point] = distance_vector / dist * sign
        else:
            grad_dir[tid_point] = normal[tid_point]  # exactly on the surface: fall back to face normal
    else:
        sdf[tid_point] = max_dist


@wp.kernel
def mesh_sdf_field_wn(
    # inputs
    mesh: wp.uint64,
    points: wp.array1d(dtype=wp.vec3),
    max_dist: float,
    # outputs
    sdf: wp.array1d(dtype=wp.float32),
    normal: wp.array1d(dtype=wp.vec3),
    grad_dir: wp.array1d(dtype=wp.vec3),
):
    """Same as :func:`mesh_sdf_field` but resolves inside/outside with the generalized WINDING
    NUMBER (``mesh_query_point_sign_winding_number``) instead of the pseudo-normal. Robust for
    meshes that are not perfectly watertight / consistently wound. Requires the mesh to be built
    with ``support_winding_number=True``.
    """
    tid_point = wp.tid()
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    # accuracy=2.0, threshold=0.5 are warp's defaults for the winding-number sign query
    res = wp.mesh_query_point_sign_winding_number(
        mesh, points[tid_point], max_dist, sign, face_index, face_u, face_v, 2.0, 0.5
    )
    if res:
        closest = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
        distance_vector = points[tid_point] - closest
        dist = wp.length(distance_vector)
        normal[tid_point] = wp.mesh_eval_face_normal(mesh, face_index)
        sdf[tid_point] = dist * sign
        if dist > 1.0e-8:
            grad_dir[tid_point] = distance_vector / dist * sign
        else:
            grad_dir[tid_point] = normal[tid_point]
    else:
        sdf[tid_point] = max_dist


def calc_sdf_field(
    points_wp: wp.array1d(dtype=wp.vec3),
    mesh_id: int,
    max_dist: float = 1e6,
    use_winding_number: bool = False,
):
    """Launch the SDF kernel for a flat array of query points against one mesh.

    Allocates the output arrays and dispatches either :func:`mesh_sdf_field` (pseudo-
    normal sign) or :func:`mesh_sdf_field_wn` (winding-number sign) over all query
    points. The query points are assumed to be in the mesh's local frame.

    Args:
        points_wp: Query points as a Warp ``vec3`` array of length ``N``.
        mesh_id: Warp mesh id to query against.
        max_dist: Maximum distance to search; points farther than this are assigned
            ``max_dist`` and receive no normal/gradient. Defaults to ``1e6``.
        use_winding_number: If ``True`` use the generalized winding number to decide
            inside/outside (robust for non-watertight meshes); otherwise use the
            pseudo-normal. Defaults to ``False``.

    Returns:
        tuple: ``(distance_wp, normal_wp, grad_dir_wp)`` Warp arrays of length ``N``:
        the signed distance (``float32``), the closest-face outward normal (``vec3``),
        and the exact SDF spatial gradient ``d(sdf)/dx`` (``vec3``).
    """

    distance_wp = wp.ones([len(points_wp)], dtype=wp.float32, device=points_wp.device)
    normal_wp = wp.ones([len(points_wp)], dtype=wp.vec3, device=points_wp.device)
    grad_dir_wp = wp.zeros([len(points_wp)], dtype=wp.vec3, device=points_wp.device)

    wp.launch(
        kernel=mesh_sdf_field_wn if use_winding_number else mesh_sdf_field,
        dim=[len(points_wp)],
        inputs=[
            mesh_id,
            points_wp,
            max_dist,
            distance_wp,
            normal_wp,
            grad_dir_wp,
        ],
        device=points_wp.device,
    )
    return distance_wp, normal_wp, grad_dir_wp


class CalcSdfField(torch.autograd.Function):
    """Differentiable wrapper around :func:`calc_sdf_field` w.r.t. the query points.

    The bare :func:`calc_sdf_field` crosses the torch<->warp boundary via ``wp.from_torch`` /
    ``wp.to_torch`` with no autograd hook, so ``sdf``/``normal`` are detached from ``points`` and
    the hand contact points receive NO gradient from the object SDF (the force-closure / contact
    energies then can't pull the hand onto the surface). This op restores that gradient using the
    analytical identity  ``d(sdf)/dx = ∇sdf = outward surface normal``.

    The backward uses the kernel's ``grad_dir`` output -- the EXACT SDF gradient
    ``d(sdf)/dx = sign * (x - closest)/|x - closest|`` -- which is correct at faces AND edges/
    vertices (matching the isaac.locoma reference and kaolin). The returned ``normal`` (outward
    face normal) is passed through as a detached constant for callers that need the surface
    orientation, but is NOT used for the query-point gradient.
    """

    @staticmethod
    def forward(ctx, points, mesh_id, max_dist=1e6, use_winding_number=False):
        pts_wp = wp.from_torch(points.detach().contiguous(), dtype=wp.vec3)
        distance_wp, normal_wp, grad_dir_wp = calc_sdf_field(
            points_wp=pts_wp, mesh_id=mesh_id, max_dist=max_dist, use_winding_number=use_winding_number
        )
        sdf = wp.to_torch(distance_wp).clone()
        normal = wp.to_torch(normal_wp).clone()
        grad_dir = wp.to_torch(grad_dir_wp).clone()
        ctx.save_for_backward(grad_dir)
        return sdf, normal

    @staticmethod
    def backward(ctx, grad_sdf, grad_normal):
        (grad_dir,) = ctx.saved_tensors
        grad_points = grad_sdf.unsqueeze(-1) * grad_dir  # d(sdf)/dx = exact ∇sdf (correct at edges too)
        return grad_points, None, None, None  # (points, mesh_id, max_dist, use_winding_number)


@wp.kernel
def mesh_sdf_field_batched(
    # inputs
    meshes: wp.array2d(dtype=wp.uint64),  # [n_env, n_link]
    points: wp.array3d(dtype=wp.vec3),  # [n_env, n_link, n_point]  -- ALREADY in each link's local frame
    max_dist: float,
    # outputs
    sdf: wp.array3d(dtype=wp.float32),  # [n_env, n_link, n_point]  signed (exterior +, interior -)
    grad_dir: wp.array3d(dtype=wp.vec3),  # [n_env, n_link, n_point]  exact d(sdf)/d(point)
):
    """Batched signed-distance field of pre-transformed local query points against per-(env,link)
    meshes, returning the EXACT analytical spatial gradient ``grad_dir = sign*(x-closest)/|x-closest|``
    (guarded at ``dist<1e-8``). This is the batched analog of :func:`mesh_sdf_field`; the caller
    (``CalcSdfFieldBatched``) uses ``grad_dir`` for the backward instead of warp's tape adjoint, so
    the query-point gradient is well defined even for points exactly on the surface (warp's autodiff
    of ``wp.length`` / the quaternion transform there produces NaN, which froze the optimizer)."""
    e, l, p = wp.tid()
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    res = wp.mesh_query_point_sign_normal(meshes[e, l], points[e, l, p], max_dist, sign, face_index, face_u, face_v)
    if res:
        closest = wp.mesh_eval_position(meshes[e, l], face_index, face_u, face_v)
        distance_vector = points[e, l, p] - closest
        dist = wp.length(distance_vector)
        sdf[e, l, p] = dist * sign
        if dist > 1.0e-8:
            grad_dir[e, l, p] = distance_vector / dist * sign
        else:
            grad_dir[e, l, p] = wp.mesh_eval_face_normal(meshes[e, l], face_index)
    else:
        sdf[e, l, p] = max_dist
        grad_dir[e, l, p] = wp.vec3(0.0, 0.0, 0.0)


class CalcSdfFieldBatched(torch.autograd.Function):
    """Differentiable batched hand-mesh SDF w.r.t. the (already torch-transformed) local query points.

    Replaces the pose-adjoint ``CalcObjDistances``: instead of differentiating the SDF w.r.t. each
    link's position/quaternion through warp's tape (which emits NaN for surface-coincident points and
    needs ``matrix_to_quaternion``), the caller transforms the query points into each link's local
    frame in PyTorch (rotation MATRICES, NaN-free) and this op supplies the analytical
    ``d(sdf)/d(local point) = grad_dir``. torch autograd then composes it back to the link poses /
    joints. Same signed convention as before (exterior +, interior -)."""

    @staticmethod
    def forward(ctx, meshes_wp, points, max_dist=1e6):
        # points: torch [n_env, n_link, n_point, 3] (local frame). warp reads the trailing 3 as vec3.
        n_env, n_link, n_point = points.shape[0], points.shape[1], points.shape[2]
        dev = str(points.device)
        pts_wp = wp.from_torch(points.detach().contiguous(), dtype=wp.vec3)  # [n_env, n_link, n_point]
        sdf_wp = wp.zeros((n_env, n_link, n_point), dtype=wp.float32, device=dev)
        grad_wp = wp.zeros((n_env, n_link, n_point), dtype=wp.vec3, device=dev)
        wp.launch(
            kernel=mesh_sdf_field_batched,
            dim=(n_env, n_link, n_point),
            inputs=[meshes_wp, pts_wp, float(max_dist)],
            outputs=[sdf_wp, grad_wp],
            device=dev,
        )
        sdf = wp.to_torch(sdf_wp).clone()
        grad_dir = wp.to_torch(grad_wp).clone()
        ctx.save_for_backward(grad_dir)
        return sdf

    @staticmethod
    def backward(ctx, grad_sdf):
        (grad_dir,) = ctx.saved_tensors
        grad_points = grad_sdf.unsqueeze(-1) * grad_dir  # d(sdf)/d(local point) = exact grad_dir
        return None, grad_points, None  # (meshes_wp, points, max_dist)


@wp.kernel
def calc_sdf_field_batched(
    object_meshes: wp.array2d(dtype=wp.uint64),  # Shape n_envs x n_objects
    object_positions: wp.array2d(dtype=wp.vec3),  # Shape n_envs x n_objects
    object_rotations: wp.array2d(dtype=wp.quat),  # Shape n_envs x n_objects
    lookup_points: wp.array2d(dtype=wp.vec3),  # Shape n_envs x n_points x 3
    env_ids_wp: wp.array1d(dtype=wp.uint64),  # Shape n_envs
    distances: wp.array3d(dtype=wp.float32),  # Shape n_envs x n_objects x n_points
    normals: wp.array3d(dtype=wp.vec3),  # Shape n_envs x n_objects x n_points
):
    max_dist = float(1e6)
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
        # Epsilon-guarded length. This op's gradient (w.r.t. object_positions/rotations) is produced
        # by warp's tape adjoint, and the adjoint of wp.length is v/|v| = NaN when the query point
        # lands exactly on the mesh surface (distance_vector == 0). That NaN flowed into the JOINT
        # gradient of E_pen (link poses depend on the joints via FK) and froze the optimizer
        # (acceptance -> 0, fingers stuck at init). sqrt(dot(v,v)+eps) is smooth at v == 0.
        dist = wp.sqrt(wp.dot(distance_vector, distance_vector) + 1.0e-8)
        normals[tid_env, tid_obj_mesh_id, tid_point] = wp.mesh_eval_face_normal(
            object_meshes[env_idx, tid_obj_mesh_id], face_index
        )
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
    """Signed distances and world-frame normals from query points to posed object meshes.

    For each environment, object and query point, transforms the query point into the
    object's local frame, evaluates the signed distance (exterior positive, interior
    negative) and the closest-face normal, then rotates that normal back into the world
    frame.

    Args:
        object_meshes: Warp mesh-id array of shape ``(n_envs, n_objects)``.
        object_positions: Object positions, shape ``(n_envs, n_objects, 3)`` in meters.
        object_rotations: Object orientations as quaternions, shape
            ``(n_envs, n_objects, 4)``.
        lookup_points: Query points in world frame, shape ``(n_envs, n_points, 3)`` in
            meters.
        max_dist: Maximum search distance; farther points are set to ``max_dist``.
        env_ids: Optional environment indices selecting which meshes to query; defaults
            to all environments.

    Returns:
        tuple: ``(distances, normals)`` torch tensors of shape
        ``(n_envs, n_objects, n_points)`` and ``(n_envs, n_objects, n_points, 3)``.
    """
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

    object_rotations = object_rotations.contiguous()
    object_rotations_wp = wp.from_torch(object_rotations, dtype=wp.quat)

    if env_ids is None:
        env_ids = torch.arange(object_positions.shape[0], device=lookup_points.device)

    env_ids_wp = wp.from_torch(env_ids, dtype=wp.uint64)

    wp.launch(
        kernel=calc_sdf_field_batched,
        dim=[
            object_positions.shape[0],  # n_envs
            object_positions.shape[1],  # n_objects
            lookup_points.shape[1],  # n_points
        ],
        inputs=[
            object_meshes,
            object_positions_wp,
            object_rotations_wp,
            lookup_points_wp,
            env_ids_wp,
            distances_wp,
            normals_wp,
            # max_dist,
        ],
        device=lookup_points_wp.device,
    )
    # convert back to torch
    distances = wp.to_torch(distances_wp)
    normals = wp.to_torch(normals_wp)
    return distances, normals


class CalcObjDistances(torch.autograd.Function):
    """Differentiable object-SDF w.r.t. object pose (position and orientation).

    Wraps :func:`calc_sdf_field_batched` and, unlike :class:`CalcSdfFieldBatched`,
    propagates gradients into the object ``position``/``rotation`` via Warp's tape
    adjoint (the query ``lookup_points`` are treated as fixed and receive no gradient).
    Returns signed distances and world-frame normals with the same convention as
    :func:`calc_obj_distances` (exterior positive, interior negative).
    """

    @staticmethod
    def forward(ctx, object_meshes, object_positions, object_rotations, lookup_points):
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

        object_rotations = object_rotations.contiguous()
        object_rotations_wp = wp.from_torch(object_rotations, dtype=wp.quat)

        env_ids = torch.arange(object_positions.shape[0], device=lookup_points.device)

        env_ids_wp = wp.from_torch(env_ids, dtype=wp.uint64)

        wp.launch(
            kernel=calc_sdf_field_batched,
            dim=[
                object_positions.shape[0],  # n_envs
                object_positions.shape[1],  # n_objects
                lookup_points.shape[1],  # n_points
            ],
            inputs=[
                object_meshes,
                object_positions_wp,
                object_rotations_wp,
                lookup_points_wp,
                env_ids_wp,
            ],
            outputs=[
                distances_wp,
                normals_wp,
            ],
            device=lookup_points_wp.device,
        )
        # convert back to torch
        distances = wp.to_torch(distances_wp)
        normals = wp.to_torch(normals_wp)
        ctx.save_for_backward(object_positions, object_rotations, lookup_points, distances, normals)
        ctx.mesh_ids = object_meshes
        return distances, normals

    @staticmethod
    def backward(ctx, grad_distances, grad_normals):
        object_positions, object_rotations, lookup_points, distances, normals = ctx.saved_tensors
        object_meshes = ctx.mesh_ids

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

        object_rotations = object_rotations.contiguous()
        object_rotations_wp = wp.from_torch(object_rotations, dtype=wp.quat)

        env_ids = torch.arange(object_positions.shape[0], device=lookup_points.device)

        env_ids_wp = wp.from_torch(env_ids, dtype=wp.uint64)

        distances_wp.grad = wp.from_torch(grad_distances.contiguous(), dtype=wp.float32)
        normals_wp.grad = wp.from_torch(grad_normals, dtype=wp.vec3)

        wp.launch(
            kernel=calc_sdf_field_batched,
            dim=[
                object_positions.shape[0],  # n_envs
                object_positions.shape[1],  # n_objects
                lookup_points.shape[1],  # n_points
            ],
            inputs=[object_meshes, object_positions_wp, object_rotations_wp, lookup_points_wp, env_ids_wp],
            outputs=[
                distances_wp,
                normals_wp,
            ],
            adj_inputs=[None, object_positions_wp.grad, object_rotations_wp.grad, None, None],
            adj_outputs=[
                distances_wp.grad,
                normals_wp.grad,
            ],
            adjoint=True,
            device=lookup_points_wp.device,
        )

        # forward takes 4 inputs (object_meshes, object_positions, object_rotations, lookup_points),
        # so backward must return exactly 4 gradients. It previously returned 5 (an extra trailing
        # None from an older signature), which raises "returned an incorrect number of gradients"
        # whenever this op is actually backpropagated through. lookup_points (the object surface
        # points queried against the hand) are fixed, so their gradient is None.
        return (
            None,
            wp.to_torch(object_positions_wp.grad),
            wp.to_torch(object_rotations_wp.grad),
            None,
        )

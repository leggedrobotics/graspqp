# Copyright (c) 2025 ETH Zurich, René Zurbrügg
#
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for working with USD transform (xform) operations.

This module provides utilities for manipulating USD transform operations (xform ops) on prims.
Transform operations in USD define how geometry is positioned, oriented, and scaled in 3D space.

The utilities in this module help standardize transform stacks, clear operations, and manipulate
transforms in a consistent way across different USD assets.
"""

from __future__ import annotations

from pxr import Gf, Sdf, Usd, UsdGeom

def resolve_prim_pose(
    prim: Usd.Prim, ref_prim: Usd.Prim | None = None
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Resolve the pose of a prim with respect to another prim.

    Note:
        This function ignores scale and skew by orthonormalizing the transformation
        matrix at the final step. However, if any ancestor prim in the hierarchy
        has non-uniform scale, that scale will still affect the resulting position
        and orientation of the prim (because it's baked into the transform before
        scale removal).

        In other words: scale **is not removed hierarchically**. If you need
        completely scale-free poses, you must walk the transform chain and strip
        scale at each level. Please open an issue if you need this functionality.

    Args:
        prim: The USD prim to resolve the pose for.
        ref_prim: The USD prim to compute the pose with respect to.
            Defaults to None, in which case the world frame is used.

    Returns:
        A tuple containing the position (as a 3D vector) and the quaternion orientation
        in the (w, x, y, z) format.

    Raises:
        ValueError: If the prim or ref prim is not valid.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>> from pxr import Usd, UsdGeom
        >>>
        >>> # Get prim
        >>> stage = sim_utils.get_current_stage()
        >>> prim = stage.GetPrimAtPath("/World/ImportedAsset")
        >>>
        >>> # Resolve pose
        >>> pos, quat = sim_utils.resolve_prim_pose(prim)
        >>> print(f"Position: {pos}")
        >>> print(f"Orientation: {quat}")
        >>>
        >>> # Resolve pose with respect to another prim
        >>> ref_prim = stage.GetPrimAtPath("/World/Reference")
        >>> pos, quat = sim_utils.resolve_prim_pose(prim, ref_prim)
        >>> print(f"Position: {pos}")
        >>> print(f"Orientation: {quat}")
    """
    # check if prim is valid
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath().pathString}' is not valid.")
    # get prim xform
    xform = UsdGeom.Xformable(prim)
    prim_tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # sanitize quaternion
    # this is needed, otherwise the quaternion might be non-normalized
    prim_tf.Orthonormalize()

    if ref_prim is not None:
        # if reference prim is the root, we can skip the computation
        if ref_prim.GetPath() != Sdf.Path.absoluteRootPath:
            # get ref prim xform
            ref_xform = UsdGeom.Xformable(ref_prim)
            ref_tf = ref_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            # make sure ref tf is orthonormal
            ref_tf.Orthonormalize()
            # compute relative transform to get prim in ref frame
            prim_tf = prim_tf * ref_tf.GetInverse()

    # extract position and orientation
    prim_pos = [*prim_tf.ExtractTranslation()]
    prim_quat = [prim_tf.ExtractRotationQuat().real, *prim_tf.ExtractRotationQuat().imaginary]
    return tuple(prim_pos), tuple(prim_quat)


def resolve_prim_scale(prim: Usd.Prim) -> tuple[float, float, float]:
    """Resolve the scale of a prim in the world frame.

    At an attribute level, a USD prim's scale is a scaling transformation applied to the prim with
    respect to its parent prim. This function resolves the scale of the prim in the world frame,
    by computing the local to world transform of the prim. This is equivalent to traversing up
    the prim hierarchy and accounting for the rotations and scales of the prims.

    For instance, if a prim has a scale of (1, 2, 3) and it is a child of a prim with a scale of (4, 5, 6),
    then the scale of the prim in the world frame is (4, 10, 18).

    Args:
        prim: The USD prim to resolve the scale for.

    Returns:
        The scale of the prim in the x, y, and z directions in the world frame.

    Raises:
        ValueError: If the prim is not valid.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>> from pxr import Usd, UsdGeom
        >>>
        >>> # Get prim
        >>> stage = sim_utils.get_current_stage()
        >>> prim = stage.GetPrimAtPath("/World/ImportedAsset")
        >>>
        >>> # Resolve scale
        >>> scale = sim_utils.resolve_prim_scale(prim)
        >>> print(f"Scale: {scale}")
    """
    # check if prim is valid
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath().pathString}' is not valid.")
    # compute local to world transform
    xform = UsdGeom.Xformable(prim)
    world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # extract scale
    return tuple([*(v.GetLength() for v in world_transform.ExtractRotationMatrix())])


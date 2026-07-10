# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Rotation-representation helpers.

Utilities for converting the continuous 6D rotation representation (two 3-vectors)
predicted/optimized by the models into a proper ``SO(3)`` rotation matrix.
"""

import roma
import torch


def robust_compute_rotation_matrix_from_ortho6d(poses):
    """Convert a continuous 6D rotation representation into a rotation matrix.

    Uses a symmetric (robust) Gram-Schmidt orthonormalization so both predicted
    direction vectors contribute to the result, rather than forcing the second vector
    to be orthogonal to the first. This yields a valid, differentiable ``SO(3)`` matrix
    even when the two raw vectors are not orthogonal.

    Args:
        poses: Tensor of shape ``(batch, 6)`` whose first 3 columns are the raw ``x``
            direction and last 3 columns the raw ``y`` direction.

    Returns:
        torch.Tensor: Rotation matrices of shape ``(batch, 3, 3)``.
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3
    return roma.special_gramschmidt(torch.stack([x_raw, y_raw], -1), epsilon=1e-5)  # batch*3*3

# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Tests for graspqp.utils.transforms (ortho6d -> rotation matrix)."""

import torch

from graspqp.utils.transforms import robust_compute_rotation_matrix_from_ortho6d as ortho6d_to_R


def _is_rotation(R, atol=1e-5):
    eye = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(R)
    orthonormal = torch.allclose(R @ R.transpose(-1, -2), eye, atol=atol)
    proper = torch.allclose(torch.det(R), torch.ones(R.shape[0], device=R.device, dtype=R.dtype), atol=atol)
    return orthonormal and proper


def test_identity_ortho6d_maps_to_identity():
    poses = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    R = ortho6d_to_R(poses)
    assert R.shape == (1, 3, 3)
    assert torch.allclose(R, torch.eye(3)[None], atol=1e-5)


def test_output_is_valid_rotation_for_random_batch():
    torch.manual_seed(0)
    poses = torch.randn(64, 6)
    R = ortho6d_to_R(poses)
    assert R.shape == (64, 3, 3)
    assert _is_rotation(R)


def test_non_orthogonal_input_still_yields_valid_rotation():
    # x and y nearly parallel -- the "robust" Gram-Schmidt must still return an SO(3) matrix.
    poses = torch.tensor([[1.0, 0.0, 0.0, 0.99, 0.01, 0.0]])
    R = ortho6d_to_R(poses)
    assert _is_rotation(R)


def test_gradients_flow_through():
    poses = torch.randn(8, 6, requires_grad=True)
    R = ortho6d_to_R(poses)
    R.sum().backward()
    assert poses.grad is not None
    assert torch.isfinite(poses.grad).all()


def test_deterministic():
    poses = torch.randn(4, 6)
    assert torch.equal(ortho6d_to_R(poses), ortho6d_to_R(poses))

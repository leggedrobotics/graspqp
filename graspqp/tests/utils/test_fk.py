# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Tests for graspqp.utils.fk (Robotiq 2F-140 coupled forward kinematics)."""

import pytest
import torch

from graspqp.utils.fk import (
    ROBOTIQ_2F_CFG140_model_joint_names as JOINT_NAMES,
    robotiq2f140_fk,
)


def test_output_shape_matches_joint_names():
    driven = torch.tensor([[0.4], [0.1], [0.0]])
    joints = robotiq2f140_fk(driven)
    assert joints.shape == (3, len(JOINT_NAMES))  # (N, 6)


def test_driven_angle_is_clamped():
    # limits are [-0.05, 0.8]; the driven joint is the first output column.
    assert robotiq2f140_fk(torch.tensor([[2.0]]))[0, 0].item() == pytest.approx(0.8)
    assert robotiq2f140_fk(torch.tensor([[-1.0]]))[0, 0].item() == pytest.approx(-0.05)


def test_joint_order_reindexes_columns():
    driven = torch.tensor([[0.3]])
    full = robotiq2f140_fk(driven)
    reordered = robotiq2f140_fk(driven, joint_order=JOINT_NAMES)
    assert torch.allclose(full, reordered)

    subset = ["right_inner_knuckle_joint", "finger_joint"]
    picked = robotiq2f140_fk(driven, joint_order=subset)
    assert picked.shape == (1, 2)
    assert torch.allclose(picked[:, 0], full[:, JOINT_NAMES.index(subset[0])])
    assert torch.allclose(picked[:, 1], full[:, JOINT_NAMES.index(subset[1])])


def test_unknown_joint_name_raises():
    with pytest.raises(ValueError):
        robotiq2f140_fk(torch.tensor([[0.3]]), joint_order=["not_a_joint"])


def test_deterministic():
    driven = torch.tensor([[0.25]])
    assert torch.allclose(robotiq2f140_fk(driven), robotiq2f140_fk(driven))

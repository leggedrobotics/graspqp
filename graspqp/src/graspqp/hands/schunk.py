"""Schunk EGU-50 two-finger gripper hand model implementation.

This module provides the kinematic and jacobian calculations for the Schunk EGU-50
parallel gripper. The gripper has two fingers that move symmetrically, controlled
by a single prismatic joint that drives both fingers in opposite directions.
"""

import torch
from graspqp.core import HandModel


def calculate_joints(joint_angles: torch.Tensor, hand_model):
    """Calculate forward kinematics for Schunk parallel gripper.

    Computes the poses of gripper links given joint angles. The Schunk gripper
    has symmetric fingers controlled by a single prismatic joint - the left finger
    moves in the positive direction while the right finger moves in the negative
    direction by the same amount.

    Args:
        joint_angles: Joint angles tensor with shape (..., 1) containing the
                     prismatic joint position for the left finger
        hand_model: HandModel instance containing the kinematic chain

    Returns:
        Forward kinematics result containing poses of all gripper links
    """
    # Create symmetric joint angles: left finger positive, right finger negative
    joint_angles = torch.stack(
        [
            joint_angles[..., 0],
            -joint_angles[..., 0],
        ],
        axis=-1,
    )
    return hand_model.chain.forward_kinematics(joint_angles)


def calculate_jacobian(joint_angles: torch.Tensor, hand_model):
    """Calculate jacobian matrix for Schunk parallel gripper.

    Computes the jacobian relating joint velocities to end-effector velocities.
    For the parallel gripper, this combines the jacobians of both fingers,
    accounting for their symmetric motion (opposite directions).

    Args:
        joint_angles: Joint angles tensor with shape (..., 1) containing the
                     prismatic joint position for the left finger
        hand_model: HandModel instance containing the kinematic chain

    Returns:
        Combined jacobian tensor with shape (..., 6, 1) relating the single
        prismatic joint to the relative motion between the fingers
    """
    # Create symmetric joint angles: left finger positive, right finger negative
    joint_angles = torch.stack(
        [
            joint_angles[..., 0],  # left finger (positive direction)
            -joint_angles[..., 0],  # right finger (negative direction)
        ],
        axis=-1,
    )
    # Calculate jacobians for both fingers
    jacobian = hand_model.chain.jacobian(joint_angles)

    # Combine jacobians: difference accounts for relative motion between fingers
    return (jacobian[..., 0] - jacobian[..., 1]).unsqueeze(-1)


def getHandModel(device: str, asset_dir: str, **kwargs) -> HandModel:
    params = dict(
        mjcf_path=f"{asset_dir}/schunk_2f/schunk.urdf",
        mesh_path=f"{asset_dir}/schunk_2f/meshes",
        contact_points_path=f"{asset_dir}/schunk_2f/contact_points.json",
        penetration_points_path=f"{asset_dir}/schunk_2f/penetration_points.json",
        device=device,
        n_surface_points=512,
        forward_axis="z",
        up_axis="x",
        use_collision_if_possible=True,
        only_use_collision=True,
        default_state=torch.tensor([0.0], device=device),
        joint_filter=["egu_50_prismatic_1"],
        joint_calc_fnc=calculate_joints,
        jacobian_fnc=calculate_jacobian,
    )
    params.update(kwargs)
    return HandModel(**params)

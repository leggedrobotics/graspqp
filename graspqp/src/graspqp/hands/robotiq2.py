# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Robotiq 2F-140 two-finger adaptive gripper model.

Builds a :class:`graspqp.core.HandModel` for the Robotiq 2F-140 gripper. The gripper
is driven by a single actuated joint (``finger_joint``); the remaining knuckle/finger
joints of the four-bar linkage are coupled and reconstructed from the driven angle via
the learned closed-form model in :func:`graspqp.utils.fk.robotiq2f140_fk`.
"""

import torch

from graspqp.core import HandModel
from graspqp.utils.fk import robotiq2f140_fk

# ['finger_joint', 'left_inner_finger_joint', 'left_inner_knuckle_joint', 'right_outer_knuckle_joint', 'right_inner_finger_joint', 'right_inner_knuckle_joint']


def calculate_joints(joint_angles: torch.Tensor, hand_model: HandModel):
    """Forward kinematics from the single driven joint of the 2F-140 gripper.

    Reconstructs all coupled linkage joints from the driven angle (ordered to match
    ``hand_model.joints_names``) and runs them through the kinematic chain.

    Args:
        joint_angles: Driven joint angle, shape ``(..., 1)`` in radians.
        hand_model: The :class:`~graspqp.core.HandModel` providing the kinematic chain
            and joint ordering.

    Returns:
        Link poses produced by ``hand_model.chain.forward_kinematics``.
    """
    fk_angles = robotiq2f140_fk(joint_angles, joint_order=hand_model.joints_names)

    return hand_model.chain.forward_kinematics(fk_angles)


def calculate_jacobian(joint_angles: torch.Tensor, hand_model):
    """Jacobian of the 2F-140 gripper w.r.t. its single driven joint.

    The driven ``finger_joint`` and the mirrored ``right_outer_knuckle_joint`` columns
    of the chain Jacobian are subtracted to yield the motion produced by the single
    actuated DoF.

    Args:
        joint_angles: Driven joint angle, shape ``(..., 1)`` in radians.
        hand_model: The :class:`~graspqp.core.HandModel` providing the kinematic chain.

    Returns:
        torch.Tensor: Jacobian reduced to the single actuated DoF, shape ``(..., 1)``
        in the last dimension.
    """
    joint_angles = robotiq2f140_fk(joint_angles, joint_order=hand_model.joints_names)
    jacobian = hand_model.chain.jacobian(joint_angles)
    return (jacobian[..., 0] - jacobian[..., 3]).unsqueeze(-1)


def getHandModel(device: str, asset_dir: str, **kwargs) -> HandModel:
    """Build the Robotiq 2F-140 gripper model.

    Loads the 2F-140 URDF, meshes and contact/penetration point definitions and
    registers the coupled forward-kinematics and Jacobian callbacks driven by the
    single ``finger_joint``.

    Args:
        device: Torch device for the model tensors (e.g. ``"cuda"``).
        asset_dir: Root assets directory; hand files are read from
            ``{asset_dir}/robotiq2``.
        **kwargs: Additional :class:`~graspqp.core.HandModel` arguments that override
            the defaults set here.

    Returns:
        HandModel: The configured Robotiq 2F-140 gripper model.
    """
    params = dict(
        mjcf_path=f"{asset_dir}/robotiq2/robotiq_2f140.urdf",
        mesh_path=f"{asset_dir}/robotiq2/meshes",
        contact_points_path=f"{asset_dir}/robotiq2/contact_points.json",
        penetration_points_path=f"{asset_dir}/robotiq2/penetration_points.json",
        device=device,
        n_surface_points=512,
        forward_axis="z",
        up_axis="x",
        use_collision_if_possible=True,
        default_state=torch.tensor([0.0], device=device),
        joint_calc_fnc=calculate_joints,
        jacobian_fnc=calculate_jacobian,
        joint_filter=["finger_joint"],
    )
    params.update(kwargs)
    return HandModel(**params)

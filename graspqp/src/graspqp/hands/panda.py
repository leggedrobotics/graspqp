# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Franka Emika Panda parallel-jaw gripper model.

Builds a :class:`graspqp.core.HandModel` for the Panda's two-finger parallel gripper.
The gripper is controlled by a single actuated DoF; both fingers mirror the same
commanded opening (in meters), which is expanded to the two physical prismatic joints
in the forward-kinematics and Jacobian callbacks.
"""

import torch

from graspqp.core import HandModel


def calculate_joints(joint_angles: torch.Tensor, hand_model):
    """Forward kinematics for the Panda gripper from its single actuated DoF.

    Args:
        joint_angles: Actuated opening, shape ``(..., 1)`` in meters, applied equally
            to the left and right finger.
        hand_model: The :class:`~graspqp.core.HandModel` providing the kinematic chain.

    Returns:
        Link poses produced by ``hand_model.chain.forward_kinematics``.
    """
    joint_angles = torch.stack(
        [
            joint_angles[..., 0],  # leftfinger,
            joint_angles[..., 0],  # rightfinger,
        ],
        axis=-1,
    )
    return hand_model.chain.forward_kinematics(joint_angles)


def calculate_jacobian(joint_angles: torch.Tensor, hand_model):
    """Jacobian of the Panda gripper w.r.t. its single actuated DoF.

    Both finger columns of the chain Jacobian are summed so the returned Jacobian maps
    the single commanded opening to the combined finger motion.

    Args:
        joint_angles: Actuated opening, shape ``(..., 1)`` in meters.
        hand_model: The :class:`~graspqp.core.HandModel` providing the kinematic chain.

    Returns:
        torch.Tensor: Jacobian reduced to the single actuated DoF, shape ``(..., 1)``
        in the last dimension.
    """
    joint_angles = torch.stack(
        [
            joint_angles[..., 0],  # leftfinger,
            joint_angles[..., 0],  # rightfinger,
        ],
        axis=-1,
    )
    jacobian = hand_model.chain.jacobian(joint_angles)
    return (jacobian[..., 0] + jacobian[..., 1]).unsqueeze(-1)


def getHandModel(device: str, asset_dir: str, **kwargs) -> HandModel:
    """Build the Franka Panda parallel-jaw gripper model.

    Loads the Panda gripper URDF, meshes and contact/penetration point definitions and
    registers the single-DoF forward-kinematics and Jacobian callbacks. The default
    state opens the gripper to ``0.04`` m per finger.

    Args:
        device: Torch device for the model tensors (e.g. ``"cuda"``).
        asset_dir: Root assets directory; hand files are read from
            ``{asset_dir}/panda``.
        **kwargs: Additional :class:`~graspqp.core.HandModel` arguments that override
            the defaults set here.

    Returns:
        HandModel: The configured Panda gripper model.
    """
    params = dict(
        mjcf_path=f"{asset_dir}/panda/franka_panda.urdf",
        mesh_path=f"{asset_dir}/panda/meshes",
        contact_points_path=f"{asset_dir}/panda/contact_points.json",
        penetration_points_path=f"{asset_dir}/panda/penetration_points.json",
        device=device,
        n_surface_points=512,
        forward_axis="z",
        up_axis="x",
        use_collision_if_possible=True,
        default_state=torch.tensor([0.04], device=device),
        joint_filter=["panda_finger_joint1"],
        joint_calc_fnc=calculate_joints,
        jacobian_fnc=calculate_jacobian,
    )
    params.update(kwargs)
    return HandModel(**params)

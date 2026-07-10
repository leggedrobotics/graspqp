# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Forward kinematics for the coupled Robotiq 2F-140 linkage.

The Robotiq 2F-140 gripper is a single-DoF four-bar linkage: only ``finger_joint`` is
driven, and the remaining knuckle/finger joints follow through the mechanism. This
module loads a small learned model (``robotiq2f_fk.pth``) that maps the driven angle to
the five coupled joint angles, exposed via :func:`robotiq2f140_fk`.
"""

import os

import torch

file_dir = os.path.dirname(__file__)
WEIGHTS_PATH = os.path.join(
    file_dir,
    "../../../assets/robotiq2/robotiq2f_fk.pth",
)
global ROBOTIQ_2F_CFG140_model
ROBOTIQ_2F_CFG140_model = None

ROBOTIQ_2F_CFG140_model_joint_names = [
    "finger_joint",
    "left_inner_knuckle_joint",
    "right_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
]


def robotiq2f140_fk(driven_angle: torch.Tensor, joint_order=None) -> torch.Tensor:
    """Reconstruct all coupled joint angles of the Robotiq 2F-140 from its driven angle.

    The single driven angle is clamped to the mechanism's limits ``[-0.05, 0.8]`` rad,
    passed through the learned coupling model to obtain the five dependent joint angles,
    and concatenated with the driven angle to form the full 6-joint state.

    The model's native joint order is
    ``[finger_joint, left_inner_knuckle_joint, right_inner_knuckle_joint,
    right_outer_knuckle_joint, left_inner_finger_joint, right_inner_finger_joint]``
    (see :data:`ROBOTIQ_2F_CFG140_model_joint_names`).

    Args:
        driven_angle: Driven ``finger_joint`` angle, shape ``(N, 1)`` in radians.
        joint_order: Optional list of joint names; if given, the returned columns are
            reordered/selected to match it. Each name must be one of the model's joint
            names.

    Returns:
        torch.Tensor: Full joint angles, shape ``(N, 6)`` in radians, in the model's
        native order or in ``joint_order`` if provided.

    Raises:
        ValueError: If ``joint_order`` contains a name not present in the model.
    """
    # hard clip driven_angle to limits
    driven_angle = torch.clamp(driven_angle, min=-0.05, max=0.8)
    global ROBOTIQ_2F_CFG140_model
    if ROBOTIQ_2F_CFG140_model is None:
        ROBOTIQ_2F_CFG140_model = torch.load(WEIGHTS_PATH, weights_only=False)
        ROBOTIQ_2F_CFG140_model.eval()
    ROBOTIQ_2F_CFG140_model.to(driven_angle.device)
    joints = torch.cat([driven_angle, ROBOTIQ_2F_CFG140_model(driven_angle)], dim=-1)

    if joint_order is not None:
        joint_order_idxs = []
        for joint_name in joint_order:
            if joint_name in ROBOTIQ_2F_CFG140_model_joint_names:
                joint_order_idxs.append(ROBOTIQ_2F_CFG140_model_joint_names.index(joint_name))
            else:
                raise ValueError(f"Joint name {joint_name} not found in model.")

        return joints[..., joint_order_idxs]
    return joints


if __name__ == "__main__":
    # Test the forward kinematics function
    driven_angle = torch.tensor([[0.4]], dtype=torch.float32)
    full_chain_states = robotiq2f140_fk(driven_angle)

    # joint names

    print("Full joints position:", full_chain_states)

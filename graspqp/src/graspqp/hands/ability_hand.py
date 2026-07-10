# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""PSYONIC Ability Hand model.

Builds a :class:`graspqp.core.HandModel` for the 6-DoF PSYONIC Ability Hand. The
four fingers (index, middle, ring, little) are each mechanically coupled: the distal
joint (``*_q2``) follows the proximal joint (``*_q1``) through a fixed transmission
ratio, so only the proximal joint is actuated. The thumb has two independent joints
(``thumb_q1``, ``thumb_q2``). The coupling is expressed here in
:func:`get_all_joint_angles` and threaded into the model via custom forward-kinematics
and Jacobian callbacks.
"""

import json
import os

import torch

from graspqp.core import HandModel

# Transmission ratio coupling each finger's distal joint (`*_q2`) to its
# proximal joint (`*_q1`): q2 = mult * q1 + offset.
def get_all_joint_angles(joint_angles: torch.Tensor):
    """Expand the 6 actuated DoF to the 10 physical joint angles of the Ability Hand.

    The four fingers are underactuated: each distal joint follows its proximal joint
    through the fixed transmission ``q2 = mult * q1 + offset``. The thumb's two joints
    are passed through unchanged.

    Args:
        joint_angles: Actuated joint angles, shape ``(..., 6)`` in radians, ordered
            ``[index_q1, middle_q1, ring_q1, little_q1, thumb_q1, thumb_q2]``.

    Returns:
        torch.Tensor: Full physical joint angles, shape ``(..., 10)`` in radians,
        ordered ``[index_q1, index_q2, middle_q1, middle_q2, ring_q1, ring_q2,
        little_q1, little_q2, thumb_q1, thumb_q2]``.
    """
    mult, offset = 1.05851325, 0.0
    joint_angles = torch.stack(
        [
            joint_angles[..., 0],  # index_q1
            joint_angles[..., 0] * mult + offset,  # index_q2
            joint_angles[..., 1],  # middle_q1
            joint_angles[..., 1] * mult + offset,  # middle_q2
            joint_angles[..., 2],  # ring_q1
            joint_angles[..., 2] * mult + offset,  # ring_q2
            joint_angles[..., 3],  # little_q1
            joint_angles[..., 3] * mult + offset,  # little_q2
            joint_angles[..., 4],  # thumb_q1
            joint_angles[..., 5],  # thumb_q2
        ],
        axis=-1,
    )
    return joint_angles


def calculate_joints(joint_angles: torch.Tensor, hand_model):
    """Forward kinematics from the 6 actuated DoF, accounting for finger coupling.

    Args:
        joint_angles: Actuated joint angles, shape ``(..., 6)`` in radians.
        hand_model: The :class:`~graspqp.core.HandModel` providing the kinematic chain.

    Returns:
        Link poses produced by ``hand_model.chain.forward_kinematics`` for the full
        set of physical joints.
    """
    return hand_model.chain.forward_kinematics(get_all_joint_angles(joint_angles))


def calculate_jacobian(joint_angles: torch.Tensor, hand_model):
    """Jacobian w.r.t. the 6 actuated DoF, folding in the finger coupling.

    The chain Jacobian is computed for all 10 physical joints and then reduced to the
    6 actuated columns: each finger's coupled distal-joint column is added (scaled by
    the transmission ratio) into its proximal-joint column, and the two thumb columns
    are kept independent.

    Args:
        joint_angles: Actuated joint angles, shape ``(..., 6)`` in radians.
        hand_model: The :class:`~graspqp.core.HandModel` providing the kinematic chain.

    Returns:
        torch.Tensor: Jacobian reduced to the 6 actuated DoF.
    """
    mult = 1.05851325
    jacobian = hand_model.chain.jacobian(get_all_joint_angles(joint_angles))
    # modify the jacobian to account for the fact that the thumb_q2 joint is not used
    active_jacobian = jacobian[..., [0, 2, 4, 6, 8, 9]]
    active_jacobian[..., :-2] = active_jacobian[..., :-2] + jacobian[..., [1, 3, 5, 7]] * mult
    return active_jacobian


def getHandModel(device: str, asset_dir: str, grasp_type: str = "all", **kwargs) -> HandModel:
    """Build the PSYONIC Ability Hand model.

    Loads the Ability Hand URDF, meshes and contact/penetration point definitions and
    wires up the finger-coupling forward-kinematics and Jacobian callbacks. The hand is
    controlled through 6 actuated DoF ``[index_q1, middle_q1, pinky_q1, ring_q1,
    thumb_q1, thumb_q2]``.

    Args:
        device: Torch device for the model tensors (e.g. ``"cuda"``).
        asset_dir: Root assets directory; hand files are read from
            ``{asset_dir}/ability_hand``.
        grasp_type: Selects which subset of contact links is active. ``"all"`` (or
            ``"default"``) uses every link; any other value is looked up in
            ``eigengrasps.json`` to restrict the active contact links to a named
            grasp/eigengrasp preset.
        **kwargs: Additional :class:`~graspqp.core.HandModel` arguments that override
            the defaults set here.

    Returns:
        HandModel: The configured Ability Hand model.

    Raises:
        ValueError: If ``grasp_type`` is not ``"all"`` and either ``eigengrasps.json``
            is missing or does not contain the requested grasp type.
    """
    contact_links = None
    if grasp_type == "default":
        grasp_type = "all"

    if grasp_type is not None and grasp_type != "all":
        eigengrasp_file = f"{asset_dir}/ability_hand/eigengrasps.json"
        if not os.path.exists(eigengrasp_file):
            raise ValueError(f"eigengrasps.json not found at {eigengrasp_file}")
        json_data = json.load(open(eigengrasp_file))
        if grasp_type not in json_data:
            raise ValueError(
                f"grasp type {grasp_type} not found in eigengrasps.json. Available grasp types are {list(json_data.keys())}"
            )
        contact_links = json_data[grasp_type]

    params = dict(
        mjcf_path=f"{asset_dir}/ability_hand/ability_hand.urdf",
        mesh_path=f"{asset_dir}/ability_hand/urdf_meshes",
        contact_points_path=f"{asset_dir}/ability_hand/contact_points.json",
        penetration_points_path=f"{asset_dir}/ability_hand/penetration_points.json",
        contact_links=contact_links,
        device=device,
        n_surface_points=512,
        forward_axis="z",
        up_axis="x",
        grasp_axis="y",
        use_collision_if_possible=True,
        default_state=torch.tensor(
            [
                0.3,  # index_q1
                0.3,  # middle_q1
                0.3,  # pinky_q1
                0.3,  # ring_q1
                1,  # thumb_q1
                0,  # thumb_q2
            ],
            dtype=torch.float,
            device=device,
        ),
        joint_filter=[
            "index_q1",
            "middle_q1",
            "pinky_q1",
            "ring_q1",
            "thumb_q1",
            "thumb_q2",
        ],
        joint_calc_fnc=calculate_joints,
        jacobian_fnc=calculate_jacobian,
        grasp_type=grasp_type,
    )
    params.update(kwargs)
    return HandModel(**params)

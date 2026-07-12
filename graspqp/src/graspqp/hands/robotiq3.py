# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Robotiq 3-Finger adaptive gripper model.

Builds a :class:`graspqp.core.HandModel` for the Robotiq 3-Finger gripper in its flat
(non-coupled) configuration, where the joints are treated as directly actuated. The
default state loads a slightly flexed three-finger pose.
"""

import torch

from graspqp.core import HandModel


def getHandModel(device: str, asset_dir: str, **kwargs) -> HandModel:
    """Build the Robotiq 3-Finger gripper model.

    Loads the 3-Finger (flat) URDF, meshes and contact/penetration point definitions.

    Args:
        device: Torch device for the model tensors (e.g. ``"cuda"``).
        asset_dir: Root assets directory; hand files are read from
            ``{asset_dir}/robotiq3``.
        **kwargs: Additional :class:`~graspqp.core.HandModel` arguments that override
            the defaults set here.

    Returns:
        HandModel: The configured Robotiq 3-Finger gripper model.
    """
    params = dict(
        mjcf_path=f"{asset_dir}/robotiq3/robotiq_3finger_flat.urdf",
        mesh_path=f"{asset_dir}/robotiq3/meshes",
        contact_points_path=f"{asset_dir}/robotiq3/contact_points.json",
        penetration_points_path=f"{asset_dir}/robotiq3/penetration_points.json",
        device=device,
        n_surface_points=512,
        forward_axis="z",
        up_axis="x",
        use_collision_if_possible=True,
        default_state=torch.tensor([0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.3, 0.3, 0.3, 0.0, 0.0], device=device),
    )
    params.update(kwargs)
    return HandModel(**params)

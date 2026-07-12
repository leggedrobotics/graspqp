# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Allegro Hand model.

Builds a :class:`graspqp.core.HandModel` for the 16-DoF Wonik Allegro Hand (four
fully actuated 4-DoF fingers). Unlike the coupled/underactuated grippers, every joint
is directly actuated, so no custom kinematics callbacks are needed.
"""

import torch

from graspqp.core import HandModel


def getHandModel(device: str, asset_dir: str, **kwargs) -> HandModel:
    """Build the Allegro Hand model.

    Loads the Allegro URDF, meshes and contact/penetration point definitions and
    initializes the 16 joints (4 per finger) to a slightly flexed default pose.

    Args:
        device: Torch device for the model tensors (e.g. ``"cuda"``).
        asset_dir: Root assets directory; hand files are read from
            ``{asset_dir}/allegro``.
        **kwargs: Additional :class:`~graspqp.core.HandModel` arguments that override
            the defaults set here.

    Returns:
        HandModel: The configured Allegro Hand model.
    """
    params = dict(
        mjcf_path=f"{asset_dir}/allegro/allegro_hand.urdf",
        mesh_path=f"{asset_dir}/allegro/meshes",
        contact_points_path=f"{asset_dir}/allegro/contact_points.json",
        penetration_points_path=f"{asset_dir}/allegro/penetration_points.json",
        device=device,
        n_surface_points=512,
        forward_axis="z",
        up_axis="x",
        grasp_axis="y",
        use_collision_if_possible=True,
        default_state=torch.tensor(
            [
                0.0,
                0.2,
                0.5,
                0.5,
                0,
                0.2,
                0.5,
                0.5,
                0.0,
                0.2,
                0.5,
                0.5,
                1.0,
                0.5,
                0.5,
                0.2,
            ],
            device=device,
        ),
    )
    params.update(kwargs)
    return HandModel(**params)

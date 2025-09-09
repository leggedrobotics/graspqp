from graspqp.core import HandModel
import torch
import json
import os


def getHandModel(
    device: str, asset_dir: str, grasp_type: str = "all", **kwargs
) -> HandModel:
    contact_links = None

    if grasp_type is not None and grasp_type != "all":
        eigengrasp_file = f"{asset_dir}/shadow_hand/eigengrasps.json"
        if not os.path.exists(eigengrasp_file):
            raise ValueError(f"eigengrasps.json not found at {eigengrasp_file}")
        json_data = json.load(open(eigengrasp_file))
        if grasp_type not in json_data:
            raise ValueError(
                f"grasp type {grasp_type} not found in eigengrasps.json. Available grasp types are {list(json_data.keys())}"
            )
        contact_links = json_data[grasp_type]

    params = dict(
        mjcf_path=f"{asset_dir}/shadow_hand/shadow_hand.urdf",
        mesh_path=f"{asset_dir}/shadow_hand/meshes",
        contact_points_path=f"{asset_dir}/shadow_hand/contact_points.json",
        penetration_points_path=f"{asset_dir}/shadow_hand/penetration_points.json",
        contact_links=contact_links,
        device=device,
        n_surface_points=512,
        forward_axis="z",
        up_axis="x",
        grasp_axis = "y",
        use_collision_if_possible=True,
        default_state=torch.tensor(
            [
                0,  # robot0_WRJ1
                0,  # robot0_WRJ0
                0.1,  # robot0_FFJ3
                0,  # robot0_FFJ2
                0.6,  # robot0_FFJ1
                0,  # robot0_FFJ0
                0,  # robot0_LFJ4
                -0.2,  # robot0_LFJ3
                0,  # robot0_LFJ2
                0.6,  # robot0_LFJ1
                0,  # robot0_LFJ0
                0,  # robot0_MFJ3
                0.0,  # robot0_MFJ2
                0.6,  # robot0_MFJ1
                0,  # robot0_MFJ0
                -0.1,  # robot0_RFJ3
                0,  # robot0_RFJ2
                0.6,  # robot0_RFJ1
                0,  # robot0_RFJ0
                0,  # robot0_THJ4
                1.2,  # robot0_THJ3
                0.0,  # robot0_THJ2
                -0.2,  # robot0_THJ1
                0.0,  # robot0_THJ0
            ],
            device=device,
            dtype=torch.float,
        ),
        grasp_type=grasp_type,
    )
    params.update(kwargs)
    return HandModel(**params)

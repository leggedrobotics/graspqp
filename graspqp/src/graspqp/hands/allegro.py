import torch
from graspqp.core import HandModel


def getHandModel(device: str, asset_dir: str, **kwargs) -> HandModel:
    params = dict(
        mjcf_path=f"{asset_dir}/allegro/allegro_hand.urdf",
        mesh_path=f"{asset_dir}/allegro/meshes",
        contact_points_path=f"{asset_dir}/allegro/contact_points.json",
        penetration_points_path=f"{asset_dir}/allegro/penetration_points.json",
        device=device,
        n_surface_points=512,
        forward_axis="z",
        up_axis="x",
        grasp_axis = "y",
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

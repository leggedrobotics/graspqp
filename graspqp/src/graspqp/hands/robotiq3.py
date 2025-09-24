import torch

from graspqp.core import HandModel


def getHandModel(device: str, asset_dir: str, **kwargs) -> HandModel:
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

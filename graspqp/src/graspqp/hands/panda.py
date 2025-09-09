import torch
from graspqp.core import HandModel


def calculate_joints(joint_angles: torch.Tensor, hand_model):
    joint_angles = torch.stack(
        [
            joint_angles[..., 0],  # leftfinger,
            joint_angles[..., 0],  # rightfinger,
        ],
        axis=-1,
    )
    return hand_model.chain.forward_kinematics(joint_angles)


def calculate_jacobian(joint_angles: torch.Tensor, hand_model):
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

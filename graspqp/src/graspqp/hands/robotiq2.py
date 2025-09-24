import torch

from graspqp.core import HandModel
from graspqp.utils.fk import robotiq2f140_fk

# ['finger_joint', 'left_inner_finger_joint', 'left_inner_knuckle_joint', 'right_outer_knuckle_joint', 'right_inner_finger_joint', 'right_inner_knuckle_joint']


def calculate_joints(joint_angles: torch.Tensor, hand_model: HandModel):

    fk_angles = robotiq2f140_fk(joint_angles, joint_order=hand_model.joints_names)

    return hand_model.chain.forward_kinematics(fk_angles)


def calculate_jacobian(joint_angles: torch.Tensor, hand_model):
    joint_angles = robotiq2f140_fk(joint_angles, joint_order=hand_model.joints_names)
    jacobian = hand_model.chain.jacobian(joint_angles)
    return (jacobian[..., 0] - jacobian[..., 3]).unsqueeze(-1)


def getHandModel(device: str, asset_dir: str, **kwargs) -> HandModel:
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

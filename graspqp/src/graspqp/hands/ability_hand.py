from graspqp.core import HandModel
import torch
import os
import json

def get_all_joint_angles(joint_angles: torch.Tensor):
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
    return hand_model.chain.forward_kinematics(get_all_joint_angles(joint_angles))


def calculate_jacobian(joint_angles: torch.Tensor, hand_model):
    mult = 1.05851325
    jacobian = hand_model.chain.jacobian(get_all_joint_angles(joint_angles))
    # modify the jacobian to account for the fact that the thumb_q2 joint is not used
    active_jacobian = jacobian[..., [0, 2, 4, 6, 8, 9]]
    active_jacobian[..., :-2] = (
        active_jacobian[..., :-2] + jacobian[..., [1, 3, 5, 7]] * mult
    )
    return active_jacobian


def getHandModel(device: str, asset_dir: str, grasp_type:str = "all", **kwargs) -> HandModel:
    contact_links = None
    
    if grasp_type is not None and grasp_type != "all":
        eigengrasp_file = f"{asset_dir}/ability_hand/eigengrasps.json"
        if not os.path.exists(eigengrasp_file):
            raise ValueError(f"eigengrasps.json not found at {eigengrasp_file}")
        json_data = json.load(open(eigengrasp_file))
        if grasp_type not in json_data:
            raise ValueError(f"grasp type {grasp_type} not found in eigengrasps.json. Available grasp types are {list(json_data.keys())}")
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
        grasp_axis = "y",
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

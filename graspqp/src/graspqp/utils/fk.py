import torch
import os

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
    """
    Computes the forward kinematics of the Robotiq 2F-140 gripper.

    Args:
        driven_angle: A tensor of shape (N, 3) representing the joint angles of the gripper.

    Returns:
        A tensor of shape (N, 3) representing the end effector position in world coordinates.
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

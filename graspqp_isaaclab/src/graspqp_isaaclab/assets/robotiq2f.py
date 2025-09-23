import math
import os

import isaaclab.sim as sim_utils
from graspqp_isaaclab.models.hand_model_cfg import HandModelCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from graspqp.utils.fk import robotiq2f140_fk

##
# Configuration
##

ROBOTIQ_2F_ACTUATED_JOINT_NAMES = [
    "finger_joint",
]


def cal_joint_pos(joint_pos):
    """Set the joint position of the Robotiq 2F."""
    return robotiq2f140_fk(joint_pos[:, [0]])


ROBOTIQ_2F_CFG = HandModelCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            os.path.dirname(__file__),
            "Robotiq2f",
            "robotiq_2f140.usd",
        ),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=ROBOTIQ_2F_ACTUATED_JOINT_NAMES,
            effort_limit=20 * 0.125,  # 20N actuation force at finger length of 0.125m
            velocity_limit=0.88,  # linear speed of 0.110m/s at length of 0.125m -> 0.110/0.125 = 0.88 rad/s
            stiffness=100.0,
            damping=0.0,
            friction=0.0,
        ),
        "implicit": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_inner_knuckle_joint",
                "right_inner_knuckle_joint",
                "right_outer_knuckle_joint",
                "left_inner_finger_joint",
                "right_inner_finger_joint",
            ],
            effort_limit=1000,  # 20N actuation force at finger length of 0.125m
            velocity_limit=0.88,  # linear speed of 0.110m/s at length of 0.125m -> 0.110/0.125 = 0.88 rad/s
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
    actuated_joints_expr=ROBOTIQ_2F_ACTUATED_JOINT_NAMES,
    mimic_joints={
        "right_outer_knuckle_joint": {
            "parent": "finger_joint",
            "offset": 0.0,
            "multiplier": -1.0,
        },
    },
    init_fnc=cal_joint_pos,
    hand_model_name="robotiq2f",
)
"""Configuration of Robotiq 2F robot."""

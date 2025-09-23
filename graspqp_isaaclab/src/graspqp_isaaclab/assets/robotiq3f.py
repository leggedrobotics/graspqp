import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from graspqp_isaaclab.models.hand_model_cfg import HandModelCfg

##
# Configuration
##

ROBOTIQ_3F_ACTUATED_JOINT_NAMES = [
    "RIQ_palm_RIQ_link_0_joint",
    "RIQ_palm_RIQ_link_0_joint_b",
    "RIQ_palm_RIQ_link_1_joint_a",
    "RIQ_link_0_RIQ_link_1_joint_c",
    "RIQ_link_0_RIQ_link_1_joint_b",
    "RIQ_link_1_RIQ_link_2_joint_a",
    "RIQ_link_1_RIQ_link_2_joint_c",
    "RIQ_link_1_RIQ_link_2_joint_b",
    "RIQ_link_2_RIQ_link_3_joint_a",
    "RIQ_link_2_RIQ_link_3_joint_c",
    "RIQ_link_2_RIQ_link_3_joint_b",
]


ROBOTIQ_3F_CFG = HandModelCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            os.path.dirname(__file__),
            "Robotiq3f",
            "robotiq_3f.usd",
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
            solver_position_iteration_count=8,
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
            joint_names_expr=ROBOTIQ_3F_ACTUATED_JOINT_NAMES,
            effort_limit=20 * 0.125,  # 20N actuation force at finger length of 0.125m
            velocity_limit=0.88,  # linear speed of 0.110m/s at length of 0.125m -> 0.110/0.125 = 0.88 rad/s
            stiffness=1000.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
    actuated_joints_expr=ROBOTIQ_3F_ACTUATED_JOINT_NAMES,
    mimic_joints={},
    hand_model_name="robotiq3f",
)
"""Configuration of Shadow Hand robot."""

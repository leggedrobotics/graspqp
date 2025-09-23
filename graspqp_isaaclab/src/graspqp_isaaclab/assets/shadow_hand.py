import os

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from graspqp_isaaclab.models.hand_model_cfg import HandModelCfg

##
# Configuration
##

SHADOW_HAND_CFG = HandModelCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            os.path.dirname(__file__),
            "shadow_hand",
            "shadow_hand.usd",
        ),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.00,
            stabilization_threshold=0.0005,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["robot0_WR.*", "robot0_(FF|MF|RF|LF|TH)J(3|2|1)", "robot0_(LF|TH)J4", "robot0_THJ0"],
            effort_limit={
                # ".*": .78,
                "robot0_WRJ1": 4.785,
                "robot0_WRJ0": 2.175,
                "robot0_(FF|MF|RF|LF)J1": 0.7245,
                "robot0_FFJ(3|2)": 0.9,
                "robot0_MFJ(3|2)": 0.9,
                "robot0_RFJ(3|2)": 0.9,
                "robot0_LFJ(4|3|2)": 0.9,
                "robot0_THJ4": 2.3722,
                "robot0_THJ3": 1.45,
                "robot0_THJ(2|1)": 0.99,
                "robot0_THJ0": 0.81,
            },
            stiffness={
                "robot0_WRJ.*": 100.0,
                "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 100.0,
                "robot0_(LF|TH)J4": 100.0,
                "robot0_THJ0": 100.0,
            },
            damping={
                "robot0_WRJ.*": 0.05,
                "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 0.01,
                "robot0_(LF|TH)J4": 0.01,
                "robot0_THJ0": 0.01,
            },
            velocity_limit={
                ".*": 3.0,
            },
        ),
    },
    actuated_joints_expr=[
        "robot0_WRJ1",
        "robot0_WRJ0",
        "robot0_FFJ3",
        "robot0_MFJ3",
        "robot0_RFJ3",
        "robot0_LFJ4",
        "robot0_THJ4",
        "robot0_FFJ2",
        "robot0_MFJ2",
        "robot0_RFJ2",
        "robot0_LFJ3",
        "robot0_THJ3",
        "robot0_FFJ1",
        "robot0_MFJ1",
        "robot0_RFJ1",
        "robot0_LFJ2",
        "robot0_THJ2",
        "robot0_FFJ0",
        "robot0_MFJ0",
        "robot0_RFJ0",
        "robot0_LFJ1",
        "robot0_THJ1",
        "robot0_LFJ0",
        "robot0_THJ0",
    ],
    soft_joint_pos_limit_factor=1.0,
    hand_model_name="shadow_hand",
)

import math
import os

import isaaclab.sim as sim_utils
from graspqp_isaaclab.models.hand_model_cfg import HandModelCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ALLEGRO_HAND_ACTUATED_JOINT_NAMES = [
    "index_joint_0",
    "middle_joint_0",
    "ring_joint_0",
    "thumb_joint_0",
    "index_joint_1",
    "middle_joint_1",
    "ring_joint_1",
    "thumb_joint_1",
    "index_joint_2",
    "middle_joint_2",
    "ring_joint_2",
    "thumb_joint_2",
    "index_joint_3",
    "middle_joint_3",
    "ring_joint_3",
    "thumb_joint_3",
]

##
# Configuration
##
ALLEGRO_HAND_CFG = HandModelCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            os.path.dirname(__file__),
            "allegro",
            "allegro_hand.usd",
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
        variants={"Fingertips": "Default"},
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        # rot=(0.257551, 0.283045, 0.683330, -0.621782),
        joint_pos={"thumb_joint_0": 0.28},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["index.*", "middle.*", "ring.*", "thumb.*"],
            effort_limit=1.6,
            velocity_limit=2.0,
            stiffness=100.0,
            damping=0.22,
            friction=0.15,
            armature=0.003,
        ),
    },
    actuated_joints_expr=ALLEGRO_HAND_ACTUATED_JOINT_NAMES,
    soft_joint_pos_limit_factor=1.0,
    hand_model_name="allegro",
)

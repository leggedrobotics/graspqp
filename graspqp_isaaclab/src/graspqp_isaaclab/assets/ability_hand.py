import math
import os

import isaaclab.sim as sim_utils
from graspqp_isaaclab.models.hand_model_cfg import HandModelCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

DYNAARM_ARM_JOINT_NAMES = ["SH_ROT", "SH_FLE", "EL_FLE", "FA_ROT", "WRIST_1", "WRIST_2"]

##
# Configuration
##

ABILITY_HAND_ACTUATED_JOINT_NAMES = [
    "index_q1",
    "middle_q1",
    "ring_q1",
    "pinky_q1",
    "thumb_q1",
    "thumb_q2",
]
ABILITY_HAND_MIMIC_JOINT_NAMES = ["index_q2", "middle_q2", "pinky_q2", "ring_q2"]


ABILITY_HAND_DEFAULT = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.4),
    # rot=(0.0, 1.0, 0.0, 0.0),
    joint_pos={
        "index_q1": 0.3,
        "middle_q1": 0.3,
        "pinky_q1": 0.3,
        "ring_q1": 0.3,
        "thumb_q1": 0.3,
        "thumb_q2": 0.2,
        "index_q2": 0.3,
        "middle_q2": 0.3,
        "pinky_q2": 0.3,
        "ring_q2": 0.3,
    },
)

ALL_ABILITY_HAND_JOINT_NAMES = [
    "index_q1",
    "middle_q1",
    "pinky_q1",
    "ring_q1",
    "thumb_q1",
    "index_q2",
    "middle_q2",
    "pinky_q2",
    "ring_q2",
    "thumb_q2",
]


FINGER_Q1_P_GAIN = 10
FINGER_Q1_D_GAIN = 0.1
FINGER_Q1_FRICTION = 0.05
FINGER_Q1_ARMATURE = 0.01
FINGER_Q1_EFFORT_LIMIT = 9.3 / 2 * 0.102 * 1e2  # Finger length is 0.102m, max linear force is 9.3N/2

FINGER_Q2_P_GAIN = 1.5 * FINGER_Q1_P_GAIN
FINGER_Q2_D_GAIN = 0.1 * 1.5
FINGER_Q2_FRICTION = 0.05
FINGER_Q2_ARMATURE = 0
FINGER_Q2_EFFORT_LIMIT = 1e5 * FINGER_Q1_EFFORT_LIMIT


THUMB_Q1_P_GAIN = FINGER_Q1_P_GAIN
THUMB_Q1_D_GAIN = FINGER_Q1_D_GAIN
THUMB_Q1_FRICTION = FINGER_Q1_FRICTION
THUMB_Q1_ARMATURE = FINGER_Q1_ARMATURE
THUMB_Q1_EFFORT_LIMIT = FINGER_Q1_EFFORT_LIMIT

THUMB_Q2_P_GAIN = FINGER_Q2_P_GAIN
THUMB_Q2_D_GAIN = FINGER_Q2_D_GAIN
THUMB_Q2_FRICTION = FINGER_Q2_FRICTION
THUMB_Q2_ARMATURE = FINGER_Q2_ARMATURE
THUMB_Q2_EFFORT_LIMIT = FINGER_Q2_EFFORT_LIMIT

ABILITY_HAND_GRIPPER_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=ABILITY_HAND_ACTUATED_JOINT_NAMES + ABILITY_HAND_MIMIC_JOINT_NAMES,
    stiffness={
        "index_q1": FINGER_Q1_P_GAIN,
        "middle_q1": FINGER_Q1_P_GAIN,
        "pinky_q1": FINGER_Q1_P_GAIN,
        "ring_q1": FINGER_Q1_P_GAIN,
        "thumb_q1": THUMB_Q1_P_GAIN,
        "thumb_q2": THUMB_Q2_P_GAIN,
    },
    damping={
        "index_q1": FINGER_Q1_D_GAIN,
        "middle_q1": FINGER_Q1_D_GAIN,
        "pinky_q1": FINGER_Q1_D_GAIN,
        "ring_q1": FINGER_Q1_D_GAIN,
        "thumb_q1": THUMB_Q1_D_GAIN,
        "thumb_q2": THUMB_Q2_D_GAIN,
    },
    friction={
        "index_q1": FINGER_Q1_FRICTION,
        "middle_q1": FINGER_Q1_FRICTION,
        "pinky_q1": FINGER_Q1_FRICTION,
        "ring_q1": FINGER_Q1_FRICTION,
        "thumb_q1": THUMB_Q1_FRICTION,
        "thumb_q2": THUMB_Q2_FRICTION,
    },
    armature={
        "index_q1": FINGER_Q1_ARMATURE,
        "middle_q1": FINGER_Q1_ARMATURE,
        "pinky_q1": FINGER_Q1_ARMATURE,
        "ring_q1": FINGER_Q1_ARMATURE,
        "thumb_q1": THUMB_Q1_ARMATURE,
        "thumb_q2": THUMB_Q2_ARMATURE,
    },
)

ABILITY_HAND_CFG = HandModelCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            os.path.dirname(__file__),
            "AbilityHand",
            "AbilityHandMimicFlat.usd",
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
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0, torsional_patch_radius=0.002),
    ),
    init_state=ABILITY_HAND_DEFAULT,
    actuators={
        "hand": ABILITY_HAND_GRIPPER_ACTUATOR_CFG,
    },
    actuated_joints_expr=ABILITY_HAND_ACTUATED_JOINT_NAMES,
    mimic_joints={
        "index_q2": {
            "parent": "index_q1",
            "offset": 0.0,
            "multiplier": 1.0,
        },
        "middle_q2": {
            "parent": "middle_q1",
            "offset": 0.0,
            "multiplier": 1.0,
        },
        "pinky_q2": {
            "parent": "pinky_q1",
            "offset": 0.0,
            "multiplier": 1.0,
        },
        "ring_q2": {
            "parent": "ring_q1",
            "offset": 0.0,
            "multiplier": 1.0,
        },
    },
    hand_model_name="ability_hand",
)

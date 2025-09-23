import os

import isaaclab.sim as sim_utils
from graspqp_isaaclab.models.hand_model_cfg import HandModelCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

SCHUNK_2F_ACTUATED_JOINT_NAMES = [
    "egu_50_prismatic_1",
]


SCHUNK_2F_CFG = HandModelCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            os.path.dirname(__file__),
            "Schunk2f",
            "schunk.usd",
        ),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
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
            joint_names_expr=SCHUNK_2F_ACTUATED_JOINT_NAMES,
            effort_limit=20 * 0.125,  # 20N actuation force at finger length of 0.125m
            velocity_limit=0.88,  # linear speed of 0.110m/s at length of 0.125m -> 0.110/0.125 = 0.88 rad/s
            stiffness=100.0,
            damping=0.0,
            friction=0.0,
        ),
        "implicit": ImplicitActuatorCfg(
            joint_names_expr=[
                "egu_50_prismatic_2",
            ],
            effort_limit=1000,  # 20N actuation force at finger length of 0.125m
            velocity_limit=0.88,  # linear speed of 0.110m/s at length of 0.125m -> 0.110/0.125 = 0.88 rad/s
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
    actuated_joints_expr=SCHUNK_2F_ACTUATED_JOINT_NAMES,
    mimic_joints={
        "egu_50_prismatic_2": {
            "parent": "egu_50_prismatic_1",
            "offset": 0.0,
            "multiplier": -1.0,
        },
    },
    hand_model_name="schunk2f",
)
"""Configuration of Schunk 2F robot."""

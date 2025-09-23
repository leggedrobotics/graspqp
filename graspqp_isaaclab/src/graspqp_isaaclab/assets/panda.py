import os

import isaaclab.sim as sim_utils
from graspqp_isaaclab.models.hand_model_cfg import HandModelCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

PANDA_CFG = HandModelCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            os.path.dirname(__file__),
            "Panda",
            "franka_panda.usd",
        ),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.03},
    ),
    actuators={
        "finger": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint1"],
            effort_limit=70.0,
            velocity_limit=0.1,
            stiffness=1000.0,
            damping=1,
            friction=0.01,
        ),
        "mimic": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint2"],
            effort_limit=70.0,
            velocity_limit=0.1,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
    actuated_joints_expr=["panda_finger_joint1"],
    mimic_joints={
        "panda_finger_joint2": {
            "parent": "panda_finger_joint1",
            "offset": 0.0,
            "multiplier": 1.0,
        },
    },
    hand_model_name="panda",
)
"""Configuration of Shadow Hand robot."""

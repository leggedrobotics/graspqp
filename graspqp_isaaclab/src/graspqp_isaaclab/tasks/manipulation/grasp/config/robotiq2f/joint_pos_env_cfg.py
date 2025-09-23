# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
from graspqp_isaaclab.assets.robotiq2f import ROBOTIQ_2F_CFG
from graspqp_isaaclab.models.hand_model_cfg import HandModelCfg
from graspqp_isaaclab.tasks.manipulation.grasp import mdp
from graspqp_isaaclab.tasks.manipulation.grasp.config.object_mining_env import \
    ObjectGraspMiningEnvCfg
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class robotiq2fObjectGraspMiningEnvCfg(ObjectGraspMiningEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = HandModelCfg.from_articulation_cfg(
            ROBOTIQ_2F_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"), hand_model_name="robotiq2"
        )
        # override actions
        self.actions.hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=ROBOTIQ_2F_CFG.actuated_joints_expr,
            scale=1.0,
            use_default_offset=True,
            preserve_order=True,
            class_type=mdp.FixedJointPositionAction,  # Resets in isaacsim are broken
        )
        self.observations.joint_pos.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            name="robot", joint_names=ROBOTIQ_2F_CFG.actuated_joints_expr, preserve_order=True
        )
        # self.scene.hand_mesh_sensor = robotiq3f_MESH_TRACKER_CFG

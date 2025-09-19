# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
# from isaaclab_assets.shadow:ha import SHADOW_HAND_CFG

# from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG  # isort:skip
from graspqp_isaaclab.assets.shadow_hand import SHADOW_HAND_CFG

from graspqp_isaaclab.tasks.manipulation.grasp.config.object_mining_env import (
    ObjectGraspMiningEnvCfg,
)
from graspqp_isaaclab.tasks.manipulation.grasp import mdp

from isaaclab.utils import configclass

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

from graspqp_isaaclab.models.hand_model_cfg import HandModelCfg


@configclass
class shadow_handObjectGraspMiningEnvCfg(ObjectGraspMiningEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = SHADOW_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override actions
        self.actions.hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=SHADOW_HAND_CFG.actuated_joints_expr,
            scale=1.0,
            use_default_offset=True,
            preserve_order=True,
            class_type=mdp.FixedJointPositionAction,  # Resets in isaacsim are broken
        )
        self.observations.joint_pos.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            name="robot", joint_names=SHADOW_HAND_CFG.actuated_joints_expr, preserve_order=True
        )

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
# from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG, ALLEGRO_HAND_ACTUATED_JOINT_NAMES, ALLEGRO_STANDALONE_CFG
from graspqp_isaaclab.assets.allegro import ALLEGRO_HAND_CFG

# from grasp_mining.tasks.manipulation.grasp.grasp_env_cfg import GraspEnvCfg
from grasp_mining.tasks.manipulation.grasp.config.handle_mining_env import HandleGraspMiningEnvCfg

# from grasp_mining.tasks.manipulation.grasp.config.object_mining_env import ObjectGraspMiningEnvCfg, ObjectLiftEnvCfg
from graspqp_isaaclab.tasks.manipulation.grasp.config.object_mining_env import ObjectGraspMiningEnvCfg


from graspqp_isaaclab.tasks.manipulation.grasp.config.object_mining_env import ObjectGraspMiningEnvCfg

from graspqp_isaaclab.tasks.manipulation.grasp import mdp
from isaaclab.utils import configclass

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg


# from isaaclab.envs import mdp
from isaaclab.utils import configclass

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

from grasp_mining.models.hand_model_cfg import HandModelCfg

# import torch

# from isaaclab.managers import CommandTermCfg
# from isaaclab.sim.spawners.multi_asset.multi_asset_cfg import MultiAssetCfg
# import glob

##
# Environment configuration
##


@configclass
class allegroObjectGraspMiningEnvCfg(ObjectGraspMiningEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = HandModelCfg.from_articulation_cfg(
            ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"), hand_model_name="allegro"
        )
        # # override actions
        # self.actions.hand_action = mdp.JointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=ALLEGRO_HAND_CFG.actuated_joints_expr,
        #     scale=1.0,
        #     use_default_offset=True,
        # )
        # self.observations.joint_pos.joint_pos.params["asset_cfg"] = SceneEntityCfg(
        #     name="robot", joint_names=ALLEGRO_HAND_CFG.actuated_joints_expr
        # )

        self.actions.hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=ALLEGRO_HAND_CFG.actuated_joints_expr,
            scale=1.0,
            use_default_offset=True,
            # preserve_order=True,
            class_type=mdp.FixedJointPositionAction,  # Resets in isaacsim are broken
        )
        self.observations.joint_pos.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            name="robot",
            joint_names=ALLEGRO_HAND_CFG.actuated_joints_expr,
            # preserve_order=True
        )

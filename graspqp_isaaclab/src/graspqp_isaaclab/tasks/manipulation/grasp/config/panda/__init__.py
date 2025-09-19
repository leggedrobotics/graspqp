# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import joint_pos_env_cfg

##
# Register Gym environments.
##

# gym.register(
#     id="Isaac-Grasp-AbilityHand-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": joint_pos_env_cfg.AbilityHandGraspEnvCfg,
#     },
# )


gym.register(
    id="Isaac-Object-Grasp-Mining-panda-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.PandaObjectGraspMiningEnvCfg,
    },
)

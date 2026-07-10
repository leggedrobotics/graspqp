# Copyright (c) 2025 ETH Zurich, René Zurbrügg
#
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from graspqp_isaaclab.models.object_model_cfg import RigidObjectModelCfg
from graspqp_isaaclab.tasks.manipulation.grasp.grasp_mining_env import GraspEnvCfg as GraspMiningEnvCfg
from isaaclab.utils import configclass


@configclass
class ObjectGraspMiningEnvCfg(GraspMiningEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.obj = RigidObjectModelCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/path/to/data/dexgrasp_remeshed/tiny/core-bottle-d8b6c270d29c58c55627157b31e16dc2/core-bottle-d8b6c270d29c58c55627157b31e16dc2.usd",
                
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=5,
                    max_contact_impulse=20,
                    solver_velocity_iteration_count=0,
                    solver_position_iteration_count=12,
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    rest_offset=0.0,
                    contact_offset=5e-3, # collision_enabled=False
                ),
                activate_contact_sensors=False,
                scale=(1.0, 1.0, 1.0)
            ),
            init_state=RigidObjectModelCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            mesh_target_cfg=RigidObjectModelCfg.MeshTargetCfg(
                target_prim_expr="/World/envs/env_.*/Object/geometry",
            ),
        )

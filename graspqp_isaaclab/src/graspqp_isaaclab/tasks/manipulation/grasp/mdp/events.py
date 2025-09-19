# # Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# # Copyright (c) 2022-2024, The ORBIT Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# from __future__ import annotations

import torch


from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_inv

from isaaclab.envs import ManagerBasedEnv


def reset_state(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()
    root_states[:, :3] += env.scene.env_origins[env_ids].clone()

    # # set into the physics simulation
    asset.write_root_pose_to_sim(root_states[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_states[:, 7:13], env_ids=env_ids)

    # make sure to have no external forces
    asset.set_external_force_and_torque(
        forces=torch.zeros(0, 3, device=env_ids.device),
        torques=torch.zeros(0, 3, device=env_ids.device),
        env_ids=env_ids,
    )

    if isinstance(asset, Articulation):  # TODO move to base class
        asset.set_joint_position_target(torch.zeros(len(env_ids), 1, device=env_ids.device), env_ids=env_ids)
        asset.write_joint_state_to_sim(
            torch.zeros(len(env_ids), 1, device=env_ids.device),
            torch.zeros(len(env_ids), 1, device=env_ids.device),
            env_ids=env_ids,
        )


def pull_object(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    max_force: float = 5.0,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("obj"),
) -> torch.Tensor | None:
    obj = env.scene[asset_cfg.name]

    if direction == "random":
        direction = torch.rand(3, device=obj.device)
        direction = direction - 0.5

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=obj.device)

    if len(env_ids) == 0:
        return None

    # extract the used quantities (to enable type-hinting)
    obj: RigidObject = env.scene[asset_cfg.name]

    rot_w = quat_inv(obj.data.root_quat_w[env_ids])

    forces = quat_apply(
        rot_w,
        torch.tensor(
            [
                [max_force * direction[0], max_force * direction[1], max_force * direction[2]],
            ],
            device=obj.device,
        ).expand(env_ids.shape[0], 1, 3),
    )

    obj.set_external_force_and_torque(
        forces=forces,
        torques=torch.zeros(env_ids.shape[0], 1, 3, device=obj.device),
        env_ids=env_ids,
    )
    return forces

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING

from graspqp_isaaclab.tasks.manipulation.grasp import mdp


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import math

SHOW = False
GRAVITY = False

CTRL_FREQ = 1.0 / 30.0
sim_dt = 1.0 / 200.0
CHECK_FREQ = 1.0 / 3.0  # 3 times per s
CHECK_INTERVAL = math.floor(CHECK_FREQ / CTRL_FREQ)


##
# Scene definition
##
@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    # grasp_tracker: GraspTrackerCfg = GraspTrackerCfg()

    obj: RigidObjectCfg | ArticulationCfg = MISSING

    # Visualization tools
    # agent_visualizer: EmptyComponentCfg = EmptyComponentCfg()

    # Evaluation tools
    # eval_visualizer: EmptyComponentCfg = EmptyComponentCfg()


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    hand_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class JointPosCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=".*")},
        )
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ObjectPositionCfg(ObsGroup):
        """Observations for object position."""

        # observation terms (order preserved)
        object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg(name="obj")})
        object_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg(name="obj")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    joint_pos: JointPosCfg = JointPosCfg()
    object_pos: ObjectPositionCfg = ObjectPositionCfg()
    policy: JointPosCfg = JointPosCfg()  # TODO: remove


def _pull_config(max_force, direction, start_time):

    return EventTerm(
        func=mdp.pull_object,
        mode="interval",
        is_global_time=True,
        interval_range_s=(start_time, start_time),
        is_single_shot=True,
        params={
            "max_force": max_force,
            "direction": direction,
            "asset_cfg": SceneEntityCfg("obj"),
        },
    )


def reset_full_state(
    env,
    env_ids: torch.Tensor,
):
    reset_object(env, env_ids, SceneEntityCfg("obj", body_names=".*"))
    reset_robot(env, env_ids, SceneEntityCfg("robot"))


def reset_object(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("obj"),
):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.scene.device)
    # # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()
    root_states[:, :3] += env.scene.env_origins[env_ids].clone()

    # # set into the physics simulation
    asset.write_root_pose_to_sim(root_states[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_states[:, 7:13], env_ids=env_ids)

    if isinstance(asset, RigidObject):
        asset.set_external_force_and_torque(
            forces=torch.zeros(len(env_ids), 3, device=env_ids.device),
            torques=torch.zeros(len(env_ids), 3, device=env_ids.device),
            env_ids=env_ids,
        )


def reset_robot(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.scene.device)
    # # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()
    root_states[:, :3] += env.scene.env_origins[env_ids].clone()

    joint_pose = asset.data.default_joint_pos[env_ids].clone()
    asset.set_joint_position_target(joint_pose, env_ids=env_ids)
    asset.write_root_pose_to_sim(root_states[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_states[:, 7:13], env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pose, torch.zeros_like(joint_pose), env_ids=env_ids)


max_force = 3


@configclass
class PullEventCfg:
    """Configuration for events."""

    reset_robot = EventTerm(
        func=reset_robot,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # reset
    object_position = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("obj", body_names=".*"),
        },
    )

    pull_object_up = _pull_config(max_force, (0.0, 0.0, 1.0), 0.25)
    reset_pull_up = _pull_config(0.0, (0.0, 0.0, 1.0), 0.75)

    pull_object_down = _pull_config(max_force, (0.0, 0.0, -1.0), 1.0)
    reset_pull_down = _pull_config(0.0, (0.0, 0.0, -1.0), 1.5)

    pull_object_x = _pull_config(max_force, (1.0, 0.0, 0.0), 1.75)
    reset_pull_x = _pull_config(0.0, (1.0, 0.0, 0.0), 2.25)

    pull_object_n_x = _pull_config(max_force, (-1.0, 0.0, 0.0), 2.5)
    reset_pull_n_x = _pull_config(0.0, (-1.0, 0.0, 0.0), 3.0)

    pull_object_y = _pull_config(max_force, (0.0, 1.0, 0.0), 3.25)
    reset_pull_y = _pull_config(0.0, (0.0, 1.0, 0.0), 3.75)

    pull_object_n_y = _pull_config(max_force, (0.0, -1.0, 0.0), 4.0)
    reset_pull_n_y = _pull_config(0.0, (0.0, -1.0, 0.0), 4.5)

    reset_everything = EventTerm(
        func=reset_full_state,
        mode="interval",
        is_global_time=True,
        interval_range_s=(1.5, 1.5),
    )


##
# Environment configuration
##


def object_com_error(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("obj")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if isinstance(asset, Articulation):
        return asset.data.joint_pos[..., 0].abs()
    else:
        com_error = (asset.data.body_pos_w[:, -1] - env.scene.env_origins).norm(dim=-1)
    return com_error


def object_com_error_th(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("obj"), threshold=0.1, reset_frequency: int = 5
) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2-kernel."""
    com_error = object_com_error(env, asset_cfg) > threshold
    hand_error = env.scene["robot"].data.root_pos_w[:, -1] > 1.0
    return torch.logical_or(com_error, hand_error) & (env.episode_length_buf % CHECK_INTERVAL == 0)


def time_out(env) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return (env.episode_length_buf >= env.max_episode_length) & (env.episode_length_buf % CHECK_INTERVAL == 0)


# @configclass
# class TerminationsCfg:
#     """Termination terms for the MDP."""

#     time_out = DoneTerm(func=time_out, time_out=True)
#     com_error = DoneTerm(func=object_com_error_th, params={"threshold": 0.03})


@configclass
class TimeoutTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=time_out, time_out=True)


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    object_com_error = RewTerm(func=object_com_error_th, weight=-1.0, params={"threshold": 0.03})


@configclass
class GraspEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: GraspSceneCfg = GraspSceneCfg(num_envs=32, env_spacing=0.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    rewards: RewardsCfg = RewardsCfg()

    terminations: TimeoutTerminationsCfg = TimeoutTerminationsCfg()
    events: PullEventCfg = PullEventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.episode_length_s = 4.8
        self.viewer.eye = (-3.0, 0.0, 2.0)
        # simulation settings
        self.sim.dt = sim_dt
        self.decimation = int(CTRL_FREQ / self.sim.dt)
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            compliant_contact_stiffness=1e5,
        )

        self.scene.replicate_physics = False
        self.sim.physx.gpu_max_rigid_patch_count = 1024 * 1024 * 6
        # self.sim.physx.gpu_collision_stack_size = 2**31
        self.sim.physx.gpu_collision_stack_size = 2**31

        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 1024 * 8
        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024 * 8
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 8
        self.sim.physx.gpu_temp_buffer_capacity = 16 * 1024 * 1024 * 8
        self.sim.physx.gpu_heap_capacity = 16 * 1024 * 1024 * 8

        # self.sim.physx.solver_type = 1
        self.sim.physx.enable_ccd = True
        self.sim.physx.bounce_threshold_velocity = 0.1
        # self.sim.physx.enable_stabilization = False
        self.sim.physx.friction_correlation_distance = 0.005

        self.seed = 42

        self.check_interval = CHECK_INTERVAL

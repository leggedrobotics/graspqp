"""Environment registry module for managing Isaac Lab environment configurations.

This module provides the infrastructure for creating and configuring environments
with multiple assets, grippers, and simulation parameters. It handles environment-asset
mapping, gripper configuration, and dynamic parameter updates from command line arguments.
"""

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from graspqp_isaaclab.spawners.multi_asset.multi_asset_cfg import MultiAssetCfg
from graspqp_isaaclab.tasks import *  # noqa: F401
from graspqp_isaaclab.utils.data import resolve_assets
from isaaclab.utils import configclass
from isaaclab_tasks.utils import parse_env_cfg


class SelectionFunction:
    """Selection function for mapping environments to assets.

    This class provides a callable interface for distributing assets across
    multiple environments using round-robin assignment.

    Args:
        num_envs: Number of simulation environments
    """

    def __init__(self, num_envs: int):
        self._num_envs = num_envs
        self.__name__ = "SelectionFunction"

    def __call__(self, idx, num_assets):
        """Map environment index to asset index using round-robin.

        Args:
            idx: Environment index
            num_assets: Total number of available assets

        Returns:
            int: Asset index for the given environment
        """
        n_envs_per_asset = self._num_envs // num_assets

        return (idx // n_envs_per_asset) % num_assets


def get_env_cfg(args_cli, collapse_grippers=False):
    """Create environment configuration with multi-asset support.

    Configures Isaac Lab environments with multiple objects and grippers,
    handling asset distribution, color assignment, and parameter updates
    from command line arguments.

    Args:
        args_cli: Command line arguments containing simulation parameters
        collapse_grippers: Whether to use a single gripper for all objects

    Returns:
        tuple: (env_cfg, asset_mapping) where env_cfg is the configured
               environment and asset_mapping maps indices to asset names

    Raises:
        AssertionError: If number of assets doesn't match expected count
    """
    if collapse_grippers:
        args_cli.num_envs = args_cli.num_envs // args_cli.n_grasps_per_env

    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric, device=args_cli.device
    )

    _, assets_cfg = resolve_assets(
        usd_paths=args_cli.usd_files, default_object_cfg=env_cfg.scene.obj.copy(), collapse_grippers=collapse_grippers
    )

    if args_cli.static_show:
        for asset in assets_cfg:
            asset.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

    if collapse_grippers:
        env_cfg.scene.obj.collision_group = -1  # Everything collides with objects (i.e. all hands collide with objects)

        for i in range(args_cli.n_grasps_per_env):
            # Update actions
            setattr(env_cfg.scene, f"robot_{i}", env_cfg.scene.robot.replace(prim_path="{ENV_REGEX_NS}/" + f"Robot_{i}"))
            setattr(env_cfg.actions, f"hand_action_{i}", env_cfg.actions.hand_action.replace(asset_name="robot_" + f"{i}"))
            getattr(env_cfg.scene, f"robot_{i}").collision_group = i  # Make sure hands don't collide

        @configclass
        class EmptyCfg:
            pass

        # Disable observations, rewards, events, and terminations as they are not used in multi-asset setups
        env_cfg.observations = EmptyCfg()
        env_cfg.rewards = EmptyCfg()
        env_cfg.events = EmptyCfg()
        env_cfg.terminations = EmptyCfg()

        # remove initial robot used to clone
        delattr(env_cfg.scene, "robot")
        delattr(env_cfg.actions, "hand_action")

    selection_func = SelectionFunction(args_cli.num_envs)

    # Wrap with multi asset cfg
    env_cfg.scene.obj.spawn = MultiAssetCfg(
        assets_cfg=assets_cfg,
        randomize=False,
        selection_func=selection_func,
    )
    asset_mapping = [selection_func(i, len(assets_cfg)) for i in range(args_cli.num_envs)]
    if args_cli.energy_type and hasattr(env_cfg.scene, "hand_mesh_sensor"):
        print("Setting Energy Type to", args_cli.energy_type)
        env_cfg.scene.grasp_tracker.energy_method = args_cli.energy_type
        # env_cfg.scene.hand_mesh_sensor.energy_type = [args_cli.energy_type]

    if hasattr(args_cli, "train_energy_type") and args_cli.train_energy_type and hasattr(env_cfg.scene, "hand_mesh_sensor"):
        print("Setting Train Energy Type to", args_cli.train_energy_type)
        env_cfg.scene.grasp_tracker.energy_type = args_cli.train_energy_type
        args_cli.energy_type = args_cli.train_energy_type
        print("Setting Energy Type to", args_cli.energy_type)

    return env_cfg, asset_mapping

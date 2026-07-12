# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""
Static grasp agents for executing pre-defined grasp configurations.

This module provides agents that execute static, pre-computed grasps without
learning or adaptation. Used for evaluation and testing of grasp datasets.
"""

import os

import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.utils.math import subtract_frame_transforms
from .base import Agent
import copy


class StaticGraspAgent(Agent):
    """
    Agent that executes pre-defined static grasps with optional closing motion.

    Applies fixed hand poses and joint configurations to evaluate grasp quality
    without learning. Supports grasp closing with configurable velocities.
    """

    def __init__(
        self,
        env,
        hand_poses,
        joint_positions,
        hand: Articulation,
        env_ids=None,
        closing_vel=None,
        closing_distance=15e-3,
        energy=None,
    ):
        """
        Initialize static grasp agent with pre-defined configurations.

        Args:
            env: Isaac Lab environment
            hand_poses: Hand pose configurations (position + orientation)
            joint_positions: Joint angle configurations
            hand: Articulated hand asset
            env_ids: Environment IDs to control (default: all)
            closing_vel: Joint commands for grasp closing
            closing_distance: Distance threshold for grasp closing
            energy: Grasp quality energy values
        """
        super().__init__(env)

        if env_ids is None:
            env_ids = slice(None)

        self._closing_vel = closing_vel
        self._closing_distance = closing_distance
        self._env_ids = env_ids

        self._hand_poses = hand_poses
        self._joint_positions = joint_positions.clone()

        self._hand = hand
        self.num_envs = len(self._joint_positions)

        self._dimension = self._joint_positions.shape[1]
        self._iter = 0
        self.energy = energy

    def reset(self):
        """Reset hand to initial grasp configuration."""
        # clamp joint positions
        joint_positions = self._joint_positions.clone()
        positions = self._hand_poses[:, :3] + self.env.scene.env_origins[self._env_ids]
        orientations = self._hand_poses[:, 3:]
        velocities = torch.zeros((len(self._hand_poses), 6), device=self._hand.device)

        # convert slice to range
        env_ids_sequence = (
            None
            if self._env_ids.start is None
            else torch.arange(self._env_ids.start, self._env_ids.stop, device=self._env.device)
        )

        self._hand.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids_sequence)
        self._hand.write_root_velocity_to_sim(velocities, env_ids=env_ids_sequence)

        self._hand.data.default_root_state[self._env_ids] = torch.cat(
            [positions - self.env.scene.env_origins[self._env_ids], orientations, 0 * positions, 0 * positions], dim=-1
        )
        self._hand.set_default_joint_positions(
            joint_positions, joint_ids=self._hand.data.actuated_joint_indices, env_ids=self._env_ids
        )

        self._hand.write_joint_state_to_sim(
            joint_positions,
            0 * joint_positions,
            env_ids=env_ids_sequence,
            joint_ids=self._hand.data.actuated_joint_indices,
        )
        self.env.action_manager.reset()
        self._last_joint_pos = None
        self._last_object_pose = None

    @property
    def current_joint_pos(self):
        return self._hand.data.joint_pos[self._env_ids, self._hand.data.actuated_joint_indices]

    def get_actions(self):
        """Get actions for grasp execution (closing velocities if specified)."""
        self._iter += 1
        actions = torch.zeros((self.num_envs, self._dimension), device=self.env.device)

        if self._closing_vel is not None:
            if self._closing_vel.shape != actions.shape:
                pass
            else:
                actions = self._closing_vel
        self._last_joint_pos = self.current_joint_pos.clone()

        obj = self.env.scene["obj"]

        if isinstance(obj, Articulation):
            # handles
            pose = obj.data.body_pose_w[self._env_ids, obj.data.body_names.index("geometry")].clone()
            pose[:, :3] -= self.env.scene.env_origins[self._env_ids]
            self._last_object_pose = pose
        else:
            pose = obj.data.body_pose_w[self._env_ids, -1].clone()
            pose[:, :3] -= self.env.scene.env_origins[self._env_ids]
            self._last_object_pose = pose

        return actions

    def update_envs(self, observations, rewards):
        """Update environments (no-op for static agent)."""
        pass

    def reset_envs(self, envs, finished):
        """Reset specific environments (no-op for static agent)."""
        pass

    def finished(self):
        """Check if agent is finished (always False for continuous execution)."""
        return False

    def save(self, path, suffix="", mask=None, values=None):
        """Save grasp configurations and results to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        to_save_params = {}
        joint_params = self._joint_positions[mask] if mask is not None else self._joint_positions

        if self._last_joint_pos is None:
            raise ValueError(
                "Cannot save grasp parameters before executing the agent and obtaining last joint positions."
            )

        joint_params_closed = self._last_joint_pos[mask] if mask is not None else self._last_joint_pos
        object_pose = self._last_object_pose[mask] if mask is not None else self._last_object_pose

        closing_vel = self._closing_vel[mask] if mask is not None else self._closing_vel
        hand_poses = self._hand_poses[mask] if mask is not None else self._hand_poses
        closing_hand_poses = hand_poses.clone()
        if len(closing_hand_poses) > 0:
            closing_hand_poses = torch.cat(
                subtract_frame_transforms(
                    object_pose[:, :3],
                    object_pose[:, 3:],
                    closing_hand_poses[:, :3],
                    closing_hand_poses[:, 3:],
                ),
                dim=-1,
            )

        for joint_name, param in zip(self._hand.data.actuated_joint_names, joint_params.T):
            to_save_params[joint_name] = param.detach().cpu()

        grasp_vel = {}
        for joint_name, param in zip(self._hand.data.actuated_joint_names, closing_vel.T):
            grasp_vel[joint_name] = param.detach().cpu()

        to_save_params["root_pose"] = hand_poses[:, :7].detach().cpu()
        if values is None:
            values = torch.zeros(len(hand_poses), device=self._hand.device)

        to_save_closed_params = {}
        to_save_closed_params["root_pose"] = closing_hand_poses[:, :7].detach().cpu()

        for joint_name, param in zip(self._hand.data.actuated_joint_names, joint_params_closed.T):
            to_save_closed_params[joint_name] = param.detach().cpu()

        data = {
            "parameters": to_save_params,
            "closed_parameters": to_save_closed_params,
            "grasp_velocities": grasp_vel,
            "full_grasp_velocities": grasp_vel,
            "values": values.detach().cpu(),
            # Poses are written straight to the articulation root (see reset), i.e. the gripper
            # root frame -- not the graspqp "grasp_frame". Absent field => gripper root.
            "root_frame": "gripper_root",
            # "values": values,
        }  # , "object_config": self.object.cfg.to_dict(), "values": values}

        path = path if suffix == "" else path.replace(".pt", f"_{suffix}.pt")
        torch.save(data, path)
        print("Saved Grasp data to", path, "Number of grasps:", len(hand_poses))
        return path


class StaticShowGraspAgent(Agent):
    """
    Agent for visualizing multiple static grasps across different hands.

    Similar to StaticGraspAgent but supports multiple hand assets for
    comparative visualization of grasp configurations.
    """

    def __init__(
        self,
        env,
        hand_poses,
        joint_positions,
        hands: list[Articulation],
        env_ids=None,
        closing_vel=None,
        closing_distance=15e-3,
    ):
        """
        Initialize multi-hand static grasp visualization agent.

        Args:
            env: Isaac Lab environment
            hand_poses: Hand pose configurations
            joint_positions: Joint angle configurations
            hands: List of articulated hand assets
            env_ids: Environment IDs to control
            closing_vel: Joint velocities for grasp closing
            closing_distance: Distance threshold for grasp closing
        """
        super().__init__(env)

        if env_ids is None:
            env_ids = slice(None)

        self._closing_vel = closing_vel
        self._closing_distance = closing_distance
        self._env_ids = env_ids

        self._hand_poses = hand_poses
        self._joint_positions = joint_positions

        self._hands = hands
        self.num_envs = len(self._joint_positions)

        self._dimension = self._joint_positions.shape[1]

    def reset(self):
        """Reset all hands to their initial grasp configurations."""
        # clamp joint positions
        joint_positions = self._joint_positions
        # joint_positions = (
        #     convert_actuated_to_full_dof(self._joint_positions) if False else self._joint_positions.clone()
        # )
        positions = self._hand_poses[:, :3] + self.env.scene.env_origins[self._env_ids]
        orientations = self._hand_poses[:, 3:]
        velocities = torch.zeros((len(self._hand_poses), 6), device=self._env.device)

        # convert slice to range
        env_ids_sequence = (
            None
            if self._env_ids.start is None
            else torch.arange(self._env_ids.start, self._env_ids.stop, device=self._env.device)
        )

        all_poses = torch.cat([positions, orientations], dim=-1)
        all_joint_positions = joint_positions
        if isinstance(self._hands, list):
            for hand_idx, hand in enumerate(self._hands):
                pose = all_poses[[hand_idx]]
                joint_position = all_joint_positions[[hand_idx]]

                hand.write_root_pose_to_sim(pose, env_ids=env_ids_sequence)
                hand.write_root_velocity_to_sim(velocities[[hand_idx]], env_ids=env_ids_sequence)

                position = pose[:, :3]
                hand.data.default_root_state[env_ids_sequence] = torch.cat(
                    [position - self.env.scene.env_origins[env_ids_sequence], pose[:, 3:], 0 * position, 0 * position],
                    dim=-1,
                )

                hand.write_joint_state_to_sim(
                    joint_position,
                    0 * joint_position,
                    env_ids=env_ids_sequence,
                    joint_ids=hand.data.actuated_joint_indices,
                )
                hand.set_default_joint_positions(
                    joint_position, joint_ids=hand.data.actuated_joint_indices, env_ids=self._env_ids
                )
        else:
            self._hands.write_root_pose_to_sim(all_poses, env_ids=env_ids_sequence)
            self._hands.write_root_velocity_to_sim(velocities, env_ids=env_ids_sequence)
            self._hands.data.default_root_state[self._env_ids] = torch.cat(
                [positions - self.env.scene.env_origins[self._env_ids], orientations, 0 * positions, 0 * positions],
                dim=-1,
            )
            self._hands.set_default_joint_positions(
                joint_positions, joint_ids=self._hands.data.actuated_joint_indices, env_ids=self._env_ids
            )
            self._hands.write_joint_state_to_sim(
                joint_positions,
                0 * joint_positions,
                env_ids=env_ids_sequence,
                joint_ids=self._hands.data.actuated_joint_indices,
            )

    def get_actions(self):
        """Get zero actions (no movement for visualization)."""
        actions = (
            torch.zeros((1, self.num_envs * self._dimension), device=self.env.device)
            if isinstance(self._hands, list)
            else torch.zeros((self.num_envs, self._dimension), device=self.env.device)
        ) + self._closing_vel

        return actions

    def update_envs(self, observations, rewards):
        """Update environments (no-op for visualization agent)."""
        pass

    def reset_envs(self, envs, finished):
        """Reset specific environments (no-op for visualization agent)."""
        pass

    def finished(self):
        """Check if agent is finished (always False for continuous visualization)."""
        return False

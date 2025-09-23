"""
Agent evaluation wrapper for grasp quality assessment and statistics collection.
"""

import os
import time

import torch
from graspqp_isaaclab.utils.eval import (RunningStatistics,
                                         calc_entropy_for_grasps,
                                         calc_unique_grasps)

from .base import Agent


class AgentEvalWrapper(Agent):
    """
    Evaluation wrapper for dexterous grasping agents.

    This class wraps existing agents to provide comprehensive evaluation capabilities
    including grasp success tracking, diversity metrics, entropy calculations, and
    statistical analysis across multiple precision levels.

    The wrapper evaluates grasps using different pulling directions (x, y, z axes)
    and tracks various metrics such as:
    - Success rates per object/configuration
    - Grasp diversity and uniqueness at different precision levels
    - Joint and pose entropy for grasp variety assessment
    - Penetration depth analysis for collision detection

    Args:
        agent: The base agent to wrap for evaluation
        asset_mapping (torch.Tensor): Mapping of environments to asset configurations
        print_interval (int, optional): Interval in seconds for printing statistics. Defaults to 5.
        min_evals (int, optional): Minimum number of evaluations required. Defaults to 10.
        output_folders (list, optional): List of output directories for saving results. Defaults to None.

    Attributes:
        _agent: The wrapped agent instance
        _asset_mapping: Tensor mapping environments to assets
        _statistics: RunningStatistics instance for tracking performance
        _min_evals: Minimum evaluations required before completion
        steps: Current step counter
        observations: List of collected observations
        rewards: List of collected rewards
    """

    def __init__(
        self,
        agent,
        asset_mapping: torch.Tensor,
        print_interval=5,
        min_evals=10,
        output_folders=None,
    ):
        """
        Initialize the evaluation wrapper.

        Args:
            agent: The agent to wrap for evaluation
            asset_mapping (torch.Tensor): Mapping of environment IDs to asset configurations
            print_interval (int, optional): Time interval (seconds) between statistics printouts
            min_evals (int, optional): Minimum number of evaluation episodes required
            output_folders (list, optional): Directories for saving evaluation results
        """
        super().__init__(agent.env)
        self._agent = agent
        self._asset_mapping = asset_mapping
        # statistics
        self._statistics = RunningStatistics(self.env.num_envs, asset_mapping)

        self._tic = time.time()
        self._print_interval = print_interval

        self._pending_envs = None
        self._statistics.set_configs([cfg for cfg in self._env.scene["obj"].cfg.spawn.assets_cfg])
        self._min_evals = min_evals
        self.steps = 0
        self.observations = []
        self.rewards = []

        self._results = []
        self._output_folders = output_folders

    def reset(self):
        """
        Reset the evaluation wrapper and underlying agent.

        Resets all statistics, the wrapped agent, and initializes all environments
        as pending for evaluation.
        """
        self._agent.reset()
        self._statistics.reset()
        self._pending_envs = torch.arange(self.env.num_envs, device=self.env.device)

    def get_actions(self):
        """
        Get actions from the wrapped agent.

        Returns:
            torch.Tensor: Actions computed by the underlying agent
        """
        actions = self._agent.get_actions()
        return actions

    def reset_envs(self, envs, succeeded):
        """
        Reset specific environments and evaluate grasp performance.

        This method processes completed grasp attempts by:
        1. Analyzing rewards across different pulling directions (x, y, z axes)
        2. Determining success/failure for each axis and overall performance
        3. Updating statistics with success rates and additional metrics
        4. Computing grasp diversity metrics (entropy, uniqueness) when sufficient data exists
        5. Saving successful and failed grasps to output files

        Args:
            envs (torch.Tensor): Environment IDs to reset
            succeeded (torch.Tensor): Boolean tensor indicating environment success status
        """
        # TODO
        # reset_separation = int(
        #     self.env.cfg.events.reset_everything.interval_range_s[0] / (self.env.cfg.decimation * self.env.cfg.sim.dt)
        # )
        reset_separation = 50
        # Split evaluation into different phases (typically: x-axis, y-axis, z-axis pulls + final)
        diff_steps = [reset_separation] * 3
        diff_steps += [len(self.rewards) - sum(diff_steps)]

        rewards = torch.cat(self.rewards, -1)

        # Analyze success/failure for each pulling direction
        rewards_each_pull = rewards.split(diff_steps, -1)
        fails = []
        for pull in rewards_each_pull:
            # A grasp fails if any reward in the pulling sequence is negative
            fails.append(pull.min(axis=-1).values < 0.0)

        # Success per axis (x, y, z) - invert failure flags
        succ_per_axis = ~torch.stack(fails, -1)[..., :3]
        self._results.append(~torch.stack(fails, -1)[..., :3])

        self._statistics.update(envs, (~torch.stack(fails, -1)[..., :3]).any(dim=-1))
        self._statistics.update_info(envs, x_axis=succ_per_axis[:, 0], y_axis=succ_per_axis[:, 1], z_axis=succ_per_axis[:, 2])
        self._statistics.update_info(envs, all_axis=succ_per_axis.all(dim=-1))

        # observations = torch.cat(self.observations, -1)
        self.rewards = []
        self.observations = []

        # COMPREHENSIVE EVALUATION: Compute advanced metrics when sufficient trials exist
        finished = self._statistics.trials.min().item() > 0
        if finished:
            # Calculate final statistics including entropy and grasp uniqueness metrics
            for agent in self._agent._agents:
                env_ids = agent._env_ids

                non_failing_envs = self._statistics.sucesses[env_ids] >= 0.5 * self._statistics.trials[env_ids].clamp(min=1)

                joint_positions = agent._joint_positions
                hand_poses = agent._hand_poses

                joints_entropy, position_entropy, orientation_entropy = calc_entropy_for_grasps(
                    joint_positions[non_failing_envs],
                    hand_poses[non_failing_envs],
                    self.env,
                )
                self._statistics.update_info(env_ids, joints_entropy=joints_entropy, fix=True)
                self._statistics.update_info(env_ids, position_entropy=position_entropy, fix=True)
                self._statistics.update_info(env_ids, orientation_entropy=orientation_entropy, fix=True)

                # Define precision levels for grasp uniqueness evaluation
                # These thresholds determine when two grasps are considered "unique"
                precision_joints = [0.5, 0.2, 0.1]  # Joint position precision (radians)
                precision_positions = [2e-1, 2e-2, 1e-2]  # Hand position precision (meters)
                precision_deg = [  # Hand orientation precision (degrees)
                    45 * 180 / torch.pi,
                    5 * 180 / torch.pi,
                    2.5 * 180 / torch.pi,
                ]
                names = ["20cm_05_450deg", "2cm_050deg", "1cm_025deg"]  # Descriptive names for precision levels

                for name, precision_joint, precision_pos, precision_deg in zip(
                    names, precision_joints, precision_positions, precision_deg
                ):
                    unique_grasps, unique_working_grasps = calc_unique_grasps(
                        joint_positions.clone(),
                        hand_poses.clone(),
                        precision_pos,
                        precision_joint,
                        precision_deg,
                        non_failing_envs,
                    )
                    self._statistics.update_info(env_ids, **{f"good_grasps_{name}": unique_working_grasps}, fix=True)
                    self._statistics.update_info(
                        env_ids,
                        **{f"grasps_{name}": unique_grasps},
                        fix=True,
                    )

    def update_envs(self, observations, rewards):
        """
        Update environments with new observations and rewards.

        Collects observations and rewards for later analysis during reset_envs.
        Also updates the underlying agent and increments step counter.

        Args:
            observations (torch.Tensor): Environment observations
            rewards (torch.Tensor): Environment rewards
        """
        self.steps += 1
        self._agent.update_envs(observations, rewards)
        self.observations.append(observations)
        self.rewards.append(rewards)

    def finished(self, suffix=""):
        """
        Check if evaluation is complete and save final results.

        Determines if sufficient evaluations have been completed based on min_evals.
        When finished, saves comprehensive statistics to CSV files and individual
        grasp data (successful/failed) to PyTorch files.

        Args:
            suffix (str, optional): Suffix to append to output filenames. Defaults to "".

        Returns:
            bool: True if evaluation is complete, False otherwise
        """
        if len(self._results) < self._min_evals:
            return False

        self._statistics.to_csv(
            folders=self._output_folders,
            file_name=("dexgrasp_eval_isaac_sim.csv" if suffix == "" else f"dexgrasp_eval_isaac_sim_{suffix}.csv"),
        )
        # print full statistics
        self._statistics.print_statistics(full_statistics=True)
        self._statistics.print_statistics(full_statistics=False)

        for idx, agent in enumerate(self._agent._agents):
            env_ids = agent._env_ids
            # also save all unique grasps
            # find successfull graasps
            non_failing_envs = self._statistics.sucesses[env_ids] >= 0.5 * self._statistics.trials[env_ids].clamp(min=1)
            # find non colliding graspsp

            out = os.path.join(self._output_folders[idx], "succ_grasps.pt")
            agent.save(
                out,
                suffix=suffix,
                mask=non_failing_envs,
                values=agent.energy[non_failing_envs],
            )
            # failed grasps
            failed = ~non_failing_envs
            out = os.path.join(self._output_folders[idx], "failed_grasps.pt")
            agent.save(out, suffix=suffix, mask=failed, values=agent.energy[failed])

        return True

    def get_statistics(self):
        """
        Get comprehensive evaluation statistics as a DataFrame.

        Returns:
            pandas.DataFrame: Statistics including success rates, trials, entropy metrics,
                            and grasp uniqueness measures across different precision levels
        """
        df = self._statistics.get_df()
        return df

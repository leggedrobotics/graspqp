import os
import time

import torch
from graspqp_isaaclab.utils.eval import RunningStatistics, calc_entropy_for_grasps, calc_unique_grasps

from .base import Agent


class AgentEvalWrapper(Agent):
    """
    Evaluation wrapper for dexterous grasping agents.

    Wraps an existing agent to track grasp success, diversity metrics, entropy,
    and statistical analysis across multiple precision levels. Evaluates grasps
    using different pulling directions (x, y, z axes).

    Args:
        agent: The base agent to wrap for evaluation.
        asset_mapping (torch.Tensor): Mapping of environments to asset configurations.
        print_interval (int): Interval in seconds for printing statistics.
        min_evals (int): Minimum number of evaluations required before finishing.
        output_folders (list): Output directories for saving results.
    """

    def __init__(
        self,
        agent,
        asset_mapping: torch.Tensor,
        print_interval=5,
        min_evals=10,
        output_folders=None,
    ):
        super().__init__(agent.env)
        self._agent = agent
        self._asset_mapping = asset_mapping
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
        """Reset all statistics, the wrapped agent, and mark all environments as pending."""
        self._agent.reset()
        self._statistics.reset()
        self._pending_envs = torch.arange(self.env.num_envs, device=self.env.device)

    def get_actions(self):
        """Return actions from the wrapped agent."""
        return self._agent.get_actions()

    def reset_envs(self, envs, succeeded):
        """
        Reset specific environments and evaluate grasp performance.

        Processes completed grasp attempts by analyzing rewards across pulling directions
        (x, y, z axes), updating success statistics, and computing diversity metrics
        (entropy, uniqueness at multiple precision levels).

        Args:
            envs (torch.Tensor): Environment IDs to reset.
            succeeded (torch.Tensor): Boolean tensor indicating environment success status.
        """
        reset_separation = 50
        diff_steps = [reset_separation] * 3
        diff_steps += [len(self.rewards) - sum(diff_steps)]

        rewards = torch.cat(self.rewards, -1)
        rewards_each_pull = rewards.split(diff_steps, -1)
        fails = []
        for pull in rewards_each_pull:
            # Strip first step, which sometimes might be lingering from previous phase
            fails.append(pull[:, 1:].min(axis=-1).values < 0.0)

        succ_per_axis = ~torch.stack(fails, -1)[..., :3]
        self._results.append(~torch.stack(fails, -1)[..., :3])

        self._statistics.update(envs, (~torch.stack(fails, -1)[..., :3]).any(dim=-1))
        self._statistics.update_info(
            envs, x_axis=succ_per_axis[:, 0], y_axis=succ_per_axis[:, 1], z_axis=succ_per_axis[:, 2]
        )
        self._statistics.update_info(envs, all_axis=succ_per_axis.all(dim=-1))

        self.rewards = []
        self.observations = []

        finished = self._statistics.trials.min().item() > 0
        if finished:
            for agent in self._agent._agents:
                env_ids = agent._env_ids

                non_failing_envs = self._statistics.sucesses[env_ids] >= 0.5 * self._statistics.trials[env_ids].clamp(min=1)

                joint_positions = agent._joint_positions.clone()
                hand_poses = agent._hand_poses

                joints_entropy, position_entropy, orientation_entropy = calc_entropy_for_grasps(
                    joint_positions[non_failing_envs].clone(),
                    hand_poses[non_failing_envs].clone(),
                    self.env,
                )
                self._statistics.update_info(env_ids, joints_entropy=joints_entropy, fix=True)
                self._statistics.update_info(env_ids, position_entropy=position_entropy, fix=True)
                self._statistics.update_info(env_ids, orientation_entropy=orientation_entropy, fix=True)

                precision_joints_list = [90 * torch.pi / 180, 45 * torch.pi / 180, 5 * torch.pi / 180]
                precision_positions_list = [0.2, 0.05, 0.01]
                precision_degs_list = [90 * torch.pi / 180, 45 * torch.pi / 180, 5 * torch.pi / 180]
                names = ["20cm_90deg_90deg", "20cm_45deg_45deg", "2cm_45deg_45deg", "1cm_5deg_5deg"]

                for name, precision_joint, precision_pos, precision_deg in zip(
                    names, precision_joints_list, precision_positions_list, precision_degs_list
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
                    self._statistics.update_info(env_ids, **{f"grasps_{name}": unique_grasps}, fix=True)

    def update_envs(self, observations, rewards):
        """
        Collect observations and rewards for later analysis during reset_envs.

        Args:
            observations (torch.Tensor): Environment observations.
            rewards (torch.Tensor): Environment rewards.
        """
        self.steps += 1
        self._agent.update_envs(observations, rewards.clone())
        self.observations.append(observations)
        self.rewards.append(rewards.clone())

    def finished(self, suffix=""):
        """
        Check if evaluation is complete and save final results to CSV and .pt files.

        Args:
            suffix (str): Suffix to append to output filenames.

        Returns:
            bool: True if evaluation is complete, False otherwise.
        """
        if len(self._results) < self._min_evals:
            return False

        self._statistics.to_csv(
            folders=self._output_folders,
            file_name=("dexgrasp_eval_isaac_sim.csv" if suffix == "" else f"dexgrasp_eval_isaac_sim_{suffix}.csv"),
        )
        self._statistics.print_statistics(full_statistics=False)

        for idx, agent in enumerate(self._agent._agents):
            env_ids = agent._env_ids
            non_failing_envs = self._statistics.sucesses[env_ids] >= 0.5 * self._statistics.trials[env_ids].clamp(min=1)

            out = os.path.join(self._output_folders[idx], "succ_grasps.pt")
            agent.save(out, suffix=suffix, mask=non_failing_envs, values=agent.energy[non_failing_envs])

            failed = ~non_failing_envs
            out = os.path.join(self._output_folders[idx], "failed_grasps.pt")
            agent.save(out, suffix=suffix, mask=failed, values=agent.energy[failed])

        return True

    def get_statistics(self):
        """Return evaluation statistics as a DataFrame."""
        return self._statistics.get_df()

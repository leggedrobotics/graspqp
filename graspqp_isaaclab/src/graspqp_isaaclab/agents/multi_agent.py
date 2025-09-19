"""
Multi-agent wrapper for coordinating multiple grasping agents across different assets.

This module provides a unified interface for managing multiple specialized agents,
each handling different object types or configurations within the same environment.
"""

import torch
from isaaclab.sim import SimulationContext
from .base import Agent


class MultiAgentWrapper(Agent):
    """
    Wrapper for coordinating multiple agents across different asset types.

    Routes environment interactions to appropriate specialized agents based on
    asset mapping, enabling different grasping strategies for different objects.

    Args:
        agents (list[Agent]): List of specialized agents for different asset types
        asset_mapping (torch.Tensor): Maps environment IDs to corresponding agent indices
    """

    def __init__(self, agents: list[Agent], asset_mapping: torch.Tensor):
        """Initialize multi-agent wrapper with validation and environment partitioning."""
        if len(agents) != asset_mapping.max().item() + 1:
            raise ValueError("Number of agents must match the number of assets.")

        super().__init__(agents[0].env)

        self._agents = agents
        self._asset_mapping = asset_mapping
        # Calculate environment distribution across agents
        self._envs_per_asset = torch.bincount(asset_mapping).tolist()
        self._envs_cumsum = torch.cumsum(torch.tensor([0] + self._envs_per_asset), dim=0)

    def reset(self):
        """Reset all managed agents."""
        for agent in self._agents:
            agent.reset()

    def get_actions(self):
        """Collect and concatenate actions from all agents."""
        actions = [agent.get_actions() for agent in self._agents]
        actions = torch.cat(actions, dim=0)
        return actions

    def update_envs(self, observations, rewards):
        """Forward observations and rewards to all agents."""
        for agent in self._agents:
            agent.update_envs(observations, rewards)

    def reset_envs(self, envs, succeeded):
        """Route environment resets to appropriate agents based on environment ranges."""
        for idx, agent in enumerate(self._agents):
            # Find environments belonging to this agent's asset range
            valid_envs_mask = (envs >= self._envs_cumsum[idx]) & (envs < self._envs_cumsum[idx + 1])
            if valid_envs_mask.any():
                # Convert global env IDs to local agent env IDs
                agent.reset_envs(envs[valid_envs_mask] - self._envs_cumsum[idx], succeeded[valid_envs_mask])

    def finished(self):
        """Check if all agents have completed their tasks."""
        return all([agent.finished() for agent in self._agents])

    def set_debug_vis(self, debug_vis: bool):
        """Enable/disable debug visualization for all agents."""
        for agent in self._agents:
            agent.set_debug_vis(debug_vis)

        if not debug_vis:
            draw_interface = SimulationContext.instance().draw_interface
            draw_interface._clear()

    def debug_vis_callback(self):
        """Execute debug visualization callbacks for all agents."""
        for agent in self._agents:
            agent.debug_vis_callback()

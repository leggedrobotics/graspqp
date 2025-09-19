class Agent:
    """
    Base class for an agent interacting with an environment.

    Attributes:
        _env: The environment the agent interacts with.
        _debug_vis: A boolean indicating whether debug visualization is enabled.

    Methods:
        env: Returns the environment.
        device: Returns the device of the environment.
        finished: Indicates whether the agent has finished its task.
        get_actions: Abstract method to get actions from the agent.
        reset_envs: Abstract method to reset environments.
        update_envs: Abstract method to update environments with observations and rewards.
        reset: Abstract method to reset the agent.
        set_debug_vis: Sets the debug visualization flag.
        debug_vis_callback: Callback for debug visualization.
    """

    def __init__(self, env):
        """
        Initializes the agent with the given environment.

        Args:
            env: The environment the agent will interact with.
        """
        self._env = env
        self._debug_vis = False

    @property
    def env(self):
        """
        Returns the environment.

        Returns:
            The environment the agent interacts with.
        """
        return self._env

    @property
    def device(self):
        """
        Returns the device of the environment.

        Returns:
            The device of the environment.
        """
        return self._env.device

    def finished(self):
        """
        Indicates whether the agent has finished its task.

        Returns:
            False, indicating the agent has not finished its task.
        """
        return False

    def get_actions(self):
        """
        Abstract method to get actions from the agent.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError()

    def reset_envs(self, envs, finished):
        """
        Abstract method to reset environments.

        Args:
            envs: The environments to reset.
            finished: A flag indicating whether the environments are finished.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError()

    def update_envs(self, observations, rewards):
        """
        Abstract method to update environments with observations and rewards.

        Args:
            observations: The observations from the environments.
            rewards: The rewards from the environments.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError()

    def reset(self):
        """
        Abstract method to reset the agent.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError()

    def set_debug_vis(self, debug_vis: bool):
        """
        Sets the debug visualization flag.

        Args:
            debug_vis: A boolean indicating whether debug visualization is enabled.
        """
        self._debug_vis = debug_vis

    def debug_vis_callback(self):
        """
        Callback for debug visualization.
        """
        pass

# agents/randomAgent.py

"""
This module defines a RandomAgent class, which inherits from AbstractAgent.
The RandomAgent selects actions randomly from the available action space.
"""

import os
from typing import Any, Dict
import gym
import numpy as np
from agents.abstractAgent import AbstractAgent


class RandomAgent(AbstractAgent):
    """
    An agent that selects actions randomly, serving as a baseline.
    It does not learn or use the state information to make decisions.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """
        Initializes the RandomAgent.

        Args:
            observation_space (gym.Space): The environment's observation space.
            action_space (gym.Space): The environment's action space.
        """
        super().__init__(observation_space, action_space)
        self.history = ""  # For logging actions if needed.

    def get_action(self, state: Any, is_training: bool = True) -> int:
        """
        Selects a random action from the action space.

        Args:
            state: The current state of the environment (ignored).
            is_training: Flag indicating training mode (ignored).

        Returns:
            int: A randomly selected action index.
        """
        return self.action_space.sample()

    def update(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        **kwargs: Any,
    ) -> None:
        """
        The RandomAgent does not learn, so this method is empty.
        """
        pass

    def on_episode_start(self) -> None:
        """
        Hook for start-of-episode logic. No action needed for this agent.
        """
        pass

    def on_episode_end(self) -> None:
        """
        Hook for end-of-episode logic. No action needed for this agent.
        """
        pass

    def get_update_info(self) -> Dict[str, Any]:
        """
        Returns an empty dictionary as this agent has no metrics to log.
        """
        return {}

    def save_model(self, path: str, filename: str = "history.txt") -> None:
        """Saves the history of actions and states to a text file."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, filename), "w") as f:
            f.write(self.history)

    @classmethod
    def load_model(cls, path: str, filename: str, **kwargs: Any) -> "RandomAgent":
        """The RandomAgent is stateless and cannot be loaded."""
        raise NotImplementedError(
            "The RandomAgent is stateless and does not support loading."
        )

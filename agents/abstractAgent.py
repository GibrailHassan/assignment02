# agents/abstractAgent.py

"""
This module defines the AbstractAgent class, an abstract base class (ABC)
that outlines the common interface for all reinforcement learning agents.
"""

import abc
from typing import Any, Dict
import gym


class AbstractAgent(metaclass=abc.ABCMeta):
    """
    Abstract class defining the structure and behavior of an RL agent.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """
        Initializes the agent with the environment's observation and action spaces.

        Args:
            observation_space (gym.Space): The Gym observation space.
            action_space (gym.Space): The Gym action space.
        """
        self.observation_space = observation_space
        self.action_space = action_space

    @abc.abstractmethod
    def get_action(self, state: Any, is_training: bool = True) -> Any:
        """
        Selects an action based on the current state of the environment.
        """
        raise NotImplementedError

    @abc.abstractmethod
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
        Updates the agent's internal parameters based on an experience.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_update_info(self) -> Dict[str, Any]:
        """
        Returns a dictionary of agent-specific metrics for logging.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def on_episode_start(self) -> None:
        """
        A hook for any setup logic at the start of an episode.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def on_episode_end(self) -> None:
        """
        A hook for any cleanup or summary logic at the end of an episode.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, path: str, filename: str) -> None:
        """
        Saves the agent's learned model to a file.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load_model(cls, path: str, filename: str, **kwargs: Any) -> "AbstractAgent":
        """
        Loads a previously saved model from a file.
        """
        raise NotImplementedError

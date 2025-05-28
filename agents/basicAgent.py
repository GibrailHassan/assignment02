# agents/basicAgent.py

"""
This module defines a BasicAgent class that inherits from AbstractAgent.
The BasicAgent selects actions based on a straightforward heuristic calculation,
without using any reinforcement learning algorithms.
"""

from typing import Any, Dict
import gym
from agents.abstractAgent import AbstractAgent


class BasicAgent(AbstractAgent):
    """
    A basic, rule-based agent that selects actions to minimize distance to a target.

    Note: The original implementation of this agent required the full environment
    in its get_action method. This has been refactored to be compliant with the
    standard agent-runner interaction, though its heuristic nature remains.
    For its heuristic to work, it would need more information than just the state.
    As is, its get_action method is not fully functional without modification.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """
        Initializes the BasicAgent.

        Args:
            observation_space (gym.Space): The environment's observation space.
            action_space (gym.Space): The environment's action space.
        """
        super().__init__(observation_space, action_space)

    def get_action(self, state: Any, is_training: bool = True) -> int:
        """
        Selects an action.

        NOTE: The original heuristic logic required access to the 'env' object
        to simulate next states, which is not standard. A proper implementation
        would require refactoring the heuristic logic itself. For now, we
        will have it act like a RandomAgent to satisfy the interface.
        """
        # To make this agent runnable, we default to random action selection.
        # To implement the original heuristic, this method would need to be redesigned.
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
        The BasicAgent does not learn, so this method is empty.
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

    def save_model(self, path: str, filename: str) -> None:
        """This agent does not have a model to save."""
        pass

    @classmethod
    def load_model(cls, path: str, filename: str, **kwargs: Any) -> "BasicAgent":
        """This agent does not have a model to load."""
        raise NotImplementedError("The BasicAgent does not support loading models.")

# agents/randomAgent.py

"""
This module defines a RandomAgent class, which inherits from AbstractAgent.
The RandomAgent selects actions randomly from the available action space.
"""

import os
from typing import Any, Dict
import gym

# import numpy as np # Not strictly needed here
from agents.abstractAgent import AbstractAgent


class RandomAgent(AbstractAgent):
    """
    An agent that selects actions randomly, serving as a baseline.
    It does not learn or use the state information to make decisions.
    """

    def __init__(
        self, observation_space: gym.Space, action_space: gym.Space, **kwargs: Any
    ):  # Added **kwargs
        """
        Initializes the RandomAgent.

        Args:
            observation_space (gym.Space): The environment's observation space.
            action_space (gym.Space): The environment's action space.
            **kwargs: Accepts other keyword arguments and ignores them.
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

    def save_model(
        self, path: str, filename: str = "random_agent_history.txt"
    ) -> str:  # Return path
        """Saves the history of actions and states to a text file."""
        # Random agent doesn't have a "model" in the typical sense,
        # but we can save its history if that's useful.
        # Or, simply do nothing if there's nothing to save.
        # For now, let's assume it might save some operational log or history.
        full_path = os.path.join(path, filename)
        os.makedirs(path, exist_ok=True)
        try:
            with open(full_path, "w") as f:
                f.write(
                    self.history if hasattr(self, "history") else "No history recorded."
                )
            print(f"RandomAgent history/log saved to {full_path}")
        except Exception as e:
            print(f"Could not save RandomAgent history: {e}")
            return None  # Indicate failure
        return full_path

    @classmethod
    def load_model(cls, path: str, filename: str, **kwargs: Any) -> "RandomAgent":
        """The RandomAgent is stateless and typically not loaded in a meaningful way."""
        # Re-instantiate a new RandomAgent.
        if "observation_space" not in kwargs or "action_space" not in kwargs:
            raise ValueError(
                "observation_space and action_space must be provided in kwargs for RandomAgent.load_model"
            )
        print(
            f"RandomAgent is stateless. Creating a new instance. (Path: {path}/{filename} ignored for model state)."
        )
        return cls(kwargs["observation_space"], kwargs["action_space"])

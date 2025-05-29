"""
This module defines a BasicAgent class that inherits from AbstractAgent.
The BasicAgent selects actions based on a straightforward heuristic calculation,
without using any reinforcement learning algorithms.
"""

from typing import Any, Dict
import gym
import os  # Added for save_model path creation

# import numpy as np # Not strictly needed here if get_action is simplified
from agents.abstractAgent import AbstractAgent
import mlflow  # For MLflow artifact logging


class BasicAgent(AbstractAgent):
    """
    A basic, rule-based agent.
    Its original heuristic logic for MoveToBeacon is not fully implemented here
    as it required direct environment access in get_action.
    It currently acts randomly to satisfy the interface.
    """

    def __init__(
        self, observation_space: gym.Space, action_space: gym.Space, **kwargs: Any
    ):  # Added **kwargs
        """
        Initializes the BasicAgent.

        Args:
            observation_space (gym.Space): The environment's observation space.
            action_space (gym.Space): The environment's action space.
            **kwargs: Accepts other keyword arguments and ignores them.
        """
        super().__init__(observation_space, action_space)

    def get_action(self, state: Any, is_training: bool = True) -> int:
        """
        Selects an action. Currently defaults to random.
        The original heuristic (minimizing distance to beacon) would require
        access to the environment object or more complex state information.
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

    def save_model(
        self, path: str, filename: str = "basic_agent_info.txt"
    ) -> str:  # Return path
        """This agent does not have a trainable model, but can save info."""
        full_path = os.path.join(path, filename)
        os.makedirs(path, exist_ok=True)
        try:
            with open(full_path, "w") as f:
                f.write("BasicAgent: No trainable parameters.")
            print(f"BasicAgent info saved to {full_path}")
            if mlflow.active_run():
                mlflow.log_artifact(full_path, artifact_path="agent_info")
        except Exception as e:
            print(f"Could not save BasicAgent info: {e}")
            return None
        return full_path

    @classmethod
    def load_model(cls, path: str, filename: str, **kwargs: Any) -> "BasicAgent":
        """This agent does not have a model state to load."""
        if "observation_space" not in kwargs or "action_space" not in kwargs:
            raise ValueError(
                "observation_space and action_space must be provided in kwargs for BasicAgent.load_model"
            )
        print(
            f"BasicAgent is stateless. Creating a new instance. (Path: {path}/{filename} ignored for model state)."
        )
        return cls(kwargs["observation_space"], kwargs["action_space"])

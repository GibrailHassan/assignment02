# agents/tableAgent.py

"""
Defines a base class for table-based reinforcement learning agents.
"""

import os
import random
from typing import Tuple, Dict, Any, Type, TypeVar

import torch
import numpy as np
import gym
from agents.abstractAgent import AbstractAgent

T = TypeVar("T", bound="TableBasedAgent")


class TableBasedAgent(AbstractAgent):
    """
    An abstract base class for agents that use a Q-table.

    This class dynamically computes the Q-table size from the environment's
    observation space and correctly handles state spaces with negative values
    by applying an offset to map states to non-negative indices.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.1,
        **kwargs: Any,  # Added to accept and ignore extra params like is_training
    ):
        super().__init__(
            observation_space, action_space
        )  # AbstractAgent.__init__ doesn't take kwargs

        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min

        state_dims_for_q_table = (
            observation_space.high - observation_space.low + 1
        ).astype(int)
        num_actions = action_space.n
        q_table_shape = tuple(state_dims_for_q_table) + (num_actions,)
        self.q_table: torch.Tensor = torch.zeros(q_table_shape)

        self.index_offset = -observation_space.low.astype(int)
        # print(f"Initialized Q-table with shape: {q_table_shape}") # Already printed by children

    def _get_q_table_indices(self, state: np.ndarray) -> tuple:
        indices = tuple((state + self.index_offset).astype(int))
        return indices

    def get_action(self, state: np.ndarray, is_training: bool = True) -> int:
        current_epsilon: float = self.epsilon if is_training else 0.0
        if random.random() < current_epsilon:
            return self.action_space.sample()

        state_indices = self._get_q_table_indices(state)
        q_values_for_state: torch.Tensor = self.q_table[state_indices]
        best_action: int = torch.argmax(q_values_for_state).item()
        return best_action

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError

    def on_episode_start(self) -> None:
        pass

    def on_episode_end(self) -> None:
        # Only decay epsilon if the agent is set to training mode (relevant for DQNAgent, good practice here too)
        # Assuming an 'is_training' attribute might be set by some agents, or rely on external call context.
        # For TableBasedAgent, epsilon decay is usually per episode.
        # We can check if an 'is_training' attribute exists from kwargs or is set by a child.
        # For now, we'll assume if on_episode_end is called, it's part of a training loop where decay is desired.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_update_info(self) -> Dict[str, float]:
        return {"epsilon": self.epsilon}

    def save_model(self, path: str, filename: str = "q_table.pt") -> str:  # Return path
        os.makedirs(path, exist_ok=True)
        model_path: str = os.path.join(path, filename)
        torch.save(
            {"q_table": self.q_table, "index_offset": self.index_offset}, model_path
        )
        print(f"Q-table saved locally to {model_path}")

        if mlflow.active_run():
            try:
                mlflow.log_artifact(model_path, artifact_path="q_table_models")
                print(
                    f"Q-table also logged as MLflow artifact to 'q_table_models/{filename}'."
                )
            except Exception as e:
                print(f"Warning: Failed to log Q-table to MLflow: {e}")
        return model_path

    @classmethod
    def load_model(
        cls: Type[T], path: str, filename: str = "q_table.pt", **kwargs: Any
    ) -> T:
        model_path: str = os.path.join(path, filename)

        data = torch.load(model_path, weights_only=False)

        q_table = data["q_table"]
        index_offset = data["index_offset"]

        observation_actual_shape: tuple = index_offset.shape
        low = -index_offset
        q_table_state_dims_sizes = np.array(q_table.shape[:-1])
        high_corrected = q_table_state_dims_sizes - 1 - index_offset

        observation_space = gym.spaces.Box(
            low=low, high=high_corrected, shape=observation_actual_shape, dtype=np.int32
        )
        action_space = gym.spaces.Discrete(q_table.shape[-1])

        # Ensure these are passed to the constructor
        kwargs_for_init = kwargs.copy()  # Avoid modifying original kwargs dict
        kwargs_for_init["observation_space"] = observation_space
        kwargs_for_init["action_space"] = action_space

        # Remove params that might not be accepted by __init__ if they were just for load_model
        # For TableBasedAgent, most params are part of __init__ or handled by **kwargs

        instance: T = cls(**kwargs_for_init)

        instance.q_table = q_table
        # Ensure the loaded instance also gets the index_offset
        instance.index_offset = index_offset

        print(f"Q-table loaded from {model_path}")

        return instance

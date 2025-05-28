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
    ):
        super().__init__(observation_space, action_space)

        # --- Hyperparameters ---
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min

        # --- Dynamic Q-Table Initialization ---
        # Calculate the size of each dimension of the state space.
        # e.g., for a range of [-10, 10], the size is 10 - (-10) + 1 = 21.
        state_dims = (observation_space.high - observation_space.low + 1).astype(int)

        # The number of actions is the size of the discrete action space.
        num_actions = action_space.n

        # Combine state dimensions and action dimension for the full Q-table shape.
        q_table_shape = tuple(state_dims) + (num_actions,)
        self.q_table: torch.Tensor = torch.zeros(q_table_shape)

        # --- State to Index Offset ---
        # To handle negative state values, we create an offset.
        # A state of -10 should map to index 0. So, offset = -(-10) = 10.
        self.index_offset = -observation_space.low.astype(int)

        print(f"Initialized Q-table with shape: {q_table_shape}")

    def _get_q_table_indices(self, state: np.ndarray) -> tuple:
        """Converts an environment state to valid Q-table indices."""
        # Add the offset to shift all state values to be non-negative.
        indices = tuple((state + self.index_offset).astype(int))
        return indices

    def get_action(self, state: np.ndarray, is_training: bool = True) -> int:
        """
        Selects an action using an epsilon-greedy policy.
        """
        current_epsilon: float = self.epsilon if is_training else 0.0
        if random.random() < current_epsilon:
            return self.action_space.sample()

        state_indices = self._get_q_table_indices(state)
        q_values_for_state: torch.Tensor = self.q_table[state_indices]
        best_action: int = torch.argmax(q_values_for_state).item()
        return best_action

    # ... (on_episode_start, on_episode_end, get_update_info methods remain the same) ...

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        **kwargs: Any,
    ) -> None:
        """
        A placeholder for the update method, to be implemented by child classes.
        """
        raise NotImplementedError

    def on_episode_start(self) -> None:
        pass

    def on_episode_end(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_update_info(self) -> Dict[str, float]:
        return {"epsilon": self.epsilon}

    def save_model(self, path: str, filename: str = "q_table.pt") -> None:
        os.makedirs(path, exist_ok=True)
        model_path: str = os.path.join(path, filename)
        # We save both the table and the index offset for correct loading.
        torch.save(
            {"q_table": self.q_table, "index_offset": self.index_offset}, model_path
        )
        print(f"Q-table saved to {model_path}")

    @classmethod
    def load_model(
        cls: Type[T], path: str, filename: str = "q_table.pt", **kwargs: Any
    ) -> T:
        model_path: str = os.path.join(path, filename)
        data = torch.load(model_path)
        q_table = data["q_table"]
        index_offset = data["index_offset"]

        # Reconstruct observation and action spaces from the loaded data
        state_dims = np.array(q_table.shape[:-1])
        low = -index_offset
        high = state_dims - index_offset - 1
        observation_space = gym.spaces.Box(low=low, high=high, dtype=np.int32)
        action_space = gym.spaces.Discrete(q_table.shape[-1])

        kwargs["observation_space"] = observation_space
        kwargs["action_space"] = action_space

        instance: T = cls(**kwargs)
        instance.q_table = q_table
        instance.index_offset = index_offset
        print(f"Q-table loaded from {model_path}")

        return instance

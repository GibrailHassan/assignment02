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

        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min

        # state_dims represents the size of each dimension for Q-table indexing
        state_dims_for_q_table = (
            observation_space.high - observation_space.low + 1
        ).astype(int)
        num_actions = action_space.n
        q_table_shape = tuple(state_dims_for_q_table) + (num_actions,)
        self.q_table: torch.Tensor = torch.zeros(q_table_shape)

        # index_offset is used to map environment states (which can be negative)
        # to non-negative Q-table indices.
        self.index_offset = -observation_space.low.astype(int)
        print(f"Initialized Q-table with shape: {q_table_shape}")

    def _get_q_table_indices(self, state: np.ndarray) -> tuple:
        # Apply offset and convert to integer tuple for indexing
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
        # This method must be implemented by child classes (QLearningAgent, SARSAAgent)
        raise NotImplementedError

    def on_episode_start(self) -> None:
        # Hook for logic at the start of an episode (e.g., resetting agent's episode-specific state)
        pass

    def on_episode_end(self) -> None:
        # Hook for logic at the end of an episode (e.g., decaying epsilon)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_update_info(self) -> Dict[str, float]:
        # Returns agent-specific information for logging by the runner
        return {"epsilon": self.epsilon}

    def save_model(self, path: str, filename: str = "q_table.pt") -> None:
        os.makedirs(path, exist_ok=True)
        model_path: str = os.path.join(path, filename)
        # Save both the Q-table and the index_offset needed for correct loading
        torch.save(
            {"q_table": self.q_table, "index_offset": self.index_offset}, model_path
        )
        print(f"Q-table saved to {model_path}")

    @classmethod
    def load_model(
        cls: Type[T], path: str, filename: str = "q_table.pt", **kwargs: Any
    ) -> T:
        model_path: str = os.path.join(path, filename)

        # Explicitly set weights_only=False as the saved file contains more than just weights
        # (it contains a dictionary with q_table and index_offset, which are NumPy arrays when saved by older PyTorch)
        # and was likely saved with an older PyTorch version or without weights_only=True.
        data = torch.load(model_path, weights_only=False)

        q_table = data["q_table"]
        index_offset = data[
            "index_offset"
        ]  # This is a NumPy array, e.g., array([10, 10])

        # Reconstruct the observation_space for the agent
        # The shape of an observation is the number of dimensions in the state vector.
        # This can be derived from the length of the index_offset array.
        observation_actual_shape: tuple = (
            index_offset.shape
        )  # e.g., (2,) for a 2D state vector

        # 'low' bounds for the observation space
        low = -index_offset  # e.g., array([-10, -10])

        # 'high' bounds for the observation space
        # state_dims_for_q_table_sizes = np.array(q_table.shape[:-1]) # e.g., array([21, 21])
        # high_corrected = low + state_dims_for_q_table_sizes - 1 # e.g., array([10, 10])

        # A more direct way to get high assuming symmetric space around 0 for index_offset
        # If observation_space.low was [-10,-10] and observation_space.high was [10,10]
        # then index_offset is [10,10].
        # state_dims_for_q_table = (high - low + 1)
        # high = state_dims_for_q_table - 1 + low
        # high = (q_table.shape[:-1]) - 1 - index_offset # This seems more direct
        q_table_state_dims_sizes = np.array(q_table.shape[:-1])
        high_corrected = q_table_state_dims_sizes - 1 - index_offset

        # Create the gym.spaces.Box object
        # The 'shape' argument must match the shape of 'low' and 'high', and represent one observation.
        observation_space = gym.spaces.Box(
            low=low, high=high_corrected, shape=observation_actual_shape, dtype=np.int32
        )

        # Reconstruct the action_space
        action_space = gym.spaces.Discrete(q_table.shape[-1])

        # Update kwargs with the reconstructed spaces for agent initialization
        kwargs["observation_space"] = observation_space
        kwargs["action_space"] = action_space

        # Create the agent instance
        instance: T = cls(**kwargs)

        # Assign the loaded Q-table and index_offset
        instance.q_table = q_table
        instance.index_offset = index_offset  # Ensure the instance has this attribute

        print(f"Q-table loaded from {model_path}")

        return instance

"""
Defines a base class for table-based reinforcement learning agents,
now with MLflow model artifact logging.
"""

import os
import random
from typing import Tuple, Dict, Any, Type, TypeVar

import torch
import numpy as np
import gym
import mlflow  # Import MLflow

from agents.abstractAgent import AbstractAgent

T = TypeVar("T", bound="TableBasedAgent")


class TableBasedAgent(AbstractAgent):
    """
    An abstract base class for agents that use a Q-table.

    This class dynamically computes the Q-table size from the environment's
    observation space and correctly handles state spaces with negative values
    by applying an offset to map states to non-negative indices.
    It also handles saving the Q-table locally and logging it as an MLflow artifact.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,  # Per-episode decay for table-based
        epsilon_min: float = 0.1,
        is_training: bool = True,  # Added to control epsilon decay during eval
        **kwargs: Any,
    ):
        super().__init__(observation_space, action_space)

        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.initial_epsilon: float = epsilon  # Store initial epsilon
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay  # Multiplicative per-episode decay
        self.epsilon_min: float = epsilon_min
        self.is_training = is_training  # Store is_training status

        state_dims_for_q_table = (
            observation_space.high - observation_space.low + 1
        ).astype(int)
        num_actions = action_space.n
        q_table_shape = tuple(state_dims_for_q_table) + (num_actions,)
        self.q_table: torch.Tensor = torch.zeros(q_table_shape)

        self.index_offset = -observation_space.low.astype(int)
        # print(f"Initialized Q-table with shape: {q_table_shape}") # Children can print if needed

    def _get_q_table_indices(self, state: np.ndarray) -> tuple:
        indices = tuple((state + self.index_offset).astype(int))
        return indices

    def get_action(self, state: np.ndarray, is_training: bool = True) -> int:
        # Use the agent's own is_training status if the argument isn't overriding
        effective_is_training = (
            is_training if is_training is not None else self.is_training
        )
        current_epsilon: float = self.epsilon if effective_is_training else 0.0

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
        # Per-episode epsilon decay, only if training
        if self.is_training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_update_info(self) -> Dict[str, float]:
        return {"epsilon": self.epsilon}

    def save_model(
        self, path: str, filename: str = "q_table_checkpoint.pt"
    ) -> str | None:
        """
        Saves the Q-table and index_offset locally and logs the file as an MLflow artifact.

        Args:
            path (str): The directory path to save the model in.
            filename (str): The name for the saved model file.

        Returns:
            str | None: The full path to the locally saved model file, or None if saving failed.
        """
        if not path:  # If model_save_dir was empty in config
            print(
                "Local model saving path not provided, skipping local save for TableBasedAgent."
            )
            return None

        os.makedirs(path, exist_ok=True)
        model_file_path: str = os.path.join(path, filename)

        try:
            torch.save(
                {
                    "q_table": self.q_table,
                    "index_offset": self.index_offset,
                    "epsilon": self.epsilon,  # Save current epsilon for potential resume
                },
                model_file_path,
            )
            print(
                f"TableBasedAgent model (q_table, offset, epsilon) saved locally to {model_file_path}"
            )

            # --- MLflow Artifact Logging ---
            if mlflow.active_run():
                try:
                    # Log the .pt file as a generic artifact.
                    # The artifact_path creates a subdirectory in the MLflow run's artifacts.
                    mlflow.log_artifact(
                        model_file_path, artifact_path="table_agent_models"
                    )
                    print(
                        f"TableBasedAgent model also logged as MLflow artifact to 'table_agent_models/{filename}'."
                    )
                except Exception as e:
                    print(
                        f"Warning: Failed to log TableBasedAgent model to MLflow: {e}"
                    )
            return model_file_path
        except Exception as e:
            print(f"Error saving TableBasedAgent model locally: {e}")
            return None

    @classmethod
    def load_model(
        cls: Type[T], path: str, filename: str = "q_table_checkpoint.pt", **kwargs: Any
    ) -> T:
        """
        Loads a Q-table, index_offset, and epsilon from a file.
        Requires observation_space and action_space in kwargs to reconstruct the agent.
        """
        model_file_path: str = os.path.join(path, filename)

        if not os.path.exists(model_file_path):
            print(f"Error: Model file not found at {model_file_path}")
            raise FileNotFoundError(f"Model file not found at {model_file_path}")

        try:
            data = torch.load(model_file_path, weights_only=False)
        except Exception as e:
            print(f"Error loading data from {model_file_path} with torch.load: {e}")
            raise

        q_table = data.get("q_table")
        index_offset = data.get("index_offset")
        loaded_epsilon = data.get("epsilon")  # Get saved epsilon

        if q_table is None or index_offset is None:
            raise ValueError(
                f"Model file {model_file_path} is missing 'q_table' or 'index_offset'."
            )

        # observation_space and action_space must be passed via kwargs by the caller (main.py)
        # as they are needed to initialize the agent before loading state.
        if "observation_space" not in kwargs or "action_space" not in kwargs:
            raise ValueError(
                "observation_space and action_space must be provided in kwargs for TableBasedAgent.load_model"
            )

        # Create the agent instance using the provided spaces and other params from kwargs
        instance: T = cls(**kwargs)  # This will call __init__

        # Now, overwrite the newly initialized q_table, index_offset, and epsilon
        instance.q_table = q_table
        instance.index_offset = index_offset
        if loaded_epsilon is not None:
            instance.epsilon = loaded_epsilon
            print(f"Loaded epsilon: {instance.epsilon}")

        # If not training (i.e., loading for evaluation), set epsilon to min
        if hasattr(instance, "is_training") and not instance.is_training:
            instance.epsilon = instance.epsilon_min
            print(
                f"Evaluation mode: Epsilon set to min_epsilon: {instance.epsilon_min}"
            )

        print(f"TableBasedAgent model loaded from {model_file_path}")
        return instance

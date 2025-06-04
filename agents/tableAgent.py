"""
Defines a base class for table-based reinforcement learning agents,
such as Q-Learning and SARSA. This class provides common functionality
including Q-table initialization, management of state indexing (especially
for environments with negative state values), epsilon-greedy action selection,
and model persistence (saving/loading the Q-table and related parameters),
including logging the model as an MLflow artifact.
"""

import os
import random
from typing import Tuple, Dict, Any, Type, TypeVar, Optional  # Added Optional
import abc  # Abstract Base Classes module for defining abstract methods
import torch
import numpy as np
import gym
import mlflow  # Import MLflow for artifact logging

# Import the abstract parent class
from agents.abstractAgent import AbstractAgent

# Define a type variable for generic subclasses of TableBasedAgent.
# This is used for the load_model classmethod to correctly type its return value.
T_TableBasedAgent = TypeVar("T_TableBasedAgent", bound="TableBasedAgent")


class TableBasedAgent(AbstractAgent):
    """
    An abstract base class for reinforcement learning agents that use a Q-table.

    This class handles:
    - Dynamic Q-table sizing based on the environment's observation space.
    - Mapping of potentially negative state values from the environment to
      non-negative indices suitable for Q-table access using an `index_offset`.
    - Epsilon-greedy action selection strategy for balancing exploration and exploitation.
    - Per-episode epsilon decay.
    - Saving the Q-table, `index_offset`, and current `epsilon` to a local file
      and logging this file as an MLflow artifact.
    - Loading the Q-table, `index_offset`, and `epsilon` from a file to reconstruct
      the agent's state.

    Subclasses (like `QLearningAgent` and `SARSAAgent`) must implement the
    `update` method, which defines the specific learning rule of the algorithm.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,  # Expects a Box space for state_dims calculation
        action_space: gym.spaces.Discrete,  # Expects a Discrete space for num_actions
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,  # Gamma
        epsilon: float = 1.0,  # Initial exploration rate
        epsilon_decay: float = 0.995,  # Multiplicative decay factor per episode
        epsilon_min: float = 0.1,  # Minimum exploration rate
        is_training: bool = True,  # Agent mode: True for training, False for evaluation
        **kwargs: Any,  # Absorb any additional parameters from config
    ):
        """
        Initializes the TableBasedAgent.

        Args:
            observation_space (gym.spaces.Box): The environment's observation space.
                Expected to be a `gym.spaces.Box` to determine the dimensions and
                range (low/high) of states for Q-table sizing.
            action_space (gym.spaces.Discrete): The environment's action space.
                Expected to be `gym.spaces.Discrete` to determine the number of actions.
            learning_rate (float, optional): The learning rate (alpha) for updates.
                                            Defaults to 0.1.
            discount_factor (float, optional): The discount factor (gamma) for future
                                               rewards. Defaults to 0.99.
            epsilon (float, optional): The initial probability for choosing a random
                                       action (exploration). Defaults to 1.0.
            epsilon_decay (float, optional): The multiplicative factor by which epsilon
                                             is decayed at the end of each training episode.
                                             Defaults to 0.995.
            epsilon_min (float, optional): The minimum value epsilon can decay to.
                                           Defaults to 0.1.
            is_training (bool, optional): Flag indicating if the agent is in training mode.
                                          This affects epsilon decay and action selection.
                                          Defaults to True.
            **kwargs (Any): Additional keyword arguments passed from configuration,
                            which are ignored by this base class constructor but might
                            be used by subclasses if they override `__init__`.
        """
        super().__init__(observation_space, action_space)

        # Learning parameters
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor

        # Epsilon-greedy parameters
        self.initial_epsilon: float = epsilon  # Store the initial epsilon for reference
        self.epsilon: float = epsilon  # Current epsilon value, decays during training
        self.epsilon_decay: float = epsilon_decay  # Per-episode decay factor
        self.epsilon_min: float = epsilon_min

        self.is_training: bool = is_training  # Agent's operational mode

        # --- Q-Table Initialization ---
        # The Q-table stores Q(s,a) values. Its dimensions depend on the state space
        # and action space.
        # We assume a discrete state space that can be derived from observation_space.high/low.
        if not isinstance(observation_space, gym.spaces.Box):
            raise ValueError(
                "TableBasedAgent currently requires a gym.spaces.Box observation space "
                "to determine Q-table dimensions from space.low and space.high."
            )
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                "TableBasedAgent requires a gym.spaces.Discrete action space."
            )

        # Calculate the size of each dimension in the state space for the Q-table.
        # (high - low + 1) gives the number of discrete steps in each dimension.
        state_dims_for_q_table = (
            observation_space.high - observation_space.low + 1
        ).astype(
            int
        )  # Ensure integer dimensions

        num_actions: int = action_space.n  # Number of possible actions

        # Shape of the Q-table: (dim1_size, dim2_size, ..., dimN_size, num_actions)
        q_table_shape: tuple = tuple(state_dims_for_q_table) + (num_actions,)

        # Initialize Q-table with zeros. Using PyTorch tensor for potential GPU usage
        # or easier integration with PyTorch-based components if ever needed, though
        # for simple table-based methods, NumPy might also suffice.
        self.q_table: torch.Tensor = torch.zeros(
            q_table_shape
        )  # dtype=torch.float32 by default

        # Calculate the offset needed to map state values (which can be negative)
        # to non-negative Q-table indices.
        # E.g., if state[0] can range from -10 to 10, and observation_space.low[0] is -10,
        # then index_offset[0] will be 10.
        # state_value + index_offset = q_table_index
        self.index_offset: np.ndarray = -observation_space.low.astype(int)

        # Logging initialization details (optional, can be verbose)
        # print(f"Initialized Q-table with shape: {q_table_shape}")
        # print(f"Calculated index offset for Q-table: {self.index_offset}")

    def _get_q_table_indices(self, state: np.ndarray) -> tuple:
        """
        Converts a raw state from the environment to a tuple of Q-table indices.

        This method applies the `self.index_offset` to ensure that all state
        components are mapped to non-negative integers suitable for indexing
        the Q-table.

        Args:
            state (np.ndarray): The raw state vector from the environment.

        Returns:
            tuple: A tuple of integer indices corresponding to the state,
                   which can be used to access the Q-table.
        """
        # Apply the offset and convert to integers
        indices = tuple((state + self.index_offset).astype(int))
        return indices

    def get_action(self, state: np.ndarray, is_training: Optional[bool] = None) -> int:
        """
        Selects an action using an epsilon-greedy strategy based on the Q-table.

        If `is_training` is True (or self.is_training is True and is_training is None):
        - With probability epsilon, a random action is chosen (exploration).
        - With probability 1-epsilon, the action with the highest Q-value for the
          current state is chosen (exploitation).
        If not in training mode, it always chooses the greedy action (exploits).

        Args:
            state (np.ndarray): The current state of the environment.
            is_training (Optional[bool], optional): Overrides the agent's internal
                                                 `self.is_training` flag for this
                                                 specific action selection. If None,
                                                 `self.is_training` is used.
                                                 Defaults to None.

        Returns:
            int: The index of the selected action.
        """
        # Determine the effective training mode for this action selection
        effective_is_training: bool = (
            is_training if is_training is not None else self.is_training
        )

        # Use current epsilon if training, otherwise use a very small value (or 0) for exploitation
        # self.epsilon is decayed during training. For pure exploitation in eval,
        # one might consider current_epsilon = 0.0 directly.
        # However, using self.epsilon (which might be self.epsilon_min if fully decayed)
        # or a specific eval_epsilon (e.g., 0.0 or 0.01) is common.
        current_epsilon: float = (
            self.epsilon if effective_is_training else self.epsilon_min
        )
        # For truly greedy evaluation, one might use:
        # current_epsilon = self.epsilon if effective_is_training else 0.0

        # Epsilon-greedy decision
        if random.random() < current_epsilon:
            # Exploration: Select a random action from the action space
            return self.action_space.sample()
        else:
            # Exploitation: Select the action with the highest Q-value for the current state
            state_indices: tuple = self._get_q_table_indices(state)
            q_values_for_state: torch.Tensor = self.q_table[state_indices]

            # Find the action (index) with the maximum Q-value
            # torch.argmax returns the index of the maximum value.
            # .item() converts a single-element tensor to a Python number.
            best_action: int = torch.argmax(q_values_for_state).item()
            return best_action

    @abc.abstractmethod  # Ensure subclasses implement their specific update rule
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
        Updates the Q-table based on an experience tuple.
        This method must be implemented by subclasses (e.g., QLearningAgent, SARSAAgent)
        to define their specific learning rule.
        """
        raise NotImplementedError(
            "Subclasses like QLearningAgent or SARSAAgent must implement the update method."
        )

    def on_episode_start(self) -> None:
        """
        Hook for logic at the start of an episode.
        For TableBasedAgent, this currently does nothing but can be overridden.
        """
        pass  # No specific actions needed at the start of an episode by default.

    def on_episode_end(self) -> None:
        """
        Hook for logic at the end of an episode.
        Performs epsilon decay if the agent is in training mode.
        """
        # Decay epsilon at the end of each episode if in training mode
        if self.is_training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            # Optional: Log or print epsilon decay
            # print(f"End of episode: Epsilon decayed to {self.epsilon:.4f}")

    def get_update_info(self) -> Dict[str, float]:
        """
        Returns agent-specific information for logging, such as the current epsilon value.

        Returns:
            Dict[str, float]: A dictionary containing the current "epsilon".
        """
        return {"epsilon": self.epsilon}

    def save_model(
        self, path: str, filename: str = "q_table_checkpoint.pt"
    ) -> Optional[str]:
        """
        Saves the Q-table, index_offset, and current epsilon value locally.
        Also logs the saved model file as an artifact to MLflow if a run is active.

        Args:
            path (str): The directory path where the model file should be saved.
            filename (str, optional): The name for the saved model file.
                                     Defaults to "q_table_checkpoint.pt".

        Returns:
            Optional[str]: The full path to the locally saved model file if successful,
                           otherwise None if saving failed or was skipped (e.g., no path).
        """
        if not path:  # If model_save_dir was empty in config, path might be empty
            print(
                "TableBasedAgent: Local model saving path not provided. Skipping local save."
            )
            # Consider if MLflow logging should still occur if local save is skipped.
            # For now, it's tied to a successful local save attempt having a file path.
            return None

        os.makedirs(path, exist_ok=True)  # Ensure the directory exists
        model_file_path: str = os.path.join(path, filename)

        try:
            # Prepare data to be saved
            data_to_save = {
                "q_table": self.q_table,  # The learned Q-values
                "index_offset": self.index_offset,  # Needed to correctly map states to indices upon loading
                "epsilon": self.epsilon,  # Save current epsilon for potential resume or inspection
                # Optionally save other relevant hyperparameters if needed for consistent loading
                # "learning_rate": self.learning_rate,
                # "discount_factor": self.discount_factor,
                # "epsilon_min": self.epsilon_min,
                # "epsilon_decay": self.epsilon_decay,
            }
            torch.save(data_to_save, model_file_path)
            print(
                f"TableBasedAgent model (Q-table, offset, epsilon) saved locally to: {model_file_path}"
            )

            # --- MLflow Artifact Logging ---
            if mlflow.active_run():  # Check if an MLflow run is currently active
                try:
                    # Log the .pt file as a generic artifact.
                    # The artifact_path creates a subdirectory in the MLflow run's artifacts.
                    mlflow.log_artifact(
                        model_file_path, artifact_path="table_agent_models"
                    )
                    print(
                        f"TableBasedAgent model also logged as MLflow artifact to 'table_agent_models/{filename}'."
                    )
                except Exception as e_mlflow:
                    print(
                        f"Warning: Failed to log TableBasedAgent model to MLflow: {e_mlflow}"
                    )
            return (
                model_file_path  # Return the path to the successfully saved local file
            )
        except Exception as e_save:
            print(
                f"Error saving TableBasedAgent model locally to {model_file_path}: {e_save}"
            )
            return None

    @classmethod
    def load_model(
        cls: Type[T_TableBasedAgent],
        path: str,
        filename: str = "q_table_checkpoint.pt",
        **kwargs: Any,
    ) -> T_TableBasedAgent:
        """
        Loads a Q-table, index_offset, and epsilon from a file to reconstruct the agent.

        This class method requires `observation_space` and `action_space` to be provided
        in `kwargs`. These are used to initialize a new instance of the agent class
        before its Q-table and other parameters are overwritten with the loaded data.
        Other parameters from `kwargs` are also passed to the constructor.

        Args:
            cls (Type[T_TableBasedAgent]): The agent class (e.g., QLearningAgent, SARSAAgent).
            path (str): The directory path where the model file is located.
            filename (str, optional): The name of the model file to load.
                                     Defaults to "q_table_checkpoint.pt".
            **kwargs (Any): Must include `observation_space` (gym.spaces.Box) and
                            `action_space` (gym.spaces.Discrete) for agent
                            re-instantiation. Other parameters for the constructor
                            (like `learning_rate`, `is_training`) should also be
                            provided as needed.

        Returns:
            T_TableBasedAgent: An instance of the agent, initialized with the loaded state.

        Raises:
            FileNotFoundError: If the model file is not found.
            ValueError: If essential data ('q_table', 'index_offset') is missing from
                        the loaded file, or if 'observation_space' or 'action_space'
                        are not provided in kwargs.
        """
        model_file_path: str = os.path.join(path, filename)

        if not os.path.exists(model_file_path):
            print(f"Error: Model file not found at {model_file_path}")
            raise FileNotFoundError(f"Model file not found at {model_file_path}")

        # Load the data dictionary from the file.
        # `weights_only=False` is important if you're saving more than just weights (e.g., pickled objects, though here it's tensors).
        # For PyTorch tensors, `map_location` can be useful if loading on a different device.
        try:
            # For table-based agents, device mapping is less critical than for NNs,
            # but good practice if Q-table might grow very large.
            # Default map_location will load to CPU if saved on CPU, GPU if saved on GPU (and GPU available).
            data: Dict[str, Any] = torch.load(model_file_path, weights_only=False)
        except Exception as e_load:
            print(
                f"Error loading data from {model_file_path} with torch.load: {e_load}"
            )
            raise

        # Extract the necessary components from the loaded data
        q_table_loaded: Optional[torch.Tensor] = data.get("q_table")
        index_offset_loaded: Optional[np.ndarray] = data.get("index_offset")
        epsilon_loaded: Optional[float] = data.get("epsilon")
        # One could also load and restore other hyperparams if saved:
        # learning_rate_loaded = data.get("learning_rate")

        if q_table_loaded is None or index_offset_loaded is None:
            raise ValueError(
                f"Model file {model_file_path} is missing 'q_table' or 'index_offset'."
            )

        # `observation_space` and `action_space` MUST be passed via kwargs by the caller (e.g., main.py)
        # as they are required to initialize the agent instance before loading its state.
        if "observation_space" not in kwargs or "action_space" not in kwargs:
            raise ValueError(
                "TableBasedAgent.load_model requires 'observation_space' and 'action_space' "
                "to be provided in kwargs for agent re-instantiation."
            )

        # Create a new agent instance using the provided spaces and other params from kwargs.
        # This will call the subclass's __init__ method (e.g., QLearningAgent.__init__).
        # If `is_training` is not in kwargs, it will use the default from __init__ (usually True).
        # For evaluation, `is_training=False` should be passed in kwargs.
        agent_instance: T_TableBasedAgent = cls(**kwargs)

        # Now, overwrite the newly initialized q_table, index_offset, and epsilon
        # with the values loaded from the file.
        agent_instance.q_table = q_table_loaded
        agent_instance.index_offset = index_offset_loaded

        if epsilon_loaded is not None:
            agent_instance.epsilon = epsilon_loaded
            # print(f"Loaded epsilon: {agent_instance.epsilon:.4f}")
        else:
            # If epsilon wasn't saved, it will retain its initial value from __init__
            # or might need to be set explicitly for evaluation.
            print(
                "Warning: Epsilon not found in saved model. Using epsilon from __init__."
            )

        # If the agent is being loaded for evaluation (i.e., not training),
        # set its epsilon to the minimum exploration rate for greedy action selection.
        if hasattr(agent_instance, "is_training") and not agent_instance.is_training:
            agent_instance.epsilon = agent_instance.epsilon_min
            print(
                f"Agent loaded in evaluation mode: Epsilon explicitly set to min_epsilon ({agent_instance.epsilon_min:.4f})."
            )

        # Restore other hyperparams if they were saved and are part of the agent's state
        # if learning_rate_loaded is not None:
        #     agent_instance.learning_rate = learning_rate_loaded

        print(
            f"TableBasedAgent model (Q-table, offset, epsilon) loaded successfully from {model_file_path}"
        )
        return agent_instance

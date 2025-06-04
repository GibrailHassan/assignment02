"""
This module defines a RandomAgent class, which inherits from AbstractAgent.
The RandomAgent selects actions randomly from the available action space of the
environment. It serves as a crucial baseline for evaluating the performance of
more sophisticated learning agents; a learning agent should ideally perform
better than a random one.
"""

import os  # For path manipulation when saving agent information
from typing import Any, Dict, Optional, Type  # For type hinting
import gym  # For type hinting of observation and action spaces
import mlflow  # For MLflow artifact logging, if an active run exists

# Import the parent class from which RandomAgent inherits
from agents.abstractAgent import AbstractAgent

# Define a type variable for the class itself, used in the load_model classmethod
RandomAgentType = Type["RandomAgent"]


class RandomAgent(AbstractAgent):
    """
    An agent that selects actions uniformly at random from the action space.

    This agent does not learn from its experiences or use state information to
    make decisions. Its primary purpose is to provide a simple baseline against
    which the performance of learning-based agents can be compared.
    It implements the `AbstractAgent` interface, but its learning-related
    methods (like `update`) are no-ops.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        **kwargs: Any,  # Allow for unused parameters passed from config
    ) -> None:
        """
        Initializes the RandomAgent.

        Args:
            observation_space (gym.Space): The environment's observation space.
                                           This is stored but not used for action selection.
            action_space (gym.Space): The environment's action space, from which
                                      actions will be randomly sampled.
            **kwargs (Any): Accepts other keyword arguments from the configuration
                            which are ignored by this agent.
        """
        super().__init__(observation_space, action_space)
        # The 'history' attribute mentioned in the original file was not clearly defined
        # in its usage. If intended for logging actions/states, it would need
        # to be populated in get_action or update. For simplicity, it's omitted
        # here unless a more specific logging mechanism is implemented.
        # self.history: str = ""

    def get_action(self, state: Any, is_training: bool = True) -> int:
        """
        Selects a random action from the environment's action space.

        The current state of the environment and the training mode are ignored
        as the agent's behavior is purely random.

        Args:
            state (Any): The current state of the environment (ignored).
            is_training (bool, optional): Flag indicating training mode (ignored).
                                          Defaults to True.

        Returns:
            int: A randomly selected action. The return type assumes a discrete
                 action space, which is common for `action_space.sample()`.
                 If the action space were different (e.g., Box), the return type
                 would match that space's sample type.
        """
        # Sample a random action from the provided action space
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
        The RandomAgent does not learn, so this method is a no-op (does nothing).

        Args:
            state (Any): The state from which the action was taken.
            action (Any): The action taken.
            reward (float): The reward received.
            next_state (Any): The state transitioned to.
            done (bool): Whether the episode terminated.
            **kwargs (Any): Additional arguments.
        """
        pass  # Non-learning agent, no update logic needed.

    def on_episode_start(self) -> None:
        """
        Hook for logic at the start of an episode. No action needed for RandomAgent.
        """
        pass  # No episode-specific setup required.

    def on_episode_end(self) -> None:
        """
        Hook for logic at the end of an episode. No action needed for RandomAgent.
        """
        pass  # No episode-specific cleanup or processing required.

    def get_update_info(self) -> Dict[str, Any]:
        """
        Returns an empty dictionary as this agent has no learning-specific metrics to log.

        Returns:
            Dict[str, Any]: An empty dictionary.
        """
        return {}  # No metrics like epsilon or loss to report.

    def save_model(
        self, path: str, filename: str = "random_agent_info.txt"
    ) -> Optional[str]:
        """
        Saves a placeholder information file, as RandomAgent has no trainable model.

        This method fulfills the `AbstractAgent` interface requirement. It saves
        a simple text file indicating that the agent is random and doesn't have
        trainable parameters. This info file is also logged to MLflow if an
        active run exists.

        Args:
            path (str): The directory path to save the information file.
            filename (str, optional): The name for the saved information file.
                                     Defaults to "random_agent_info.txt".

        Returns:
            Optional[str]: The full path to the saved information file if successful,
                           otherwise None.
        """
        if not path:  # Handle cases where an empty path might be provided
            print(
                "Warning: Save path for RandomAgent info not provided. Skipping save."
            )
            return None
        os.makedirs(path, exist_ok=True)  # Ensure the directory exists

        full_path = os.path.join(path, filename)
        try:
            with open(full_path, "w") as f:
                f.write(
                    "RandomAgent: This agent selects actions randomly and has no trainable parameters.\n"
                )
                f.write(f"Observation Space: {self.observation_space}\n")
                f.write(f"Action Space: {self.action_space}\n")
                # if hasattr(self, "history") and self.history: # If history logging were implemented
                #     f.write("\n--- Action History (if recorded) ---\n")
                #     f.write(self.history)
            print(f"RandomAgent information saved to {full_path}")

            # Log this info file as an artifact to MLflow if a run is active
            if mlflow.active_run():
                try:
                    mlflow.log_artifact(full_path, artifact_path="agent_info")
                    print(
                        f"Logged RandomAgent info file to MLflow artifacts (under 'agent_info/')."
                    )
                except Exception as e_mlflow:
                    print(
                        f"Warning: Could not log RandomAgent info file to MLflow: {e_mlflow}"
                    )
            return full_path
        except Exception as e:
            print(f"Error saving RandomAgent information to {full_path}: {e}")
            return None

    @classmethod
    def load_model(
        cls: Type[RandomAgentType], path: str, filename: str, **kwargs: Any
    ) -> RandomAgentType:
        """
        "Loads" a RandomAgent. Since it's stateless, this effectively re-instantiates it.

        This agent does not have a trainable model state to load from a file.
        This method fulfills the `AbstractAgent` interface. It requires
        `observation_space` and `action_space` to be provided in `kwargs`
        to properly initialize a new instance.

        Args:
            cls (Type[RandomAgentType]): The RandomAgent class itself.
            path (str): The directory path where an info file might exist (ignored for state).
            filename (str): The name of an info file (ignored for state).
            **kwargs (Any): Must include `observation_space` and `action_space`
                            for re-instantiation. Other kwargs are passed to __init__.

        Returns:
            RandomAgentType: A new instance of RandomAgent.

        Raises:
            ValueError: If `observation_space` or `action_space` is not in `kwargs`.
        """
        if "observation_space" not in kwargs or "action_space" not in kwargs:
            raise ValueError(
                "observation_space and action_space must be provided in kwargs "
                "for RandomAgent.load_model to instantiate the agent."
            )

        # Log that we are creating a new instance as RandomAgent is stateless.
        print(
            f"RandomAgent is stateless regarding trainable parameters. Creating a new instance. "
            f"(Path '{os.path.join(path, filename)}' is noted but not used to load model state)."
        )

        # Create and return a new instance of the agent using the provided spaces and other params.
        return cls(**kwargs)

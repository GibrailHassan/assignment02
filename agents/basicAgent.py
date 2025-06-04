"""
This module defines a BasicAgent class that inherits from AbstractAgent.
The BasicAgent selects actions based on a straightforward heuristic calculation,
without using any reinforcement learning algorithms. It's primarily intended
as a simple baseline, particularly for environments like "MoveToBeacon".
"""

from typing import Any, Dict, Optional, Type
import gym
import os  # For path manipulation during model/info saving
import mlflow  # For MLflow artifact logging, if an active run exists

from agents.abstractAgent import AbstractAgent  # Parent class

# Define a type variable for the class itself, used in the load_model classmethod
BasicAgentType = Type["BasicAgent"]


class BasicAgent(AbstractAgent):
    """
    A basic, rule-based or heuristic agent.

    This agent does not implement any learning algorithms. Instead, its action
    selection is based on pre-defined rules or simple heuristics.
    For the "MoveToBeacon" environment, its original design intended to choose
    actions that minimize the direct distance to the beacon. However, in the
    current generic implementation within the `AbstractAgent` framework, if
    such environment-specific logic isn't directly available through the `state`
    or simple calculations, it might default to simpler behavior (like random action)
    to satisfy the interface requirements.

    It serves as a non-learning baseline to compare against more sophisticated
    RL agents.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        **kwargs: Any,  # Allow for unused parameters passed from config
    ) -> None:
        """
        Initializes the BasicAgent.

        Args:
            observation_space (gym.Space): The environment's observation space.
                                           Although not heavily used by this simple
                                           agent, it's part of the AbstractAgent interface.
            action_space (gym.Space): The environment's action space, used for
                                      sampling random actions if no specific
                                      heuristic is applicable.
            **kwargs (Any): Accepts other keyword arguments from the configuration
                            which are ignored by this agent.
        """
        super().__init__(observation_space, action_space)
        # No specific learning parameters or models to initialize for BasicAgent.

    def get_action(self, state: Any, is_training: bool = True) -> int:
        """
        Selects an action based on a simple heuristic or defaults to random.

        The original intent for environments like 'MoveToBeacon' might involve
        calculating which action minimizes distance to a target. If such specific
        logic isn't implemented here (as it might require direct environment
        access or more complex state parsing not assumed by the generic `state` object),
        this method will default to selecting a random action from the action space.

        Args:
            state (Any): The current state of the environment. This agent might
                         use it if a simple heuristic can be derived from it,
                         otherwise, it's ignored.
            is_training (bool, optional): Flag indicating training mode. Ignored by
                                          this non-learning agent. Defaults to True.

        Returns:
            int: A randomly selected action index from the environment's action space.
        """
        # Placeholder for heuristic logic.
        # For example, if state was a simple [dx, dy] and actions were directions:
        #   if state[0] < 0: action = "move_right"
        #   ... etc.
        # Since no specific heuristic is implemented here based on a generic 'state',
        # it defaults to random action.
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
        The BasicAgent does not learn from experience, so this method is a no-op.

        Args:
            state (Any): The state from which the action was taken.
            action (Any): The action taken.
            reward (float): The reward received.
            next_state (Any): The state transitioned to.
            done (bool): Whether the episode terminated.
            **kwargs (Any): Additional arguments.
        """
        pass  # Non-learning agent, no update needed.

    def on_episode_start(self) -> None:
        """
        Hook for logic at the start of an episode. No action needed for this agent.
        """
        pass  # No episode-specific setup needed.

    def on_episode_end(self) -> None:
        """
        Hook for logic at the end of an episode. No action needed for this agent.
        """
        pass  # No episode-specific cleanup or processing needed.

    def get_update_info(self) -> Dict[str, Any]:
        """
        Returns an empty dictionary as this agent has no learning-specific metrics.

        Returns:
            Dict[str, Any]: An empty dictionary.
        """
        return {}  # No specific metrics like epsilon or loss to report.

    def save_model(
        self, path: str, filename: str = "basic_agent_info.txt"
    ) -> Optional[str]:
        """
        Saves a placeholder information file, as this agent has no trainable model.

        While there's no "model" in the traditional RL sense to save, this method
        can save a simple text file indicating that this agent is rule-based or
        heuristic and doesn't have trainable parameters. This fulfills the
        `AbstractAgent` interface requirement. The info file is also logged to
        MLflow if an active run exists.

        Args:
            path (str): The directory path to save the information file.
            filename (str, optional): The name for the saved information file.
                                     Defaults to "basic_agent_info.txt".

        Returns:
            Optional[str]: The full path to the saved information file if successful,
                           otherwise None.
        """
        # Ensure the directory exists
        if not path:  # Handle cases where an empty path might be provided
            print("Warning: Save path for BasicAgent info not provided. Skipping save.")
            return None
        os.makedirs(path, exist_ok=True)

        full_path = os.path.join(path, filename)
        try:
            with open(full_path, "w") as f:
                f.write(
                    "BasicAgent: This agent is rule-based or heuristic and has no trainable parameters.\n"
                )
                f.write(f"Observation Space: {self.observation_space}\n")
                f.write(f"Action Space: {self.action_space}\n")
            print(f"BasicAgent information saved to {full_path}")

            # Log this info file as an artifact to MLflow if a run is active
            if mlflow.active_run():
                try:
                    mlflow.log_artifact(full_path, artifact_path="agent_info")
                    print(
                        f"Logged BasicAgent info file to MLflow artifacts (under 'agent_info/')."
                    )
                except Exception as e_mlflow:
                    print(
                        f"Warning: Could not log BasicAgent info file to MLflow: {e_mlflow}"
                    )
            return full_path
        except Exception as e:
            print(f"Error saving BasicAgent information to {full_path}: {e}")
            return None

    @classmethod
    def load_model(
        cls: Type[BasicAgentType], path: str, filename: str, **kwargs: Any
    ) -> BasicAgentType:
        """
        "Loads" a BasicAgent. Since it's stateless, this effectively re-instantiates it.

        This agent does not have a trainable model state to load from a file.
        This method fulfills the `AbstractAgent` interface. It requires
        `observation_space` and `action_space` to be provided in `kwargs`
        to properly initialize a new instance.

        Args:
            cls (Type[BasicAgentType]): The BasicAgent class itself.
            path (str): The directory path where an info file might exist (ignored for state).
            filename (str): The name of an info file (ignored for state).
            **kwargs (Any): Must include `observation_space` and `action_space`
                            for re-instantiation. Other kwargs are passed to __init__.

        Returns:
            BasicAgentType: A new instance of BasicAgent.

        Raises:
            ValueError: If `observation_space` or `action_space` is not in `kwargs`.
        """
        if "observation_space" not in kwargs or "action_space" not in kwargs:
            raise ValueError(
                "observation_space and action_space must be provided in kwargs "
                "for BasicAgent.load_model to instantiate the agent."
            )

        # Log that we are creating a new instance as BasicAgent is stateless regarding trainable parameters.
        # The file at path/filename might contain descriptive info but isn't used to restore state.
        print(
            f"BasicAgent is stateless regarding trainable parameters. Creating a new instance. "
            f"(Path '{os.path.join(path, filename)}' is noted but not used to load model state)."
        )

        # Create and return a new instance of the agent.
        # observation_space and action_space are extracted by __init__ via kwargs.
        return cls(**kwargs)

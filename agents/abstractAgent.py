"""
This module defines the AbstractAgent class, an abstract base class (ABC)
that outlines the common interface and essential methods for all reinforcement
learning (RL) agents within this framework.

By inheriting from this class, concrete agent implementations (e.g., QLearningAgent,
DQNAgent) are enforced to provide a consistent API for interaction with the
environment and the training/evaluation runner. This promotes modularity and
interchangeability of different RL algorithms.
"""

import abc  # Abstract Base Classes module
from typing import Any, Dict, TypeVar
import gym  # For type hinting of observation and action spaces

# Type variable for generic agent class, used for the load_model classmethod return type
AgentType = TypeVar("AgentType", bound="AbstractAgent")


class AbstractAgent(metaclass=abc.ABCMeta):
    """
    Abstract base class defining the core structure and required methods for an RL agent.

    All RL agents in this project should inherit from this class and implement
    its abstract methods. This ensures a common interface for tasks such as
    action selection, model updates, saving/loading models, and episode lifecycle hooks.

    Attributes:
        observation_space (gym.Space): The observation space of the environment.
                                       Defines the structure of states the agent can perceive.
        action_space (gym.Space): The action space of the environment.
                                  Defines the set of actions the agent can take.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """
        Initializes the AbstractAgent.

        This constructor should be called by all subclasses to store the
        environment's observation and action spaces.

        Args:
            observation_space (gym.Space): The Gym observation space object,
                                           which defines the format of states.
            action_space (gym.Space): The Gym action space object, which defines
                                      the format of actions.
        """
        self.observation_space: gym.Space = observation_space
        self.action_space: gym.Space = action_space

    @abc.abstractmethod
    def get_action(self, state: Any, is_training: bool = True) -> Any:
        """
        Selects an action based on the current state of the environment and training mode.

        This method encapsulates the agent's policy (how it chooses actions).
        The `is_training` flag allows the agent to behave differently during
        training (e.g., with exploration) versus evaluation (e.g., purely exploitative).

        Args:
            state (Any): The current state observation from the environment. The type
                         can vary depending on the environment (e.g., np.ndarray, Dict).
            is_training (bool, optional): A flag indicating whether the agent is
                                          currently in training mode. Defaults to True.
                                          If True, the agent might apply exploration
                                          strategies (e.g., epsilon-greedy).
                                          If False, the agent should act greedily or
                                          deterministically based on its learned policy.

        Returns:
            Any: The action selected by the agent. The type depends on the
                 environment's action space (e.g., int for Discrete, np.ndarray for Box).
        """
        raise NotImplementedError("Subclasses must implement the get_action method.")

    @abc.abstractmethod
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
        Updates the agent's internal model or parameters based on an experience tuple.

        This method is central to the learning process. It takes a transition
        (state, action, reward, next_state, done) and uses it to update the
        agent's knowledge (e.g., Q-values, network weights).

        Args:
            state (Any): The state from which the action was taken.
            action (Any): The action taken in that state.
            reward (float): The reward received from the environment after taking the action.
            next_state (Any): The state transitioned to after taking the action.
            done (bool): A flag indicating whether the episode terminated after this transition.
            **kwargs (Any): Additional keyword arguments that might be necessary for
                            specific algorithms (e.g., `next_action` for SARSA).
        """
        raise NotImplementedError("Subclasses must implement the update method.")

    @abc.abstractmethod
    def get_update_info(self) -> Dict[str, Any]:
        """
        Returns a dictionary of agent-specific metrics for logging.

        This allows the agent to expose internal values (e.g., current epsilon,
        loss values, step counts) that are useful for monitoring its training
        progress. These metrics are typically logged by the Runner.

        Returns:
            Dict[str, Any]: A dictionary where keys are metric names (str) and
                            values are their corresponding values (e.g., float, int).
        """
        raise NotImplementedError(
            "Subclasses must implement the get_update_info method."
        )

    @abc.abstractmethod
    def on_episode_start(self) -> None:
        """
        A hook for any logic that needs to be executed at the start of a new episode.

        This can be used for resetting episode-specific internal states of the agent,
        adjusting parameters, or any other setup required before an episode begins.
        """
        raise NotImplementedError(
            "Subclasses must implement the on_episode_start method."
        )

    @abc.abstractmethod
    def on_episode_end(self) -> None:
        """
        A hook for any logic that needs to be executed at the end of an episode.

        This can be used for end-of-episode cleanup, learning updates that occur
        only at episode boundaries (e.g., some forms of epsilon decay for table-based agents),
        or summarizing episode performance from the agent's perspective.
        """
        raise NotImplementedError(
            "Subclasses must implement the on_episode_end method."
        )

    @abc.abstractmethod
    def save_model(self, path: str, filename: str) -> str | None:
        """
        Saves the agent's learned model (e.g., Q-table, neural network weights) to a file.

        Args:
            path (str): The directory path where the model file should be saved.
            filename (str): The name of the file for the saved model.

        Returns:
            str | None: The full path to the saved model file if successful,
                        otherwise None if saving failed or was skipped.
        """
        raise NotImplementedError("Subclasses must implement the save_model method.")

    @classmethod
    @abc.abstractmethod
    def load_model(
        cls: type[AgentType], path: str, filename: str, **kwargs: Any
    ) -> AgentType:
        """
        Loads a previously saved model from a file and returns an instance of the agent.

        This is a class method, meaning it should be callable on the class itself
        (e.g., `DQNAgent.load_model(...)`) to create an instance from a saved state.

        Args:
            cls (type[AgentType]): The agent class itself.
            path (str): The directory path where the model file is located.
            filename (str): The name of the model file to load.
            **kwargs (Any): Additional keyword arguments required to reconstruct
                            the agent, such as `observation_space`, `action_space`,
                            or specific configurations (e.g., `network_config` for DQNAgent).
                            These are crucial because the saved file might only contain
                            weights or internal state, not the full agent structure.

        Returns:
            AgentType: An instance of the agent, initialized with the loaded model state.
        """
        raise NotImplementedError(
            "Subclasses must implement the load_model classmethod."
        )

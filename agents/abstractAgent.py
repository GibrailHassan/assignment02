"""
This module defines the AbstractAgent class, an abstract base class (ABC)
that outlines the common interface for all reinforcement learning agents.
It enforces the implementation of essential methods like getting an action,
updating the agent's knowledge, and saving/loading its model.
"""

import abc


class AbstractAgent(metaclass=abc.ABCMeta):
    """
    Abstract class defining the structure and behavior of an RL agent.
    All concrete agent implementations should inherit from this class.
    """

    def __init__(self, state_shape, action_shape):
        """
        Initializes the agent with the shapes of the state and action spaces.

        Args:
            state_shape (tuple): The shape of the observation/state space.
            action_shape (tuple or int): The shape or size of the action space.
        """
        self.state_shape = state_shape
        self.action_shape = action_shape

    @abc.abstractmethod
    def get_action(self, state):
        """
        Selects an action based on the current state of the environment.

        Args:
            state: The current state observation from the environment.

        Returns:
            The action selected by the agent.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, state, action, reward, next_state, done, next_action=None):
        """
        Updates the agent's internal parameters (e.g., Q-table, neural network weights)
        based on the observed transition (state, action, reward, next_state) and
        whether the episode has terminated. For on-policy methods like SARSA,
        the next_action is also required.

        Args:
            state: The state from which the action was taken.
            action: The action taken by the agent.
            reward (float): The reward received after taking the action.
            next_state: The state transitioned to after the action.
            done (bool): A flag indicating whether the episode has terminated.
            next_action (optional): The action taken in the next_state (for on-policy methods).
                                   Defaults to None for off-policy methods.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, path, filename):
        """
        Serializes and saves the agent's learned model (e.g., Q-table, network weights)
        to a file.

        Args:
            path (str): The directory path where the model should be saved.
            filename (str): The name of the file for the saved model.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load_model(cls, path, filename):
        """
        Loads a previously saved model from a file. This is a class method
        as it should be able to create an instance of the agent from the saved data.

        Args:
            path (str): The directory path from where the model should be loaded.
            filename (str): The name of the file containing the saved model.

        Returns:
            An instance of the agent with the loaded model parameters.
        """
        raise NotImplementedError

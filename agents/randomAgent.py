"""
This module defines a RandomAgent class, which inherits from AbstractAgent.
The RandomAgent selects actions randomly from the available action space,
without any learning or state consideration. It also keeps a history of
its actions and corresponding states which can be saved to a file.
"""

import numpy as np
import os
from agents.abstractAgent import AbstractAgent
from env.env_discrete import ACTION_DIRECTION


# Definition of the RandomAgent class
class RandomAgent(AbstractAgent):
    """
    Random agent that keeps track of its actions and states in a history string.
    """

    def __init__(self, state_shape, action_shape):
        """
        Initializes the RandomAgent with the state and action space shapes.

        Args:
            state_shape (tuple): The shape of the state space.
            action_shape (tuple): The shape of the action space.
        """
        super().__init__(state_shape, action_shape)

        self.history = (
            ""  # Initialize an empty string to store the history of actions and states.
        )

    def get_action(self, state):
        """
        Selects a random action from the action space.

        Args:
            state: The current state of the environment.

        Returns:
            int: A randomly selected action.
        """
        self.history += f"From State {state}"
        action = np.random.randint(
            len(ACTION_DIRECTION)
        )  # Randomly choose an action index.
        self.history += f"Move {ACTION_DIRECTION[action]}\n"  # Append the action and its direction to the history.
        return action

    def update(self, state, action, reward, next_state, done):
        pass  # The RandomAgent does not learn, so the update method is intentionally left empty.

    def save_model(self, path, filename="history.txt"):
        """Saves the history of actions and states to a text file."""
        with open(
            os.path.join(path, filename), "w"
        ) as f:  # Open the file in write mode.
            f.write(self.history)  # Write the history string to the file.

    def load_model(self, path, filename):
        raise NotImplementedError("Error: Random Agent is stateless!")

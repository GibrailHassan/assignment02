"""
This module defines a BasicAgent class that inherits from AbstractAgent.
The BasicAgent selects actions based on a straightforward calculation of
the angle between the agent and the target, without using any
reinforcement learning algorithms.
"""

from agents import AbstractAgent


# Definition of the BasicAgent class
class BasicAgent(AbstractAgent):
    """
    A basic agent that selects actions based on a simple calculation
    of the angle to the target, without reinforcement learning.
    """

    def __init__(self, state_shape, action_shape):
        """
        Initializes the BasicAgent with the state and action space shapes.

        Args:
            state_shape (tuple): The shape of the state space.
            action_shape (tuple): The shape of the action space.
        """
        super().__init__(state_shape, action_shape)

    def get_action(self, env):
        """
        Selects an action by evaluating all possible actions and
        choosing the one that minimizes the distance to the beacon.

        Args:
            env: The environment in which the agent operates, used to simulate next states.

        Returns:
            int: The action that minimizes the distance to the beacon.
        """
        min_dist = float("inf")  # Initialize minimum distance with a very large value.
        best_action = -1  # Initialize best action as invalid.

        # Iterate over each possible action in the action space.
        for (
            action
        ) in (
            env.action_shape
        ):  # Assuming env.action_shape is a collection of valid actions.
            next_dist = env.roll_to_next_state(
                action
            )  # Simulate the next state for the given action.
            dist = (
                (next_dist[0] ** 2) + (next_dist[1] ** 2)
            ) ** 0.5  # Calculate Euclidean distance.

            # Check if the current action results in a smaller distance to the beacon.
            if dist < min_dist:
                min_dist = dist  # Update the minimum distance.
                best_action = action  # Update the best action.

        return best_action  # Return the action that leads to the minimum distance.

    def update(self, state, action, reward, next_state, done):
        """This agent does not learn from experience, so the update method is empty."""
        pass  # The BasicAgent does not learn, so the update method is intentionally left empty.

    def save_model(self, path, filename):
        """This agent does not have a model to save, so the method is empty."""
        pass  # The BasicAgent does not maintain a model, so the save_model method is intentionally left empty.

    @classmethod
    def load_model(cls, path, filename):
        """This agent does not have a model to load, so the method raises an exception."""
        raise NotImplementedError(
            "The BasicAgent does not support loading models."
        )  # The BasicAgent does not maintain a model, so loading is not supported.

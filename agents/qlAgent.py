"""
Implements the Q-Learning algorithm by inheriting from the TableBasedAgent.

This file defines the QLearningAgent, which contains the specific update rule
for the Q-Learning (off-policy) temporal difference control algorithm. All
common functionalities like action selection and model management are handled
by the parent TableBasedAgent class.
"""

from typing import Any
import torch
import numpy as np
from agents.tableAgent import TableBasedAgent


class QLearningAgent(TableBasedAgent):
    """
    An agent that learns to make decisions using the Q-Learning algorithm.

    This agent inherits from TableBasedAgent and provides the concrete implementation
    of the off-policy Q-Learning update rule.
    """

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
        Updates the Q-table using the Q-Learning update rule.

        The formula for the update is:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a'(Q(s', a')) - Q(s, a)]

        This is an "off-policy" method because it updates the Q-value based on the
        maximum possible value in the next state (greedy action), regardless of
        which action is actually chosen by the current exploration policy.

        Args:
            state (np.ndarray): The state from which the action was taken.
            action (int): The index of the action that was taken.
            reward (float): The reward received from the environment.
            next_state (np.ndarray): The state transitioned to after the action.
            done (bool): A flag indicating if the episode has terminated.
            **kwargs: Additional arguments (not used in this implementation).
        """
        # Get the current Q-value for the state-action pair
        current_q_value: torch.Tensor = self.q_table[tuple(state)][action]

        # Get the maximum Q-value for the next state (the greedy, off-policy part)
        # If the episode is done, the value of the next state is 0.
        max_next_q_value: torch.Tensor = (
            torch.max(self.q_table[tuple(next_state)])
            if not done
            else torch.tensor(0.0)
        )

        # Calculate the target Q-value (also known as the TD target)
        target_q_value: float = reward + self.discount_factor * max_next_q_value

        # Calculate the new Q-value using the learning rate
        new_q_value: torch.Tensor = current_q_value + self.learning_rate * (
            target_q_value - current_q_value
        )

        # Update the Q-table with the new value
        self.q_table[tuple(state)][action] = new_q_value

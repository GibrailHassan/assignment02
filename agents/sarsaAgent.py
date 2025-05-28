# agents/sarsaAgent.py

"""
Implements the SARSA algorithm by inheriting from the TableBasedAgent.

This file defines the SARSAAgent, which contains the specific update rule for
the SARSA (State-Action-Reward-State-Action) on-policy temporal difference
control algorithm. All common functionalities are handled by the parent
TableBasedAgent class.
"""

from typing import Any
import torch
import numpy as np
from agents.tableAgent import TableBasedAgent


class SARSAAgent(TableBasedAgent):
    """
    An agent that learns to make decisions using the SARSA algorithm.

    This agent inherits from TableBasedAgent and provides the concrete implementation
    of the on-policy SARSA update rule.
    """

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action: int,
        **kwargs: Any,
    ) -> None:
        """
        Updates the Q-table using the SARSA update rule.

        The formula for the update is:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]

        This is an "on-policy" method because it updates the Q-value based on the
        action (a') that was actually chosen by the policy in the next state (s'),
        rather than the greedy/optimal action. This means it learns the value of
        its own exploration-inclusive policy.

        Args:
            state (np.ndarray): The state from which the action was taken.
            action (int): The index of the action that was taken.
            reward (float): The reward received from the environment.
            next_state (np.ndarray): The state transitioned to after the action.
            done (bool): A flag indicating if the episode has terminated.
            next_action (int): The action taken in the next state, which is required
                               for the on-policy SARSA update.
            **kwargs: Additional arguments (not used in this implementation).
        """
        # Get the current Q-value for the state-action pair.
        current_q_value: torch.Tensor = self.q_table[tuple(state)][action]

        # Get the Q-value for the next state-action pair (the on-policy part).
        # If the episode is done, the value of the next state is 0.
        next_q_value: torch.Tensor = (
            self.q_table[tuple(next_state)][next_action]
            if not done
            else torch.tensor(0.0)
        )

        # Calculate the target Q-value (TD target).
        target_q_value: float = reward + self.discount_factor * next_q_value

        # Calculate the new Q-value using the learning rate.
        new_q_value: torch.Tensor = current_q_value + self.learning_rate * (
            target_q_value - current_q_value
        )

        # Update the Q-table with the new value.
        self.q_table[tuple(state)][action] = new_q_value

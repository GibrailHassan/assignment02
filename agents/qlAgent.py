"""
Implements the Q-Learning algorithm by inheriting from the TableBasedAgent.

This file defines the QLearningAgent, which contains the specific update rule
for the Q-Learning (off-policy) temporal difference (TD) control algorithm.
Q-Learning aims to find the optimal action-selection policy by learning an
action-value function, Q(s, a), which estimates the expected return starting
from state s, taking action a, and thereafter following the optimal policy.

All common functionalities for table-based agents, such as Q-table initialization,
epsilon-greedy action selection, handling of observation/action spaces, and
model saving/loading, are managed by the parent `TableBasedAgent` class.
This class focuses solely on the unique Q-Learning update logic.
"""

from typing import Any  # For type hinting of arbitrary state/action types
import torch
import numpy as np  # For state representation if needed, though primarily uses PyTorch tensors

# Import the parent class from which QLearningAgent inherits
from agents.tableAgent import TableBasedAgent


class QLearningAgent(TableBasedAgent):
    """
    An agent that learns to make decisions using the Q-Learning algorithm.

    This agent maintains a Q-table, where `Q[state][action]` stores the estimated
    expected future reward for taking `action` in `state`. It uses an off-policy
    approach, meaning it learns the optimal Q-values regardless of the policy
    being followed during exploration (e.g., epsilon-greedy).

    Inherits from `TableBasedAgent` for common Q-table operations, action selection
    mechanisms (epsilon-greedy), and model persistence.
    """

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        **kwargs: Any,  # Accepts additional keyword arguments (e.g., next_action for SARSA, ignored here)
    ) -> None:
        """
        Updates the Q-table entry for the given state-action pair using the Q-Learning rule.

        The Q-Learning update rule is:
        Q(s, a) <- Q(s, a) + alpha * [reward + gamma * max_a'(Q(s', a')) - Q(s, a)]
        where:
        - Q(s, a) is the current Q-value for the state-action pair.
        - alpha (self.learning_rate) is the learning rate.
        - reward is the immediate reward received.
        - gamma (self.discount_factor) is the discount factor for future rewards.
        - max_a'(Q(s', a')) is the maximum Q-value for the next state (s') over all
          possible next actions (a'). This is the "off-policy" part, as it considers
          the optimal action from the next state, not necessarily the action taken
          by the current exploration policy.
        - If `done` is True (terminal state), then max_a'(Q(s', a')) is considered 0.

        Args:
            state (np.ndarray): The state from which the action was taken.
                                The `TableBasedAgent` handles converting this to
                                appropriate Q-table indices.
            action (int): The index of the action that was taken.
            reward (float): The reward received from the environment after taking the action.
            next_state (np.ndarray): The state transitioned to after taking the action.
            done (bool): A flag indicating if the episode has terminated after this transition.
            **kwargs (Any): Additional keyword arguments that might be passed by the
                            runner (e.g., `next_action` for SARSA compatibility,
                            but it's not used by Q-Learning).
        """
        # Retrieve the Q-table indices for the current state
        current_state_indices = self._get_q_table_indices(state)
        # Get the current Q-value for the state-action pair from the Q-table
        current_q_value: torch.Tensor = self.q_table[current_state_indices][action]

        # Determine the value of the next state (max Q-value in the next state)
        if done:
            # If the episode is done, there are no future rewards from next_state, so its value is 0.
            max_next_q_value: torch.Tensor = torch.tensor(
                0.0, device=self.q_table.device
            )
        else:
            # If not done, find the maximum Q-value among all actions in the next_state.
            # This is the core of the off-policy update: learn from the best possible next action.
            next_state_indices = self._get_q_table_indices(next_state)
            max_next_q_value = torch.max(self.q_table[next_state_indices])

        # Calculate the TD Target: reward + gamma * max_a'(Q(s', a'))
        # This is the value estimate towards which we want to move our current Q(s,a).
        target_q_value: float = reward + self.discount_factor * max_next_q_value.item()

        # Calculate the new Q-value using the Q-Learning update rule (TD error scaled by learning rate)
        # TD Error = target_q_value - current_q_value
        new_q_value: torch.Tensor = current_q_value + self.learning_rate * (
            target_q_value - current_q_value
        )

        # Update the Q-table with the newly calculated Q-value
        self.q_table[current_state_indices][action] = new_q_value

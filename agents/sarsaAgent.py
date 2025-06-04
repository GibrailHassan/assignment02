"""
Implements the SARSA algorithm by inheriting from the TableBasedAgent.

This file defines the SARSAAgent, which contains the specific update rule for
the SARSA (State-Action-Reward-State-Action) on-policy temporal difference (TD)
control algorithm. SARSA learns an action-value function, Q(s, a), by updating
its estimates based on the actual action taken in the next state, rather than
the greedy optimal action (as in Q-Learning). This makes it an "on-policy"
learning method, as it evaluates and improves the policy it is currently following.

All common functionalities for table-based agents, such as Q-table management,
epsilon-greedy action selection, and model persistence, are handled by the
parent `TableBasedAgent` class. This class focuses solely on the SARSA update logic.
"""

from typing import Any  # For type hinting of arbitrary state/action types
import torch
import numpy as np  # For state representation if needed

# Import the parent class from which SARSAAgent inherits
from agents.tableAgent import TableBasedAgent


class SARSAAgent(TableBasedAgent):
    """
    An agent that learns to make decisions using the SARSA algorithm.

    SARSA is an on-policy temporal difference learning algorithm. It estimates
    the action-value function Q(s,a) for the current policy (including its
    exploration strategy). The name SARSA reflects the quintuple of experience
    it uses for updates: (S)tate, (A)ction, (R)eward, (S)tate_next, (A)ction_next.

    This agent inherits from `TableBasedAgent` for Q-table operations,
    epsilon-greedy action selection, and model management.
    """

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action: int,  # Crucial for SARSA: the action taken in the next state
        **kwargs: Any,  # Accepts additional keyword arguments (ignored here)
    ) -> None:
        """
        Updates the Q-table entry for the given state-action pair using the SARSA rule.

        The SARSA update rule is:
        Q(s, a) <- Q(s, a) + alpha * [reward + gamma * Q(s', a') - Q(s, a)]
        where:
        - Q(s, a) is the current Q-value for the state-action pair.
        - alpha (self.learning_rate) is the learning rate.
        - reward is the immediate reward received.
        - gamma (self.discount_factor) is the discount factor for future rewards.
        - Q(s', a') is the Q-value for the next state (s') and the action (a')
          that was actually taken in that next state according to the current policy.
          This is the key difference from Q-Learning and makes SARSA on-policy.
        - If `done` is True (terminal state), then Q(s', a') is considered 0.

        Args:
            state (np.ndarray): The state from which the action was taken.
                                The `TableBasedAgent` handles converting this to
                                appropriate Q-table indices.
            action (int): The index of the action that was taken in the current state.
            reward (float): The reward received from the environment after taking the action.
            next_state (np.ndarray): The state transitioned to after taking the action.
            done (bool): A flag indicating if the episode has terminated after this transition.
            next_action (int): The action that was actually taken in `next_state`,
                               determined by the agent's policy (e.g., epsilon-greedy)
                               in that next state. This is essential for the SARSA update.
            **kwargs (Any): Additional keyword arguments that might be passed by the
                            runner (ignored in this specific update implementation).
        """
        # Retrieve the Q-table indices for the current state
        current_state_indices = self._get_q_table_indices(state)
        # Get the current Q-value for the (state, action) pair from the Q-table
        current_q_value: torch.Tensor = self.q_table[current_state_indices][action]

        # Determine the Q-value for the (next_state, next_action) pair
        if done:
            # If the episode is done, the value contribution from the next state is 0,
            # as there are no further actions or rewards.
            next_q_value_for_next_action: torch.Tensor = torch.tensor(
                0.0, device=self.q_table.device
            )
        else:
            # If not done, get the Q-value for the specific action (next_action)
            # that was taken in the next_state.
            next_state_indices = self._get_q_table_indices(next_state)
            next_q_value_for_next_action = self.q_table[next_state_indices][next_action]

        # Calculate the TD Target: reward + gamma * Q(s', a')
        # This is the value estimate towards which we want to move our current Q(s,a).
        # It uses the Q-value of the action actually taken in the next state.
        target_q_value: float = (
            reward + self.discount_factor * next_q_value_for_next_action.item()
        )

        # Calculate the new Q-value using the SARSA update rule (TD error scaled by learning rate)
        # TD Error = target_q_value - current_q_value
        new_q_value: torch.Tensor = current_q_value + self.learning_rate * (
            target_q_value - current_q_value
        )

        # Update the Q-table with the newly calculated Q-value
        self.q_table[current_state_indices][action] = new_q_value

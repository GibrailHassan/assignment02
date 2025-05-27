# agents/dqnAgent.py

"""
Implementation of a Deep Q-Network (DQN) agent.

This agent uses a neural network to approximate the Q-value function and
employs Experience Replay and a target network to stabilize learning, as
described in the assignment.
"""

import numpy as np
import os
import torch
import random
from collections import deque, namedtuple
from torch import nn
import torch.optim as optim
from typing import List, Tuple, Any, Callable # Added Callable

from agents.abstractAgent import AbstractAgent
from agents.NN_model import Model, CNNModel, init_weights

# A named tuple to represent a single transition in the environment.
# This makes the code in the replay memory more readable.
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayMemory:
    """
    A fixed-size replay memory to store transitions for experience replay.

    Experience replay allows the agent to learn from a buffer of past
    experiences, which helps to de-correlate samples and stabilize learning.
    The assignment suggests using a list or deque. [cite: 26]
    """

    def __init__(self, capacity: int):
        """
        Initializes the ReplayMemory.

        Args:
            capacity (int): The maximum number of transitions to store.
        """
        self.memory = deque([], maxlen=capacity)

    def add(self, *args: Any) -> None:
        """Adds a new transition to the memory."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Samples a random batch of transitions from the memory. [cite: 29]

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            A list of sampled transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Returns the current number of transitions in the memory."""
        return len(self.memory)


class DQNAgent(AbstractAgent):
    """
    A Deep Q-Network (DQN) agent that uses a neural network to approximate the Q-value function.
    """

    def __init__(
        self,
        state_shape: tuple,
        action_shape: tuple,
        use_cnn: bool = False,
        **kwargs: Any,
    ):
        """
        Initializes the DQNAgent.

        Args:
            state_shape (tuple): The shape of the state space.
            action_shape (tuple): The shape of the action space.
            use_cnn (bool): If True, a CNN will be used. Otherwise, an MLP is used.
            **kwargs: Hyperparameters for the agent.
        """
        super().__init__(state_shape, action_shape)

        self.use_cnn = use_cnn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Set hyperparameters from kwargs or use defaults
        self.batch_size = kwargs.get("batch_size", 32)
        self.learning_rate = kwargs.get("learning_rate", 0.00025)
        self.discount_factor = kwargs.get("discount_factor", 0.99)
        self.epsilon = kwargs.get("epsilon", 1.0)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.9999)
        self.epsilon_min = kwargs.get("epsilon_min", 0.1)
        self.target_update_freq = kwargs.get("target_update_freq", 1000)
        self._step_counter = 0

        # Create online and target networks as required by the assignment [cite: 32]
        self.online_nn = self._create_network().to(self.device)
        self.target_nn = self._create_network().to(self.device)
        self.target_nn.load_state_dict(self.online_nn.state_dict())
        self.target_nn.eval()  # Target network is only for inference

        self.optimizer = optim.Adam(self.online_nn.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(kwargs.get("memory_capacity", 100000))
        self.loss_fn = nn.MSELoss()

    def _create_network(self) -> nn.Module:
        """Factory method to create a neural network based on configuration."""
        if self.use_cnn:
            # For CNN, state_shape is (channels, height, width)
            num_channels = self.state_shape[0]
            network = CNNModel(
                input_channels=num_channels, num_actions=len(self.action_shape)
            )
        else:
            network = Model(output=len(self.action_shape))

        network.apply(init_weights)
        return network

    def get_action(self, state: np.ndarray) -> int:
        """Selects an action using an epsilon-greedy policy. [cite: 36]"""
        # Exploration: choose a random action
        if np.random.rand() < self.epsilon:
            return random.choice(self.action_shape)

        # Exploitation: use the online network to find the best action [cite: 39]
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.online_nn.forward(state_tensor)
        return torch.argmax(q_values).item()

    def _get_sample_from_memory(self) -> Tuple[torch.Tensor, ...]:
        """Samples a batch from replay memory and prepares it for training."""
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(
            self.device
        )
        actions = (
            torch.tensor(batch.action, dtype=torch.int64).view(-1, 1).to(self.device)
        )
        rewards = (
            torch.tensor(batch.reward, dtype=torch.float32).view(-1, 1).to(self.device)
        )
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(
            self.device
        )
        dones = (
            torch.tensor(batch.done, dtype=torch.float32).view(-1, 1).to(self.device)
        )

        return states, actions, rewards, next_states, dones

    def _calculate_loss(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Calculates the MSE loss for a batch of transitions."""
        states, actions, rewards, next_states, dones = batch

        # Get Q-values for current states from the online network: Q(s_t, a_t)
        current_q_values = self.online_nn.forward(states).gather(1, actions)

        # Calculate the target Q-values using the target network: y_i = R_i + gamma * max_a' Q_target(s_{t+1}, a')
        with torch.no_grad():
            next_q_values, _ = self.target_nn.forward(next_states).max(
                dim=1, keepdim=True
            )
            target_q_values = rewards + self.discount_factor * next_q_values * (
                1 - dones
            )

        # Compute the Mean Squared Error loss
        loss = self.loss_fn(current_q_values, target_q_values)
        return loss

    def update(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        *args: Any,
        **kwargs: Any,
    ) -> float | None:
        """Stores a transition in memory and performs a learning step."""
        self.memory.add(state, action, reward, next_state, done)
        self._step_counter += 1

        # Only start learning after there are enough samples in memory [cite: 42]
        if len(self.memory) < self.batch_size:
            return None

        # Sample a batch and calculate loss [cite: 44, 45]
        batch = self._get_sample_from_memory()
        loss = self._calculate_loss(batch)

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_nn.parameters(), 1.0)
        self.optimizer.step()

        # Update epsilon with a linear decay
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        # Periodically update the target network [cite: 33, 47]
        if self._step_counter % self.target_update_freq == 0:
            self.reset_target_nn()

        return self.epsilon

    def reset_target_nn(self) -> None:
        """Copies the weights from the online network to the target network."""
        self.target_nn.load_state_dict(self.online_nn.state_dict())

    def save_model(self, path: str, filename: str = "dqn_agent.pt") -> None:
        """Saves the online network's state dictionary."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.online_nn.state_dict(), os.path.join(path, filename))
        print(f"DQN model saved to {os.path.join(path, filename)}")

    @classmethod
    def load_model(
        cls, path: str, filename: str = "dqn_agent.pt", **kwargs: Any
    ) -> "DQNAgent":
        """Loads a model's state dictionary and creates a new agent instance."""
        instance = cls(**kwargs)
        model_state_dict = torch.load(os.path.join(path, filename))
        instance.online_nn.load_state_dict(model_state_dict)
        instance.target_nn.load_state_dict(model_state_dict)
        print(f"DQN model loaded from {os.path.join(path, filename)}")
        return instance

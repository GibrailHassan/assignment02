# agents/dqnAgent.py

"""
Implementation of a Deep Q-Network (DQN) agent, with optional Double DQN (DDQN) support.

This agent uses a neural network to approximate the Q-value function and
employs Experience Replay and a target network to stabilize learning.
The DDQN modification helps to reduce overestimation of Q-values.
"""

import os
import random
from collections import deque, namedtuple
from typing import List, Tuple, Any, Dict

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.abstractAgent import AbstractAgent

# Assuming NN_model.py is in the same directory or a correctly configured path
from .NN_model import Model, CNNModel

# A named tuple to represent a single transition in the environment.
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayMemory:
    """
    A fixed-size replay memory to store transitions for experience replay.
    """

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def add(self, *args: Any) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        if len(self.memory) < batch_size:
            return []  # Not enough samples
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQNAgent(AbstractAgent):
    """
    A Deep Q-Network (DQN) agent that uses a neural network to approximate
    the Q-value function. Can be configured to use Double DQN (DDQN) logic.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_cnn: bool = False,
        enable_ddqn: bool = False,  # New parameter to enable DDQN
        **kwargs: Any,
    ):
        """
        Initializes the DQNAgent.

        Args:
            observation_space (gym.Space): The environment's observation space.
            action_space (gym.Space): The environment's action space.
            use_cnn (bool): If True, a CNN will be used. Otherwise, an MLP is used.
            enable_ddqn (bool): If True, DDQN target calculation will be used.
            **kwargs: Hyperparameters for the agent.
        """
        super().__init__(observation_space, action_space)

        self.use_cnn = use_cnn
        self.enable_ddqn = enable_ddqn  # Store the DDQN flag
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"DQNAgent using device: {self.device}{' (DDQN enabled)' if self.enable_ddqn else ''}"
        )

        # Hyperparameters
        self.batch_size = kwargs.get("batch_size", 32)
        self.learning_rate = kwargs.get("learning_rate", 0.00025)
        self.discount_factor = kwargs.get("discount_factor", 0.99)
        self.initial_epsilon = kwargs.get("epsilon", 1.0)
        self.epsilon = self.initial_epsilon

        # Linear epsilon decay per step is assumed.
        # epsilon_decay_rate is the amount to subtract per step.
        self.epsilon_decay_rate = kwargs.get("epsilon_decay", 0.00001)
        self.epsilon_min = kwargs.get("epsilon_min", 0.1)
        self.target_update_freq = kwargs.get("target_update_freq", 1000)
        self.is_training = kwargs.get("is_training", True)

        self._step_counter = 0

        # Create online and target networks
        self.online_nn = self._create_network().to(self.device)
        self.target_nn = self._create_network().to(self.device)
        self.target_nn.load_state_dict(self.online_nn.state_dict())
        self.target_nn.eval()

        self.optimizer = optim.AdamW(self.online_nn.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(kwargs.get("memory_capacity", 100000))
        self.loss_fn = nn.MSELoss()

    def _create_network(self) -> nn.Module:
        """Factory method to create a neural network based on configuration."""
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise ValueError("DQNAgent currently only supports Discrete action spaces.")
        num_actions = self.action_space.n

        if not isinstance(self.observation_space, gym.spaces.Box):
            raise ValueError("DQNAgent requires a gym.spaces.Box observation space.")

        obs_shape = self.observation_space.shape

        if self.use_cnn:
            if len(obs_shape) == 3:
                if (
                    obs_shape[0] <= 4
                    and obs_shape[0] < obs_shape[1]
                    and obs_shape[0] < obs_shape[2]
                ):
                    input_channels = obs_shape[0]
                    # print(f"CNN using (C,H,W) format, input_channels: {input_channels} from shape {obs_shape}")
                elif (
                    obs_shape[2] <= 4
                    and obs_shape[2] < obs_shape[0]
                    and obs_shape[2] < obs_shape[1]
                ):
                    input_channels = obs_shape[2]
                    # print(f"CNN using (H,W,C) format, input_channels: {input_channels} from shape {obs_shape}. Will permute in agent.")
                else:
                    input_channels = obs_shape[0]
                    # print(f"CNN input_channels heuristically set to {input_channels} from shape {obs_shape} (assuming C,H,W). Verify if correct.")
            else:
                raise ValueError(
                    f"CNN expects a 3D observation space (e.g. C,H,W or H,W,C), got {obs_shape}"
                )

            network = CNNModel(input_channels=input_channels, num_actions=num_actions)
        else:  # MLP
            input_features = int(np.prod(obs_shape))
            network = Model(input_features=input_features, output=num_actions)
        return network

    def get_action(self, state: np.ndarray, is_training: bool = True) -> int:
        """Selects an action using an epsilon-greedy policy."""
        # Use self.is_training if is_training argument is not overriding
        effective_is_training = (
            is_training if is_training is not None else self.is_training
        )
        current_epsilon = self.epsilon if effective_is_training else 0.0

        if np.random.rand() < current_epsilon:
            return self.action_space.sample()

        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)

        obs_shape = self.observation_space.shape
        if self.use_cnn:
            if state_tensor.ndim == 3 and len(obs_shape) == 3:
                if (
                    obs_shape[2] < obs_shape[0] and obs_shape[2] < obs_shape[1]
                ):  # Input is (H, W, C)
                    state_tensor = state_tensor.permute(2, 0, 1)  # -> (C, H, W)
            elif (
                state_tensor.ndim == 2 and len(obs_shape) == 3 and obs_shape[0] == 1
            ):  # (H,W) for single channel (C,H,W)
                state_tensor = state_tensor.unsqueeze(0)
            # After potential permute/unsqueeze, check final channel dim for C,H,W
            if (
                state_tensor.ndim != 3
                or state_tensor.shape[0] != self.online_nn.conv1.in_channels
            ):
                expected_c = self.online_nn.conv1.in_channels
                # This error might be too strict if state is already (C,H,W) and just needs batching
                # print(f"Warning: CNN get_action input state dim {state_tensor.ndim} or channels {state_tensor.shape[0]} mismatch. Expected (C,H,W) with C={expected_c} for network input. State shape: {state.shape}")

        if state_tensor.ndim == (
            3 if self.use_cnn else len(obs_shape)
        ):  # If it's a single processed state
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            q_values = self.online_nn(state_tensor)
        return torch.argmax(q_values).item()

    def _get_sample_from_memory(
        self,
    ) -> (
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ):
        """Samples a batch from replay memory and prepares it for training."""
        transitions = self.memory.sample(self.batch_size)
        if not transitions:
            return None

        batch = Transition(*zip(*transitions))

        states_np = np.array(batch.state)
        next_states_np = np.array(batch.next_state)

        obs_shape = self.observation_space.shape

        if self.use_cnn:
            # Ensure states are (B, C, H, W)
            if states_np.ndim == 4 and len(obs_shape) == 3:
                if (
                    obs_shape[2] < obs_shape[0] and obs_shape[2] < obs_shape[1]
                ):  # Input is (B, H, W, C)
                    states = (
                        torch.as_tensor(states_np, dtype=torch.float32)
                        .permute(0, 3, 1, 2)
                        .to(self.device)
                    )
                    next_states = (
                        torch.as_tensor(next_states_np, dtype=torch.float32)
                        .permute(0, 3, 1, 2)
                        .to(self.device)
                    )
                else:  # Assume already (B, C, H, W)
                    states = torch.as_tensor(states_np, dtype=torch.float32).to(
                        self.device
                    )
                    next_states = torch.as_tensor(
                        next_states_np, dtype=torch.float32
                    ).to(self.device)
            elif (
                states_np.ndim == 3 and len(obs_shape) == 3 and obs_shape[0] == 1
            ):  # Batch of (H,W) for single channel (C,H,W)
                states = (
                    torch.as_tensor(states_np, dtype=torch.float32)
                    .unsqueeze(1)
                    .to(self.device)
                )
                next_states = (
                    torch.as_tensor(next_states_np, dtype=torch.float32)
                    .unsqueeze(1)
                    .to(self.device)
                )
            else:  # Should not happen if env state is consistent
                # This might indicate an issue with how states are collected or preprocessed
                # For example, if some states are (H,W) and others are (C,H,W) in the same batch
                print(
                    f"Warning: Inconsistent state shapes in batch for CNN. states_np.shape: {states_np.shape}, obs_shape: {obs_shape}"
                )
                # Fallback or raise error
                states = torch.as_tensor(states_np, dtype=torch.float32).to(
                    self.device
                )  # Hope for the best or error out
                next_states = torch.as_tensor(next_states_np, dtype=torch.float32).to(
                    self.device
                )

        else:  # MLP
            states = torch.as_tensor(states_np, dtype=torch.float32).to(self.device)
            next_states = torch.as_tensor(next_states_np, dtype=torch.float32).to(
                self.device
            )
            if states.ndim > 2:
                states = states.view(states.size(0), -1)
            if next_states.ndim > 2:
                next_states = next_states.view(next_states.size(0), -1)

        actions = (
            torch.as_tensor(batch.action, dtype=torch.int64).view(-1, 1).to(self.device)
        )
        rewards = (
            torch.as_tensor(batch.reward, dtype=torch.float32)
            .view(-1, 1)
            .to(self.device)
        )
        dones = (
            torch.as_tensor(batch.done, dtype=torch.bool).view(-1, 1).to(self.device)
        )

        return states, actions, rewards, next_states, dones

    def _calculate_loss(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Calculates the MSE loss for a batch of transitions, supporting DDQN."""
        states, actions, rewards, next_states, dones = batch

        # Get Q-values for current states from the online network: Q_online(s_t, a_t)
        current_q_values = self.online_nn(states).gather(1, actions)

        with torch.no_grad():
            if self.enable_ddqn:
                # --- DDQN Target Calculation ---
                # 1. Select best actions for next_states using the online_nn
                online_next_q_values = self.online_nn(next_states)
                best_next_actions = online_next_q_values.argmax(dim=1, keepdim=True)

                # 2. Evaluate these best_next_actions using the target_nn
                # Q_target(s_{t+1}, argmax_a' Q_online(s_{t+1}, a'))
                target_next_q_values_for_actions = self.target_nn(next_states).gather(
                    1, best_next_actions
                )
            else:
                # --- Standard DQN Target Calculation ---
                # max_a' Q_target(s_{t+1}, a')
                next_q_values_target_net = self.target_nn(next_states)
                target_next_q_values_for_actions, _ = next_q_values_target_net.max(
                    dim=1, keepdim=True
                )

            # Compute the target Q-values: R_i + gamma * Q_target_selected
            # Ensure dones is a boolean tensor for correct masking (~dones will be 0 for terminal, 1 otherwise)
            target_q_values = (
                rewards
                + self.discount_factor * target_next_q_values_for_actions * (~dones)
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
        **kwargs: Any,
    ) -> None:
        """Stores a transition in memory and performs a learning step."""
        self.memory.add(state, action, reward, next_state, done)

        # Only start learning after there are enough samples in memory
        # or after a specific number of initial steps (if learn_after_steps is implemented)
        if len(self.memory) < self.batch_size:
            return

        batch_data = self._get_sample_from_memory()
        if batch_data is None:  # Not enough samples yet from memory.sample()
            return

        loss = self._calculate_loss(batch_data)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_nn.parameters(), 1.0
        )  # Gradient clipping
        self.optimizer.step()

        self._step_counter += 1

        # Linear Epsilon decay per step, only if training
        if self.is_training:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay_rate)

        # Periodically update the target network
        if self._step_counter % self.target_update_freq == 0:
            self.reset_target_nn()

        return None  # Update method doesn't need to return epsilon for the runner

    def reset_target_nn(self) -> None:
        """Copies the weights from the online network to the target network."""
        self.target_nn.load_state_dict(self.online_nn.state_dict())

    # --- Methods to satisfy AbstractAgent ---
    def on_episode_start(self) -> None:
        """
        Hook for start-of-episode logic.
        """
        pass  # Epsilon decay is per step in the current setup

    def on_episode_end(self) -> None:
        """
        Hook for end-of-episode logic.
        """
        # If using per-episode epsilon decay, it would happen here.
        # For example (if self.epsilon_decay_rate was multiplicative per episode):
        # if self.is_training:
        #    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)
        pass

    def get_update_info(self) -> Dict[str, Any]:
        """
        Returns agent-specific metrics for logging by the runner.
        """
        return {"epsilon": self.epsilon, "steps": self._step_counter}

    def save_model(self, path: str, filename: str = "dqn_online_nn.pt") -> None:
        """Saves the online network's state dictionary."""
        os.makedirs(path, exist_ok=True)
        model_file_path = os.path.join(path, filename)
        torch.save(self.online_nn.state_dict(), model_file_path)
        print(f"DQN model (online_nn) saved to {model_file_path}")

    @classmethod
    def load_model(
        cls, path: str, filename: str = "dqn_online_nn.pt", **kwargs: Any
    ) -> "DQNAgent":
        """Loads a model's state dictionary and creates a new agent instance."""
        if "observation_space" not in kwargs or "action_space" not in kwargs:
            raise ValueError(
                "observation_space and action_space must be provided in kwargs for DQNAgent.load_model"
            )

        # When loading, agent is typically not training unless specified.
        kwargs.setdefault("is_training", False)
        instance = cls(**kwargs)
        model_file_path = os.path.join(path, filename)

        model_state_dict = torch.load(model_file_path, map_location=instance.device)

        instance.online_nn.load_state_dict(model_state_dict)
        instance.target_nn.load_state_dict(model_state_dict)
        instance.online_nn.to(instance.device)
        instance.target_nn.to(instance.device)

        # If loaded for evaluation (is_training is False), set to eval mode and min epsilon.
        if not instance.is_training:
            instance.online_nn.eval()
            instance.target_nn.eval()
            instance.epsilon = instance.epsilon_min

        print(f"DQN model loaded from {model_file_path} to device {instance.device}")
        return instance

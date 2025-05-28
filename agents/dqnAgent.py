# agents/dqnAgent.py

"""
Implementation of a Deep Q-Network (DQN) agent.

This agent uses a neural network to approximate the Q-value function and
employs Experience Replay and a target network to stabilize learning.
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
from .NN_model import Model, CNNModel  # init_weights is called within Model/CNNModel

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def add(self, *args: Any) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        if len(self.memory) < batch_size:
            return []
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQNAgent(AbstractAgent):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_cnn: bool = False,
        **kwargs: Any,
    ):
        super().__init__(observation_space, action_space)

        self.use_cnn = use_cnn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQNAgent using device: {self.device}")

        self.batch_size = kwargs.get("batch_size", 32)
        self.learning_rate = kwargs.get("learning_rate", 0.00025)
        self.discount_factor = kwargs.get("discount_factor", 0.99)
        self.initial_epsilon = kwargs.get("epsilon", 1.0)
        self.epsilon = self.initial_epsilon

        self.epsilon_decay_rate = kwargs.get("epsilon_decay", 0.00001)
        self.epsilon_min = kwargs.get("epsilon_min", 0.1)
        self.target_update_freq = kwargs.get("target_update_freq", 1000)
        self.is_training = kwargs.get(
            "is_training", True
        )  # Added for clarity in epsilon decay

        self._step_counter = 0

        self.online_nn = self._create_network().to(self.device)
        self.target_nn = self._create_network().to(self.device)
        self.target_nn.load_state_dict(self.online_nn.state_dict())
        self.target_nn.eval()

        self.optimizer = optim.AdamW(self.online_nn.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(kwargs.get("memory_capacity", 100000))
        self.loss_fn = nn.MSELoss()

    def _create_network(self) -> nn.Module:
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise ValueError("DQNAgent currently only supports Discrete action spaces.")
        num_actions = self.action_space.n

        if not isinstance(self.observation_space, gym.spaces.Box):
            raise ValueError("DQNAgent requires a gym.spaces.Box observation space.")

        obs_shape = self.observation_space.shape

        if self.use_cnn:
            if len(obs_shape) == 3 and (
                obs_shape[0] > 3 or obs_shape[1] > 3
            ):  # Likely (H, W, C)
                input_channels = obs_shape[2]
                if (
                    input_channels > obs_shape[0] or input_channels > obs_shape[1]
                ):  # Heuristic for C,H,W
                    print(
                        f"Warning: CNN input_channels ({input_channels}) derived from obs_shape[-1] {obs_shape} seems large. Ensure observation is (H,W,C)."
                    )
            elif len(obs_shape) == 3 and obs_shape[0] <= 3:  # Likely (C, H, W)
                input_channels = obs_shape[0]
            else:
                raise ValueError(
                    f"CNN expects a 3D observation space (H, W, C) or (C,H,W), got {obs_shape}"
                )

            network = CNNModel(input_channels=input_channels, num_actions=num_actions)
        else:  # MLP
            input_features = int(np.prod(obs_shape))
            network = Model(input_features=input_features, output=num_actions)
        return network

    def get_action(self, state: np.ndarray, is_training: bool = True) -> int:
        current_epsilon = self.epsilon if is_training else 0.0
        if np.random.rand() < current_epsilon:
            return self.action_space.sample()

        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)

        # Ensure state_tensor is (C, H, W) if use_cnn, then add batch dim
        if self.use_cnn:
            if (
                state_tensor.ndim == 3
                and self.observation_space.shape[2] == state_tensor.shape[2]
            ):  # (H, W, C)
                state_tensor = state_tensor.permute(2, 0, 1)  # -> (C, H, W)
            elif (
                state_tensor.ndim != 3
                or state_tensor.shape[0] != self.online_nn.conv1.in_channels
            ):
                raise ValueError(
                    f"CNN input state dim {state_tensor.ndim} or channels {state_tensor.shape[0]} mismatch. Expected (C,H,W) with C={self.online_nn.conv1.in_channels}"
                )

        if state_tensor.ndim == len(self.observation_space.shape) or (
            self.use_cnn
            and state_tensor.ndim == 2
            and self.observation_space.shape[2] == 1
        ):  # (H,W) for single channel CNN
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
        transitions = self.memory.sample(self.batch_size)
        if not transitions:
            return None

        batch = Transition(*zip(*transitions))

        states_np = np.array(batch.state)
        next_states_np = np.array(batch.next_state)

        # Permute if CNN and states are (B, H, W, C)
        if (
            self.use_cnn
            and states_np.ndim == 4
            and states_np.shape[3] < states_np.shape[1]
        ):  # (B, H, W, C)
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
        else:  # MLP or already (B, C, H, W)
            states = torch.as_tensor(states_np, dtype=torch.float32).to(self.device)
            next_states = torch.as_tensor(next_states_np, dtype=torch.float32).to(
                self.device
            )

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
        states, actions, rewards, next_states, dones = batch
        current_q_values = self.online_nn(states).gather(1, actions)
        with torch.no_grad():
            next_q_values_target_net = self.target_nn(next_states)
            best_next_actions_q_values, _ = next_q_values_target_net.max(
                dim=1, keepdim=True
            )
            target_q_values = (
                rewards + self.discount_factor * best_next_actions_q_values * (~dones)
            )
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
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) < self.batch_size:
            return

        batch_data = self._get_sample_from_memory()
        if batch_data is None:
            return

        loss = self._calculate_loss(batch_data)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_nn.parameters(), 1.0)
        self.optimizer.step()

        self._step_counter += 1

        if self.is_training:  # Use the instance variable
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay_rate)

        if self._step_counter % self.target_update_freq == 0:
            self.reset_target_nn()

        return None

    def reset_target_nn(self) -> None:
        self.target_nn.load_state_dict(self.online_nn.state_dict())

    def on_episode_start(self) -> None:
        pass

    def on_episode_end(self) -> None:
        pass

    def get_update_info(self) -> Dict[str, Any]:
        return {"epsilon": self.epsilon, "steps": self._step_counter}

    def save_model(self, path: str, filename: str = "dqn_online_nn.pt") -> None:
        os.makedirs(path, exist_ok=True)
        model_file_path = os.path.join(path, filename)
        torch.save(self.online_nn.state_dict(), model_file_path)
        print(f"DQN model (online_nn) saved to {model_file_path}")

    @classmethod
    def load_model(
        cls, path: str, filename: str = "dqn_online_nn.pt", **kwargs: Any
    ) -> "DQNAgent":
        if "observation_space" not in kwargs or "action_space" not in kwargs:
            raise ValueError(
                "observation_space and action_space must be provided in kwargs for DQNAgent.load_model"
            )

        # Pass is_training=False if loading for evaluation, or get from kwargs
        kwargs.setdefault("is_training", False)
        instance = cls(**kwargs)
        model_file_path = os.path.join(path, filename)

        model_state_dict = torch.load(model_file_path, map_location=instance.device)

        instance.online_nn.load_state_dict(model_state_dict)
        instance.target_nn.load_state_dict(model_state_dict)
        instance.online_nn.to(instance.device)
        instance.target_nn.to(instance.device)

        # Set to eval mode if not training
        if not instance.is_training:
            instance.online_nn.eval()
            instance.target_nn.eval()
            instance.epsilon = instance.epsilon_min  # Use min epsilon for eval

        print(f"DQN model loaded from {model_file_path} to device {instance.device}")
        return instance

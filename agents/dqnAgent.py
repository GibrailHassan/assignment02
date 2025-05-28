# agents/dqnAgent.py

"""
Implementation of a Deep Q-Network (DQN) agent, with optional Double DQN (DDQN) support.
This agent now receives its neural networks via dependency injection.
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

# MLflow import for model logging
import mlflow

from agents.abstractAgent import AbstractAgent
from networks.base import BaseNetwork  # Import BaseNetwork from the new networks module

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
            return []
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQNAgent(AbstractAgent):
    """
    A Deep Q-Network (DQN) agent that uses externally provided neural networks.
    Can be configured for standard DQN or Double DQN (DDQN) logic.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        online_network: BaseNetwork,  # Injected online network
        target_network: BaseNetwork,  # Injected target network
        enable_ddqn: bool = False,
        learning_rate: float = 0.00025,
        discount_factor: float = 0.99,
        initial_epsilon: float = 1.0,
        epsilon_decay_rate: float = 0.00001,
        epsilon_min: float = 0.1,
        target_update_freq: int = 1000,
        memory_capacity: int = 100000,
        batch_size: int = 32,
        is_training: bool = True,
        **kwargs: Any,
    ):
        """
        Initializes the DQNAgent.

        Args:
            observation_space (gym.Space): The environment's observation space.
            action_space (gym.Space): The environment's action space.
            online_network (BaseNetwork): The primary Q-network.
            target_network (BaseNetwork): The target Q-network.
            enable_ddqn (bool): If True, DDQN target calculation will be used.
            learning_rate (float): Learning rate for the optimizer.
            discount_factor (float): Gamma for future rewards.
            initial_epsilon (float): Starting value for epsilon.
            epsilon_decay_rate (float): Amount to linearly decay epsilon by per step.
            epsilon_min (float): Minimum epsilon value.
            target_update_freq (int): Frequency (in steps) to update the target network.
            memory_capacity (int): Maximum size of the replay memory.
            batch_size (int): Batch size for sampling from replay memory.
            is_training (bool): Indicates if the agent is in training mode.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(
            observation_space, action_space
        )  # Pass observation_space and action_space

        self.enable_ddqn = enable_ddqn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"DQNAgent using device: {self.device}{' (DDQN enabled)' if self.enable_ddqn else ''}"
        )

        # Ensure the injected networks are nn.Module instances
        if not isinstance(online_network, nn.Module) or not isinstance(
            target_network, nn.Module
        ):
            raise TypeError(
                "online_network and target_network must be instances of torch.nn.Module (or BaseNetwork)."
            )

        self.online_nn = online_network.to(self.device)
        self.target_nn = target_network.to(self.device)

        self.target_nn.load_state_dict(self.online_nn.state_dict())
        self.target_nn.eval()

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self.is_training = is_training

        self._step_counter = 0

        self.optimizer = optim.AdamW(self.online_nn.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(memory_capacity)
        self.loss_fn = nn.MSELoss()

    def get_action(self, state: np.ndarray, is_training: bool = True) -> int:
        effective_is_training = (
            is_training if is_training is not None else self.is_training
        )
        current_epsilon = self.epsilon if effective_is_training else 0.0

        if np.random.rand() < current_epsilon:
            # Ensure action_space is Discrete for .sample()
            if not isinstance(self.action_space, gym.spaces.Discrete):
                raise TypeError(
                    "DQNAgent get_action requires a Discrete action space to sample from."
                )
            return self.action_space.sample()

        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)

        # The BaseNetwork's forward method should handle permutations if necessary
        # Add batch dimension if it's a single processed state
        if state_tensor.ndim == len(self.observation_space.shape):
            state_tensor = state_tensor.unsqueeze(0)

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

        states_np = np.array(batch.state, dtype=np.float32)  # Ensure float32 for states
        next_states_np = np.array(batch.next_state, dtype=np.float32)

        states = torch.as_tensor(states_np).to(self.device)
        next_states = torch.as_tensor(next_states_np).to(self.device)

        actions = (
            torch.as_tensor(batch.action, dtype=torch.int64).view(-1, 1).to(self.device)
        )
        rewards = (
            torch.as_tensor(batch.reward, dtype=torch.float32)
            .view(-1, 1)
            .to(self.device)
        )
        # Ensure dones are boolean, then convert to float for multiplication if needed, or use ~ for bool negation
        dones = (
            torch.as_tensor(batch.done, dtype=torch.bool).view(-1, 1).to(self.device)
        )

        return states, actions, rewards, next_states, dones

    def _calculate_loss(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        states, actions, rewards, next_states, dones = batch

        current_q_values = self.online_nn(states).gather(1, actions)

        with torch.no_grad():
            if self.enable_ddqn:
                online_next_q_values = self.online_nn(next_states)
                best_next_actions = online_next_q_values.argmax(dim=1, keepdim=True)
                target_next_q_values_for_actions = self.target_nn(next_states).gather(
                    1, best_next_actions
                )
            else:
                next_q_values_target_net = self.target_nn(next_states)
                target_next_q_values_for_actions, _ = next_q_values_target_net.max(
                    dim=1, keepdim=True
                )

            target_q_values = (
                rewards
                + self.discount_factor * target_next_q_values_for_actions * (~dones)
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

        if self.is_training:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay_rate)

        if self._step_counter > 0 and self._step_counter % self.target_update_freq == 0:
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

    def save_model(self, path: str, filename: str = "dqn_online_nn.pt") -> str:
        os.makedirs(path, exist_ok=True)
        model_file_path = os.path.join(path, filename)
        torch.save(self.online_nn.state_dict(), model_file_path)
        print(f"DQN model (online_nn) saved locally to {model_file_path}")

        if mlflow.active_run():
            try:
                mlflow.pytorch.log_model(
                    pytorch_model=self.online_nn,
                    artifact_path="pytorch_models",
                )
                print(
                    f"DQN model also logged as MLflow PyTorch artifact to 'pytorch_models'."
                )
            except Exception as e:
                print(f"Warning: Failed to log DQN model to MLflow: {e}")
        return model_file_path

    @classmethod
    def load_model(
        cls, path: str, filename: str = "dqn_online_nn.pt", **kwargs: Any
    ) -> "DQNAgent":
        if "observation_space" not in kwargs or "action_space" not in kwargs:
            raise ValueError(
                "observation_space and action_space must be provided in kwargs for DQNAgent.load_model"
            )

        # Networks must be provided or reconstructed.
        # The parameters for network reconstruction (network_config) should be in kwargs.
        online_network = kwargs.pop("online_network", None)
        target_network = kwargs.pop("target_network", None)

        if not online_network or not target_network:
            network_config = kwargs.pop("network_config", None)
            if not network_config or not isinstance(network_config, dict):
                raise ValueError(
                    "If online_network and target_network are not provided directly, "
                    "'network_config' (dict with 'name' and 'params') must be in kwargs for DQNAgent.load_model."
                )

            from networks.factory import create_network  # Local import

            obs_space = kwargs["observation_space"]
            act_space = kwargs["action_space"]

            online_network = create_network(
                name=network_config.get("name"),
                observation_space=obs_space,
                action_space=act_space,
                params=network_config.get("params", {}),
            )
            target_network = create_network(
                name=network_config.get("name"),
                observation_space=obs_space,
                action_space=act_space,
                params=network_config.get("params", {}),
            )

        # Ensure is_training is properly set for the new instance, defaulting to False for loaded models
        kwargs.setdefault("is_training", False)

        # Create the instance, passing the (now ensured) networks
        instance = cls(
            online_network=online_network, target_network=target_network, **kwargs
        )

        model_file_path = os.path.join(path, filename)

        try:
            model_state_dict = torch.load(model_file_path, map_location=instance.device)
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_file_path}")
            raise
        except Exception as e:
            print(f"Error loading model state_dict from {model_file_path}: {e}")
            raise

        instance.online_nn.load_state_dict(model_state_dict)
        instance.target_nn.load_state_dict(model_state_dict)

        if not instance.is_training:
            instance.online_nn.eval()
            instance.target_nn.eval()
            instance.epsilon = instance.epsilon_min

        print(f"DQN model loaded from {model_file_path} to device {instance.device}")
        return instance

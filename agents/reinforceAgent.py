# agents/reinforceAgent.py
"""
Implements the REINFORCE (Monte Carlo Policy Gradient) algorithm.

REINFORCE is a policy-based reinforcement learning algorithm that learns a
parameterized policy directly. It updates the policy parameters by performing
gradient ascent on an objective function related to the expected return.
This implementation collects experiences over a full episode and then performs
an update using the collected trajectory.
"""

import os
from typing import List, Tuple, Any, Dict, Optional, Type

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # For softmax

import mlflow

from agents.abstractAgent import AbstractAgent
from networks.base import BaseNetwork  # For type hinting the policy network

# Define a type variable for the REINFORCEAgent class itself
REINFORCEAgentType = Type["REINFORCEAgent"]


class REINFORCEAgent(AbstractAgent):
    """
    REINFORCE (Monte Carlo Policy Gradient) Agent.

    This agent learns a policy that maps states to action probabilities.
    It updates its policy network based on the discounted returns obtained
    at the end of each episode.

    Attributes:
        policy_network (BaseNetwork): The neural network representing the policy.
        optimizer (optim.Optimizer): Optimizer for training the policy network.
        learning_rate (float): Learning rate for the optimizer.
        discount_factor (float): Gamma, for discounting future rewards.
        is_training (bool): Flag indicating if the agent is in training mode.
        device (torch.device): CPU or CUDA.

        # Episode buffer
        _rewards_buffer (List[float]): Stores rewards for the current episode.
        _log_probs_buffer (List[torch.Tensor]): Stores log probabilities of actions taken.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        policy_network: BaseNetwork,  # The policy network (actor)
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        is_training: bool = True,
        **kwargs: Any,  # To absorb other params from config
    ):
        """
        Initializes the REINFORCEAgent.

        Args:
            observation_space (gym.Space): Environment's observation space.
            action_space (gym.Space): Environment's action space (must be Discrete).
            policy_network (BaseNetwork): The neural network to be used as the policy.
                                          Its output layer should produce logits for actions.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            discount_factor (float, optional): Discount factor (gamma). Defaults to 0.99.
            is_training (bool, optional): Agent's mode. Defaults to True.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(observation_space, action_space)

        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                "REINFORCEAgent currently supports Discrete action spaces only."
            )

        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy_network: BaseNetwork = policy_network.to(self.device)
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.is_training: bool = is_training

        self.optimizer: optim.Optimizer = optim.AdamW(
            self.policy_network.parameters(), lr=self.learning_rate
        )

        # Buffers for collecting episode data
        self._rewards_buffer: List[float] = []
        self._log_probs_buffer: List[torch.Tensor] = (
            []
        )  # Store log_prob of chosen actions

        print(f"REINFORCEAgent initialized on device: {self.device}")
        print(f"Policy Network: {self.policy_network.__class__.__name__}")

    def get_action(self, state: np.ndarray, is_training: Optional[bool] = None) -> int:
        """
        Selects an action based on the policy network's output probabilities.

        Args:
            state (np.ndarray): The current state observation.
            is_training (Optional[bool], optional): Overrides self.is_training if provided.
                                                 During evaluation (not is_training), it typically
                                                 takes the action with the highest probability.
                                                 During training, it samples from the distribution.
                                                 Defaults to None.

        Returns:
            int: The selected action index.
        """
        effective_is_training: bool = (
            is_training if is_training is not None else self.is_training
        )

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.ndim == len(
            self.observation_space.shape
        ):  # e.g. (C,H,W) or (Features,)
            state_tensor = state_tensor.unsqueeze(
                0
            )  # Add batch dim: (1, C,H,W) or (1, Features)

        self.policy_network.eval()  # Set to eval mode for deterministic output if needed for sampling
        with torch.no_grad():
            action_logits: torch.Tensor = self.policy_network(
                state_tensor
            )  # (1, NumActions)
            action_probs: torch.Tensor = F.softmax(
                action_logits, dim=-1
            )  # Convert logits to probabilities
        self.policy_network.train()  # Set back to train mode

        if effective_is_training:
            # Sample an action based on the probability distribution
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample()
            # Store log probability of the chosen action for the update step
            # This needs to be done here if we want to store it for later use in update
            # self._log_probs_buffer.append(action_distribution.log_prob(action)) # Moved to update
        else:
            # Evaluation: choose the action with the highest probability (greedy)
            action = torch.argmax(action_probs, dim=-1)

        return action.item()

    def _store_transition_outcome(
        self, action: int, state_for_log_prob: np.ndarray, reward: float
    ):
        """Helper to store log_prob and reward, called from update()"""
        if not self.is_training:
            return

        state_tensor = torch.as_tensor(
            state_for_log_prob, dtype=torch.float32, device=self.device
        )
        if state_tensor.ndim == len(self.observation_space.shape):
            state_tensor = state_tensor.unsqueeze(0)

        # We need to re-calculate log_prob for the *chosen* action for the state it was chosen in.
        # The action was chosen in `get_action` based on `state`.
        # The `update` method receives `state`, `action`, `reward`, `next_state`, `done`.
        # The `state` in `update` is the state where `action` was taken.

        self.policy_network.train()  # Ensure network is in train mode for consistent behavior if it has dropout etc.
        action_logits: torch.Tensor = self.policy_network(
            state_tensor
        )  # (1, NumActions)
        action_probs: torch.Tensor = F.softmax(action_logits, dim=-1)
        action_distribution = torch.distributions.Categorical(action_probs)

        # Action tensor for log_prob
        action_tensor = torch.tensor([action], device=self.device)
        log_prob = action_distribution.log_prob(action_tensor)

        self._log_probs_buffer.append(log_prob)
        self._rewards_buffer.append(reward)

    def update(
        self,
        state: Any,  # State where action was taken
        action: Any,  # Action taken
        reward: float,
        next_state: Any,
        done: bool,
        **kwargs: Any,
    ) -> None:
        """
        Stores the reward and log probability of the action taken.
        The actual learning update happens at the end of the episode in `on_episode_end`.

        Args:
            state (Any): The state from which the action was taken.
            action (Any): The action taken in that state.
            reward (float): The reward received.
            next_state (Any): The state transitioned to.
            done (bool): Whether the episode terminated.
            **kwargs (Any): Additional arguments.
        """
        if not self.is_training:
            return

        # Store reward and log_prob of the action taken in 'state'
        self._store_transition_outcome(action, state, reward)

        # If the episode is done, trigger the learning update
        if done:
            self._learn_from_episode()

    def _learn_from_episode(self):
        """
        Performs the REINFORCE update at the end of an episode.
        Uses the collected rewards and log probabilities.
        """
        if not self.is_training or not self._rewards_buffer:  # Ensure there's data
            self._clear_buffers()  # Clear buffers even if not learning from this episode
            return

        # --- Calculate Discounted Returns (G_t) ---
        # G_t = R_t + gamma * R_{t+1} + gamma^2 * R_{t+2} + ...
        discounted_returns = []
        cumulative_return = 0.0
        for r in reversed(self._rewards_buffer):  # Iterate backwards from T-1 to 0
            cumulative_return = r + self.discount_factor * cumulative_return
            discounted_returns.insert(0, cumulative_return)  # Insert at the beginning

        # Convert to tensor and normalize returns (optional but often helpful)
        returns_tensor = torch.tensor(
            discounted_returns, dtype=torch.float32, device=self.device
        )
        # Normalization: (returns - mean(returns)) / (std(returns) + epsilon)
        # This can stabilize training by keeping returns in a consistent range.
        eps_norm = np.finfo(
            np.float32
        ).eps.item()  # Small epsilon for numerical stability
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (
            returns_tensor.std() + eps_norm
        )

        # --- Calculate Policy Loss ---
        # Loss = - sum_t (G_t * log_prob(A_t | S_t))
        # We want to perform gradient *ascent* on the objective J = sum_t (G_t * log_prob),
        # so for gradient *descent*, we minimize -J.
        policy_loss_terms = []
        for log_prob, G_t in zip(self._log_probs_buffer, returns_tensor):
            policy_loss_terms.append(-log_prob * G_t)  # Loss for this timestep

        # Sum up losses for all timesteps in the episode
        self.optimizer.zero_grad()
        policy_loss: torch.Tensor = torch.cat(
            policy_loss_terms
        ).sum()  # Concatenate and sum scalar tensors

        # --- Perform Optimization ---
        policy_loss.backward()  # Compute gradients
        self.optimizer.step()  # Update policy network parameters

        self._clear_buffers()  # Clear buffers for the next episode

    def _clear_buffers(self):
        """Clears the episode buffers."""
        self._rewards_buffer.clear()
        self._log_probs_buffer.clear()

    def on_episode_start(self) -> None:
        """Clears episode buffers at the start of a new episode."""
        self._clear_buffers()

    def on_episode_end(self) -> None:
        """
        Performs the learning update if it hasn't been triggered by `done` in `update`.
        This ensures an update happens even if `done` wasn't the last signal.
        However, current design calls _learn_from_episode if done in update().
        This hook could be used for additional end-of-episode processing if needed.
        """
        # The current implementation calls _learn_from_episode when done is True in the update method.
        # If an episode could end without 'done' being true in the *last* call to update()
        # (e.g., if Runner terminates episode externally), then this might be a place for it.
        # For now, assuming `done` in `update()` is reliable.
        # if self.is_training and self._rewards_buffer: # If any data was collected and not processed
        #     self._learn_from_episode()
        self._clear_buffers()  # Ensure buffers are clear for safety.

    def get_update_info(self) -> Dict[str, Any]:
        """Returns agent-specific metrics (e.g., learning rate)."""
        # REINFORCE typically doesn't have an epsilon.
        # Could log average return or policy loss if calculated and stored.
        return {"learning_rate": self.learning_rate}

    def save_model(
        self, path: str, filename: str = "reinforce_policy_net.pt"
    ) -> Optional[str]:
        """Saves the policy network's state_dict."""
        if not path:
            print(
                "REINFORCEAgent: Local model saving path not provided. Skipping local save."
            )
            if mlflow.active_run():
                try:
                    mlflow.pytorch.log_model(
                        self.policy_network, "policy_network_models"
                    )
                    print(
                        "REINFORCEAgent policy network logged to MLflow (local save skipped)."
                    )
                except Exception as e_mlflow:
                    print(
                        f"Warning: Failed to log REINFORCE policy network to MLflow: {e_mlflow}"
                    )
            return None

        os.makedirs(path, exist_ok=True)
        model_file_path = os.path.join(path, filename)
        try:
            torch.save(self.policy_network.state_dict(), model_file_path)
            print(f"REINFORCEAgent policy network saved locally to: {model_file_path}")
            if mlflow.active_run():
                mlflow.pytorch.log_model(self.policy_network, "policy_network_models")
                print("REINFORCEAgent policy network also logged as MLflow artifact.")
            return model_file_path
        except Exception as e_save:
            print(f"Error saving REINFORCEAgent policy network: {e_save}")
            return None

    @classmethod
    def load_model(
        cls: type,
        path: str,
        filename: str = "reinforce_policy_net.pt",
        **kwargs: Any,
    ) -> "REINFORCEAgent":
        """Loads a REINFORCEAgent model."""
        if "observation_space" not in kwargs or "action_space" not in kwargs:
            raise ValueError(
                "load_model requires 'observation_space' and 'action_space' in kwargs."
            )

        policy_network_to_load: Optional[BaseNetwork] = kwargs.pop(
            "policy_network", None
        )

        if not policy_network_to_load:
            policy_network_config = kwargs.pop("policy_network_config", None)
            if not policy_network_config or not isinstance(policy_network_config, dict):
                raise ValueError(
                    "If 'policy_network' is not directly provided, 'policy_network_config' "
                    "(dict with 'name' and 'params') must be in kwargs for load_model."
                )

            from networks.factory import create_network as create_nn_from_factory_local

            obs_space = kwargs["observation_space"]
            act_space = kwargs["action_space"]
            net_name = policy_network_config.get("name")
            if not net_name:
                raise ValueError("'policy_network_config' must contain 'name'.")

            print(
                f"Reconstructing policy network '{net_name}' via factory for REINFORCEAgent loading..."
            )
            policy_network_to_load = create_nn_from_factory_local(
                name=net_name,
                observation_space=obs_space,
                action_space=act_space,  # Ensure output layer matches num_actions
                params=policy_network_config.get("params", {}),
            )
            print("Policy network reconstructed.")

        kwargs.setdefault("is_training", False)  # Default to eval mode when loading

        agent_instance = cls(policy_network=policy_network_to_load, **kwargs)

        model_file_path = os.path.join(path, filename)
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found at {model_file_path}")

        try:
            state_dict = torch.load(model_file_path, map_location=agent_instance.device)
            agent_instance.policy_network.load_state_dict(state_dict)
            print(f"REINFORCEAgent policy network loaded from {model_file_path}")
        except Exception as e_load:
            print(f"Error loading REINFORCEAgent policy network state_dict: {e_load}")
            raise

        if not agent_instance.is_training:
            agent_instance.policy_network.eval()
            print("REINFORCEAgent loaded in evaluation mode.")

        return agent_instance

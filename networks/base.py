"""
Defines an abstract base class for all neural network models.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any  # Removed Tuple as it's not used here directly
import gym  # Added gym for Space type hint


class BaseNetwork(nn.Module, ABC):
    """
    An abstract base class for all network architectures.

    Ensures that any network used by an agent inherits from torch.nn.Module
    and implements a forward pass. It also expects observation and action spaces
    for potential automatic input/output sizing during initialization.
    """

    def __init__(
        self, observation_space: gym.Space, action_space: gym.Space, **kwargs: Any
    ) -> None:
        """
        Initializes the BaseNetwork.

        Args:
            observation_space (gym.Space): The environment's observation space.
            action_space (gym.Space): The environment's action space.
            **kwargs (Any): Additional keyword arguments that specific network
                            architectures might accept.
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def forward(
        self, x: torch.Tensor, *args: Any
    ) -> torch.Tensor:  # Return type often torch.Tensor
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor to the network.
            *args (Any): Additional arguments specific to the network's forward pass,
                         which might be used by more complex architectures (e.g., LSTMs
                         requiring hidden states).

        Returns:
            torch.Tensor: The output tensor from the network (e.g., Q-values, policy logits).
        """
        raise NotImplementedError

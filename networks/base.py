# networks/base.py
"""
Defines an abstract base class (`BaseNetwork`) for all neural network models
used within the reinforcement learning framework.

This module provides a common interface that all specific network architectures
(e.g., MLPNetwork, CNNNetwork) must adhere to. By inheriting from `BaseNetwork`,
these architectures are guaranteed to be `torch.nn.Module` instances and to
implement a `forward` pass. The base class also standardizes the initialization
with observation and action spaces, which can be crucial for dynamically
configuring network input and output dimensions.
"""

from abc import ABC, abstractmethod  # Import Abstract Base Class support
import torch
import torch.nn as nn  # PyTorch's neural network module
from typing import Any  # For type hinting arbitrary additional arguments
import gym  # For type hinting of observation and action spaces (gym.Space)


class BaseNetwork(nn.Module, ABC):  # Inherits from PyTorch's nn.Module and ABC
    """
    An abstract base class for all neural network architectures in the project.

    This class ensures that any network used by an agent (e.g., for Q-value
    approximation, policy representation, or value function estimation)
    inherits from `torch.nn.Module` (making it a standard PyTorch model) and
    implements a `forward` method defining how input is processed to produce output.

    It also standardizes the constructor to accept `observation_space` and
    `action_space` from the environment. This allows network architectures to
    dynamically configure their input and output layer sizes based on the
    specific environment they are being used with.

    Attributes:
        observation_space (gym.Space): The environment's observation space.
        action_space (gym.Space): The environment's action space.
    """

    def __init__(
        self,
        observation_space: gym.Space,  # The observation space of the environment
        action_space: gym.Space,  # The action space of the environment
        **kwargs: Any,  # Allow for additional, architecture-specific keyword arguments
    ) -> None:
        """
        Initializes the BaseNetwork.

        This constructor should be called by all subclasses. It stores the
        observation and action spaces and calls the `torch.nn.Module` constructor.

        Args:
            observation_space (gym.Space): The environment's observation space,
                which defines the structure and dimensions of the states the
                network will receive as input.
            action_space (gym.Space): The environment's action space, which can
                be used to determine the number of output units (e.g., for
                Q-values per action or action probabilities).
            **kwargs (Any): Additional keyword arguments that specific network
                            architectures might accept (e.g., number of hidden
                            layers, filter sizes). These are captured here to allow
                            flexibility in subclass constructors but are not directly
                            used by `BaseNetwork` itself beyond passing to `super()`.
        """
        super().__init__()  # Initialize the parent nn.Module

        # Store the observation and action spaces as instance attributes
        # These can be used by subclasses to determine input/output dimensions
        self.observation_space: gym.Space = observation_space
        self.action_space: gym.Space = action_space

        # kwargs are not explicitly used by BaseNetwork's __init__ beyond this,
        # but passing them to super() could be relevant if nn.Module had such args,
        # or if a more complex hierarchy of base network classes was used.
        # For now, they are primarily for subclasses to pick up from their own __init__.

    @abstractmethod  # Decorator to mark this method as abstract
    def forward(self, x: torch.Tensor, *args: Any) -> torch.Tensor:
        """
        Defines the forward pass of the network. This method must be implemented by all subclasses.

        The forward pass describes how an input tensor `x` (representing a state
        or batch of states) is transformed into an output tensor (e.g., Q-values,
        action probabilities, state value).

        Args:
            x (torch.Tensor): The input tensor to the network. Its shape should be
                              compatible with the network's design, typically derived
                              from `self.observation_space`.
            *args (Any): Additional positional arguments that a specific network's
                         forward pass might require (e.g., hidden states for an RNN).

        Returns:
            torch.Tensor: The output tensor from the network. The shape and meaning
                          depend on the network's purpose (e.g., (batch_size, num_actions)
                          for Q-values in a discrete action space).

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

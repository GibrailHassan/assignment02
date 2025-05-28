# networks/base.py
"""
Defines an abstract base class for all neural network models.
"""
from abc import ABC, abstractmethod
import torch
from torch import nn


class BaseNetwork(nn.Module, ABC):
    """
    An abstract base class for all network architectures.

    Ensures that any network used by an agent inherits from torch.nn.Module
    and implements a forward pass.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor to the network.

        Returns:
            torch.Tensor: The output tensor from the network.
        """
        raise NotImplementedError

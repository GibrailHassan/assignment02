# networks/architectures.py
"""
Defines specific neural network architectures used by deep RL agents.
"""
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.base import BaseNetwork


# (The init_weights function remains the same)
def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class MLP(BaseNetwork):
    """
    A generic Multi-Layer Perceptron (MLP) for processing vector-based states.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: int,
        hidden_layers: List[int] = [64, 64],
    ):
        super().__init__()
        # The input size is the product of the input_shape dimensions
        input_size = int(torch.prod(torch.tensor(input_shape)).item())

        layers = []
        # Dynamically create hidden layers based on the hidden_layers list
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size  # For the next layer

        # Add the final output layer
        layers.append(nn.Linear(input_size, output_shape))
        self.network = nn.Sequential(*layers)
        self.network.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CNN(BaseNetwork):
    """
    A Convolutional Neural Network (CNN) for processing 32x32 image-based features.
    """

    def __init__(self, input_shape: Tuple[int, ...], output_shape: int):
        super().__init__()
        # input_shape is expected to be (channels, height, width)
        input_channels, height, width = input_shape
        if height != 32 or width != 32:
            raise ValueError("This CNN architecture is designed for 32x32 inputs.")

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 16x16
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 8x8
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 4x4
        )
        linear_input_size = 64 * 4 * 4
        self.classifier = nn.Sequential(
            nn.Linear(linear_input_size, 512), nn.ReLU(), nn.Linear(512, output_shape)
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)

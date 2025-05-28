# agents/NN_model.py

"""
Defines the neural network architectures used by deep reinforcement learning agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# from typing import Tuple # Not strictly needed if only used in comments


def init_weights(m: nn.Module) -> None:
    """
    Initializes the weights of a neural network layer using Xavier uniform initialization.
    Initializes biases to zero if they exist.
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:  # Check if bias exists
            nn.init.zeros_(m.bias)


class Model(nn.Module):  # MLP
    """
    A simple Multi-Layer Perceptron (MLP) for processing vector-based states.
    """

    def __init__(self, input_features: int, output: int, h1: int = 64, h2: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MLP expects a batch of 1D vectors (B, Features)
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        elif x.ndim == 1 and self.fc1.in_features > 1:
            x = x.unsqueeze(0)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


class CNNModel(nn.Module):
    """
    A Convolutional Neural Network (CNN) for processing image-based features.
    Expects input tensor in (B, C, H, W) format.
    """

    def __init__(self, input_channels: int, num_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flattened_size = 64 * 4 * 4

        self.fc_stack = nn.Sequential(
            nn.Linear(self.flattened_size, 512), nn.ReLU(), nn.Linear(512, num_actions)
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.shape[1] != self.conv1.in_channels:
            raise ValueError(
                f"CNNModel input channel mismatch. Expected {self.conv1.in_channels}, got {x.shape[1]} from input shape {x.shape}"
            )

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        # Use reshape() instead of view() for robustness
        x = x.reshape(x.size(0), -1)

        if x.shape[1] != self.flattened_size:
            raise ValueError(
                f"CNNModel flattened size mismatch. Expected {self.flattened_size}, got {x.shape[1]}"
            )

        return self.fc_stack(x)

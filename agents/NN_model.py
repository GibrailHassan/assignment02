# agents/NN_model.py

"""
Defines the neural network architectures used by deep reinforcement learning agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def init_weights(m: nn.Module) -> None:
    """
    Initializes the weights of a neural network layer using Xavier uniform initialization.
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class Model(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for processing vector-based states.
    """

    def __init__(self, output: int, h1: int = 64, h2: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(2, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


class CNNModel(nn.Module):
    """
    A Convolutional Neural Network (CNN) for processing 32x32 image-based features.
    """

    def __init__(self, input_channels: int, num_actions: int):
        """
        Initializes the CNN model.
        """
        super().__init__()
        # Convolutional layers to extract spatial features
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 16x16

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 8x8

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 4x4

        # The flattened size will be 64 channels * 4 * 4 = 1024
        linear_input_size = 64 * 4 * 4

        # Fully connected layers to map features to action Q-values
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512), nn.ReLU(), nn.Linear(512, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the CNN.

        This method is designed to handle both a single 3D state tensor
        (C, H, W) and a 4D batch of state tensors (B, C, H, W).
        """
        # ***** THIS IS THE FIX *****
        # Check if the input is a single state (3 dimensions) and add a
        # batch dimension if it is. The network expects a 4D tensor.
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Pass input through the first convolutional and pooling layer
        x = self.pool1(F.relu(self.conv1(x)))

        # Pass through the second convolutional and pooling layer
        x = self.pool2(F.relu(self.conv2(x)))

        # Pass through the third convolutional and pooling layer
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten the output for the fully connected layers
        flattened = x.view(x.size(0), -1)

        # Pass flattened features through the fully connected layers
        return self.fc(flattened)

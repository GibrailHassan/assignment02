"""
This module defines the neural network model architecture used by the DQNAgent
and a utility function for initializing its weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# run1 : dropout=0.1, lr=0.001
# run2 : dropout=0.2, lr=0.0005
# run2 : dropout=0.2, lr=0.0001
# run2 : dropout=0.1, lr=0.005
# run3 : dropout=0.1, lr=0.0005, discount_factor=0.99 (25.05.-17:29)
# run3 : dropout=0.1 (l2), lr=0.0005, discount_factor=0.97 (25.05. )


# create model class for nn module
class Model(nn.Module):
    """
    A simple feedforward neural network model.
    It consists of two fully connected hidden layers with ReLU activations
    and an output layer. Dropout is applied after the second hidden layer.
    """

    # input layer:
    def __init__(self, output, h1=128, h2=64):
        """
        Initializes the neural network model.

        Args:
            output (int): The number of output units (e.g., number of actions).
            h1 (int): The number of units in the first hidden layer. Defaults to 128.
            h2 (int): The number of units in the second hidden layer. Defaults to 64.
        """
        super().__init__()
        # Convolutional layers (commented out, can be used for image-based inputs)
        # self.conv = nn.Sequential(
        # nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(inplace=True),
        # nn.Conv2d(32, 64, 4, stride=2),    nn.ReLU(inplace=True),
        # nn.Conv2d(64, 64, 3, stride=1),    nn.ReLU(inplace=True),
        #
        # nn.Linear(3136, 512), nn.ReLU(inplace=True),
        # nn.Linear(512, output))

        # First fully connected layer (input size 2, assuming [x, y] state)
        self.fc1 = nn.Linear(2, h1)
        # Dropout layer (commented out, can be added after fc1 if needed)
        # self.dropout1 = nn.Dropout(p=0.1)
        # Second fully connected layer
        self.fc2 = nn.Linear(h1, h2)
        # Dropout layer after the second hidden layer
        self.dropout2 = nn.Dropout(p=0.1)
        # Output layer
        self.out = nn.Linear(h2, output)

    # forward path
    def fw(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor of the network.
        """
        x = F.relu(self.fc1(x))
        # x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.out(x)
        return x


def init_weights(m):
    """
    Initializes the weights of linear layers using Xavier (Glorot) uniform initialization
    and biases to zero. This is a common practice for ReLU-activated networks.

    Args:
        m (nn.Module): The module (layer) to initialize.
    """
    if isinstance(m, nn.Linear):
        # Xavier (Glorot) uniform initialization, good for relu
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

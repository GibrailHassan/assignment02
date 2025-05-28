# networks/architectures.py
"""
Defines specific neural network architectures used by deep RL agents,
inheriting from the BaseNetwork.
"""
from typing import List, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

from .base import BaseNetwork  # Use relative import


def init_weights_xavier(m: nn.Module) -> None:
    """
    Initializes the weights of Linear or Conv2D layers using Xavier uniform initialization.
    Initializes biases to zero if they exist.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MLPNetwork(BaseNetwork):
    """
    A generic Multi-Layer Perceptron (MLP) for processing vector-based states.
    It dynamically creates layers based on the observation_space, action_space,
    and a list of hidden layer sizes.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_layers: List[int] = [64, 64],
        **kwargs: Any,
    ):
        """
        Initializes the MLPNetwork.

        Args:
            observation_space (gym.Space): The environment's observation space.
                                           Expected to be a gym.spaces.Box.
            action_space (gym.Space): The environment's action space.
                                      Expected to be gym.spaces.Discrete for this example.
            hidden_layers (List[int]): A list specifying the size of each hidden layer.
            **kwargs (Any): Additional keyword arguments (captured by BaseNetwork).
        """
        super().__init__(observation_space, action_space, **kwargs)

        if not isinstance(observation_space, gym.spaces.Box):
            raise ValueError("MLPNetwork requires a Box observation space.")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                "This MLPNetwork example assumes a Discrete action space for output size."
            )

        # Calculate input features from the product of observation space shape dimensions
        input_features = int(np.prod(observation_space.shape))
        num_actions = action_space.n

        layers = []
        current_dim = input_features
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_actions))

        self.network = nn.Sequential(*layers)
        self.network.apply(init_weights_xavier)  # Apply initialization to all layers

    def forward(self, x: torch.Tensor, *args: Any) -> torch.Tensor:
        """
        Performs a forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor. Expected to be a batch of 1D vectors (B, Features)
                              or a single 1D vector (Features).
            *args (Any): Ignored in this implementation.

        Returns:
            torch.Tensor: Output tensor (e.g., Q-values for each action).
        """
        # MLP expects a batch of 1D vectors (B, Features)
        # If input x is (B, C, H, W) or (B, H, W, C), it needs flattening.
        # If input x is (Features) for a single sample, it needs batching.

        # Get the expected input feature size from the first linear layer
        expected_input_features = self.network[0].in_features

        if x.ndim > 2:  # Has more than Batch and Feature dimensions (e.g. image data)
            x = x.view(x.size(0), -1)  # Flatten extra dimensions
        elif x.ndim == 1:  # Single unbatched sample
            if x.size(0) == expected_input_features:
                x = x.unsqueeze(0)  # Add batch dim: (Features) -> (1, Features)
            else:
                raise ValueError(
                    f"MLPNetwork unbatched input feature size mismatch. Expected {expected_input_features}, got {x.size(0)} from input shape {x.shape}"
                )
        # If x.ndim == 2 (already B, Features), check if features match

        if x.size(-1) != expected_input_features:
            # Attempt to reshape if it's a batched input with wrong trailing dimensions
            if x.ndim == 2 and x.size(0) > 0:  # B, Something
                try:
                    x = x.view(x.size(0), expected_input_features)
                except RuntimeError as e:
                    raise ValueError(
                        f"MLPNetwork input feature size mismatch after attempting reshape. Expected {expected_input_features}, got {x.size(-1)} from input shape {x.shape}. Original error: {e}"
                    )
            else:
                raise ValueError(
                    f"MLPNetwork input feature size mismatch. Expected {expected_input_features}, got {x.size(-1)} from input shape {x.shape}"
                )

        return self.network(x)


class CNNNetwork(BaseNetwork):
    """
    A Convolutional Neural Network (CNN) for processing image-based features.
    It dynamically determines input channel format (CHW vs HWC) from observation_space
    and permutes if necessary to feed data in (B, C, H, W) format to Conv2D layers.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        # Example CNN-specific params; more can be added via kwargs from config
        conv_channels: List[int] = [32, 64, 64],
        kernel_sizes: List[int] = [5, 5, 3],  # For conv layers
        pool_kernel_sizes: List[int] = [2, 2, 2],  # For pooling layers
        strides_conv: List[int] = [1, 1, 1],  # For conv layers
        strides_pool: List[int] = [2, 2, 2],  # For pooling layers
        paddings_conv: List[int] = [2, 2, 1],  # For conv layers
        fc_hidden_size: int = 512,
        **kwargs: Any,
    ):
        super().__init__(observation_space, action_space, **kwargs)

        if (
            not isinstance(observation_space, gym.spaces.Box)
            or len(observation_space.shape) != 3
        ):
            raise ValueError(
                "CNNNetwork requires a 3D Box observation space (C, H, W) or (H, W, C)."
            )
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                "This CNNNetwork example assumes a Discrete action space for output size."
            )

        obs_shape = observation_space.shape
        self.input_format_is_chw: bool

        # Determine input channels and original H, W based on observation_space.shape
        # Heuristic: if first dim is small (<=4), assume (C,H,W)
        if (
            obs_shape[0] <= 4
            and obs_shape[0] < obs_shape[1]
            and obs_shape[0] < obs_shape[2]
        ):
            self.input_format_is_chw = True
            c, h, w = obs_shape[0], obs_shape[1], obs_shape[2]
        # Heuristic: if last dim is small (<=4), assume (H,W,C)
        elif (
            obs_shape[2] <= 4
            and obs_shape[2] < obs_shape[0]
            and obs_shape[2] < obs_shape[1]
        ):
            self.input_format_is_chw = False
            h, w, c = obs_shape[0], obs_shape[1], obs_shape[2]
        else:  # Fallback or ambiguous case
            # Defaulting to CHW if cannot clearly determine. User might need to ensure env provides CHW.
            print(
                f"Warning: CNN input format ambiguous for shape {obs_shape}. Assuming (C,H,W) where C={obs_shape[0]}."
            )
            self.input_format_is_chw = True
            c, h, w = obs_shape[0], obs_shape[1], obs_shape[2]

        num_actions = action_space.n

        conv_layers_list = []
        current_channels = c
        current_h, current_w = h, w

        # Ensure parameter lists have the same length for convolutional stack
        num_conv_stacks = len(conv_channels)
        if not (
            len(kernel_sizes) == num_conv_stacks
            and len(strides_conv) == num_conv_stacks
            and len(paddings_conv) == num_conv_stacks
            and len(pool_kernel_sizes) == num_conv_stacks
            and len(strides_pool) == num_conv_stacks
        ):
            raise ValueError(
                "CNN parameter lists (channels, kernels, strides, paddings, pool_kernels, pool_strides) must have the same number of elements."
            )

        for i in range(num_conv_stacks):
            conv_layers_list.append(
                nn.Conv2d(
                    current_channels,
                    conv_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides_conv[i],
                    padding=paddings_conv[i],
                )
            )
            conv_layers_list.append(nn.ReLU())
            conv_layers_list.append(
                nn.MaxPool2d(kernel_size=pool_kernel_sizes[i], stride=strides_pool[i])
            )
            current_channels = conv_channels[i]

            # Calculate H, W after conv
            current_h = (
                current_h + 2 * paddings_conv[i] - kernel_sizes[i]
            ) // strides_conv[i] + 1
            current_w = (
                current_w + 2 * paddings_conv[i] - kernel_sizes[i]
            ) // strides_conv[i] + 1
            # Calculate H, W after pool
            current_h = (current_h - pool_kernel_sizes[i]) // strides_pool[i] + 1
            current_w = (current_w - pool_kernel_sizes[i]) // strides_pool[i] + 1

            if current_h <= 0 or current_w <= 0:
                raise ValueError(
                    f"CNN dimensions became non-positive after layer {i+1} (H={current_h}, W={current_w}). Check CNN parameters."
                )

        self.features = nn.Sequential(*conv_layers_list)
        self.flattened_size = current_channels * current_h * current_w
        if self.flattened_size <= 0:
            raise ValueError(
                f"Calculated flattened_size for CNN is {self.flattened_size}. Check CNN parameters and input dimensions."
            )

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, fc_hidden_size),
            nn.ReLU(),
            nn.Linear(fc_hidden_size, num_actions),
        )
        self.apply(init_weights_xavier)  # Apply initialization to all layers

    def forward(self, x: torch.Tensor, *args: Any) -> torch.Tensor:
        """
        Performs a forward pass through the CNN.
        Input x is permuted to (B, C, H, W) if it's in (B, H, W, C) or (H, W, C) format.
        """
        # Ensure input is 4D (B, C, H, W) or 3D (C, H, W / H, W, C)
        if x.ndim < 3 or x.ndim > 4:
            raise ValueError(
                f"CNNNetwork expects 3D or 4D input, got {x.ndim}D with shape {x.shape}"
            )

        # Handle permutation and batching
        if not self.input_format_is_chw:  # Original observation_space was (H,W,C)
            if x.ndim == 4:  # Batched (B, H, W, C)
                x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
            elif x.ndim == 3:  # Single (H,W,C)
                x = x.permute(2, 0, 1).unsqueeze(0)  # Convert to (1,C,H,W)
        elif self.input_format_is_chw and x.ndim == 3:  # Single (C,H,W)
            x = x.unsqueeze(0)  # Add batch dim -> (1,C,H,W)

        # Now x should be (B, C, H, W)
        if (
            x.shape[1] != self.features[0].in_channels
        ):  # features[0] is the first Conv2d layer
            raise ValueError(
                f"CNNNetwork input channel mismatch. Expected {self.features[0].in_channels}, got {x.shape[1]} from processed input shape {x.shape}"
            )

        x = self.features(x)
        x = x.reshape(x.size(0), -1)  # Flatten

        if x.shape[1] != self.flattened_size:
            raise ValueError(
                f"CNNNetwork flattened size mismatch. Expected {self.flattened_size}, got {x.shape[1]} after features. Output shape from features: {x.shape}"
            )

        return self.classifier(x)

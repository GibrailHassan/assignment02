# networks/architectures.py
"""
Defines specific neural network architectures used by deep reinforcement learning
agents (e.g., DQNAgent). These architectures inherit from the `BaseNetwork`
abstract class and are implemented using PyTorch (torch.nn).

Currently includes:
- `MLPNetwork`: A Multi-Layer Perceptron suitable for vector-based state inputs.
- `CNNNetwork`: A Convolutional Neural Network suitable for image-like state inputs.

Both networks dynamically adapt their input and output layers based on the
environment's observation and action spaces. They also include Xavier uniform
weight initialization.
"""
from typing import List, Tuple, Any, Optional  # Optional added for clarity
import torch
import torch.nn as nn
import torch.nn.functional as F  # Functional module for activation functions etc.
import numpy as np  # For np.prod to calculate flattened input size
import gym  # For gym.Space type hints and properties (shape, n)

# Use relative import to access BaseNetwork from the same package
from .base import BaseNetwork


def init_weights_xavier(m: nn.Module) -> None:
    """
    Initializes the weights of Linear (fully connected) or Conv2D layers
    using Xavier uniform initialization. Biases, if present, are initialized to zero.

    This initialization scheme helps in keeping the signal variance roughly
    constant across layers, which can aid in training deep networks.

    Args:
        m (nn.Module): The PyTorch module (layer) to initialize.
                       This function is typically applied recursively to a network
                       using `network.apply(init_weights_xavier)`.
    """
    # Check if the module is an instance of nn.Linear or nn.Conv2d
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        # Apply Xavier uniform initialization to the weight tensor
        nn.init.xavier_uniform_(m.weight)
        # If the layer has a bias term, initialize it to zeros
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MLPNetwork(BaseNetwork):
    """
    A generic Multi-Layer Perceptron (MLP) network, also known as a Fully
    Connected Network (FCN).

    This network is designed for processing vector-based (1D) state inputs.
    It dynamically creates a sequence of linear layers with ReLU activations,
    followed by a final linear output layer. The number and size of hidden
    layers, as well as input and output features, are configurable.

    Attributes:
        network (nn.Sequential): The sequential container holding all layers of the MLP.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_layers: List[int] = [64, 64],  # Default hidden layer sizes
        **kwargs: Any,  # Capture any additional arguments from BaseNetwork or config
    ):
        """
        Initializes the MLPNetwork.

        Args:
            observation_space (gym.Space): The environment's observation space.
                Expected to be a `gym.spaces.Box` representing a vector.
                The shape of this space determines the input features of the MLP.
            action_space (gym.Space): The environment's action space.
                Expected to be `gym.spaces.Discrete` for this MLP implementation,
                where `action_space.n` determines the number of output neurons
                (e.g., Q-values for each discrete action).
            hidden_layers (List[int], optional): A list where each element specifies
                the number of neurons in a hidden layer. Defaults to `[64, 64]`,
                creating two hidden layers with 64 neurons each.
            **kwargs (Any): Additional keyword arguments captured by `BaseNetwork`
                            or for future extensions.
        """
        super().__init__(
            observation_space, action_space, **kwargs
        )  # Call BaseNetwork constructor

        # Validate observation and action space types
        if not isinstance(observation_space, gym.spaces.Box):
            raise ValueError(
                "MLPNetwork currently requires a gym.spaces.Box observation space."
            )
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                "This MLPNetwork implementation assumes a gym.spaces.Discrete action space "
                "to determine the output layer size (number of actions)."
            )

        # Calculate the number of input features by flattening the observation space shape.
        # E.g., if observation_space.shape is (4, 2), input_features will be 8.
        # np.prod computes the product of all elements in the shape tuple.
        input_features: int = int(np.prod(observation_space.shape))

        # Number of output neurons, corresponding to the number of discrete actions.
        num_actions: int = action_space.n

        layers: List[nn.Module] = []  # List to hold all network layers
        current_dim: int = input_features  # Input dimension for the first layer

        # Dynamically create hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))  # Linear layer
            layers.append(nn.ReLU())  # ReLU activation function
            current_dim = hidden_dim  # Update current_dim for the next layer

        # Add the final output layer (linear, no activation here as it often represents Q-values or logits)
        layers.append(nn.Linear(current_dim, num_actions))

        # Create the sequential model from the list of layers
        self.network: nn.Sequential = nn.Sequential(*layers)

        # Apply Xavier uniform initialization to all linear layers in the network
        self.network.apply(init_weights_xavier)

        # print(f"MLPNetwork initialized: Input={input_features}, Hidden={hidden_layers}, Output={num_actions}")

    def forward(self, x: torch.Tensor, *args: Any) -> torch.Tensor:
        """
        Performs a forward pass through the MLP.

        The input tensor `x` is expected to be a batch of 1D vectors (Batch_size, Features).
        If the input is an image-like tensor (e.g., Batch_size, Channels, Height, Width)
        or a single unbatched vector, it will be reshaped/flattened appropriately.

        Args:
            x (torch.Tensor): The input tensor representing a state or batch of states.
                              Expected shapes:
                              - (Batch_size, Features)
                              - (Features,) for a single unbatched sample.
                              - (Batch_size, C, H, W) or other multi-dim shapes will be flattened.
            *args (Any): Additional arguments, ignored in this MLP implementation but
                         included for compatibility with `BaseNetwork.forward` signature.

        Returns:
            torch.Tensor: The output tensor from the network (e.g., Q-values for
                          each action if used in DQN). Shape: (Batch_size, Num_actions).

        Raises:
            ValueError: If input tensor dimensions or feature sizes are mismatched
                        after attempting to reshape.
        """
        # The first linear layer (self.network[0]) holds the expected input feature size.
        # This is `input_features` calculated in __init__.
        expected_input_features: int = self.network[0].in_features

        # --- Input Reshaping and Validation ---
        # Handle different input tensor dimensionalities:
        if (
            x.ndim > 2
        ):  # E.g., (Batch, Channels, Height, Width) - common for CNN inputs passed to MLP head
            # Flatten all dimensions except the batch dimension.
            # x.view(x.size(0), -1) reshapes to (Batch_size, C*H*W)
            x = x.reshape(
                x.size(0), -1
            )  # Using reshape as it's generally more flexible
        elif x.ndim == 1:  # Single unbatched sample, e.g., (Features,)
            if x.size(0) == expected_input_features:
                # Add a batch dimension: (Features) -> (1, Features)
                x = x.unsqueeze(0)
            else:
                # Feature size mismatch for unbatched input
                raise ValueError(
                    f"MLPNetwork: Unbatched input feature size mismatch. "
                    f"Expected {expected_input_features} features, but got {x.size(0)} from input shape {x.shape}."
                )
        # If x.ndim == 2, it's assumed to be (Batch_size, Features) already.

        # Final check for feature size consistency after potential reshaping
        if x.size(-1) != expected_input_features:
            # This handles cases where x.ndim was 2 but x.size(1) was still wrong,
            # or if reshaping from >2D resulted in an incorrect feature dimension.
            # An attempt to reshape might be possible if it's a batched input with product of trailing dims matching.
            if x.ndim == 2 and x.size(0) > 0:  # If it's batched (B, SomethingElse)
                try:
                    # Attempt to reshape to (Batch_size, expected_input_features)
                    # This would work if x.size(0) * SomethingElse == x.size(0) * expected_input_features
                    # (i.e. SomethingElse == expected_input_features), or if total elements match and B is maintained.
                    # More robustly, if a batched input has prod(shape[1:]) != expected_input_features:
                    if np.prod(x.shape[1:]) == expected_input_features:
                        x = x.reshape(x.size(0), expected_input_features)
                    else:
                        raise RuntimeError(
                            "Product of features does not match expected input features."
                        )
                except RuntimeError as e_reshape:  # Catch error from reshape attempt
                    raise ValueError(
                        f"MLPNetwork: Input feature size mismatch after attempting final reshape. "
                        f"Expected {expected_input_features} features, got {x.size(-1)} from processed input shape {x.shape}. "
                        f"Original reshape error: {e_reshape}"
                    )
            else:  # Should not be reached if previous logic is correct, but as a safeguard.
                raise ValueError(
                    f"MLPNetwork: Input feature size mismatch. "
                    f"Expected {expected_input_features} features, got {x.size(-1)} from input shape {x.shape}."
                )

        # Pass the processed tensor through the sequential network
        return self.network(x)


class CNNNetwork(BaseNetwork):
    """
    A Convolutional Neural Network (CNN) for processing image-based state inputs.

    This network is designed for 2D spatial data, such as screen features from a game.
    It consists of a configurable number of convolutional layers followed by
    max-pooling, then a flattening step, and finally one or more fully connected
    layers for classification or value function approximation.

    The network attempts to dynamically determine the input channel format
    (Channels-Height-Width vs. Height-Width-Channels) from the observation space shape
    and permutes the input tensor if necessary to the PyTorch standard (B, C, H, W).

    Attributes:
        input_format_is_chw (bool): True if the input observation_space shape
                                    is inferred as (Channels, Height, Width).
        features (nn.Sequential): The convolutional part of the network.
        flattened_size (int): The number of features after the convolutional layers
                              and flattening, used as input to the classifier part.
        classifier (nn.Sequential): The fully connected (MLP) part of the network.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        # --- CNN Architecture Parameters (examples, can be configured via YAML) ---
        conv_channels: List[int] = [
            32,
            64,
            64,
        ],  # Number of output channels for each conv layer
        kernel_sizes: List[int] = [5, 5, 3],  # Kernel size for each conv layer
        pool_kernel_sizes: List[int] = [2, 2, 2],  # Kernel size for each pooling layer
        strides_conv: List[int] = [1, 1, 1],  # Stride for each conv layer
        strides_pool: List[int] = [2, 2, 2],  # Stride for each pooling layer
        paddings_conv: List[int] = [
            2,
            2,
            1,
        ],  # Padding for each conv layer (e.g., to preserve size with stride 1)
        fc_hidden_size: int = 512,  # Size of the hidden layer in the fully connected part
        **kwargs: Any,  # Capture any additional arguments
    ):
        """
        Initializes the CNNNetwork.

        Args:
            observation_space (gym.Space): The environment's observation space.
                Expected to be a `gym.spaces.Box` with a 3D shape, representing
                an image (e.g., (C, H, W) or (H, W, C)).
            action_space (gym.Space): The environment's action space.
                Expected to be `gym.spaces.Discrete` for this CNN implementation,
                determining the output size of the final fully connected layer.
            conv_channels (List[int], optional): List of output channels for each conv layer.
                                                 Defaults to [32, 64, 64].
            kernel_sizes (List[int], optional): List of kernel sizes for each conv layer.
                                                Defaults to [5, 5, 3].
            pool_kernel_sizes (List[int], optional): List of kernel sizes for MaxPool2d layers.
                                                    Defaults to [2, 2, 2].
            strides_conv (List[int], optional): List of strides for conv layers. Defaults to [1, 1, 1].
            strides_pool (List[int], optional): List of strides for MaxPool2d layers. Defaults to [2, 2, 2].
            paddings_conv (List[int], optional): List of paddings for conv layers. Defaults to [2, 2, 1].
            fc_hidden_size (int, optional): Number of neurons in the hidden fully connected
                                            layer after the convolutional features. Defaults to 512.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(observation_space, action_space, **kwargs)

        # Validate observation and action space types
        if (
            not isinstance(observation_space, gym.spaces.Box)
            or len(observation_space.shape) != 3  # Expects 3D input (C,H,W) or (H,W,C)
        ):
            raise ValueError(
                "CNNNetwork requires a 3D gym.spaces.Box observation space (e.g., (C, H, W) or (H, W, C))."
            )
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                "This CNNNetwork implementation assumes a gym.spaces.Discrete action space for output size."
            )

        obs_shape: Tuple[int, ...] = (
            observation_space.shape
        )  # E.g., (3, 64, 64) or (64, 64, 3)

        # --- Determine Input Format (CHW vs HWC) and Initial Dimensions ---
        # Heuristic to infer if the input format is Channels-Height-Width (CHW) or Height-Width-Channels (HWC).
        # PyTorch Conv2D layers expect input as (Batch, Channels, Height, Width).
        c: int  # Number of input channels
        h: int  # Input height
        w: int  # Input width

        if (
            obs_shape[0] <= 4
            and obs_shape[0] < obs_shape[1]
            and obs_shape[0] < obs_shape[2]
        ):
            # If the first dimension is small (<=4) and smaller than the other two, assume CHW format.
            self.input_format_is_chw: bool = True
            c, h, w = obs_shape[0], obs_shape[1], obs_shape[2]
            # print(f"CNNNetwork: Inferred input format CHW: C={c}, H={h}, W={w}")
        elif (
            obs_shape[2] <= 4
            and obs_shape[2] < obs_shape[0]
            and obs_shape[2] < obs_shape[1]
        ):
            # If the last dimension is small (<=4) and smaller than the other two, assume HWC format.
            self.input_format_is_chw: bool = False
            h, w, c = obs_shape[0], obs_shape[1], obs_shape[2]
            # print(f"CNNNetwork: Inferred input format HWC: H={h}, W={w}, C={c}")
        else:
            # Ambiguous case or potentially 1D feature map treated as image.
            # Defaulting to CHW if unsure. User might need to ensure env provides CHW or network handles it.
            print(
                f"Warning: CNNNetwork input format is ambiguous for shape {obs_shape}. "
                f"Assuming CHW format with C={obs_shape[0]}, H={obs_shape[1]}, W={obs_shape[2]}. "
                "Ensure your environment provides data in this format or preprocess accordingly."
            )
            self.input_format_is_chw: bool = True
            c, h, w = obs_shape[0], obs_shape[1], obs_shape[2]

        num_actions: int = (
            action_space.n
        )  # Number of output neurons for discrete actions

        # --- Build Convolutional Layers ---
        conv_layers_list: List[nn.Module] = []
        current_channels: int = c
        current_h, current_w = h, w  # Track spatial dimensions after each layer

        # Ensure architecture parameter lists have consistent lengths
        num_conv_stacks: int = len(conv_channels)
        if not (
            len(kernel_sizes) == num_conv_stacks
            and len(strides_conv) == num_conv_stacks
            and len(paddings_conv) == num_conv_stacks
            and len(pool_kernel_sizes)
            == num_conv_stacks  # Assuming one pool layer per conv stack
            and len(strides_pool)
            == num_conv_stacks  # Assuming one pool layer per conv stack
        ):
            raise ValueError(
                "CNNNetwork: Parameter lists for convolutional stack (channels, kernels, strides, paddings, "
                "pool_kernels, pool_strides) must all have the same number of elements."
            )

        for i in range(num_conv_stacks):
            # Convolutional layer
            conv_layers_list.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=conv_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides_conv[i],
                    padding=paddings_conv[i],
                )
            )
            conv_layers_list.append(nn.ReLU())  # Activation

            # Max pooling layer
            conv_layers_list.append(
                nn.MaxPool2d(kernel_size=pool_kernel_sizes[i], stride=strides_pool[i])
            )
            current_channels = conv_channels[
                i
            ]  # Update current channels for next conv layer

            # --- Calculate output dimensions after this conv-pool stack ---
            # Output size for a dimension after Conv2D: floor((Input + 2*Padding - Kernel) / Stride) + 1
            current_h = (
                current_h + 2 * paddings_conv[i] - kernel_sizes[i]
            ) // strides_conv[i] + 1
            current_w = (
                current_w + 2 * paddings_conv[i] - kernel_sizes[i]
            ) // strides_conv[i] + 1
            # Output size for a dimension after MaxPool2D: floor((Input - Kernel) / Stride) + 1 (assuming default padding 0)
            current_h = (current_h - pool_kernel_sizes[i]) // strides_pool[i] + 1
            current_w = (current_w - pool_kernel_sizes[i]) // strides_pool[i] + 1

            if current_h <= 0 or current_w <= 0:
                raise ValueError(
                    f"CNNNetwork: Spatial dimensions (H={current_h}, W={current_w}) became non-positive "
                    f"after convolutional stack {i+1}. Review CNN parameters (kernels, strides, padding) "
                    f"in relation to input size (H={h}, W={w})."
                )

        # Create the sequential feature extractor (convolutional part)
        self.features: nn.Sequential = nn.Sequential(*conv_layers_list)

        # Calculate the size of the flattened feature vector after conv layers
        self.flattened_size: int = current_channels * current_h * current_w
        if (
            self.flattened_size <= 0
        ):  # Should be caught by H,W <=0 check earlier, but good safeguard
            raise ValueError(
                f"CNNNetwork: Calculated flattened_size for classifier input is non-positive ({self.flattened_size}). "
                "This usually means the feature maps became too small. "
                f"Final H={current_h}, W={current_w}, Channels={current_channels}. Check CNN parameters."
            )

        # --- Build Classifier (Fully Connected Layers) ---
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Linear(self.flattened_size, fc_hidden_size),  # First FC layer
            nn.ReLU(),  # Activation
            nn.Linear(fc_hidden_size, num_actions),  # Output layer (Q-values or logits)
        )

        # Apply Xavier initialization to all Conv2D and Linear layers in the network
        self.apply(init_weights_xavier)
        # print(f"CNNNetwork initialized: Input C={c},H={h},W={w} (format_CHW={self.input_format_is_chw}). "
        #       f"Flattened features={self.flattened_size}. Output actions={num_actions}.")

    def forward(self, x: torch.Tensor, *args: Any) -> torch.Tensor:
        """
        Performs a forward pass through the CNN.

        The input tensor `x` is expected to be image-like. If its format is
        (Batch, Height, Width, Channels) or a single (Height, Width, Channels),
        it will be permuted to the PyTorch standard (Batch, Channels, Height, Width).
        If it's a single (Channels, Height, Width), a batch dimension will be added.

        Args:
            x (torch.Tensor): The input tensor representing a state or batch of states.
                              Expected shapes: (B,C,H,W), (B,H,W,C), (C,H,W), or (H,W,C).
            *args (Any): Additional arguments, ignored in this CNN implementation.

        Returns:
            torch.Tensor: The output tensor from the network (e.g., Q-values).
                          Shape: (Batch_size, Num_actions).
        """
        # --- Input Permutation and Batching ---
        # Ensure input is 4D (B, C, H, W) or 3D (C, H, W / H, W, C for single sample)
        if x.ndim < 3 or x.ndim > 4:
            raise ValueError(
                f"CNNNetwork expects 3D or 4D input (image-like), but got {x.ndim}D input with shape {x.shape}."
            )

        # Handle permutation for HWC input and add batch dimension for single samples
        if (
            not self.input_format_is_chw
        ):  # Original observation_space was inferred as (H,W,C)
            if x.ndim == 4:  # Batched input: (B, H, W, C)
                x = x.permute(0, 3, 1, 2)  # Permute to (B, C, H, W)
            elif x.ndim == 3:  # Single unbatched sample: (H, W, C)
                x = x.permute(2, 0, 1).unsqueeze(
                    0
                )  # Permute to (C,H,W) then add batch: (1,C,H,W)
        elif (
            self.input_format_is_chw and x.ndim == 3
        ):  # Single unbatched sample in CHW format: (C,H,W)
            x = x.unsqueeze(0)  # Add batch dimension: (1,C,H,W)

        # At this point, x should be in (B, C, H, W) format.

        # --- Sanity Check: Input Channels ---
        # `self.features[0]` is the first Conv2d layer in the `self.features` sequential block.
        # Its `in_channels` attribute stores the expected number of input channels.
        expected_in_channels: int = self.features[0].in_channels
        if x.shape[1] != expected_in_channels:
            raise ValueError(
                f"CNNNetwork input channel mismatch. Network expected {expected_in_channels} channels, "
                f"but received input with {x.shape[1]} channels (processed input shape: {x.shape}). "
                f"Original observation_space shape was: {self.observation_space.shape} "
                f"(inferred as CHW: {self.input_format_is_chw})."
            )

        # --- Feature Extraction (Convolutional Layers) ---
        x = self.features(x)  # Pass through convolutional layers + pooling

        # --- Flattening ---
        # Reshape the output of conv layers to (Batch_size, FlattenedFeatures)
        # x.size(0) is the batch size. -1 infers the second dimension.
        x = x.reshape(x.size(0), -1)  # Using reshape for flexibility

        # --- Sanity Check: Flattened Size ---
        if x.shape[1] != self.flattened_size:
            # This error indicates an issue in the calculation of `self.flattened_size`
            # in `__init__` or an unexpected output shape from `self.features`.
            raise ValueError(
                f"CNNNetwork flattened feature size mismatch. Expected {self.flattened_size} features, "
                f"but got {x.shape[1]} features after flattening the output of convolutional layers. "
                f"Output shape from self.features before flatten: {x.shape} (if this was logged before reshape)."  # This x.shape here is actually *after* flatten.
            )

        # --- Classification/Regression (Fully Connected Layers) ---
        x = self.classifier(x)  # Pass through fully connected layers

        return x

# networks/factory.py
"""
A factory module for creating neural network architectures.

This module provides a centralized way to instantiate different neural network
objects based on a string name and a set of parameters, typically defined in
an experiment's configuration file (e.g., YAML). This approach decouples the
agent implementation or experiment setup logic from the concrete network
architecture implementations, making it easy to switch between different network
types or add new ones without modifying the core code that uses these networks.

To add a new network architecture:
1. Implement the network class, ensuring it inherits from `BaseNetwork`
   (defined in `networks.base.py`).
2. Place the implementation in `networks.architectures.py` or a similar module.
3. Import the new network class into this factory file.
4. Add an entry to the `NETWORK_REGISTRY` dictionary, mapping a unique
   string name (used in configuration files) to the network class.
"""

from typing import Dict, Any, Type, Optional  # Optional added for params
import gym  # For gym.Space type hint, used by network constructors

# Import the base class for type hinting and to ensure registered networks adhere to the interface
from .base import BaseNetwork

# Import all concrete network architectures that can be created by this factory
from .architectures import MLPNetwork, CNNNetwork

# Example: from .architectures import MyCustomCNN # If you were to add a new network


# --- Network Registry ---
# A dictionary that maps string identifiers (names) to their corresponding network classes.
# This registry is used by the `create_network` function to look up and instantiate
# the requested network type. The names defined here are typically used in YAML
# configuration files (e.g., under `agent.params.online_network_config.name`)
# to specify which network architecture to use.
NETWORK_REGISTRY: Dict[str, Type[BaseNetwork]] = {
    "MLPNetwork": MLPNetwork,  # Multi-Layer Perceptron
    "CNNNetwork": CNNNetwork,  # Convolutional Neural Network
    # Example: "MyCustomCNN": MyCustomCNN, # Entry for a hypothetical new network
}


def create_network(
    name: str,  # The registered name of the network architecture
    observation_space: gym.Space,  # Environment's observation space
    action_space: gym.Space,  # Environment's action space
    params: Optional[Dict[str, Any]] = None,  # Network-specific constructor parameters
) -> BaseNetwork:
    """
    Instantiates a neural network object from its registered name and parameters.

    This factory function looks up the network class associated with the given `name`
    in the `NETWORK_REGISTRY`. It then initializes an instance of this class,
    passing the `observation_space` and `action_space` (which allow the network
    to dynamically configure its input and output dimensions) and any additional
    architecture-specific parameters provided in the `params` dictionary.

    Args:
        name (str): The name of the network architecture to create.
                    This must be a key present in the `NETWORK_REGISTRY`.
        observation_space (gym.Space): The environment's observation space.
                                       Passed to the network's constructor.
        action_space (gym.Space): The environment's action space.
                                  Passed to the network's constructor.
        params (Optional[Dict[str, Any]], optional): A dictionary of parameters
            specific to the network architecture's constructor (e.g., `hidden_layers`
            for `MLPNetwork`, or `conv_channels` for `CNNNetwork`). These are
            typically defined in the experiment configuration file.
            Defaults to None, which is treated as an empty dictionary.

    Raises:
        ValueError: If the provided network `name` is not found in the `NETWORK_REGISTRY`.

    Returns:
        BaseNetwork: An initialized instance of the requested neural network,
                     ready to be used (e.g., by an RL agent).
    """
    if name not in NETWORK_REGISTRY:
        # Network name not found, raise an error with available options
        available_networks = list(NETWORK_REGISTRY.keys())
        raise ValueError(
            f"Unknown network name: '{name}'. "
            f"Available network architectures in registry are: {available_networks}"
        )

    # Use an empty dictionary for parameters if None is provided.
    # This ensures `**params_to_pass` unpacking works correctly.
    if params is None:
        params_to_pass = {}
    else:
        params_to_pass = params

    # Retrieve the network class from the registry
    network_class: Type[BaseNetwork] = NETWORK_REGISTRY[name]

    # Instantiate the network.
    # The `observation_space` and `action_space` are passed explicitly.
    # Architecture-specific parameters from the `params_to_pass` dictionary
    # (e.g., `hidden_layers`, `conv_channels`) are unpacked using `**params_to_pass`.
    try:
        network_instance = network_class(
            observation_space=observation_space,
            action_space=action_space,
            **params_to_pass,
        )
    except TypeError as e:
        # This can happen if `params_to_pass` is missing required arguments for the
        # network's constructor or if unexpected arguments are passed.
        print(
            f"Error: TypeError during instantiation of network '{name}'. "
            f"Check if all required parameters are provided in the configuration "
            f"and match the network's __init__ signature. Details: {e}"
        )
        # Propagate the error, as network creation is critical for agents that use them.
        raise
    except Exception as e:
        # Catch any other unexpected errors during network instantiation.
        print(
            f"Error: An unexpected error occurred during instantiation of network '{name}'. Details: {e}"
        )
        raise

    print(f"Successfully created network: '{name}' with parameters: {params_to_pass}")
    return network_instance

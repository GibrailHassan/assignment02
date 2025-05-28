# networks/factory.py
"""
A factory for creating neural network architectures.
This module provides a centralized way to instantiate network objects based on a
string name and parameters, typically defined in a configuration file.
"""

from typing import Dict, Any, Type
import gym  # For gym.Space type hint

# Import the base class and specific architectures
from .base import BaseNetwork
from .architectures import MLPNetwork, CNNNetwork  # Add other custom networks here

# A registry mapping string names to their corresponding network classes.
# To add a new network architecture:
# 1. Implement it in networks.architectures (inheriting from BaseNetwork).
# 2. Import it here.
# 3. Add it to this NETWORK_REGISTRY dictionary.
NETWORK_REGISTRY: Dict[str, Type[BaseNetwork]] = {
    "MLPNetwork": MLPNetwork,
    "CNNNetwork": CNNNetwork,
    # Example: "MyCustomCNN": MyCustomCNN,
}


def create_network(
    name: str,
    observation_space: gym.Space,
    action_space: gym.Space,
    params: Dict[str, Any] = None,
) -> BaseNetwork:
    """
    Instantiates a neural network object from its registered name.

    The observation_space and action_space from the environment are passed
    to the network's constructor, allowing it to dynamically configure its
    input and output dimensions. Additional architecture-specific parameters
    are passed via the 'params' dictionary.

    Args:
        name (str): The name of the network architecture to create.
                    This must be a key in the NETWORK_REGISTRY.
        observation_space (gym.Space): The environment's observation space.
        action_space (gym.Space): The environment's action space.
        params (Dict[str, Any], optional): A dictionary of parameters specific
                                           to the network architecture's constructor
                                           (e.g., hidden_layers for MLPNetwork).
                                           Defaults to None (an empty dictionary).

    Raises:
        ValueError: If the provided network name is not found in the registry.

    Returns:
        BaseNetwork: An initialized instance of the requested neural network.
    """
    if name not in NETWORK_REGISTRY:
        raise ValueError(
            f"Unknown network name: '{name}'. "
            f"Available network architectures are: {list(NETWORK_REGISTRY.keys())}"
        )

    if params is None:
        params = {}

    network_class = NETWORK_REGISTRY[name]

    # Instantiate the network, passing observation_space, action_space,
    # and any other architecture-specific parameters from the config.
    network = network_class(
        observation_space=observation_space,
        action_space=action_space,
        **params,  # Unpack architecture-specific params (e.g., hidden_layers)
    )

    print(f"Successfully created network: '{name}'")
    return network

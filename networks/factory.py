# networks/factory.py
"""
A factory for creating neural network architectures.
"""
from typing import Dict, Any, Type, Tuple
from networks.base import BaseNetwork
from networks.architectures import MLP, CNN

NETWORK_REGISTRY: Dict[str, Type[BaseNetwork]] = {
    "MLP": MLP,
    "CNN": CNN,
}


def create_network(
    name: str,
    input_shape: Tuple[int, ...],
    output_shape: int,
    params: Dict[str, Any] = None,
) -> BaseNetwork:
    """Instantiates a network object from its registered name."""
    if name not in NETWORK_REGISTRY:
        raise ValueError(f"Unknown network: '{name}'")

    if params is None:
        params = {}

    network_class = NETWORK_REGISTRY[name]
    return network_class(input_shape=input_shape, output_shape=output_shape, **params)

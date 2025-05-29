"""
A factory for creating and registering game environments.

This module provides a centralized way to instantiate environments based on a
string name, as defined in a configuration file. This approach decouples the main
application logic from the concrete environment implementations, making it easy
to add new environments without changing the core training code.
"""

from typing import Dict, Any, Type, Optional
import gym

from env.env_discrete import MoveToBeaconDiscreteEnv
from env.env_full import MoveToBeaconEnv
from env.dr_env import DefeatRoachesEnv

# A registry mapping string names to their corresponding environment classes.
# To add a new environment, simply import it and add it to this dictionary.
ENV_REGISTRY: Dict[str, Type[gym.Env]] = {
    "MoveToBeaconDiscreteEnv": MoveToBeaconDiscreteEnv,
    "MoveToBeaconEnv": MoveToBeaconEnv,
    "DefeatRoachesEnv": DefeatRoachesEnv,
}


def create_environment(name: str, params: Optional[Dict[str, Any]] = None) -> gym.Env:
    """
    Instantiates an environment object from its registered name.

    Args:
        name (str): The name of the environment to create. This must be a key
                    in the ENV_REGISTRY.
        params (Dict[str, Any], optional): A dictionary of parameters to pass
                                           to the environment's constructor.
                                           Defaults to None.

    Raises:
        ValueError: If the provided environment name is not found in the registry.

    Returns:
        gym.Env: An initialized instance of the requested environment.
    """
    if name not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment name: '{name}'. "
            f"Available environments are: {list(ENV_REGISTRY.keys())}"
        )

    # Use an empty dictionary if no parameters are provided.
    if params is None:
        params = {}

    # Look up the class in the registry and instantiate it with the given parameters.
    env_class = ENV_REGISTRY[name]
    environment = env_class(**params)

    print(f"Successfully created environment: '{name}'")
    return environment

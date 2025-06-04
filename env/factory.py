"""
A factory module for creating and registering game environments.

This module provides a centralized way to instantiate different game environment
wrappers based on a string name, typically defined in a configuration file (e.g., YAML).
This approach decouples the main application logic (e.g., in `main.py`) from the
concrete environment implementations, making it easy to add new environments or
modify existing ones without changing the core training or evaluation scripts.

To add a new environment:
1. Implement the environment class, ensuring it adheres to the `gym.Env` interface.
   Typically, this involves creating a wrapper around a PySC2 environment.
2. Import the new environment class into this file.
3. Add an entry to the `ENV_REGISTRY` dictionary, mapping a unique string name
   (used in configuration files) to the environment class.
"""

from typing import Dict, Any, Type, Optional  # For type hinting
import gym  # For gym.Env base class and type hinting

# Import all concrete environment wrapper classes that can be created by this factory
from env.env_discrete import MoveToBeaconDiscreteEnv
from env.env_full import MoveToBeaconEnv
from env.dr_env import DefeatRoachesEnv

# Example: from env.my_new_env import MyNewCustomEnv # If you were to add a new environment

# --- Environment Registry ---
# A dictionary that maps string identifiers (names) to their corresponding environment classes.
# This registry is used by the `create_environment` function to look up and instantiate
# the requested environment type. The names defined here are typically used in YAML
# configuration files (e.g., under the `environment.name` key) to specify which
# environment to use for an experiment.
ENV_REGISTRY: Dict[str, Type[gym.Env]] = {
    "MoveToBeaconDiscreteEnv": MoveToBeaconDiscreteEnv,
    "MoveToBeaconEnv": MoveToBeaconEnv,
    "DefeatRoachesEnv": DefeatRoachesEnv,
    # Example: "MyNewCustomEnv": MyNewCustomEnv, # Entry for a hypothetical new environment
}


def create_environment(name: str, params: Optional[Dict[str, Any]] = None) -> gym.Env:
    """
    Instantiates an environment object from its registered name using provided parameters.

    This factory function looks up the environment class associated with the given `name`
    in the `ENV_REGISTRY`. It then initializes an instance of this class,
    passing any environment-specific parameters contained in the `params` dictionary.

    Args:
        name (str): The registered name of the environment to create (must be a key
                    in `ENV_REGISTRY`).
        params (Optional[Dict[str, Any]], optional): A dictionary of parameters
            to be passed to the environment's constructor (`__init__` method).
            These parameters are typically specified in the 'environment.params'
            section of a YAML configuration file. If None or empty, the environment
            is initialized with its default parameters. Defaults to None.

    Raises:
        ValueError: If the provided environment `name` is not found in the `ENV_REGISTRY`.

    Returns:
        gym.Env: An initialized instance of the requested Gym-compatible environment.
    """
    if name not in ENV_REGISTRY:
        # Environment name not found, raise an error with available options
        available_environments = list(ENV_REGISTRY.keys())
        raise ValueError(
            f"Unknown environment name: '{name}'. "
            f"Available environments in registry are: {available_environments}"
        )

    # Use an empty dictionary for parameters if None is provided,
    # ensuring `**params` unpacking works correctly.
    if params is None:
        params_to_pass = {}
    else:
        params_to_pass = params

    # Retrieve the environment class from the registry
    env_class: Type[gym.Env] = ENV_REGISTRY[name]

    # Instantiate the environment.
    # All parameters from the `params_to_pass` dictionary are unpacked and passed
    # as keyword arguments to the environment's constructor.
    try:
        environment_instance = env_class(**params_to_pass)
    except TypeError as e:
        # This can happen if `params_to_pass` is missing required arguments for the
        # environment's constructor or if unexpected arguments are passed.
        print(
            f"Error: TypeError during instantiation of environment '{name}'. "
            f"Check if all required parameters are provided in the configuration "
            f"and match the environment's __init__ signature. Details: {e}"
        )
        # Propagate the error to halt execution, as environment creation is critical.
        raise
    except Exception as e:
        # Catch any other unexpected errors during environment instantiation.
        print(
            f"Error: An unexpected error occurred during instantiation of environment '{name}'. Details: {e}"
        )
        raise

    print(
        f"Successfully created environment: '{name}' with parameters: {params_to_pass}"
    )
    return environment_instance

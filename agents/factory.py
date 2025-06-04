"""
A factory module for creating and registering reinforcement learning agents.

This module provides a centralized mechanism (`create_agent` function and
`AGENT_REGISTRY`) to instantiate different types of RL agents based on a
string name. This approach decouples the main experiment setup logic
(e.g., in `main.py`) from the concrete agent implementations, making it easier
to add new agent types or modify existing ones without changing the core
experiment execution code.

To add a new agent:
1. Implement the agent class, ensuring it inherits from `AbstractAgent`.
2. Import the new agent class into this file.
3. Add an entry to the `AGENT_REGISTRY` dictionary, mapping a unique
   string name (used in configuration files) to the agent class.
"""

from typing import Dict, Any, Type
import gym  # For type hinting of observation and action spaces

# Import the abstract base class for type hinting and consistency
from agents.abstractAgent import AbstractAgent

# Import all concrete agent implementations that can be created by this factory
from agents.qlAgent import QLearningAgent
from agents.sarsaAgent import SARSAAgent
from agents.dqnAgent import DQNAgent
from agents.randomAgent import RandomAgent
from agents.basicAgent import BasicAgent
from agents.reinforceAgent import (
    REINFORCEAgent,
)  # <<< MODIFICATION: Import the new REINFORCEAgent

# --- Agent Registry ---
# A dictionary that maps string identifiers (names) to their corresponding agent classes.
# This registry is used by the `create_agent` function to look up and instantiate
# the requested agent type. The names defined here are typically used in YAML
# configuration files to specify which agent to use for an experiment.
AGENT_REGISTRY: Dict[str, Type[AbstractAgent]] = {
    "QLearningAgent": QLearningAgent,
    "SARSAAgent": SARSAAgent,
    "DQNAgent": DQNAgent,
    "RandomAgent": RandomAgent,
    "BasicAgent": BasicAgent,
    "REINFORCEAgent": REINFORCEAgent,
}


def create_agent(
    name: str,
    params: Dict[str, Any],  # Parameters for the agent's constructor
    observation_space: gym.Space,  # Environment's observation space
    action_space: gym.Space,  # Environment's action space
) -> AbstractAgent:
    """
    Instantiates an agent object from its registered name using the provided parameters.

    This factory function looks up the agent class associated with the given `name`
    in the `AGENT_REGISTRY`. It then initializes an instance of this class,
    passing the `observation_space`, `action_space`, and any agent-specific
    parameters (hyperparameters, network objects, etc.) contained in the `params`
    dictionary.

    Args:
        name (str): The registered name of the agent to create (must be a key
                    in `AGENT_REGISTRY`).
        params (Dict[str, Any]): A dictionary of parameters to be passed to the
                                 agent's constructor (`__init__` method). This
                                 dictionary is typically populated from the 'agent.params'
                                 section of a YAML configuration file and may include
                                 hyperparameters as well as injected dependencies like
                                 pre-built neural network objects for DQNAgent.
        observation_space (gym.Space): The observation space of the environment,
                                       which the agent will interact with.
        action_space (gym.Space): The action space of the environment.

    Raises:
        ValueError: If the provided agent `name` is not found in the `AGENT_REGISTRY`.

    Returns:
        AbstractAgent: An initialized instance of the requested agent.
    """
    if name not in AGENT_REGISTRY:
        # Agent name not found, raise an error with available options
        available_agents = list(AGENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown agent name: '{name}'. "
            f"Available agents in registry are: {available_agents}"
        )

    # Retrieve the agent class from the registry
    agent_class: Type[AbstractAgent] = AGENT_REGISTRY[name]

    # Instantiate the agent.
    # The `observation_space` and `action_space` are passed explicitly,
    # while all other necessary parameters (hyperparameters, injected networks for
    # agents like DQNAgent, etc.) are expected to be in the `params` dictionary
    # and are unpacked using `**params`.
    try:
        agent_instance = agent_class(
            observation_space=observation_space,
            action_space=action_space,
            **params,  # Unpack all other parameters from config + injected networks
        )
    except TypeError as e:
        # This can happen if `params` is missing required arguments for the
        # agent's constructor or if unexpected arguments are passed.
        print(
            f"Error: TypeError during instantiation of agent '{name}'. "
            f"Check if all required parameters are provided in the configuration "
            f"and match the agent's __init__ signature. Details: {e}"
        )
        # Propagate the error to halt execution, as agent creation is critical.
        raise
    except Exception as e:
        # Catch any other unexpected errors during agent instantiation.
        print(
            f"Error: An unexpected error occurred during instantiation of agent '{name}'. Details: {e}"
        )
        raise

    print(f"Successfully created agent: '{name}'")
    return agent_instance

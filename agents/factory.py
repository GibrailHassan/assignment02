# agents/factory.py

"""
A factory for creating and registering reinforcement learning agents.

This module provides a centralized way to instantiate agent objects based on a
string name, as defined in a configuration file. This decouples the main
application logic from the concrete agent implementations, making it easy
to add new agents to the project.
"""

from typing import Dict, Any, Type
import gym

from agents.abstractAgent import AbstractAgent
from agents.qlAgent import QLearningAgent
from agents.sarsaAgent import SARSAAgent
from agents.dqnAgent import DQNAgent
from agents.randomAgent import RandomAgent
from agents.basicAgent import BasicAgent

# A registry mapping string names to their corresponding agent classes.
AGENT_REGISTRY: Dict[str, Type[AbstractAgent]] = {
    "QLearningAgent": QLearningAgent,
    "SARSAAgent": SARSAAgent,
    "DQNAgent": DQNAgent,
    "RandomAgent": RandomAgent,
    "BasicAgent": BasicAgent,
}


def create_agent(
    name: str,
    params: Dict[str, Any],
    # UPDATED: Accept the full gym space objects
    observation_space: gym.Space,
    action_space: gym.Space,
) -> AbstractAgent:
    """
    Instantiates an agent object from its registered name.

    This function requires the observation and action spaces from the environment
    to properly initialize the agent.

    Args:
        name (str): The name of the agent to create. This must be a key
                    in the AGENT_REGISTRY.
        params (Dict[str, Any]): A dictionary of hyperparameters to pass to the
                                 agent's constructor.
        observation_space (gym.Space): The environment's observation space.
        action_space (gym.Space): The environment's action space.

    Raises:
        ValueError: If the provided agent name is not found in the registry.

    Returns:
        AbstractAgent: An initialized instance of the requested agent.
    """
    if name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent name: '{name}'. "
            f"Available agents are: {list(AGENT_REGISTRY.keys())}"
        )

    # UPDATED: Pass the correct keyword arguments to the agent's __init__
    full_params = {
        "observation_space": observation_space,
        "action_space": action_space,
        **params,
    }

    # Look up the class in the registry and instantiate it.
    agent_class = AGENT_REGISTRY[name]
    agent = agent_class(**full_params)

    print(f"Successfully created agent: '{name}'")
    return agent

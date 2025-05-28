# agents/factory.py

"""
A factory for creating and registering reinforcement learning agents.
"""

from typing import Dict, Any, Type
import gym

from agents.abstractAgent import AbstractAgent
from agents.qlAgent import QLearningAgent
from agents.sarsaAgent import SARSAAgent
from agents.dqnAgent import DQNAgent
from agents.randomAgent import RandomAgent
from agents.basicAgent import BasicAgent

AGENT_REGISTRY: Dict[str, Type[AbstractAgent]] = {
    "QLearningAgent": QLearningAgent,
    "SARSAAgent": SARSAAgent,
    "DQNAgent": DQNAgent,
    "RandomAgent": RandomAgent,
    "BasicAgent": BasicAgent,
}


def create_agent(
    name: str,
    params: Dict[
        str, Any
    ],  # This dict comes from main.py, already contains agent-specific params
    # and injected networks (like online_network)
    observation_space: gym.Space,
    action_space: gym.Space,
) -> AbstractAgent:
    """
    Instantiates an agent object from its registered name.
    """
    if name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent name: '{name}'. "
            f"Available agents are: {list(AGENT_REGISTRY.keys())}"
        )

    agent_class = AGENT_REGISTRY[name]

    # The 'params' dictionary already contains all necessary keyword arguments
    # that are specific to the agent (e.g., learning_rate, enable_ddqn,
    # online_network, target_network for DQNAgent).
    # The observation_space and action_space are passed as explicit positional/keyword args.
    agent = agent_class(
        observation_space=observation_space,
        action_space=action_space,
        **params,  # Unpack all other parameters from the config + injected networks
    )

    print(f"Successfully created agent: '{name}'")
    return agent

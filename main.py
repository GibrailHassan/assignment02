# main.py

"""
A unified entry point for running all reinforcement learning experiments.
"""

import sys
import yaml
from typing import Dict, Any

from absl import app
from absl import flags

from agents.factory import create_agent, AGENT_REGISTRY
from env.factory import create_environment
from runner.runner import Runner

# --- Command-Line Argument Definition ---
FLAGS = flags.FLAGS
flags.DEFINE_string("config", None, "Path to the YAML configuration file.")
flags.mark_flag_as_required("config")


def main(argv: Any) -> None:
    """
    The main function that sets up and runs the RL experiment.
    """
    # --- Configuration Loading ---
    print(f"Loading configuration from: {FLAGS.config}")
    try:
        with open(FLAGS.config, "r") as f:
            config: Dict[str, Any] = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {FLAGS.config}")
        sys.exit(1)

    env_config = config.get("environment", {})
    agent_config = config.get("agent", {})
    runner_config = config.get("runner", {})

    # --- Object Creation using Factories ---
    env = create_environment(
        name=env_config.get("name", ""), params=env_config.get("params", {})
    )

    agent_params = agent_config.get("params", {})
    load_path = runner_config.get("load_model_path")
    is_training = runner_config.get("is_training", True)

    if not is_training and load_path:
        print(
            f"Loading pre-trained agent '{agent_config.get('name')}' from: {load_path}"
        )
        agent_class = AGENT_REGISTRY[agent_config.get("name")]
        agent = agent_class.load_model(path=load_path, **agent_params)
        if hasattr(agent, "epsilon_min"):
            agent.epsilon = agent.epsilon_min
    else:
        # UPDATED: Pass the full space objects, not their shapes
        agent = create_agent(
            name=agent_config.get("name", ""),
            params=agent_params,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

    runner = Runner(agent=agent, env=env, **runner_config)

    runner.run()


if __name__ == "__main__":
    app.run(main)

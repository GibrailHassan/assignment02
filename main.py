# main.py

"""
A unified entry point for running all reinforcement learning experiments,
using enhanced utility functions for MLflow setup, configuration loading,
agent and network preparation, and agent initialization.
"""

import sys
import yaml  # Ensure PyYAML is installed
from typing import Dict, Any

from absl import app
from absl import flags

import mlflow
import gym  # For gym.spaces.Discrete in network creation for critic example

# Utility Functions
from utils.experiment_utils import (
    load_config,
    setup_mlflow_run,
    # prepare_agent_networks_and_params, # This logic is now more integrated into main
    # initialize_agent # This logic is now more integrated into main
)

# Factories
from env.factory import create_environment
from agents.factory import create_agent, AGENT_REGISTRY  # AGENT_REGISTRY for loading
from networks.factory import create_network as create_nn_from_factory
from runner.runner import Runner  # <--- ADDED THIS IMPORT

# --- Command-Line Argument Definition ---
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config", None, "Path to the YAML configuration file for the experiment."
)
flags.mark_flag_as_required("config")


def main(argv: Any) -> None:
    """
    The main function that sets up and runs the RL experiment with MLflow tracking.
    """
    # --- 1. Configuration Loading ---
    config = load_config(FLAGS.config)
    if config is None:
        sys.exit(1)

    # --- 2. MLflow Setup ---
    mlflow_experiment_name = config.get("experiment_name", "Default_RL_Experiment")

    with mlflow.start_run() as run:  # Ensures run is ended automatically
        mlflow_run_id = setup_mlflow_run(
            config, FLAGS.config
        )  # Call *inside* the 'with' block

        if mlflow_run_id is None:
            print("MLflow setup failed. Critical MLflow features might be unavailable.")
        else:
            print(f"MLflow Run ID: {mlflow_run_id} successfully initialized.")

        # Extract config sections
        env_config = config.get("environment", {})
        agent_config_yaml = config.get("agent", {})
        runner_config = config.get("runner", {}).copy()  # Use a copy

        # --- 3. Environment Creation ---
        env = create_environment(
            name=env_config.get("name", "UnknownEnv"),
            params=env_config.get("params", {}),
        )
        if env is None:
            print("Error: Environment creation failed.")
            if mlflow_run_id:
                mlflow.set_tag("run_status", "env_creation_error")
            sys.exit(1)

        # --- 4. Agent Parameter Preparation & Network Creation ---
        agent_name = agent_config_yaml.get("name", "UnknownAgent")
        # Start with parameters directly from the agent's 'params' section in YAML
        agent_constructor_params = agent_config_yaml.get("params", {}).copy()

        is_training = runner_config.get("is_training", True)
        # Add/overwrite is_training status, as some agents might use it in __init__
        agent_constructor_params["is_training"] = is_training

        # Pop network configurations from agent_constructor_params before they are passed to agent factory
        # The network objects themselves will be added back.
        online_network_config = agent_constructor_params.pop(
            "online_network_config", None
        )
        actor_network_config = agent_constructor_params.pop(
            "actor_network_config", None
        )
        critic_network_config = agent_constructor_params.pop(
            "critic_network_config", None
        )

        # For loading DQNAgent, DQNAgent.load_model expects 'network_config' in its kwargs
        # to reconstruct the network. We ensure it's there if online_network_config was present.
        # If network_config is already explicitly in YAML agent.params, that will be used.
        if not is_training and agent_name == "DQNAgent":
            if (
                online_network_config
                and "network_config" not in agent_constructor_params
            ):
                agent_constructor_params["network_config"] = online_network_config
            elif (
                "network_config" not in agent_constructor_params
                and online_network_config is None
            ):
                print(
                    "Warning: Loading DQNAgent without 'network_config' or 'online_network_config'. "
                    "Load_model might fail if it needs to reconstruct networks."
                )

        # Create and inject network objects if their configurations were provided
        if online_network_config:
            print(
                f"Creating online network from config: {online_network_config.get('name')}"
            )
            agent_constructor_params["online_network"] = create_nn_from_factory(
                name=online_network_config.get("name"),
                observation_space=env.observation_space,
                action_space=env.action_space,
                params=online_network_config.get("params", {}),
            )
            # For DQN, target network is usually identical to online initially
            print(
                f"Creating target network from config: {online_network_config.get('name')}"
            )
            agent_constructor_params["target_network"] = create_nn_from_factory(
                name=online_network_config.get("name"),
                observation_space=env.observation_space,
                action_space=env.action_space,
                params=online_network_config.get("params", {}),
            )

        if actor_network_config:  # For potential Actor-Critic agents
            print(
                f"Creating actor network from config: {actor_network_config.get('name')}"
            )
            agent_constructor_params["actor_network"] = create_nn_from_factory(
                name=actor_network_config.get("name"),
                observation_space=env.observation_space,
                action_space=env.action_space,
                params=actor_network_config.get("params", {}),
            )
        if critic_network_config:  # For potential Actor-Critic agents
            print(
                f"Creating critic network from config: {critic_network_config.get('name')}"
            )
            critic_action_space = gym.spaces.Discrete(
                1
            )  # Critic typically outputs a single value
            agent_constructor_params["critic_network"] = create_nn_from_factory(
                name=critic_network_config.get("name"),
                observation_space=env.observation_space,
                action_space=critic_action_space,
                params=critic_network_config.get("params", {}),
            )

        # --- 5. Agent Creation / Loading ---
        agent: AbstractAgent | None = (
            None  # Explicitly type AbstractAgent from agents.abstractAgent
        )
        load_model_path = runner_config.get("load_model_path")

        if not is_training and load_model_path:
            print(
                f"Loading pre-trained agent '{agent_name}' from local path: {load_model_path}"
            )
            if agent_name not in AGENT_REGISTRY:
                print(
                    f"Error: Agent name '{agent_name}' not found in AGENT_REGISTRY for loading."
                )
                if mlflow_run_id:
                    mlflow.set_tag("run_status", "config_error_agent_load")
                sys.exit(1)
            agent_class = AGENT_REGISTRY[agent_name]

            try:
                # Pass observation_space, action_space explicitly, plus all other prepared params
                agent = agent_class.load_model(
                    path=load_model_path,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    **agent_constructor_params,  # Contains other agent hyperparams, and network_config for DQNAgent
                )
            except Exception as e:
                print(f"Error loading agent model: {e}")
                if mlflow_run_id:
                    mlflow.set_tag("run_status", "load_model_error")
                    mlflow.log_param("load_model_error_msg", str(e)[:250])
                raise  # Re-raise to stop execution

            if agent and hasattr(agent, "is_training"):
                agent.is_training = False
            if agent and hasattr(agent, "epsilon") and hasattr(agent, "epsilon_min"):
                agent.epsilon = agent.epsilon_min
            print("Agent loaded successfully for evaluation.")
        else:  # Training or no model path provided for evaluation
            print(
                f"Creating new agent: '{agent_name}' for training (or evaluation without loading)."
            )
            try:
                agent = create_agent(  # This is agents.factory.create_agent
                    name=agent_name,
                    params=agent_constructor_params,  # This dict now contains injected networks if any
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                )
                print("New agent created successfully.")
            except Exception as e:
                print(f"Error creating new agent: {e}")
                if mlflow_run_id:
                    mlflow.set_tag("run_status", "agent_creation_error")
                    mlflow.log_param("agent_creation_error_msg", str(e)[:250])
                raise  # Re-raise to stop execution

        if agent is None:
            print("Error: Agent could not be initialized.")
            if mlflow_run_id:
                mlflow.set_tag("run_status", "agent_init_critical_error")
            sys.exit(1)

        # --- 6. Runner Setup and Execution ---
        runner_config["mlflow_run_id"] = mlflow_run_id
        runner_config["experiment_name"] = mlflow_experiment_name

        runner = Runner(agent=agent, env=env, **runner_config)

        try:
            print("Starting experiment run...")
            runner.run()
            if mlflow_run_id:
                mlflow.set_tag("run_status", "completed")
            print("Experiment run completed successfully.")
        except Exception as e:
            print(f"Error during runner.run(): {e}")
            if mlflow_run_id:
                mlflow.set_tag("run_status", "failed_in_runner")
                mlflow.log_param("runner_error", str(e)[:250])
            raise

    print("MLflow run finished (if active). Main script execution complete.")


if __name__ == "__main__":
    app.run(main)

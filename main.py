# main.py

"""
A unified entry point for running all reinforcement learning experiments,
using enhanced utility functions for MLflow setup, configuration loading,
agent and network preparation, and agent initialization.
"""

import sys
from typing import Dict, Any  # Removed Tuple as it's not directly used here

from absl import app
from absl import flags

# MLflow import
import mlflow

# import os # No longer directly needed here for os.path.basename

# Import from utility module
from utils.experiment_utils import (
    load_config,
    setup_mlflow_run,
    prepare_agent_networks_and_params,
    initialize_agent,
)

# Factories are now primarily used within experiment_utils, but AGENT_REGISTRY might be needed
# if initialize_agent doesn't fully encapsulate loading logic that needs the registry.
# For now, assuming initialize_agent handles it or has access.
# from agents.factory import AGENT_REGISTRY # Keep if initialize_agent needs it for some reason

from env.factory import create_environment  # Env factory is still directly used here
from runner.runner import Runner

# from networks.factory import create_network as create_nn_from_factory # Used within prepare_agent_networks_and_params

# --- Command-Line Argument Definition ---
FLAGS = flags.FLAGS
flags.DEFINE_string("config", None, "Path to the YAML configuration file.")
flags.mark_flag_as_required("config")


def main(argv: Any) -> None:
    """
    The main function that sets up and runs the RL experiment with MLflow tracking.
    """
    # --- Configuration Loading ---
    config = load_config(FLAGS.config)
    if config is None:
        sys.exit(1)  # Exit if config loading failed

    # --- MLflow Setup ---
    # The 'with' block ensures the MLflow run is properly started and ended.
    with mlflow.start_run() as run:
        # Utility function handles setting experiment, logging params, and config artifact
        mlflow_run_id = setup_mlflow_run(config, FLAGS.config)

        if mlflow_run_id is None:
            print("MLflow setup failed. Critical MLflow features might be unavailable.")
            # Decide if an error here should be fatal or just a warning.
            # For now, we'll let it proceed but some logging might not work.
        else:
            print(
                f"MLflow Run ID: {mlflow_run_id} successfully initialized for logging."
            )

        # Extract main configuration sections
        env_config = config.get("environment", {})
        agent_config_yaml = config.get("agent", {})  # The 'agent' section from YAML
        runner_config = config.get("runner", {})

        # --- Environment Creation ---
        env = create_environment(
            name=env_config.get("name", "UnknownEnv"),
            params=env_config.get("params", {}),
        )
        if env is None:  # Assuming create_environment might return None on failure
            print("Error: Environment creation failed.")
            if mlflow_run_id:
                mlflow.set_tag("run_status", "env_creation_error")
            sys.exit(1)

        # --- Agent Preparation (Networks and Parameters) ---
        is_training = runner_config.get("is_training", True)
        # This utility function handles creating networks if specified in agent_config_yaml
        # and prepares the final parameter dictionary for the agent.
        prepared_agent_params = prepare_agent_networks_and_params(
            agent_config_yaml=agent_config_yaml, env=env, is_training=is_training
        )

        # --- Agent Initialization (Create or Load) ---
        # This utility function handles the logic for creating a new agent or loading a pre-trained one.
        agent = initialize_agent(
            agent_config_yaml=agent_config_yaml,
            env=env,
            runner_config=runner_config,  # For load_model_path and is_training
            prepared_agent_params=prepared_agent_params,
        )

        if agent is None:  # If agent initialization failed
            print("Error: Agent initialization failed.")
            if mlflow_run_id:
                mlflow.set_tag("run_status", "agent_init_error")
            sys.exit(1)

        # --- Runner Setup and Execution ---
        # Pass mlflow_run_id to Runner so it can log metrics to the correct run
        runner_config["mlflow_run_id"] = mlflow_run_id
        runner = Runner(
            agent=agent,
            env=env,
            **runner_config,  # Unpack all runner params from config
        )

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
            raise  # Re-raise the exception after logging

    print("MLflow run finished (if active). Main script execution complete.")


if __name__ == "__main__":
    app.run(main)

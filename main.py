# main.py
"""
A unified entry point for running all reinforcement learning experiments.

This script orchestrates the setup and execution of RL experiments based on
a provided YAML configuration file. It leverages utility functions from
`utils.experiment_utils` for clarity, modularity, and conciseness in handling
configuration loading, MLflow setup, environment creation, agent dependency
preparation (including network instantiation), and agent initialization.
The core experiment loop is managed by the `Runner` class.

Usage:
    python main.py --config=configs/your_experiment_config.yaml
"""

import sys
from typing import Dict, Any, Optional
from absl import app, flags
import mlflow

# --- Refactored Utility Functions ---
from utils.experiment_utils import (
    load_config_or_exit,  # Loads YAML or exits on failure
    ensure_mlflow_experiment,  # Gets/Creates MLflow experiment, returns ID
    log_run_params_and_artifacts,  # Logs config and params to active MLflow run
    prepare_agent_dependencies,  # Prepares agent params, instantiates networks
    initialize_agent_or_exit,  # Creates or loads agent, exits on failure
)

# --- Core Components ---
from env.factory import create_environment  # Factory to create environment instances

# Note: AGENT_REGISTRY and create_agent from agents.factory are used internally by initialize_agent_or_exit
# Note: create_nn_from_factory from networks.factory is used internally by prepare_agent_dependencies

from runner.runner import (
    Runner,
)  # Class that manages the agent-environment interaction loop
from agents.abstractAgent import AbstractAgent  # For type hinting
import gym  # For type hinting


# --- Command-Line Argument Definition ---
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config", None, "Path to the YAML configuration file for the experiment."
)
flags.mark_flag_as_required("config")


def setup_and_run_experiment(
    agent: AbstractAgent,
    env: gym.Env,
    runner_config_yaml: Dict[str, Any],  # The 'runner' section from config
    mlflow_run_id: str,  # Active MLflow run ID
    mlflow_experiment_name: str,  # Active MLflow experiment name
) -> None:
    """
    Helper function to set up the Runner and execute the experiment run.
    This encapsulates the final stage of starting the experiment and handling
    its lifecycle, including error handling and cleanup.

    Args:
        agent (AbstractAgent): The initialized RL agent.
        env (gym.Env): The initialized environment.
        runner_config_yaml (Dict[str, Any]): Configuration parameters for the Runner.
        mlflow_run_id (str): The ID of the active MLflow run.
        mlflow_experiment_name (str): The name of the active MLflow experiment.
    """
    print("\n--- Setting up and Starting Experiment Runner ---")

    # Augment runner_config with MLflow details for the Runner's use (e.g., logging plots)
    runner_config_yaml["mlflow_run_id"] = mlflow_run_id
    runner_config_yaml["experiment_name"] = mlflow_experiment_name

    # Instantiate the Runner
    exp_runner = Runner(agent=agent, env=env, **runner_config_yaml)

    try:
        # Execute the main training/evaluation loop
        exp_runner.run()
        # If execution completes without error, mark the MLflow run as completed
        mlflow.set_tag("run_status", "completed")
        print("\n--- Experiment Run Completed Successfully ---")
    except KeyboardInterrupt:
        # Handle user interruption gracefully
        print("\nExperiment run interrupted by user (KeyboardInterrupt).")
        mlflow.set_tag("run_status", "interrupted_by_user")
        # The Runner's finally block should handle resource cleanup
    except Exception as e_runner:
        # Catch any other errors during the Runner's execution
        print(f"Error during Runner execution: {e_runner}")
        mlflow.set_tag("run_status", "failed_in_runner")
        # Log error details to MLflow for easier debugging
        mlflow.log_param("runner_error_type", type(e_runner).__name__)
        mlflow.log_param(
            "runner_error_message", str(e_runner)[:250]
        )  # Log first 250 chars
        raise  # Re-raise the exception to be handled by the top-level try-except in main
    # The Runner's `finally` block is responsible for closing the environment and TensorBoard writer.


def main(argv: Any) -> None:  # 'argv' is passed by absl.app.run
    """
    Main orchestration function for the RL experiment.
    """
    print("=====================================================")
    print("      Reinforcement Learning Experiment Pipeline     ")
    print("=====================================================")
    print(f"Using configuration file: {FLAGS.config}")

    # 1. Load Configuration (exits on failure)
    config: Dict[str, Any] = load_config_or_exit(FLAGS.config)

    # 2. Ensure MLflow Experiment Exists & Set Active Context
    experiment_name_from_config: str = config.get(
        "experiment_name", "Default_RL_Experiment"
    )
    # `ensure_mlflow_experiment` gets/creates the experiment and returns its ID.
    # It also calls `mlflow.set_experiment()` to set the active context.
    experiment_id: Optional[str] = ensure_mlflow_experiment(experiment_name_from_config)
    # If experiment_id is None here, it means ensure_mlflow_experiment had an issue.
    # `mlflow.start_run()` without `experiment_id` will then use the "Default" experiment.

    # 3. Start MLflow Run and Log Initial Details
    # The `with` statement ensures `mlflow.end_run()` is called.
    # Pass `experiment_id` to explicitly associate the run.
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=None
    ) as run:  # `run_name` can be set via tags later if desired
        active_run_id: str = run.info.run_id
        print(f"\nMLflow Run started. Active Run ID: {active_run_id}")
        print(
            f"Run associated with Experiment: '{experiment_name_from_config}' (ID: {experiment_id if experiment_id else 'Default (0)'})."
        )

        # Log config file as artifact and flattened parameters to the active run
        log_run_params_and_artifacts(config, FLAGS.config, active_run_id)

        # 4. Create Environment
        env_config_yaml: Dict[str, Any] = config.get("environment", {})
        env_name: str = env_config_yaml.get("name", "UnknownEnv")
        print(f"\n--- Initializing Environment: {env_name} ---")
        try:
            env: gym.Env = create_environment(
                name=env_name, params=env_config_yaml.get("params", {})
            )
            print(f"Environment '{env_name}' created successfully.")
        except Exception as e_env:  # Catch errors from create_environment
            print(
                f"Fatal Error: Environment '{env_name}' creation failed. Details: {e_env}"
            )
            mlflow.set_tag("run_status", "env_creation_fatal_error")
            sys.exit(1)

        # 5. Prepare Agent Dependencies (Extracts parameters & Instantiates Networks)
        agent_config_yaml: Dict[str, Any] = config.get(
            "agent", {}
        )  # Full 'agent' section from YAML
        runner_config_yaml: Dict[str, Any] = config.get(
            "runner", {}
        )  # Full 'runner' section
        is_training_mode: bool = runner_config_yaml.get("is_training", True)

        # This utility function now handles popping network configs, creating networks,
        # and preparing the dictionary of parameters for the agent.
        prepared_agent_constructor_params: Dict[str, Any] = prepare_agent_dependencies(
            agent_config_yaml=agent_config_yaml, env=env, is_training=is_training_mode
        )

        # 6. Initialize Agent (Handles loading pre-trained models or creating new ones)
        # This utility function also exits on failure.
        agent: AbstractAgent = initialize_agent_or_exit(
            agent_name=agent_config_yaml.get("name", "UnknownAgent"),
            agent_config_yaml=agent_config_yaml,  # Pass original agent YAML for load_model needs
            runner_config=runner_config_yaml,  # For load_model_path & is_training flag
            env=env,
            prepared_agent_params=prepared_agent_constructor_params,  # Contains instantiated networks etc.
        )

        # 7. Setup Runner and Execute the Experiment
        # This helper encapsulates the Runner instantiation and the main run loop with its try-except.
        setup_and_run_experiment(
            agent=agent,
            env=env,
            runner_config_yaml=runner_config_yaml,  # Pass the 'runner' section of config
            mlflow_run_id=active_run_id,
            mlflow_experiment_name=experiment_name_from_config,
        )

    # MLflow run automatically ends here due to the `with` statement.
    print("\nMLflow run has finished. Main script execution complete.")
    print("=====================================================")


if __name__ == "__main__":
    try:
        # `app.run(main)` parses flags (like --config) and calls the `main` function.
        app.run(main)
    except SystemExit as e:
        # This allows our explicit sys.exit(1) calls to terminate the program
        # without absl-py printing a generic "FATAL Flags parsing error".
        if e.code != 0:  # Only print if it's an error exit code
            print(f"Application terminated with exit code {e.code}.")
        # sys.exit(e.code) # app.run will handle the actual exit based on the exception
    except Exception as e_global:
        # Catch any other truly unexpected global exceptions not caught within main()
        print(f"An UNEXPECTED GLOBAL ERROR occurred: {e_global}")
        print("This error was not caught by the main experiment logic.")
        # Attempt to log to MLflow if possible, though context might be partially lost
        try:
            if mlflow.active_run():  # Check if a run is somehow still active
                mlflow.set_tag("run_status", "global_exception_outside_main_logic")
                mlflow.log_param("global_error_type", type(e_global).__name__)
                mlflow.log_param("global_error_message", str(e_global)[:250])
        except Exception as e_mlflow_final_log_attempt:
            print(
                f"Further error: Could not log global error to MLflow: {e_mlflow_final_log_attempt}"
            )
        sys.exit(1)  # Ensure a non-zero exit code for such critical failures

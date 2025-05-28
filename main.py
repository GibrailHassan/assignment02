# main.py

"""
A unified entry point for running all reinforcement learning experiments,
now with MLflow integration for experiment tracking.
"""

import sys
import yaml
from typing import Dict, Any

from absl import app
from absl import flags

# MLflow import
import mlflow
import os  # For logging config file path

from agents.factory import create_agent, AGENT_REGISTRY  # AGENT_REGISTRY for loading
from env.factory import create_environment
from runner.runner import Runner

# --- Command-Line Argument Definition ---
FLAGS = flags.FLAGS
flags.DEFINE_string("config", None, "Path to the YAML configuration file.")
flags.mark_flag_as_required("config")


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flattens a nested dictionary for MLflow parameter logging.
    Example: {'agent': {'lr': 0.1}} -> {'agent.lr': 0.1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            # MLflow parameters must be string, numeric, or boolean.
            # We'll convert lists/tuples to strings for logging.
            if isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
    return dict(items)


def main(argv: Any) -> None:
    """
    The main function that sets up and runs the RL experiment with MLflow tracking.
    """
    # --- Configuration Loading ---
    print(f"Loading configuration from: {FLAGS.config}")
    try:
        with open(FLAGS.config, "r") as f:
            config: Dict[str, Any] = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {FLAGS.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}")
        sys.exit(1)

    # --- MLflow Setup ---
    mlflow_experiment_name = config.get("experiment_name", "Default_RL_Experiment")
    try:
        mlflow.set_experiment(mlflow_experiment_name)
        print(f"MLflow experiment set to: '{mlflow_experiment_name}'")
    except Exception as e:
        print(
            f"Error setting MLflow experiment: {e}. Ensure MLflow server is accessible or permissions are correct."
        )
        # Optionally, decide if you want to proceed without MLflow or exit
        # For now, we'll proceed, and MLflow calls later will likely fail if setup failed.

    # Start an MLflow run. All parameters, metrics, and artifacts will be logged to this run.
    with mlflow.start_run() as run:
        mlflow_run_id = run.info.run_id
        print(f"MLflow Run ID: {mlflow_run_id}")

        # Log the configuration file itself as an artifact for reproducibility
        try:
            mlflow.log_artifact(FLAGS.config, artifact_path="configs")
            print(
                f"Logged configuration file '{FLAGS.config}' to MLflow artifacts (configs/{os.path.basename(FLAGS.config)})."
            )
        except Exception as e:
            print(f"Warning: Could not log config file to MLflow: {e}")

        # Log all parameters from the config, flattened
        try:
            flat_config = flatten_dict(config)
            # MLflow param values are limited in length; truncate if necessary
            # Also, ensure values are appropriate types (string, numeric, bool)
            params_to_log = {}
            for key, value in flat_config.items():
                if isinstance(value, (str, int, float, bool)):
                    if (
                        isinstance(value, str) and len(value) > 250
                    ):  # MLflow limit for param value length
                        params_to_log[key] = value[:247] + "..."
                    else:
                        params_to_log[key] = value
                # else: # Optionally log other types as strings or skip
                #    params_to_log[key] = str(value)[:250]

            if params_to_log:
                mlflow.log_params(params_to_log)
                print("Logged configuration parameters to MLflow.")
            else:
                print("No suitable parameters found in config to log to MLflow.")

        except Exception as e:
            print(f"Warning: Could not log parameters to MLflow: {e}")

        # Extract main configuration sections
        env_config = config.get("environment", {})
        agent_config = config.get("agent", {})
        runner_config = config.get("runner", {})

        # --- Object Creation using Factories ---
        env = create_environment(
            name=env_config.get(
                "name", "UnknownEnv"
            ),  # Provide default if name missing
            params=env_config.get("params", {}),
        )

        agent_params = agent_config.get("params", {})
        load_model_path = runner_config.get("load_model_path")
        is_training = runner_config.get("is_training", True)

        # Pass is_training to agent parameters, as some agents might use it (e.g. DQNAgent)
        agent_params["is_training"] = is_training

        if not is_training and load_model_path:
            print(
                f"Loading pre-trained agent '{agent_config.get('name')}' from local path: {load_model_path}"
            )
            agent_name_to_load = agent_config.get("name", "UnknownAgent")
            if agent_name_to_load not in AGENT_REGISTRY:
                print(
                    f"Error: Agent name '{agent_name_to_load}' not found in AGENT_REGISTRY."
                )
                sys.exit(1)
            agent_class = AGENT_REGISTRY[agent_name_to_load]

            # Pass observation_space and action_space for agent re-instantiation
            # These are needed by load_model for some agents (like TableBasedAgent)
            # to reconstruct the spaces if not fully defined by the saved model.
            # DQNAgent also requires them.
            agent_params["observation_space"] = env.observation_space
            agent_params["action_space"] = env.action_space

            agent = agent_class.load_model(path=load_model_path, **agent_params)

            # Ensure agent is in exploitation mode if not training
            if hasattr(agent, "epsilon") and hasattr(agent, "epsilon_min"):
                agent.epsilon = agent.epsilon_min
        else:
            agent = create_agent(
                name=agent_config.get("name", "UnknownAgent"),
                params=agent_params,
                observation_space=env.observation_space,
                action_space=env.action_space,
            )

        # Pass mlflow_run_id to Runner so it can log metrics to the correct run
        runner_config["mlflow_run_id"] = mlflow_run_id
        runner = Runner(
            agent=agent,
            env=env,
            **runner_config,  # Unpack all runner params from config
        )

        # --- Run the Experiment ---
        try:
            runner.run()
        except Exception as e:
            print(f"Error during runner.run(): {e}")
            # Optionally, log error to MLflow
            mlflow.set_tag("run_status", "failed")
            mlflow.log_param("run_error", str(e)[:250])  # Log error message
            raise  # Re-raise the exception after logging
        else:
            mlflow.set_tag("run_status", "completed")

        # mlflow.end_run() is automatically called when exiting the 'with mlflow.start_run()' block
    print("MLflow run finished.")


if __name__ == "__main__":
    app.run(main)

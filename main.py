# main.py

"""
A unified entry point for running all reinforcement learning experiments,
now with MLflow integration and a modular network creation pipeline.
"""

import sys
import yaml
from typing import Dict, Any

from absl import app
from absl import flags

# MLflow import
import mlflow
import os  # For logging config file path

from agents.factory import create_agent, AGENT_REGISTRY
from env.factory import create_environment
from runner.runner import Runner

# Import the new network factory
from networks.factory import (
    create_network as create_nn_from_factory,
)  # Alias for clarity


# --- Command-Line Argument Definition ---
FLAGS = flags.FLAGS
flags.DEFINE_string("config", None, "Path to the YAML configuration file.")
flags.mark_flag_as_required("config")


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flattens a nested dictionary for MLflow parameter logging.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            if isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
    return dict(items)


def main(argv: Any) -> None:
    """
    The main function that sets up and runs the RL experiment with MLflow tracking
    and modular network creation.
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
        print(f"Error setting MLflow experiment: {e}.")

    with mlflow.start_run() as run:
        mlflow_run_id = run.info.run_id
        print(f"MLflow Run ID: {mlflow_run_id}")

        try:
            mlflow.log_artifact(FLAGS.config, artifact_path="configs")
            print(
                f"Logged configuration file '{FLAGS.config}' to MLflow artifacts (configs/{os.path.basename(FLAGS.config)})."
            )
        except Exception as e:
            print(f"Warning: Could not log config file to MLflow: {e}")

        try:
            flat_config = flatten_dict(config)
            params_to_log = {}
            for key, value in flat_config.items():
                if isinstance(value, (str, int, float, bool)):
                    if isinstance(value, str) and len(value) > 250:
                        params_to_log[key] = value[:247] + "..."
                    else:
                        params_to_log[key] = value
            if params_to_log:
                mlflow.log_params(params_to_log)
                print("Logged configuration parameters to MLflow.")
        except Exception as e:
            print(f"Warning: Could not log parameters to MLflow: {e}")

        env_config = config.get("environment", {})
        agent_config_from_yaml = config.get("agent", {})  # Keep original agent config
        runner_config = config.get("runner", {})

        # --- Object Creation ---
        env = create_environment(
            name=env_config.get("name", "UnknownEnv"),
            params=env_config.get("params", {}),
        )

        # Prepare agent parameters, separating network configs if present
        agent_name = agent_config_from_yaml.get("name", "UnknownAgent")
        agent_params = agent_config_from_yaml.get(
            "params", {}
        ).copy()  # Work with a copy

        load_model_path = runner_config.get("load_model_path")
        is_training = runner_config.get("is_training", True)
        agent_params["is_training"] = (
            is_training  # Add is_training for agents that use it in __init__
        )

        # --- Network Creation (if applicable, e.g., for DQNAgent) ---
        # Network configs are now expected under agent_params in YAML
        # e.g., agent.params.online_network_config: {name: "MLPNetwork", params: {...}}
        # or agent.params.network_config for single network agents that need it for load_model

        online_network_config = agent_params.pop("online_network_config", None)
        actor_network_config = agent_params.pop(
            "actor_network_config", None
        )  # For potential Actor-Critic
        critic_network_config = agent_params.pop(
            "critic_network_config", None
        )  # For potential Actor-Critic

        # Inject networks if their configs were provided
        if online_network_config:
            agent_params["online_network"] = create_nn_from_factory(
                name=online_network_config.get("name"),
                observation_space=env.observation_space,
                action_space=env.action_space,
                params=online_network_config.get("params", {}),
            )
            # For DQN, target network is usually identical to online initially
            agent_params["target_network"] = create_nn_from_factory(
                name=online_network_config.get("name"),
                observation_space=env.observation_space,
                action_space=env.action_space,
                params=online_network_config.get("params", {}),
            )

        if actor_network_config:  # For Actor-Critic type agents
            agent_params["actor_network"] = create_nn_from_factory(
                name=actor_network_config.get("name"),
                observation_space=env.observation_space,
                action_space=env.action_space,
                params=actor_network_config.get("params", {}),
            )
        if critic_network_config:  # For Actor-Critic type agents
            agent_params["critic_network"] = create_nn_from_factory(
                name=critic_network_config.get("name"),
                observation_space=env.observation_space,  # Critic often sees state
                action_space=gym.spaces.Discrete(1),  # Critic outputs a single value
                params=critic_network_config.get("params", {}),
            )

        # If loading a model, the agent's load_model method might need network_config
        # to reconstruct networks if they are not passed directly.
        # The DQNAgent.load_model was updated to expect network_config in **kwargs.
        if not is_training and load_model_path and agent_name == "DQNAgent":
            # If online_network_config was used to define the architecture for loading
            if online_network_config:
                agent_params["network_config"] = online_network_config
            # If a generic "network_config" was provided for loading, it's already in agent_params

        # --- Agent Creation / Loading ---
        if not is_training and load_model_path:
            print(
                f"Loading pre-trained agent '{agent_name}' from local path: {load_model_path}"
            )
            if agent_name not in AGENT_REGISTRY:
                print(f"Error: Agent name '{agent_name}' not found in AGENT_REGISTRY.")
                sys.exit(1)
            agent_class = AGENT_REGISTRY[agent_name]

            # Pass necessary spaces and all other agent_params (which might include network_config for DQNAgent.load_model)
            agent = agent_class.load_model(
                path=load_model_path,
                observation_space=env.observation_space,  # Essential for load_model
                action_space=env.action_space,  # Essential for load_model
                **agent_params,  # Contains other agent hyperparams, and potentially network_config
            )

            if hasattr(agent, "is_training"):
                agent.is_training = False  # Ensure eval mode
            if hasattr(agent, "epsilon") and hasattr(agent, "epsilon_min"):
                agent.epsilon = agent.epsilon_min
        else:
            agent = create_agent(
                name=agent_name,
                params=agent_params,  # Contains agent hyperparams and injected networks if any
                observation_space=env.observation_space,
                action_space=env.action_space,
            )

        runner_config["mlflow_run_id"] = mlflow_run_id
        runner = Runner(agent=agent, env=env, **runner_config)

        try:
            runner.run()
        except Exception as e:
            print(f"Error during runner.run(): {e}")
            mlflow.set_tag("run_status", "failed")
            mlflow.log_param("run_error", str(e)[:250])
            raise
        else:
            mlflow.set_tag("run_status", "completed")

    print("MLflow run finished.")


if __name__ == "__main__":
    app.run(main)

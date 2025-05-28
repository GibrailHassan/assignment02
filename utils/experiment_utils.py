# utils/experiment_utils.py

"""
Utility functions for managing and logging experiments, particularly with MLflow,
and for setting up experiment components like configurations, agents, and networks.
"""

import os
import sys  # For sys.exit
import yaml
from typing import Dict, Any, Tuple  # Added Tuple

import mlflow
from torch.utils.tensorboard import SummaryWriter
import gym

# Assuming factories are correctly imported where these functions are called,
# or we can pass them as arguments if we want to make utils even more generic.
# For now, we'll import them here if directly used, or assume they are used by the caller (main.py).
from agents.factory import create_agent as create_agent_from_factory, AGENT_REGISTRY
from env.factory import create_environment as create_env_from_factory
from networks.factory import create_network as create_nn_from_factory
from agents.abstractAgent import AbstractAgent  # For type hinting


def flatten_dict_for_mlflow(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flattens a nested dictionary for MLflow parameter logging.
    Converts lists/tuples to strings for compatibility.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict_for_mlflow(v, new_key, sep=sep).items())
        else:
            if isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))
            elif isinstance(v, (str, int, float, bool)):
                items.append((new_key, v))
    return dict(items)


def load_config(config_file_path: str) -> Dict[str, Any] | None:
    """
    Loads and parses a YAML configuration file.

    Args:
        config_file_path (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any] | None: The loaded configuration dictionary, or None if loading fails.
    """
    print(f"Loading configuration from: {config_file_path}")
    try:
        with open(config_file_path, "r") as f:
            config: Dict[str, Any] = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}")
        return None


def setup_mlflow_run(config: Dict[str, Any], config_file_path: str) -> str | None:
    """
    Initializes an MLflow experiment and run, logging configuration.
    To be called within a `with mlflow.start_run():` block.

    Args:
        config (Dict[str, Any]): The loaded YAML configuration dictionary.
        config_file_path (str): The path to the YAML configuration file.

    Returns:
        str | None: The MLflow run ID if successful, otherwise None.
    """
    mlflow_experiment_name = config.get("experiment_name", "Default_RL_Experiment")
    try:
        # Experiment setting should ideally be outside the run context if it needs creation,
        # but set_experiment is idempotent.
        mlflow.set_experiment(mlflow_experiment_name)
        print(f"MLflow experiment set to: '{mlflow_experiment_name}'")
    except Exception as e:
        print(
            f"Error setting MLflow experiment: {e}. MLflow tracking might be disabled for this run."
        )
        return None

    active_run = mlflow.active_run()
    if not active_run:
        print(
            "Error: No active MLflow run found. `setup_mlflow_run` should be called within `with mlflow.start_run():`."
        )
        return None

    mlflow_run_id = active_run.info.run_id
    print(f"Logging to MLflow Run ID: {mlflow_run_id}")

    try:
        mlflow.log_artifact(config_file_path, artifact_path="configs")
        print(
            f"Logged configuration file '{config_file_path}' to MLflow artifacts (configs/{os.path.basename(config_file_path)})."
        )
    except Exception as e:
        print(f"Warning: Could not log config file to MLflow: {e}")

    try:
        flat_config = flatten_dict_for_mlflow(config)
        params_to_log = {}
        for key, value in flat_config.items():
            if isinstance(value, str) and len(value) > 250:
                params_to_log[key] = value[:247] + "..."
            elif isinstance(value, (str, int, float, bool)):
                params_to_log[key] = value

        if params_to_log:
            mlflow.log_params(params_to_log)
            print("Logged configuration parameters to MLflow.")
    except Exception as e:
        print(f"Warning: Could not log parameters to MLflow: {e}")

    return mlflow_run_id


def prepare_agent_networks_and_params(
    agent_config_yaml: Dict[str, Any], env: gym.Env, is_training: bool
) -> Dict[str, Any]:
    """
    Prepares agent parameters, including creating neural networks if specified in the config.

    Args:
        agent_config_yaml (Dict[str, Any]): The 'agent' section from the loaded YAML config.
        env (gym.Env): The initialized environment instance.
        is_training (bool): Flag indicating if the agent is being set up for training.

    Returns:
        Dict[str, Any]: Parameters ready to be passed to agent factory or load_model.
    """
    agent_params = agent_config_yaml.get("params", {}).copy()
    agent_params["is_training"] = is_training  # Add/overwrite is_training status

    # Network Creation (if applicable, e.g., for DQNAgent)
    online_network_config = agent_params.pop("online_network_config", None)
    actor_network_config = agent_params.pop("actor_network_config", None)
    critic_network_config = agent_params.pop("critic_network_config", None)

    if online_network_config:
        print(f"Creating online network: {online_network_config.get('name')}")
        agent_params["online_network"] = create_nn_from_factory(
            name=online_network_config.get("name"),
            observation_space=env.observation_space,
            action_space=env.action_space,
            params=online_network_config.get("params", {}),
        )
        print(
            f"Creating target network (identical to online): {online_network_config.get('name')}"
        )
        agent_params["target_network"] = create_nn_from_factory(
            name=online_network_config.get("name"),
            observation_space=env.observation_space,
            action_space=env.action_space,
            params=online_network_config.get("params", {}),
        )

    if actor_network_config:
        print(f"Creating actor network: {actor_network_config.get('name')}")
        agent_params["actor_network"] = create_nn_from_factory(
            name=actor_network_config.get("name"),
            observation_space=env.observation_space,
            action_space=env.action_space,
            params=actor_network_config.get("params", {}),
        )
    if critic_network_config:
        print(f"Creating critic network: {critic_network_config.get('name')}")
        # Critic network might have a different action_space (e.g., outputting a single value)
        # This needs to be handled based on the specific agent's needs.
        # For a simple value critic, action_space might be Discrete(1).
        critic_action_space = gym.spaces.Discrete(1)  # Example
        agent_params["critic_network"] = create_nn_from_factory(
            name=critic_network_config.get("name"),
            observation_space=env.observation_space,
            action_space=critic_action_space,
            params=critic_network_config.get("params", {}),
        )

    # For loading a DQNAgent model, ensure network_config is available if needed by load_model
    # This is for DQNAgent.load_model to reconstruct networks if they weren't passed directly.
    agent_name = agent_config_yaml.get("name")
    if not is_training and agent_name == "DQNAgent":
        if online_network_config and "network_config" not in agent_params:
            agent_params["network_config"] = online_network_config
        elif (
            "network_config" not in agent_params and online_network_config is None
        ):  # Check if it's missing entirely
            print(
                "Warning: Attempting to load DQNAgent without 'network_config' or 'online_network_config' "
                "in agent.params. DQNAgent.load_model might fail if it needs to reconstruct networks."
            )

    return agent_params


def initialize_agent(
    agent_config_yaml: Dict[str, Any],
    env: gym.Env,
    runner_config: Dict[str, Any],
    prepared_agent_params: Dict[str, Any],
) -> AbstractAgent | None:
    """
    Creates a new agent or loads a pre-trained one.

    Args:
        agent_config_yaml (Dict[str, Any]): The 'agent' section from YAML.
        env (gym.Env): The initialized environment.
        runner_config (Dict[str, Any]): The 'runner' section from YAML.
        prepared_agent_params (Dict[str, Any]): Agent parameters, potentially including networks.

    Returns:
        AbstractAgent | None: The initialized agent, or None on failure.
    """
    agent_name = agent_config_yaml.get("name", "UnknownAgent")
    load_model_path = runner_config.get("load_model_path")
    is_training = runner_config.get("is_training", True)  # Get from runner_config

    if not is_training and load_model_path:
        print(
            f"Loading pre-trained agent '{agent_name}' from local path: {load_model_path}"
        )
        if agent_name not in AGENT_REGISTRY:
            print(
                f"Error: Agent name '{agent_name}' not found in AGENT_REGISTRY for loading."
            )
            if mlflow.active_run():
                mlflow.set_tag("run_status", "config_error_agent_load")
            return None
        agent_class = AGENT_REGISTRY[agent_name]

        try:
            # Pass observation_space and action_space explicitly as they are fundamental
            # prepared_agent_params already contains other hyperparams and potentially network_config
            agent = agent_class.load_model(
                path=load_model_path,
                observation_space=env.observation_space,
                action_space=env.action_space,
                **prepared_agent_params,
            )
        except Exception as e:
            print(f"Error loading agent model: {e}")
            if mlflow.active_run():
                mlflow.set_tag("run_status", "load_model_error")
                mlflow.log_param("load_model_error_msg", str(e)[:250])
            return None  # Propagate failure

        if hasattr(agent, "is_training"):
            agent.is_training = False
        if hasattr(agent, "epsilon") and hasattr(agent, "epsilon_min"):
            agent.epsilon = agent.epsilon_min
        print("Agent loaded successfully for evaluation.")
    else:
        print(f"Creating new agent: '{agent_name}' for training.")
        agent = create_agent_from_factory(
            name=agent_name,
            params=prepared_agent_params,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        print("New agent created successfully for training.")
    return agent


def log_metrics_to_tensorboard(
    writer: SummaryWriter | None,
    metrics: Dict[str, Any],  # Value can be Any, but only int/float logged
    episode_num: int,
    prefix: str = "",
) -> None:
    """Logs a dictionary of metrics to TensorBoard."""
    if writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}{key}", value, episode_num)


def log_metrics_to_mlflow(
    metrics: Dict[str, Any],  # Value can be Any, but only int/float logged
    episode_num: int,
    prefix: str = "",
) -> None:
    """Logs a dictionary of metrics to the active MLflow run."""
    if not mlflow.active_run():
        return

    try:
        metrics_to_log = {
            f"{prefix}{key.replace('/', '_')}": value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        if metrics_to_log:
            mlflow.log_metrics(metrics_to_log, step=episode_num)
    except Exception as e:
        print(f"Warning: MLflow metric logging failed: {e}")

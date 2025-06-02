"""
Utility functions for managing and logging experiments, particularly with MLflow,
and for setting up experiment components like configurations, agents, and networks.
"""

import os
import yaml
from typing import Dict, Any

import mlflow
from torch.utils.tensorboard import SummaryWriter
import gym

# Factories and Base Agent Class
from agents.factory import create_agent as create_agent_from_factory, AGENT_REGISTRY

# from env.factory import create_environment as create_env_from_factory # Not used directly in this file
from networks.factory import create_network as create_nn_from_factory
from agents.abstractAgent import AbstractAgent


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
    """
    mlflow_experiment_name = config.get("experiment_name", "Default_RL_Experiment")
    try:
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
    # print(f"Logging to MLflow Run ID: {mlflow_run_id}") # main.py prints this

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


def prepare_agent_constructor_params(
    agent_config_yaml: Dict[str, Any],
    env: gym.Env,
    is_training: bool,  # Pass is_training explicitly
) -> Dict[str, Any]:
    """
    Prepares the dictionary of parameters for the agent's constructor.
    This includes creating neural networks if specified in the agent's YAML config.

    Args:
        agent_config_yaml (Dict[str, Any]): The 'agent' section from the loaded YAML config.
        env (gym.Env): The initialized environment instance.
        is_training (bool): Flag indicating if the agent is being set up for training.

    Returns:
        Dict[str, Any]: Parameters ready to be passed to the agent factory
                        or the agent's load_model method's **kwargs.
    """
    agent_constructor_params = agent_config_yaml.get("params", {}).copy()
    agent_constructor_params["is_training"] = is_training

    # Network Creation (if applicable, e.g., for DQNAgent)
    online_network_config = agent_constructor_params.pop("online_network_config", None)
    actor_network_config = agent_constructor_params.pop("actor_network_config", None)
    critic_network_config = agent_constructor_params.pop("critic_network_config", None)

    if online_network_config:
        print(
            f"Preparing online network from config: {online_network_config.get('name')}"
        )
        agent_constructor_params["online_network"] = create_nn_from_factory(
            name=online_network_config.get("name"),
            observation_space=env.observation_space,
            action_space=env.action_space,
            params=online_network_config.get("params", {}),
        )
        print(
            f"Preparing target network (identical to online): {online_network_config.get('name')}"
        )
        agent_constructor_params["target_network"] = create_nn_from_factory(
            name=online_network_config.get("name"),
            observation_space=env.observation_space,
            action_space=env.action_space,
            params=online_network_config.get("params", {}),
        )

    if actor_network_config:
        print(
            f"Preparing actor network from config: {actor_network_config.get('name')}"
        )
        agent_constructor_params["actor_network"] = create_nn_from_factory(
            name=actor_network_config.get("name"),
            observation_space=env.observation_space,
            action_space=env.action_space,
            params=actor_network_config.get("params", {}),
        )
    if critic_network_config:
        print(
            f"Preparing critic network from config: {critic_network_config.get('name')}"
        )
        critic_action_space = gym.spaces.Discrete(1)
        agent_constructor_params["critic_network"] = create_nn_from_factory(
            name=critic_network_config.get("name"),
            observation_space=env.observation_space,
            action_space=critic_action_space,
            params=critic_network_config.get("params", {}),
        )

    # For loading a DQNAgent model, ensure 'network_config' is present in agent_constructor_params
    # if it's needed by DQNAgent.load_model to reconstruct networks.
    agent_name = agent_config_yaml.get("name")
    if not is_training and agent_name == "DQNAgent":
        # If online_network_config was defined (meaning networks are part of this agent type)
        # and network_config isn't already explicitly set in YAML's agent.params for loading.
        if online_network_config and "network_config" not in agent_constructor_params:
            agent_constructor_params["network_config"] = online_network_config
        elif (
            "network_config" not in agent_constructor_params
            and online_network_config is None
        ):
            # This case might arise if loading a DQN model whose YAML didn't have online_network_config
            # but load_model still needs network_config. This utility assumes network_config
            # should be derived from online_network_config if not explicitly provided for loading.
            print(
                "Warning: Attempting to load DQNAgent. 'network_config' for model loading "
                "was not found in agent.params and could not be derived from 'online_network_config'. "
                "DQNAgent.load_model might fail if it needs to reconstruct networks."
            )

    return agent_constructor_params


def initialize_agent(
    agent_name: str,  # Pass agent_name directly
    runner_config: Dict[str, Any],
    env: gym.Env,
    agent_constructor_params: Dict[str, Any],  # Already contains networks if any
) -> AbstractAgent | None:
    """
    Creates a new agent or loads a pre-trained one.

    Args:
        agent_name (str): The name of the agent (from YAML config).
        runner_config (Dict[str, Any]): The 'runner' section from YAML (for load_model_path, is_training).
        env (gym.Env): The initialized environment.
        agent_constructor_params (Dict[str, Any]): Agent parameters, including any pre-built networks.

    Returns:
        AbstractAgent | None: The initialized agent, or None on failure.
    """
    load_model_path = runner_config.get("load_model_path")
    is_training = runner_config.get("is_training", True)  # Get from runner_config

    agent: AbstractAgent | None = None

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
            # Pass observation_space and action_space explicitly as they are fundamental for load_model
            # agent_constructor_params already contains other hyperparams and potentially network_config
            agent = agent_class.load_model(
                path=load_model_path,
                observation_space=env.observation_space,
                action_space=env.action_space,
                **agent_constructor_params,
            )
        except Exception as e:
            print(f"Error loading agent model: {e}")
            if mlflow.active_run():
                mlflow.set_tag("run_status", "load_model_error")
                mlflow.log_param("load_model_error_msg", str(e)[:250])
            return None

        if agent:  # Ensure agent was loaded successfully
            if hasattr(agent, "is_training"):
                agent.is_training = False
            if hasattr(agent, "epsilon") and hasattr(agent, "epsilon_min"):
                agent.epsilon = agent.epsilon_min
            print("Agent loaded successfully for evaluation.")
    else:  # Training or evaluation without loading a model
        mode = "training" if is_training else "evaluation (new instance)"
        print(f"Creating new agent: '{agent_name}' for {mode}.")
        try:
            agent = create_agent_from_factory(
                name=agent_name,
                params=agent_constructor_params,
                observation_space=env.observation_space,
                action_space=env.action_space,
            )
            print(f"New agent '{agent_name}' created successfully for {mode}.")
        except Exception as e:
            print(f"Error creating new agent '{agent_name}': {e}")
            if mlflow.active_run():
                mlflow.set_tag("run_status", "agent_creation_error")
                mlflow.log_param("agent_creation_error_msg", str(e)[:250])
            return None

    return agent


def log_metrics_to_tensorboard(
    writer: SummaryWriter | None,
    metrics: Dict[str, Any],
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
    metrics: Dict[str, Any], episode_num: int, prefix: str = ""
) -> None:
    """Logs a dictionary of metrics to the active MLflow run."""
    if not mlflow.active_run():
        return

    try:
        # Sanitize keys for MLflow (dots are ok, slashes might not be ideal)
        metrics_to_log = {
            f"{prefix}{key.replace('/', '_')}": value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        if metrics_to_log:
            mlflow.log_metrics(metrics_to_log, step=episode_num)
    except Exception as e:
        print(f"Warning: MLflow metric logging failed: {e}")

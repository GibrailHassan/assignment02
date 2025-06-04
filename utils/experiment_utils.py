# utils/experiment_utils.py
"""
Utility functions for managing and logging experiments, particularly with MLflow,
and for setting up experiment components like configurations, agents, and networks.

This module aims to centralize common tasks related to experiment setup,
configuration management, MLflow integration, and component initialization,
thereby keeping the main experiment scripts cleaner and more focused on the
core logic. It includes functions for loading configurations, ensuring MLflow
experiments exist, logging run details, preparing agent dependencies (including
network instantiation), and initializing agents (either new or loaded from disk).
"""

import os
import sys  # For sys.exit in helper functions
import yaml  # For parsing YAML configuration files
from typing import Dict, Any, Optional

import mlflow  # For MLflow integration
from torch.utils.tensorboard import SummaryWriter  # For type hinting if passed around
import gym  # For gym.Env and gym.spaces type hints

# Factories and Base Agent Class
from agents.factory import create_agent as create_agent_from_factory, AGENT_REGISTRY
from networks.factory import create_network as create_nn_from_factory
from agents.abstractAgent import AbstractAgent


def load_config_or_exit(config_file_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file. Prints an error message and exits the
    program if loading fails for any reason (e.g., file not found, invalid YAML).

    Args:
        config_file_path (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The loaded configuration dictionary if successful.
                        The program will exit if loading fails.
    """
    print(f"Attempting to load configuration from: {config_file_path}")
    try:
        with open(config_file_path, "r") as f:
            config: Optional[Dict[str, Any]] = yaml.safe_load(f)
        if config is None:  # Handles cases like an empty YAML file
            print(
                f"Error: Configuration file '{config_file_path}' is empty or parsed as None (invalid content?)."
            )
            sys.exit(1)  # Exit with an error code
        print(f"Configuration loaded successfully from '{config_file_path}'.")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_file_path}'.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file '{config_file_path}': {e}")
        sys.exit(1)
    except (
        Exception
    ) as e:  # Catch any other unexpected errors during file loading/parsing
        print(
            f"An unexpected error occurred while loading config '{config_file_path}': {e}"
        )
        sys.exit(1)


def ensure_mlflow_experiment(experiment_name: str) -> Optional[str]:
    """
    Ensures that the specified MLflow experiment exists, creating it if necessary.
    Also sets this experiment as the active one for the current MLflow client context.

    Args:
        experiment_name (str): The desired name for the MLflow experiment.

    Returns:
        Optional[str]: The ID of the MLflow experiment if it was successfully found
                       or created. Returns None if an error occurred during the
                       process, in which case MLflow might default to another
                       experiment or logging might fail.
    """
    experiment_id: Optional[str] = None
    print(f"\n--- Ensuring MLflow Experiment: '{experiment_name}' ---")
    try:
        # Attempt to get the experiment by its name
        experiment: Optional[mlflow.entities.Experiment] = (
            mlflow.get_experiment_by_name(experiment_name)
        )

        if experiment is None:  # If the experiment does not exist
            print(
                f"MLflow experiment '{experiment_name}' does not exist. Attempting to create it."
            )
            # Create the new experiment
            experiment_id = mlflow.create_experiment(experiment_name)
            print(
                f"Successfully created MLflow experiment '{experiment_name}' with ID: {experiment_id}"
            )
        else:  # If the experiment already exists
            experiment_id = experiment.experiment_id
            print(
                f"Using existing MLflow experiment: '{experiment_name}' (ID: {experiment_id})"
            )

        # Set the retrieved or newly created experiment as the active one for MLflow operations
        mlflow.set_experiment(experiment_name=experiment_name)
        print(f"MLflow active experiment set to: '{experiment_name}'.")

    except (
        mlflow.exceptions.MlflowException
    ) as e_mlflow_exp:  # Catch MLflow specific exceptions
        print(
            f"MLflow API Error during setup of experiment '{experiment_name}': {e_mlflow_exp}"
        )
        print(
            "MLflow operations might be affected or fall back to a default experiment."
        )
    except Exception as e_generic_exp:  # Catch any other unexpected errors
        print(
            f"An unexpected error occurred during MLflow experiment setup for '{experiment_name}': {e_generic_exp}"
        )

    if experiment_id is None:
        print(
            f"Warning: Could not obtain a valid experiment ID for '{experiment_name}'. Subsequent MLflow runs might be logged to the 'Default' experiment or fail if MLflow interaction is critical."
        )
    return experiment_id


def flatten_dict_for_mlflow(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flattens a nested dictionary for MLflow parameter logging.
    (Content of this function remains the same as previously documented - crucial for MLflow params)
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict_for_mlflow(v, new_key, sep=sep).items())
        else:
            if isinstance(v, (list, tuple)):  # Convert lists/tuples to strings
                items.append((new_key, str(v)))
            elif isinstance(v, (str, int, float, bool)):  # MLflow compatible types
                items.append((new_key, v))
            # Other types are silently ignored for parameter logging
    return dict(items)


def log_run_params_and_artifacts(
    config: Dict[str, Any], config_file_path: str, run_id: str
) -> None:
    """
    Logs configuration parameters and the original config file as an artifact
    to the currently active MLflow run.

    This function assumes that an MLflow run has already been started (e.g., via
    `with mlflow.start_run():`) and that `run_id` is the ID of this active run.

    Args:
        config (Dict[str, Any]): The full configuration dictionary for the experiment.
        config_file_path (str): The file path to the original YAML configuration file.
        run_id (str): The ID of the currently active MLflow run (for logging context).
    """
    print(f"\n--- Logging Parameters and Artifacts for MLflow Run ID: {run_id} ---")

    # Log the original configuration file as an artifact
    try:
        # Define a subdirectory within artifacts for better organization
        artifact_config_dir: str = "configs"
        mlflow.log_artifact(config_file_path, artifact_path=artifact_config_dir)
        print(
            f"Logged configuration file '{os.path.basename(config_file_path)}' to MLflow artifacts (under '{artifact_config_dir}/')."
        )
    except Exception as e:
        print(
            f"Warning: Could not log configuration file '{config_file_path}' as MLflow artifact: {e}"
        )

    # Log flattened configuration parameters to MLflow
    try:
        flat_config: Dict[str, Any] = flatten_dict_for_mlflow(config)
        params_to_log: Dict[str, Any] = {}
        for key, value in flat_config.items():
            # Truncate very long string values to avoid MLflow errors (limit is typically 250 chars)
            if isinstance(value, str) and len(value) > 250:
                params_to_log[key] = value[:247] + "..."  # Truncate and add ellipsis
            elif isinstance(
                value, (str, int, float, bool)
            ):  # Only log MLflow-compatible types
                params_to_log[key] = value
            # Other complex types (like None, or nested objects not flattened) are skipped for `log_params`

        if params_to_log:
            mlflow.log_params(params_to_log)
            print("Logged flattened configuration parameters to MLflow.")
        else:
            print("No suitable parameters found to log to MLflow after flattening.")
    except Exception as e:
        print(f"Warning: Could not log parameters to MLflow: {e}")


def prepare_agent_dependencies(
    agent_config_yaml: Dict[str, Any],  # The 'agent' section from the main config dict
    env: gym.Env,  # The initialized environment instance
    is_training: bool,  # Flag: True for training, False for evaluation
) -> Dict[str, Any]:
    """
    Prepares the dictionary of parameters required for agent instantiation.

    This function is responsible for:
    1. Extracting agent-specific hyperparameters from `agent_config_yaml['params']`.
    2. Instantiating neural networks (e.g., online/target for DQN, policy for REINFORCE,
       actor/critic for A2C) if their configurations are specified in `agent_config_yaml['params']`
       (e.g., `online_network_config`, `policy_network_config`, etc.).
    3. Adding these instantiated network objects to the parameter dictionary.
    4. Handling specific configuration needs for loading certain agent types (e.g.,
       ensuring `network_config` or `policy_network_config` as YAML dicts are
       available if the agent's `load_model` method requires them for reconstruction).

    Args:
        agent_config_yaml (Dict[str, Any]): The 'agent' section from the main configuration.
                                           Example: `config['agent']`.
        env (gym.Env): The initialized environment, used to determine network input/output shapes.
        is_training (bool): Indicates if the agent is being set up for training or evaluation.

    Returns:
        Dict[str, Any]: A dictionary containing all parameters (hyperparameters and
                        instantiated network objects) ready to be passed to the
                        agent's constructor or its `load_model` method.
    """
    print("\n--- Preparing Agent Dependencies (Parameters & Neural Networks) ---")
    agent_name_from_yaml: str = agent_config_yaml.get("name", "UnknownAgent")
    # Start with a copy of parameters from `agent_config_yaml['params']`
    agent_constructor_params: Dict[str, Any] = agent_config_yaml.get(
        "params", {}
    ).copy()

    # Inject the `is_training` flag, as it's crucial for agent setup and behavior
    agent_constructor_params["is_training"] = is_training
    print(
        f"Agent '{agent_name_from_yaml}' will be set up in {'TRAINING' if is_training else 'EVALUATION'} mode."
    )

    # --- Pop all potential network configurations from agent_constructor_params ---
    # These are YAML dictionaries defining network architectures, not the nn.Module objects.
    # We pop them so they aren't passed directly as `**params` if the agent expects objects.
    online_network_config: Optional[Dict] = agent_constructor_params.pop(
        "online_network_config", None
    )
    actor_network_config: Optional[Dict] = agent_constructor_params.pop(
        "actor_network_config", None
    )
    critic_network_config: Optional[Dict] = agent_constructor_params.pop(
        "critic_network_config", None
    )
    policy_network_config: Optional[Dict] = agent_constructor_params.pop(
        "policy_network_config", None
    )  # For REINFORCE, etc.

    # --- Special Handling for Loading: Pass original YAML network configs if needed by load_model ---
    # Some agent `load_model` methods might need the original YAML config dict for a network
    # (e.g., `network_config` for DQNAgent) to reconstruct the architecture before loading weights.
    # These are added to `agent_constructor_params` under specific keys expected by `load_model`.
    if not is_training:  # Only relevant when loading models (i.e., not training)
        if agent_name_from_yaml == "DQNAgent":
            # DQNAgent.load_model might expect 'network_config' (the YAML dict).
            # If 'online_network_config' was used during training, pass it as 'network_config' for loading.
            # Or, if 'network_config' was directly in the agent's params in YAML, that's fine too.
            dqn_net_config_for_load = online_network_config or agent_config_yaml.get(
                "params", {}
            ).get("network_config")
            if (
                dqn_net_config_for_load
                and "network_config" not in agent_constructor_params
            ):
                agent_constructor_params["network_config"] = dqn_net_config_for_load
                print(
                    "Added 'network_config' (YAML dict) to params for DQNAgent loading."
                )
            elif not dqn_net_config_for_load:
                print(
                    "Warning: DQNAgent loading - No 'network_config' or 'online_network_config' found in YAML to aid network reconstruction."
                )

        elif agent_name_from_yaml == "REINFORCEAgent":  # Example for REINFORCE
            # REINFORCEAgent.load_model might expect 'policy_network_config' (the YAML dict).
            if (
                policy_network_config
                and "policy_network_config" not in agent_constructor_params
            ):
                agent_constructor_params["policy_network_config"] = (
                    policy_network_config
                )
                print(
                    "Added 'policy_network_config' (YAML dict) to params for REINFORCEAgent loading."
                )
            elif not policy_network_config:
                print(
                    "Warning: REINFORCEAgent loading - No 'policy_network_config' found in YAML to aid network reconstruction."
                )

        # Add similar blocks for ActorCritic or other agents if their load_model needs specific YAML configs.

    # --- Instantiate and Inject Network Objects into agent_constructor_params ---
    # These instantiated nn.Module objects will be passed to the agent's __init__.
    if online_network_config:  # For DQN, DDQN
        net_name = online_network_config.get("name", "UnnamedOnlineNet")
        print(f"Instantiating online network: '{net_name}'...")
        agent_constructor_params["online_network"] = create_nn_from_factory(
            name=net_name,
            observation_space=env.observation_space,
            action_space=env.action_space,
            params=online_network_config.get("params", {}),
        )
        print(f"Instantiating target network (from '{net_name}' config)...")
        agent_constructor_params["target_network"] = create_nn_from_factory(
            name=net_name,
            observation_space=env.observation_space,
            action_space=env.action_space,
            params=online_network_config.get("params", {}),
        )

    if actor_network_config:  # For Actor-Critic methods
        net_name = actor_network_config.get("name", "UnnamedActorNet")
        print(f"Instantiating actor network: '{net_name}'...")
        agent_constructor_params["actor_network"] = create_nn_from_factory(
            name=net_name,
            observation_space=env.observation_space,
            action_space=env.action_space,
            params=actor_network_config.get("params", {}),
        )

    if critic_network_config:  # For Actor-Critic methods
        net_name = critic_network_config.get("name", "UnnamedCriticNet")
        print(f"Instantiating critic network: '{net_name}'...")
        # Critic network typically outputs a single value (state value).
        # Its 'action_space' for output definition can be a dummy gym.spaces.Discrete(1).
        critic_output_action_space = gym.spaces.Discrete(1)
        agent_constructor_params["critic_network"] = create_nn_from_factory(
            name=net_name,
            observation_space=env.observation_space,
            action_space=critic_output_action_space,
            params=critic_network_config.get("params", {}),
        )

    if (
        policy_network_config
    ):  # For REINFORCE or policy-based part of some Actor-Critics
        net_name = policy_network_config.get("name", "UnnamedPolicyNet")
        print(f"Instantiating policy network: '{net_name}'...")
        agent_constructor_params["policy_network"] = create_nn_from_factory(
            name=net_name,
            observation_space=env.observation_space,
            action_space=env.action_space,  # Output layer sized by env.action_space.n
            params=policy_network_config.get("params", {}),
        )

    print(
        "Agent dependencies (parameters and instantiated networks) prepared successfully."
    )
    return agent_constructor_params


def initialize_agent_or_exit(
    agent_name: str,  # Name of the agent (key from AGENT_REGISTRY)
    agent_config_yaml: Dict[str, Any],  # The full 'agent' section from original YAML
    runner_config: Dict[str, Any],  # The 'runner' section from config
    env: gym.Env,  # The initialized environment instance
    prepared_agent_params: Dict[
        str, Any
    ],  # Params including instantiated networks (from prepare_agent_dependencies)
) -> AbstractAgent:
    """
    Initializes an agent by either loading a pre-trained model (if in evaluation
    mode and `load_model_path` is provided) or creating a new agent instance.
    Prints error messages and exits the program if agent initialization fails.

    Args:
        agent_name (str): The registered name of the agent.
        agent_config_yaml (Dict[str, Any]): The 'agent' section from the original YAML config.
                                           This is passed to allow `load_model` methods to
                                           access original YAML network configs if needed for reconstruction.
        runner_config (Dict[str, Any]): The 'runner' section from the YAML config, containing
                                       `is_training` and `load_model_path`.
        env (gym.Env): The initialized environment.
        prepared_agent_params (Dict[str, Any]): The dictionary of parameters (including
                                               any instantiated network objects) to be
                                               passed to the agent's constructor or
                                               `load_model` method.

    Returns:
        AbstractAgent: The successfully initialized (loaded or newly created) agent instance.
                       The program will exit if initialization fails.
    """
    print(f"\n--- Initializing Agent: '{agent_name}' ---")
    load_model_path: Optional[str] = runner_config.get("load_model_path")
    # `is_training` should already be in `prepared_agent_params`, but also check `runner_config` for clarity.
    is_training: bool = runner_config.get("is_training", True)

    agent: Optional[AbstractAgent] = None

    if not is_training and load_model_path:  # Evaluation mode and model path provided
        print(
            f"Attempting to load pre-trained agent '{agent_name}' from path: {load_model_path}"
        )
        if agent_name not in AGENT_REGISTRY:
            print(
                f"Error: Agent name '{agent_name}' not found in AGENT_REGISTRY. Cannot load model."
            )
            mlflow.set_tag(
                "run_status", "config_error_agent_load_unknown_registry"
            )  # More specific tag
            sys.exit(1)

        agent_class = AGENT_REGISTRY[agent_name]

        # Kwargs for agent_class.load_model must include everything it needs.
        # `prepared_agent_params` already has `is_training` and instantiated networks (if any).
        # It might also have specific YAML config dicts like `network_config` or `policy_network_config`
        # added by `prepare_agent_dependencies` if the agent's load_model needs them.
        load_model_kwargs = prepared_agent_params.copy()
        # Ensure original YAML network configs are passed if load_model needs them for reconstruction
        # This logic is now more robustly handled within prepare_agent_dependencies for specific agent types
        # if agent_name == "DQNAgent":
        #     dqn_net_config_yaml = agent_config_yaml.get("params", {}).get("online_network_config") or \
        #                           agent_config_yaml.get("params", {}).get("network_config")
        #     if dqn_net_config_yaml and "network_config" not in load_model_kwargs:
        #         load_model_kwargs["network_config"] = dqn_net_config_yaml
        # elif agent_name == "REINFORCEAgent":
        #      policy_net_config_yaml = agent_config_yaml.get("params", {}).get("policy_network_config")
        #      if policy_net_config_yaml and "policy_network_config" not in load_model_kwargs:
        #         load_model_kwargs["policy_network_config"] = policy_net_config_yaml

        try:
            agent = agent_class.load_model(
                path=load_model_path,
                observation_space=env.observation_space,
                action_space=env.action_space,
                **load_model_kwargs,  # This passes all prepared params, including networks and specific configs
            )
        except FileNotFoundError as e_fnf:
            print(
                f"Fatal Error: Model file not found for agent '{agent_name}' during load. Path: {e_fnf.filename}. Details: {e_fnf}"
            )
            mlflow.set_tag("run_status", "load_model_fatal_file_not_found")
            mlflow.log_param("load_model_error_msg", str(e_fnf)[:250])
            sys.exit(1)
        except Exception as e_load:  # Catch other errors during load_model
            print(
                f"Fatal Error: Failed to load agent model for '{agent_name}'. Details: {e_load}"
            )
            mlflow.set_tag("run_status", "load_model_fatal_generic_error")
            mlflow.log_param("load_model_error_msg", str(e_load)[:250])
            sys.exit(1)  # Exit on any load failure

        if agent:
            print(f"Agent '{agent_name}' loaded successfully for evaluation.")
            # The agent's load_model method should handle setting internal flags like is_training=False
            # and adjusting exploration parameters (e.g., epsilon to epsilon_min).
        else:  # Should be caught by exceptions if load_model fails
            print(
                f"CRITICAL: Agent '{agent_name}' is None after load_model call, despite no explicit exception. This indicates a problem in load_model implementation."
            )
            mlflow.set_tag("run_status", "load_model_failed_silently_post_call")
            sys.exit(1)

    else:  # Training mode, or evaluation mode but no model path provided (create new instance)
        mode_description = (
            "training" if is_training else "evaluation (new instance without loading)"
        )
        print(f"Creating new agent instance: '{agent_name}' for {mode_description}.")
        try:
            # Use the agent factory. `prepared_agent_params` includes instantiated networks.
            agent = create_agent_from_factory(
                name=agent_name,
                params=prepared_agent_params,  # Pass all prepared params (hyperparams + networks)
                observation_space=env.observation_space,
                action_space=env.action_space,
            )
        except Exception as e_create:  # Catch errors during agent creation
            print(
                f"Fatal Error: Failed to create new agent '{agent_name}'. Details: {e_create}"
            )
            mlflow.set_tag("run_status", "agent_creation_fatal_error")
            mlflow.log_param("agent_creation_error_msg", str(e_create)[:250])
            sys.exit(1)

        if agent:
            print(
                f"New agent '{agent_name}' created successfully for {mode_description}."
            )
        else:  # Should be caught by exceptions
            print(
                f"CRITICAL: Agent '{agent_name}' is None after create_agent_from_factory call, despite no explicit exception."
            )
            mlflow.set_tag("run_status", "agent_creation_failed_silently_post_call")
            sys.exit(1)

    # Final check, though sys.exit should have been called on failure
    if agent is None:
        print("CRITICAL ERROR: Agent initialization resulted in None. Exiting.")
        sys.exit(1)

    return agent


# --- Metric Logging Utilities (TensorBoard and MLflow) ---
def log_metrics_to_tensorboard(
    writer: Optional[SummaryWriter],  # TensorBoard SummaryWriter instance
    metrics: Dict[str, Any],  # Dictionary of metrics {name: value}
    episode_num: int,  # Global step (episode number) for logging
    prefix: str = "",  # Optional prefix for metric names (e.g., "Agent/")
) -> None:
    """Logs a dictionary of numerical metrics to TensorBoard."""
    if writer is None:
        return  # Do nothing if no TensorBoard writer is provided

    for key, value in metrics.items():
        if isinstance(value, (int, float)):  # Log only numerical values
            writer.add_scalar(f"{prefix}{key}", value, global_step=episode_num)
        # Non-numerical metrics are silently ignored by this function for add_scalar


def log_metrics_to_mlflow(
    metrics: Dict[str, Any],  # Dictionary of metrics {name: value}
    episode_num: int,  # Step (episode number) for logging
    prefix: str = "",  # Optional prefix for metric names
) -> None:
    """Logs a dictionary of numerical metrics to the active MLflow run."""
    if not mlflow.active_run():
        # print("No active MLflow run, skipping MLflow metric logging.") # Can be noisy if called frequently
        return  # Do nothing if no MLflow run is active

    try:
        # Sanitize metric keys for MLflow (dots are okay, slashes might be problematic)
        # and ensure only numerical values are logged.
        metrics_to_log = {
            f"{prefix}{key.replace('/', '_')}": value
            for key, value in metrics.items()
            if isinstance(value, (int, float))  # Filter for numerical metrics
        }
        if metrics_to_log:
            # Log all metrics in the dictionary for the given step
            mlflow.log_metrics(metrics_to_log, step=episode_num)
    except Exception as e:
        # Catch potential errors during MLflow logging (e.g., network issues, type mismatches not caught)
        print(
            f"Warning: MLflow metric logging failed for step {episode_num}. Error: {e}"
        )

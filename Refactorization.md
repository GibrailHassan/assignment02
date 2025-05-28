# Refactored RL Project: Architecture and Usage Guide

## 1. Introduction: Goals of the Refactoring

This project has undergone a significant refactoring to enhance its modularity, maintainability, and ease of use. The primary goals were to:

* **Apply the DRY (Don't Repeat Yourself) Principle:** Reduce code duplication, especially in agent implementations and experiment execution.
* **Improve Modularity:** Decouple components like agents, environments, and network architectures so they can be developed and modified independently.
* **Centralize Configuration:** Move experiment parameters out of Python scripts into readable configuration files.
* **Simplify Experiment Management:** Provide a single entry point for running all experiments, making it easier to train, evaluate, and reproduce results.
* **Enhance Extensibility:** Make it straightforward to add new agents, environments, or network architectures.
* **Adhere to Software Engineering Best Practices:** Incorporate design patterns (like Factories), clear interfaces (Abstract Base Classes), and robust type hinting.

This document outlines the new project structure, explains the role of each key component, and provides guidance on how to use and extend this improved framework.

## 2. New Project Structure Overview

The refactoring has introduced some new directories and standardized the roles of existing ones:

* **`main.py` (New Root File):** The single entry point for running all experiments. It replaces all previous `run_*.py` scripts.
* **`configs/` (New Directory):** Contains YAML configuration files for each experiment. This is where you define which agent, environment, and hyperparameters to use.
* **`agents/`:**
  * `abstractAgent.py`: Defines the updated `AbstractAgent` interface, which all agents must implement. Includes new hooks like `on_episode_start`, `on_episode_end`, and `get_update_info`.
  * `tableAgent.py` (New): A base class for table-based agents (Q-Learning, SARSA), containing shared logic for Q-table management, epsilon-greedy action selection, and model persistence.
  * `qlAgent.py`, `sarsaAgent.py`: Refactored to inherit from `tableAgent.py`, now only containing their specific update rules.
  * `dqnAgent.py`: Refactored to accept pre-built neural networks (dependency injection) and implement the new `AbstractAgent` hooks.
  * `randomAgent.py`, `basicAgent.py`: Updated to implement the new `AbstractAgent` hooks.
  * `NN_model.py`: Contains the definitions for `Model` (MLP) and `CNNModel`. (Alternatively, this could be moved to `networks/architectures.py` as discussed for further modularization).
  * `factory.py` (New): A factory module (`create_agent`) to instantiate agents based on configuration.
* **`env/`:**
  * Contains environment wrappers like `MoveToBeaconDiscreteEnv`, `MoveToBeaconEnv`, and `DefeatRoachesEnv`.
  * `dr_env.py` (Updated): Includes fixes for NumPy array conversion to prevent type errors.
  * `factory.py` (New): A factory module (`create_environment`) to instantiate environments based on configuration.
* **`networks/` (Conceptual New Directory - if full network refactor is applied):**
  * `base.py`: Abstract base class for all network architectures.
  * `architectures.py`: Specific network implementations (e.g., MLP, CNN).
  * `factory.py`: Factory to create network instances.
    *(Note: If you haven't fully implemented the separate `networks/` module yet, the `DQNAgent` still uses its internal `_create_network` method based on `agents/NN_model.py` and the `use_cnn` flag.)*
* **`runner/`:**
  * `runner.py`: The `Runner` class has been refactored to have a single, generic `run()` method that can handle all agent types. It orchestrates the training/evaluation loop, logging, and model saving.
* **`logs/`:** Directory where TensorBoard logs are saved.
* **`models/`:** Directory where trained agent models are saved.

## 3. Core Components and Their Roles

### 3.1. `main.py`: The Central Hub

* **Purpose:** Serves as the sole command-line entry point for the entire project.
* **Functionality:**
    1. Parses a command-line argument (`--config`) specifying the path to a YAML configuration file.
    2. Loads the specified YAML configuration.
    3. Uses the `env.factory.create_environment` function to instantiate the environment.
    4. Uses the `agents.factory.create_agent` function to instantiate the agent (or loads a pre-trained agent if `is_training: false` and `load_model_path` are set in the config).
    5. Instantiates the `Runner` with the created agent and environment.
    6. Calls `runner.run()` to start the experiment.
* **Key Dependencies:** `yaml`, `absl-py`, `env.factory`, `agents.factory`, `runner.runner`.

### 3.2. `configs/*.yaml`: Experiment Definitions

* **Purpose:** To define all parameters for a specific experiment in a human-readable format.
* **Structure:** Each YAML file typically contains three main sections:
  * `experiment_name`: A descriptive name.
  * `environment`:
    * `name`: The registered name of the environment class (e.g., "MoveToBeaconDiscreteEnv").
    * `params`: A dictionary of parameters to pass to the environment's constructor.
  * `agent`:
    * `name`: The registered name of the agent class (e.g., "DQNAgent").
    * `params`: A dictionary of hyperparameters for the agent. This can include flags like `use_cnn` for `DQNAgent` or `enable_ddqn` if implemented.
  * `runner`:
    * `total_episodes`: Number of episodes to run.
    * `is_training`: Boolean, `true` for training, `false` for evaluation.
    * `load_model_path` (optional): Path to a saved model directory for evaluation.
    * `tensorboard_log_dir`: Directory for TensorBoard logs.
    * `model_save_dir`: Directory for saving models.
    * `save_model_each_episode_num`: Frequency of model saving.
* **Benefit:** Enables reproducible experiments and easy parameter tuning without code changes.

### 3.3. `runner/runner.py`: The Experiment Orchestrator

* **Purpose:** Manages the agent-environment interaction loop.
* **Refactored `run()` Method:**
  * Iterates for `total_episodes`.
  * In each episode:
    * Resets the environment and calls `agent.on_episode_start()`.
    * Loops until `done`:
      * Gets an `action` from the agent (passing `is_training` flag).
      * Steps the environment.
      * If `is_training`, calls `agent.update()` (passing `next_action` for on-policy compatibility).
    * Calls `agent.on_episode_end()`.
    * Calls `summarize()` to log metrics and save models.
* **`summarize()` Method:**
  * Prints episode scores to the console.
  * Logs to TensorBoard: episodic score, total score, mean score (sliding window), and agent-specific info from `agent.get_update_info()` (e.g., epsilon).
  * Handles periodic model saving based on `save_model_each_episode_num`.

### 3.4. `agents/` Module: The Learners

* **`abstractAgent.py`:**
  * The core interface. All agents *must* implement its methods:
    * `__init__(self, observation_space: gym.Space, action_space: gym.Space)`
    * `get_action(self, state: Any, is_training: bool = True) -> Any`
    * `update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool, **kwargs: Any) -> None`
    * `on_episode_start(self) -> None`
    * `on_episode_end(self) -> None`
    * `get_update_info(self) -> Dict[str, Any]`
    * `save_model(self, path: str, filename: str) -> None`
    * `load_model(cls, path: str, filename: str, **kwargs: Any) -> 'AbstractAgent'`
* **`tableAgent.py`:**
  * Base for `QLearningAgent` and `SARSAAgent`.
  * Handles Q-table initialization (now dynamically sized based on `observation_space.low` and `observation_space.high`), indexing with offset for negative states, epsilon-greedy `get_action`, `on_episode_start`, `on_episode_end` (for epsilon decay), `get_update_info`, `save_model`, and `load_model`.
* **`qlAgent.py` & `sarsaAgent.py`:**
  * Now very concise, only implementing their specific `update` rule.
* **`dqnAgent.py`:**
  * Implements the DQN algorithm.
  * Constructor takes `observation_space` and `action_space`.
  * `_create_network()`: Selects between `Model` (MLP) and `CNNModel` based on the `use_cnn` param and `observation_space.shape`. Handles correct channel determination for CNNs.
  * Tensor Permutations: Includes logic in `get_action` and `_get_sample_from_memory` to ensure tensors are in the correct `(B, C, H, W)` or `(C, H, W)` format for CNNs.
  * Implements `on_episode_start`, `on_episode_end`, and `get_update_info`. Epsilon decay is typically handled per step within its `update` method.
* **`NN_model.py` (or `networks/architectures.py`):**
  * `Model` (MLP): Constructor now takes `input_features` (calculated from `observation_space.shape`).
  * `CNNModel`: Constructor takes `input_channels`. `forward` method uses `reshape()` instead of `view()` for robustness.
* **`agents/factory.py`:**
  * `create_agent(name, params, observation_space, action_space)`: Instantiates an agent using its registered name and passes the full `observation_space` and `action_space` objects.

### 3.5. `env/` Module: The Worlds

* **Environment Wrappers (`MoveToBeaconDiscreteEnv`, `MoveToBeaconEnv`, `DefeatRoachesEnv`):**
  * These classes wrap PySC2 environments to provide a `gym.Env`-compatible interface.
  * They define `observation_space` and `action_space` (e.g., `gym.spaces.Box`, `gym.spaces.Discrete`).
  * `DefeatRoachesEnv._get_state()`: Now explicitly converts the PySC2 `NamedNDArray` to a base `np.array(..., copy=True)` and ensures a channel dimension, which helps prevent certain type interaction errors.
* **`env/factory.py`:**
  * `create_environment(name, params)`: Instantiates an environment using its registered name.

## 4. How to Use the New Framework

### 4.1. Running an Experiment (Training or Evaluation)

1. **Choose or Create a Configuration File:**
    * Select an existing YAML file from the `configs/` directory (e.g., `configs/sarsa_train_discrete.yaml`, `configs/dqn_cnn_train_defeat_roaches.yaml`).
    * Or, create a new one by copying and modifying an existing config.

2. **Modify the Configuration (if needed):**
    * Adjust `environment.params`, `agent.params`, and `runner` settings as desired.
    * For **training**, ensure `runner.is_training: true`.
    * For **evaluation**:
        * Set `runner.is_training: false`.
        * Set `runner.load_model_path` to the directory of your saved model (e.g., `models/250528_2020_train_DQNAgent`).
        * Optionally set `environment.params.is_visualize: true` to watch.

3. **Execute `main.py`:**
    * Open your terminal, activate your virtual environment (`uv` or otherwise).
    * Run the command:

        ```bash
        python main.py --config=configs/your_chosen_config_file.yaml
        ```

        (Replace `your_chosen_config_file.yaml` with the actual file name).

### 4.2. Understanding YAML Configuration

* **Structure:** Key-value pairs, indentation matters.
* **`environment.name` & `agent.name`:** Must match keys in the respective factory registries (`env/factory.py`, `agents/factory.py`).
* **`params`:** Dictionaries passed directly as `**kwargs` to the constructors.
* **Boolean Values:** Use `true` or `false` (lowercase).

### 4.3. Model Saving and Loading

* **Saving (During Training):**
  * Controlled by `runner.model_save_dir` and `runner.save_model_each_episode_num` in the config.
  * Models are saved in timestamped subdirectories (e.g., `models/250528_2100_DQNAgent/dqn_online_nn.pt`).
* **Loading (For Evaluation):**
  * Set `runner.is_training: false` and provide `runner.load_model_path` in the config.
  * `main.py` handles calling the agent's `load_model` class method.
  * The agent's `load_model` method expects the `filename` (e.g., `q_table.pt` or `dqn_online_nn.pt`) to be present in the specified `load_model_path` directory.

### 4.4. Visualizing with TensorBoard

1. Ensure `runner.tensorboard_log_dir` is set in your config (e.g., `./logs`).
2. While an experiment is running or after it has finished, open a new terminal.
3. Navigate to your project's root directory.
4. Run: `tensorboard --logdir=./logs`
5. Open the provided URL (usually `http://localhost:6006`) in a browser.
    You'll see plots for "Episodic_Score", "Mean_Score_10_Episodes", and agent-specific metrics like "Agent/epsilon".

## 5. How to Modify or Extend the Framework

The new modular design makes extensions much simpler.

### 5.1. Adding a New Agent

1. **Create Agent File:** In `agents/`, create `myNewAgent.py`.
2. **Define Agent Class:**

    ```python
    # agents/myNewAgent.py
    from agents.abstractAgent import AbstractAgent
    import gym
    from typing import Any, Dict

    class MyNewAgent(AbstractAgent):
        def __init__(self, observation_space: gym.Space, action_space: gym.Space, **agent_params: Any):
            super().__init__(observation_space, action_space)
            # Initialize your agent's specific parameters, networks, etc.
            # self.my_param = agent_params.get("my_custom_param", default_value)

        def get_action(self, state: Any, is_training: bool = True) -> Any:
            # ...
            pass
        
        def update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool, **kwargs: Any) -> None:
            # ...
            pass

        def on_episode_start(self) -> None:
            pass

        def on_episode_end(self) -> None:
            pass

        def get_update_info(self) -> Dict[str, Any]:
            return {"my_metric": 0.0} # Example

        def save_model(self, path: str, filename: str) -> None:
            # ...
            pass

        @classmethod
        def load_model(cls, path: str, filename: str, **kwargs: Any) -> 'MyNewAgent':
            # observation_space = kwargs.pop("observation_space") # Ensure these are handled
            # action_space = kwargs.pop("action_space")
            # instance = cls(observation_space, action_space, **kwargs)
            # Load weights into instance.
            # return instance
            pass
    ```

3. **Register in Factory:** Open `agents/factory.py` and add your new agent:

    ```python
    from agents.myNewAgent import MyNewAgent # Import it

    AGENT_REGISTRY = {
        # ... existing agents ...
        "MyNewAgent": MyNewAgent,
    }
    ```

4. **Create Config File:** In `configs/`, create `my_new_agent_experiment.yaml`, setting `agent.name: "MyNewAgent"` and defining its `agent.params`.

### 5.2. Adding a New Environment

1. **Create Environment File:** In `env/`, create `myNewEnv.py`.
2. **Define Environment Class:** Ensure it inherits from `gym.Env` and implements `reset`, `step`, `render`, `close`, and defines `observation_space` and `action_space`.
3. **Register in Factory:** Open `env/factory.py` and add your new environment.
4. **Create Config File:** In `configs/`, create a new YAML file setting `environment.name` to your new environment's registered name.

### 5.3. Adding a New Network Architecture (for DQN or future network-based agents)

*(This assumes you have implemented the separate `networks/` module as discussed).*

1. **Define Network:** In `networks/architectures.py`, create your new network class, inheriting from `networks.base.BaseNetwork`.
2. **Register in Factory:** In `networks/factory.py`, add your new network to the `NETWORK_REGISTRY`.
3. **Use in Config:** In your agent's YAML configuration (e.g., for a `DQNAgent` that uses the network factory), you would specify:

    ```yaml
    agent:
      name: "DQNAgent" # Or your new network-based agent
      params:
        network_name: "MyCoolNetwork" # Registered name of your new network
        network_params:
          custom_layer_size: 128 # Params for your network's constructor
        # ... other agent params ...
    ```

    Your `DQNAgent` (or other agent) would need to be modified to accept a `network_name` and `network_params` and use the `networks.factory.create_network` function.

## 6. Summary of the Refactoring

* **Improved Readability & Maintainability:** Code is cleaner, more organized, and easier to understand.
* **Reduced Redundancy (DRY):** Common logic is shared through base classes.
* **Enhanced Modularity & Decoupling:** Agents, environments, and (optionally) networks are independent components.
* **Centralized & Reproducible Experiments:** YAML configs make experiments easy to define, track, and share.
* **Scalability & Extensibility:** Adding new components is straightforward without major changes to core logic.
* **Adherence to Best Practices:** Uses abstract base classes, factories, type hinting, and clear separation of concerns.

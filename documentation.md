# Project Guide: Refactored RL Framework for StarCraft II

## 1. Project Overview

This project provides a flexible and modular framework for developing, training, and evaluating Reinforcement Learning (RL) agents on StarCraft II mini-games using the PySC2 library. It has been refactored to promote best practices in software engineering, making it easier to manage experiments, reuse components, and extend with new algorithms or environments.

**Key Features:**

* **Modularity:** Clear separation of concerns between agents, environments, neural network architectures, the experiment runner, and utility functions.
* **Configuration-Driven:** Experiments are defined and controlled by human-readable YAML files, eliminating hardcoded parameters in scripts.
* **Extensibility:** Easily add new RL agents, custom environments, or neural network architectures using factory patterns and abstract base classes.
* **Unified Experiment Management:** A single `main.py` script serves as the entry point for all training and evaluation tasks.
* **Experiment Tracking:** Integrated with MLflow for comprehensive tracking of parameters, metrics, and artifacts (including models and configuration files). TensorBoard is also supported for granular, step-by-step visualization.
* **Support for Various RL Algorithms:** Includes implementations for table-based methods (Q-Learning, SARSA) and deep learning methods (DQN, with support for DDQN).
* **Utility Functions:** Common tasks like MLflow setup, configuration loading, and metric logging are centralized in a utility module for cleaner main scripts.

## 2. Directory and File Structure

The project is organized as follows:

assignment02/
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── main.py                 # Single entry point for all experiments
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── configs/                # YAML configuration files for experiments
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── ql_train_discrete.yaml
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── sarsa_train_discrete.yaml
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── sarsa_eval_discrete.yaml
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── dqn_fcn_train_move_to_beacon_discrete.yaml  # DQN with MLP
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── dqn_cnn_train_move_to_beacon_full.yaml    # DQN with CNN
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── dqn_cnn_train_defeat_roaches.yaml       # DQN with CNN
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── ddqn_fcn_train_move_to_beacon_discrete.yaml # DDQN with MLP
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── ddqn_cnn_train_move_to_beacon_full.yaml   # DDQN with CNN
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── ddqn_cnn_train_defeat_roaches.yaml      # DDQN with CNN
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── random_agent_discrete.yaml
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── basic_agent_discrete.yaml
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── agents/
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── **init**.py
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── abstractAgent.py      # Abstract base class for all agents
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── tableAgent.py         # Base class for Q-Learning and SARSA
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── qlAgent.py            # Q-Learning agent implementation
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── sarsaAgent.py         # SARSA agent implementation
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── dqnAgent.py           # DQN and DDQN agent implementation
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── randomAgent.py        # Agent selecting random actions
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── basicAgent.py         # Heuristic-based agent
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── factory.py            # Factory to create agent instances
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── networks/                 # Neural network architectures and factory
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── **init**.py
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── base.py               # Abstract BaseNetwork class
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── architectures.py      # Concrete MLPNetwork, CNNNetwork classes
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── factory.py            # Factory to create network instances
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── env/
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── **init**.py
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── env_discrete.py       # MoveToBeaconDiscreteEnv wrapper
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── env_full.py           # MoveToBeaconEnv (visual) wrapper
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── dr_env.py             # DefeatRoachesEnv wrapper
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── utils.py              # Utility functions specific to environments (original)
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── factory.py            # Factory to create environment instances
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── runner/
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── **init**.py
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── runner.py             # Experiment runner class
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── utils/                    # NEW: General utility functions
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── **init**.py
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── experiment_utils.py   # Utilities for config, MLflow, logging, component setup
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── models/                   # Default local directory for saving trained models
<br>&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# (MLflow also stores model artifacts)
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── logs/                     # Default local directory for saving TensorBoard logs
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── mlruns/                   # Default local directory for MLflow tracking data
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── .venv/                    # Virtual environment directory (example)
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── pyproject.toml            # Project metadata and dependencies (e.g., for uv)
<br>&nbsp;&nbsp;&nbsp;&nbsp;└── uv.lock                   # Lock file for uv (or requirements.txt for pip)

*(Note: The `agents/NN_model.py` file has been effectively replaced by the `networks/` module. If you still have `agents/NN_model.py`, its contents should have been moved to `networks/architectures.py` and the file can be deleted or repurposed.)*

## 3. Core Components and Their Roles

### 3.1. `main.py`: The Central Hub

* **Purpose:** The single script to run any experiment (training or evaluation).
* **Functionality:**
    1. Parses the `--config` command-line argument.
    2. Calls `utils.experiment_utils.load_config` to load the YAML configuration.
    3. Initializes MLflow and starts a run; then calls `utils.experiment_utils.setup_mlflow_run` to log parameters and the config artifact.
    4. Uses `env.factory.create_environment` to create the environment.
    5. Calls `utils.experiment_utils.prepare_agent_networks_and_params` to handle network creation (using `networks.factory`) and prepare the full parameter set for the agent.
    6. Calls `utils.experiment_utils.initialize_agent` which in turn uses `agents.factory.create_agent` or the agent's `load_model` method.
    7. Instantiates the `Runner`.
    8. Calls `runner.run()` and handles overall run status logging to MLflow.
    9. Ensures MLflow run is ended.

### 3.2. `configs/*.yaml`: Experiment Definitions

* **Purpose:** Define all parameters for an experiment.
* **Structure:**
  * `experiment_name`: Used for MLflow experiment grouping.
  * `environment`: Contains `name` (registered in `env/factory.py`) and `params` for the environment.
  * `agent`: Contains `name` (registered in `agents/factory.py`) and `params`.
    * For network-based agents like `DQNAgent`, `agent.params` will include network configurations (e.g., `online_network_config: {name: "CNNNetwork", params: {...}}`) and algorithm-specific hyperparameters (e.g., `learning_rate`, `enable_ddqn`).
  * `runner`: Contains parameters for the `Runner` class (e.g., `total_episodes`, `is_training`, `load_model_path`, logging directories, save frequency).

### 3.3. `utils/experiment_utils.py`: Helper Functions

* **Purpose:** To make `main.py` and `runner.py` more concise by encapsulating common or complex setup and logging logic.
* **Key Functions:**
  * `load_config(config_file_path)`: Loads and parses YAML configuration.
  * `setup_mlflow_run(config, config_file_path)`: Initializes MLflow experiment, logs parameters from the config, and logs the config file itself as an artifact. To be called within an active `mlflow.start_run()` context.
  * `prepare_agent_networks_and_params(agent_config_yaml, env, is_training)`: Reads network configurations from `agent_config_yaml`, uses `networks.factory.create_network` to build network instances, and returns a consolidated dictionary of parameters for the agent (including injected networks).
  * `initialize_agent(agent_config_yaml, env, runner_config, prepared_agent_params)`: Handles the logic to either create a new agent (using `agents.factory.create_agent`) or load a pre-trained one (using `AgentClass.load_model`), based on `runner_config.is_training` and `runner_config.load_model_path`.
  * `log_metrics_to_tensorboard(writer, metrics, episode_num, prefix)`: Logs a dictionary of metrics to TensorBoard.
  * `log_metrics_to_mlflow(metrics, episode_num, prefix)`: Logs a dictionary of metrics to the active MLflow run.

### 3.4. `runner/runner.py`: The Experiment Orchestrator

* **Purpose:** Manages the agent-environment interaction loop.
* **Key Features:**
  * Uses `utils.experiment_utils.log_metrics_to_tensorboard` and `utils.experiment_utils.log_metrics_to_mlflow` within its `summarize()` method for cleaner metric logging.
  * Otherwise, its role and generic `run()` method remain as previously defined (handles episode loop, calls agent methods like `get_action`, `update`, `on_episode_start`, `on_episode_end`).
  * Triggers local model saving by calling `agent.save_model()`.

### 3.5. `agents/` Module: The Learning Algorithms

* **`abstractAgent.py`**: Defines the `AbstractAgent` interface (methods: `__init__`, `get_action`, `update`, `on_episode_start`, `on_episode_end`, `get_update_info`, `save_model`, `load_model`).
* **`tableAgent.py`**: Base class for Q-Learning and SARSA. Handles Q-table sizing, index offset, `save_model` (with MLflow artifact logging), `load_model` (with `weights_only=False` and space reconstruction).
* **`qlAgent.py`, `sarsaAgent.py`**: Inherit from `TableBasedAgent`, define specific `update` rules.
* **`dqnAgent.py`**: Implements DQN/DDQN. Receives pre-built networks. Handles replay memory, target updates. `save_model` logs to MLflow. `load_model` reconstructs networks using `network_config`.
* **`randomAgent.py`, `basicAgent.py`**: Implement `AbstractAgent` interface.
* **`factory.py`**: `create_agent` function instantiates agents, passing `observation_space`, `action_space`, and `params` (which include injected networks for `DQNAgent`).

### 3.6. `networks/` Module: Neural Network Architectures

* **`base.py`**: Defines `BaseNetwork(torch.nn.Module, ABC)`. Requires `forward` and takes `observation_space`, `action_space` in `__init__`.
* **`architectures.py`**: Contains `MLPNetwork` and `CNNNetwork` (inheriting `BaseNetwork`). They are configurable via `__init__` parameters (e.g., `hidden_layers` for MLP, `conv_channels` for CNN) and use `observation_space`/`action_space` for input/output sizing.
* **`factory.py`**: `create_network` function instantiates networks.

### 3.7. `env/` Module: The Simulation Worlds

* Contains PySC2 environment wrappers (`MoveToBeaconDiscreteEnv`, `MoveToBeaconEnv`, `DefeatRoachesEnv`) adhering to `gym.Env`.
* `DefeatRoachesEnv._get_state()` returns a base `np.array(..., copy=True)`.
* **`factory.py`**: `create_environment` function.

## 4. How to Use the Framework

### 4.1. Setup

1. Ensure Python and StarCraft II (with maps) are installed.
2. Create and activate a virtual environment.
3. Install dependencies: `pysc2`, `gym`, `numpy`, `torch`, `absl-py`, `PyYAML`, `mlflow`, `tensorboard`. Use `uv sync` with `pyproject.toml` or `pip install -r requirements.txt`.

### 4.2. Running an Experiment (Training)

1. **Choose/Create YAML Config:** In `configs/` (e.g., `configs/dqn_cnn_train_defeat_roaches.yaml`).
2. **Define Parameters:**
    * Set `experiment_name`.
    * Configure `environment` (`name`, `params`).
    * Configure `agent` (`name`, `params`). For `DQNAgent`, `params` must include `online_network_config` (with `name` like "MLPNetwork" or "CNNNetwork", and its specific `params`) and `network_config` for loading consistency.
    * Configure `runner` (`total_episodes`, `is_training: true`, etc.).
3. **Execute:**

    ```bash
    python main.py --config=configs/your_config_file.yaml
    ```

4. **Monitor:**
    * MLflow UI: `mlflow ui` (then `http://localhost:5000`).
    * TensorBoard: `tensorboard --logdir=./logs`.

### 4.3. Loading a Trained Model and Running Evaluations

1. **Create/Modify Evaluation Config:**
    * Set `runner.is_training: false`.
    * Set `runner.load_model_path` to the *directory* of the saved model.
    * For `DQNAgent`, ensure `agent.params.network_config` in YAML matches the loaded model's architecture.
2. **Execute:**

    ```bash
    python main.py --config=configs/your_eval_config_file.yaml
    ```

### 4.4. Pausing and Resuming Training

* **Current State:** Model weights are saved. Optimizer state, replay memory, episode/step counters, and current epsilon are **not** saved for exact resumption.
* **Future Enhancement:**
    1. **Saving:** Extend `agent.save_model` to include optimizer state, replay buffer, step/epsilon. `Runner` saves its own state.
    2. **Loading:** Extend `agent.load_model` to restore these states. `main.py`/`Runner` restore runner state.
    3. **YAML:** Add `runner.resume_checkpoint_path`.

## 5. How to Modify or Extend

### 5.1. Adding a New Agent

1. Create `agents/myNewAgent.py`, subclassing `AbstractAgent`.
2. Implement all abstract methods.
3. Register in `agents/factory.py`.
4. Create YAML config.

### 5.2. Adding a New Environment

1. Create `env/myNewEnv.py`, subclassing `gym.Env`.
2. Implement `gym.Env` methods and define spaces.
3. Register in `env/factory.py`.
4. Create YAML config.

### 5.3. Adding a New Neural Network Architecture

1. Create your network class in `networks/architectures.py`, subclassing `networks.base.BaseNetwork`.
2. Register in `networks/factory.py`.
3. Use in YAML config under `agent.params.online_network_config`.

----

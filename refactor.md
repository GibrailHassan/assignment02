# Refactored RL Project: Comprehensive Guide (v2)

## 1. Introduction: Goals and Achievements of Refactoring

This project has undergone a significant refactoring to enhance its modularity, maintainability, reproducibility, and ease of use for reinforcement learning experimentation. The primary goals and achievements include:

* **DRY (Don't Repeat Yourself) Principle:** Code duplication has been minimized, particularly in agent implementations (e.g., `TableBasedAgent` for Q-Learning and SARSA) and experiment execution (unified `main.py` and `Runner`).
* **Improved Modularity & Decoupling:** Core components like agents, environments, and neural network architectures are now more independent, allowing for easier isolated development and modification.
* **Centralized Configuration:** All experiment parameters (environment settings, agent hyperparameters, runner configurations) have been moved from hardcoded Python scripts into human-readable YAML files located in the `configs/` directory.
* **Unified Experiment Management:** A single `main.py` script serves as the entry point for all experiments, driven by the YAML configuration files. This replaces the numerous `run_*.py` scripts.
* **Enhanced Extensibility:** The framework now uses factory patterns for creating agents and environments, making it straightforward to add new components without altering core logic.
* **Robust Agent Interface:** The `AbstractAgent` class defines a clear contract for all agents, including new lifecycle hooks (`on_episode_start`, `on_episode_end`) and information retrieval (`get_update_info`).
* **Improved Type Safety & Readability:** Comprehensive type hinting has been added across the codebase.
* **Bug Fixes & Stability:** Addressed various issues related to tensor dimensions, Q-table indexing, model loading (e.g., `weights_only=False`), and environment state representation.
* **Support for Agent Variations:** The `DQNAgent` now includes a configurable option for Double DQN (DDQN).

This document details the refactored project structure, explains the functionality of key components, and provides instructions on how to use and extend this robust framework.

## 2. Refactored Project Directory Structure

The project now follows this organized structure:

<!--
assignment02/
├── main.py                 # Single entry point for all experiments
├── configs/                # YAML configuration files for experiments
│   ├── ql_train_discrete.yaml
│   ├── sarsa_train_discrete.yaml
│   ├── sarsa_eval_discrete.yaml
│   ├── dqn_fcn_train_move_to_beacon_discrete.yaml
│   ├── dqn_cnn_train_move_to_beacon_full.yaml
│   ├── dqn_cnn_train_defeat_roaches.yaml
│   ├── ddqn_fcn_train_move_to_beacon_discrete.yaml # Example for DDQN
│   ├── ddqn_cnn_train_move_to_beacon_full.yaml   # Example for DDQN
│   ├── ddqn_cnn_train_defeat_roaches.yaml      # Example for DDQN
│   ├── random_agent_discrete.yaml
│   └── basic_agent_discrete.yaml
├── agents/
│   ├── __init__.py
│   ├── abstractAgent.py      # Abstract base class for all agents
│   ├── tableAgent.py         # Base class for Q-Learning and SARSA
│   ├── qlAgent.py            # Q-Learning agent implementation
│   ├── sarsaAgent.py         # SARSA agent implementation
│   ├── dqnAgent.py           # DQN and DDQN agent implementation
│   ├── NN_model.py           # Neural network architectures (MLP, CNN)
│   ├── randomAgent.py        # Agent selecting random actions
│   ├── basicAgent.py         # Heuristic-based agent
│   └── factory.py            # Factory to create agent instances
├── env/
│   ├── __init__.py
│   ├── env_discrete.py       # MoveToBeaconDiscreteEnv wrapper
│   ├── env_full.py           # MoveToBeaconEnv (visual) wrapper
│   ├── dr_env.py             # DefeatRoachesEnv wrapper (updated)
│   ├── utils.py              # Utility functions for environments
│   └── factory.py            # Factory to create environment instances
├── runner/
│   ├── __init__.py
│   └── runner.py             # Experiment runner class (refactored)
├── models/                   # Directory for saving trained models
│   └── (timestamped_agent_subdirs)/ # e.g., 250528_2100_DQNAgent/
│       └── dqn_online_nn.pt       # Example saved model file
├── logs/                     # Directory for saving TensorBoard logs
│   └── (timestamped_agent_subdirs)/
├── .venv/                    # Virtual environment directory (example)
├── pyproject.toml            # Project metadata and dependencies (for uv/poetry/pdm)
├── uv.lock                   # Lock file for uv
└── requirements.txt          # (Alternative) For pip-based dependency management
-->

*(Note: The `networks/` directory discussed for further network modularization is not explicitly listed as a separate top-level directory here, assuming `agents/NN_model.py` still holds the primary network definitions for now. If fully separated, `agents/NN_model.py` would move into `networks/architectures.py` and a `networks/factory.py` would be added.)*

## 3. Core Components and Their Roles

### 3.1. `main.py`: The Central Orchestrator

* **Purpose:** Single script to launch any training or evaluation run.
* **Functionality:**
    1. Uses `absl-py` for command-line flag parsing, requiring a `--config` argument pointing to a YAML file.
    2. Loads the specified YAML configuration.
    3. Instantiates the environment using `env.factory.create_environment()`.
    4. Instantiates or loads the agent using `agents.factory.create_agent()` or `AgentClass.load_model()`, passing the environment's `observation_space` and `action_space`.
    5. Initializes the `Runner` with the agent, environment, and runner parameters from the config.
    6. Calls `runner.run()` to start the experiment.

### 3.2. `configs/*.yaml`: Experiment Blueprints

* **Purpose:** Define all parameters for an experiment, ensuring reproducibility and easy modification.
* **Key Sections:**
  * `experiment_name`: For identification.
  * `environment`: Specifies `name` (from `env.factory.ENV_REGISTRY`) and `params` for the environment constructor.
  * `agent`: Specifies `name` (from `agents.factory.AGENT_REGISTRY`) and `params` (hyperparameters, `use_cnn`, `enable_ddqn`, etc.) for the agent constructor.
  * `runner`: Configures the `Runner` (e.g., `total_episodes`, `is_training`, `load_model_path`, logging/saving directories, save frequency).

### 3.3. `runner/runner.py`: The Interaction Loop Manager

* **Purpose:** Manages the episodes and steps within an experiment.
* **Refactored `run()` Method:**
  * A single, generic method for all agent types.
  * Handles the main episode loop.
  * Calls `agent.on_episode_start()` and `agent.on_episode_end()`.
  * In each step: gets action from agent, steps environment, calls `agent.update()` (passing `next_action` for on-policy methods like SARSA).
* **`summarize()` Method:** Logs scores, agent-specific info (from `agent.get_update_info()`), and saves models periodically.
* **TensorBoard Integration:** Writes logs to the directory specified in `runner.tensorboard_log_dir`.

### 3.4. `agents/` Module: The Learning Algorithms

* **`abstractAgent.py`:**
  * Defines the mandatory interface for all agents:
    * `__init__(self, observation_space: gym.Space, action_space: gym.Space)`
    * `get_action(self, state: Any, is_training: bool = True) -> Any`
    * `update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool, **kwargs: Any) -> None`
    * `on_episode_start(self) -> None`
    * `on_episode_end(self) -> None`
    * `get_update_info(self) -> Dict[str, Any]` (e.g., for epsilon)
    * `save_model(self, path: str, filename: str) -> None`
    * `load_model(cls, path: str, filename: str, **kwargs: Any) -> 'AbstractAgent'`
* **`tableAgent.py`:**
  * New base class for `QLearningAgent` and `SARSAAgent`.
  * Constructor takes `observation_space` and `action_space` to dynamically size the Q-table.
  * Handles negative state values by calculating and using an `index_offset`.
  * `save_model` now saves `q_table` and `index_offset`.
  * `load_model` reconstructs `observation_space` and `action_space` from the loaded Q-table dimensions and `index_offset`, and uses `torch.load(..., weights_only=False)`.
  * Implements `on_episode_start`, `on_episode_end` (for epsilon decay), `get_update_info`.
* **`qlAgent.py` & `sarsaAgent.py`:**
  * Inherit from `TableBasedAgent`.
  * Primarily implement their specific `update(..., next_action=None_for_QL, ...)` rule.
* **`dqnAgent.py`:**
  * Constructor takes `observation_space`, `action_space`.
  * Includes `enable_ddqn: bool` parameter to switch between DQN and Double DQN logic in `_calculate_loss`.
  * `_create_network()`: Correctly determines `input_channels` for CNNs based on `observation_space.shape`, handling both `(C,H,W)` and `(H,W,C)` formats.
  * Tensor Permutations: Logic in `get_action` and `_get_sample_from_memory` ensures correct tensor shapes (`B,C,H,W` or `C,H,W`) for CNNs.
  * Implements `on_episode_start`, `on_episode_end`, `get_update_info`. Epsilon decay is linear per step.
* **`NN_model.py`:**
  * `Model` (MLP): Constructor updated to take `input_features`. `init_weights` is called within constructor.
  * `CNNModel`: `forward` method uses `x.reshape(x.size(0), -1)` instead of `view()` for robustness. `init_weights` is called within constructor. `init_weights` checks if `m.bias is not None`.
* **`randomAgent.py` & `basicAgent.py`:**
  * Updated to implement `on_episode_start`, `on_episode_end`, `get_update_info` from `AbstractAgent`.
* **`agents/factory.py`:**
  * `create_agent(name, params, observation_space, action_space)`: Uses the agent's registered name and correctly passes the full `gym.Space` objects and other parameters to the agent's constructor.

### 3.5. `env/` Module: The Simulation Worlds

* **Environment Wrappers (`MoveToBeaconDiscreteEnv`, `MoveToBeaconEnv`, `DefeatRoachesEnv`):**
  * Provide a `gym.Env`-compatible interface.
  * Define `observation_space` (e.g., `gym.spaces.Box`) and `action_space` (e.g., `gym.spaces.Discrete`).
  * `DefeatRoachesEnv._get_state()`: Updated to return a base `np.array(..., copy=True, dtype=self.observation_space.dtype)` and ensures the correct channel dimension (`H,W,C`) for consistency.
* **`env/factory.py`:**
  * `create_environment(name, params)`: Instantiates an environment using its registered name.

## 4. How to Use the Refactored Framework

### 4.1. Setting Up the Project

1. **Clone/Download:** Get the project files.
2. **Create Virtual Environment:**
    * Using `uv`: `uv venv` (or similar)
    * Using standard `venv`: `python -m venv .venv` then activate.
3. **Install Dependencies:**
    * Using `uv`: `uv sync` (if you have `pyproject.toml` and `uv.lock`) or `uv pip install -r requirements.txt`.
    * Using `pip`: `pip install -r requirements.txt`.
    * Ensure `PyYAML` is listed in your dependencies for config file parsing. Other key dependencies include `pysc2`, `gym`, `numpy`, `torch`, `absl-py`, `tensorboard`.
4. **Install StarCraft II & Maps:** Follow PySC2 instructions to install the game and the required mini-game map packs.

### 4.2. Running an Experiment

1. **Navigate to `configs/` directory.**
2. **Choose or Create a YAML Configuration File:**
    * Examples: `sarsa_train_discrete.yaml`, `dqn_cnn_train_defeat_roaches.yaml`, `ddqn_fcn_train_move_to_beacon_discrete.yaml`.
    * To create a new one, copy an existing file and modify its parameters.
3. **Key Configuration Parameters:**
    * `environment.name` & `environment.params`
    * `agent.name` & `agent.params` (e.g., `use_cnn`, `enable_ddqn`, learning rates, epsilon values)
    * `runner.is_training`: `true` for training, `false` for evaluation.
    * `runner.load_model_path`: For evaluation, path to the *directory* containing the saved model file (e.g., `models/250528_1953_SARSAAgent`). The agent's `load_model` method will look for the specific model file (e.g., `q_table.pt` or `dqn_online_nn.pt`) inside this directory.
    * `runner.total_episodes`
    * `runner.tensorboard_log_dir`, `runner.model_save_dir`, `runner.save_model_each_episode_num`
4. **Execute `main.py` from the project root:**

    ```bash
    python main.py --config=configs/your_config_file.yaml
    ```

### 4.3. Model Saving and Loading

* **Saving:** Automatic during training if `save_model_each_episode_num > 0`. Models are saved in `model_save_dir/TIMESTAMP_AGENTNAME/`.
  * Table-based agents save a dictionary: `{'q_table': ..., 'index_offset': ...}` as `q_table.pt`.
  * DQN agents save the `online_nn.state_dict()` as `dqn_online_nn.pt`.
* **Loading:** Set `is_training: false` and `load_model_path` in the YAML config. `main.py` calls the agent's `load_model` class method.
  * `TableBasedAgent.load_model` uses `torch.load(..., weights_only=False)` and reconstructs the observation/action spaces.
  * `DQNAgent.load_model` loads the `state_dict` into both online and target networks and sets them to `eval()` mode if not training.

### 4.4. Monitoring with TensorBoard

1. Ensure `runner.tensorboard_log_dir` is set in your config.
2. While an experiment runs (or after), open a new terminal in the project root.
3. Run: `tensorboard --logdir=./logs` (or your specified log directory).
4. Open the provided URL (e.g., `http://localhost:6006`) in a browser.
    * View "Episodic_Score", "Mean_Score_10_Episodes", "Agent/epsilon", "Agent/steps".

## 5. How to Modify or Extend the Framework

### 5.1. Adding a New Agent

1. **Create Agent File:** E.g., `agents/myNewRLAgent.py`.
2. **Define Agent Class:**
    * Inherit from `agents.abstractAgent.AbstractAgent`.
    * Implement all abstract methods (`__init__`, `get_action`, `update`, `on_episode_start`, `on_episode_end`, `get_update_info`, `save_model`, `load_model`).
    * The `__init__` should take `observation_space: gym.Space`, `action_space: gym.Space`, and `**agent_params` for hyperparameters.
3. **Register in Agent Factory:**
    * Open `agents/factory.py`.
    * Import your new agent: `from agents.myNewRLAgent import MyNewRLAgent`.
    * Add to `AGENT_REGISTRY`: `"MyNewRLAgent": MyNewRLAgent`.
4. **Create Configuration:**
    * In `configs/`, create `my_new_rl_agent_config.yaml`.
    * Set `agent.name: "MyNewRLAgent"`.
    * Define any custom hyperparameters under `agent.params`.

### 5.2. Adding a New Environment

1. **Create Environment File:** E.g., `env/myNewStarCraftEnv.py`.
2. **Define Environment Class:**
    * Inherit from `gym.Env`.
    * Implement `__init__`, `reset`, `step`, `render`, `close`.
    * Define `self.observation_space` (e.g., `gym.spaces.Box`) and `self.action_space` (e.g., `gym.spaces.Discrete`).
3. **Register in Environment Factory:**
    * Open `env/factory.py`.
    * Import your new environment: `from env.myNewStarCraftEnv import MyNewStarCraftEnv`.
    * Add to `ENV_REGISTRY`: `"MyNewStarCraftEnv": MyNewStarCraftEnv`.
4. **Create Configuration:**
    * In `configs/`, create a config file.
    * Set `environment.name: "MyNewStarCraftEnv"`.
    * Define any environment parameters under `environment.params`.

### 5.3. Adding a New Neural Network Architecture (for DQN or future agents)

1. **Define Network Class:**
    * In `agents/NN_model.py` (or `networks/architectures.py` if you create that module).
    * Inherit from `torch.nn.Module` (or `networks.base.BaseNetwork` if using that structure).
    * Implement `__init__` (taking `input_channels`/`input_features`, `num_actions`, etc.) and `forward`.
2. **Update Agent or Network Factory:**
    * **If `DQNAgent` still uses `_create_network()`:** Modify `_create_network()` in `agents/dqnAgent.py` to instantiate your new network based on a new `agent.params` flag (e.g., `network_type: "MyCustomCNN"`).
    * **If using a separate `networks/factory.py`:** Register your new network there and update your YAML config's `agent.params` to specify `network_name: "MyCustomCNN"` and any `network_params`. The agent (e.g., `DQNAgent`) would then need to be modified to accept `network_name` and `network_params` and use this network factory.

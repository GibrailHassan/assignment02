# Project Guide: Refactored RL Framework for StarCraft II

## 1. Project Overview

This project provides a flexible and modular framework for developing, training, and evaluating Reinforcement Learning (RL) agents on StarCraft II mini-games using the PySC2 library. It has been refactored to promote best practices in software engineering, making it easier to manage experiments, reuse components, and extend with new algorithms or environments.

**Key Features:**

* **Modularity:** Clear separation of concerns between agents, environments, neural network architectures, and the experiment runner.
* **Configuration-Driven:** Experiments are defined and controlled by human-readable YAML files, eliminating hardcoded parameters in scripts.
* **Extensibility:** Easily add new RL agents, custom environments, or neural network architectures using factory patterns and abstract base classes.
* **Unified Experiment Management:** A single `main.py` script serves as the entry point for all training and evaluation tasks.
* **Experiment Tracking:** Integrated with MLflow for comprehensive tracking of parameters, metrics, and artifacts (including models and configuration files). TensorBoard is also supported for granular, step-by-step visualization.
* **Support for Various RL Algorithms:** Includes implementations for table-based methods (Q-Learning, SARSA) and deep learning methods (DQN, with support for DDQN).

## 2. Directory and File Structure

The project is organized as follows:

assignment02/
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── main.py                 # Single entry point for all experiments
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── configs/                # YAML configuration files for experiments
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── ql_train_discrete.yaml
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── sarsa_train_discrete.yaml
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── sarsa_eval_discrete.yaml
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── dqn_fcn_train_move_to_beacon_discrete.yaml  # DQN with MLP
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── dqn_cnn_train_move_to_beacon_full.yaml    # DQN with CNN
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── dqn_cnn_train_defeat_roaches.yaml       # DQN with CNN
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── ddqn_fcn_train_move_to_beacon_discrete.yaml # DDQN with MLP
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── ddqn_cnn_train_move_to_beacon_full.yaml   # DDQN with CNN
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── ddqn_cnn_train_defeat_roaches.yaml      # DDQN with CNN
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── random_agent_discrete.yaml
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   └── basic_agent_discrete.yaml
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── agents/
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── **init**.py
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── abstractAgent.py      # Abstract base class for all agents
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── tableAgent.py         # Base class for Q-Learning and SARSA
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── qlAgent.py            # Q-Learning agent implementation
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── sarsaAgent.py         # SARSA agent implementation
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── dqnAgent.py           # DQN and DDQN agent implementation
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── randomAgent.py        # Agent selecting random actions
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── basicAgent.py         # Heuristic-based agent
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   └── factory.py            # Factory to create agent instances
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── networks/                 # Neural network architectures and factory
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── **init**.py
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── base.py               # Abstract BaseNetwork class
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── architectures.py      # Concrete MLPNetwork, CNNNetwork classes
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   └── factory.py            # Factory to create network instances
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── env/
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── **init**.py
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── env_discrete.py       # MoveToBeaconDiscreteEnv wrapper
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── env_full.py           # MoveToBeaconEnv (visual) wrapper
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── dr_env.py             # DefeatRoachesEnv wrapper
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── utils.py              # Utility functions for environments
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   └── factory.py            # Factory to create environment instances
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── runner/
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   ├── **init**.py
<br>&nbsp;&nbsp;&nbsp;&nbsp;│   └── runner.py             # Experiment runner class
<br>&nbsp;&nbsp;&nbsp;&nbsp;├── models/                   # Default local directory for saving trained models
<br>&nbsp;&nbsp;&nbsp;&nbsp;│                               # (MLflow also stores model artifacts)
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
    1. Parses the `--config` command-line argument to get the path to a YAML configuration file.
    2. Initializes MLflow, sets the experiment name (from the YAML's `experiment_name`), and starts an MLflow run.
    3. Logs the YAML configuration file and all its parameters to MLflow.
    4. Uses `env.factory.create_environment` to create the specified environment instance.
    5. If the agent requires neural networks (e.g., `DQNAgent`):
        * Reads network configuration (e.g., `online_network_config`) from the `agent.params` section of the YAML.
        * Uses `networks.factory.create_network` to build the network instances (e.g., `online_network`, `target_network`).
    6. Uses `agents.factory.create_agent` to create the agent instance, injecting any created networks and other parameters. If evaluating, it calls the agent's `load_model` class method.
    7. Instantiates the `Runner` with the agent, environment, and runner parameters (including the `mlflow_run_id`).
    8. Calls `runner.run()` to start the experiment.
    9. Ends the MLflow run.

### 3.2. `configs/*.yaml`: Experiment Definitions

* **Purpose:** To define all parameters for a specific experiment. This allows for reproducible and easily comparable runs.
* **Structure:**
  * `experiment_name`: Used for MLflow experiment grouping.
  * `environment`: Contains `name` (registered in `env/factory.py`) and `params` for the environment.
  * `agent`: Contains `name` (registered in `agents/factory.py`) and `params`.
    * For network-based agents like `DQNAgent`, `agent.params` will include network configurations (e.g., `online_network_config: {name: "CNNNetwork", params: {...}}`) and algorithm-specific hyperparameters (e.g., `learning_rate`, `enable_ddqn`).
  * `runner`: Contains parameters for the `Runner` class (e.g., `total_episodes`, `is_training`, `load_model_path`, logging directories, save frequency).

### 3.3. `runner/runner.py`: The Experiment Orchestrator

* **Purpose:** Manages the agent-environment interaction loop for training or evaluation.
* **Key Features:**
  * A single, generic `run()` method that works with any agent implementing the `AbstractAgent` interface.
  * Handles episode iterations and step-by-step interactions.
  * Calls agent lifecycle hooks: `agent.on_episode_start()` and `agent.on_episode_end()`.
  * Logs metrics (episodic scores, agent-specific info like epsilon) to both TensorBoard (if `tensorboard_log_dir` is set) and MLflow (if `mlflow_run_id` is provided) via its `summarize()` method.
  * Triggers local model saving by calling `agent.save_model()` periodically based on `save_model_each_episode_num`. The agent's `save_model` method is also responsible for logging the model to MLflow artifacts.

### 3.4. `agents/` Module: The Learning Algorithms

* **`abstractAgent.py`**: Defines the `AbstractAgent` interface. Key methods all agents must implement:
  * `__init__(self, observation_space, action_space, **kwargs)`
  * `get_action(self, state, is_training)`
  * `update(self, state, action, reward, next_state, done, **kwargs)`
  * `on_episode_start(self)`
  * `on_episode_end(self)` (e.g., for per-episode epsilon decay)
  * `get_update_info(self)` (for logging agent metrics like current epsilon)
  * `save_model(self, path, filename)` (should also handle `mlflow.log_artifact` or `mlflow.pytorch.log_model`)
  * `load_model(cls, path, filename, **kwargs)`
* **`tableAgent.py`**: Base class for Q-Learning and SARSA. Handles dynamic Q-table sizing based on `observation_space`, manages `index_offset` for negative states, and implements shared `save_model` (including MLflow artifact logging) and `load_model` logic (`torch.load(..., weights_only=False)`).
* **`qlAgent.py`, `sarsaAgent.py`**: Inherit from `TableBasedAgent` and primarily define their specific `update` rules.
* **`dqnAgent.py`**:
  * Implements DQN and DDQN (toggled by `enable_ddqn` param).
  * Receives pre-built `online_network` and `target_network` instances (of type `BaseNetwork`) via dependency injection in its `__init__`.
  * Handles experience replay, target network updates, and epsilon decay (typically per-step).
  * Its `save_model` method saves the `online_nn.state_dict()` locally and logs the model to MLflow using `mlflow.pytorch.log_model`.
  * Its `load_model` class method reconstructs networks using `network_config` (passed in `**kwargs`) before loading the state dictionary.
* **`randomAgent.py`, `basicAgent.py`**: Simple agents updated to implement the `AbstractAgent` interface methods (mostly with `pass` or default behavior).
* **`factory.py`**: `create_agent(name, params, observation_space, action_space)` function that instantiates agents based on their registered name, passing necessary spaces and parameters from the YAML config (including injected networks for `DQNAgent`).

### 3.5. `networks/` Module: Neural Network Architectures

* **`base.py`**: Defines `BaseNetwork(torch.nn.Module, ABC)`, an abstract class requiring a `forward` method and taking `observation_space` and `action_space` in its constructor. This allows networks to self-configure input/output layers.
* **`architectures.py`**: Contains concrete network implementations like:
  * `MLPNetwork(BaseNetwork)`: A generic MLP whose structure (e.g., `hidden_layers`) can be defined by parameters.
  * `CNNNetwork(BaseNetwork)`: A generic CNN whose structure (e.g., `conv_channels`, `kernel_sizes`, `fc_hidden_size`) can be defined by parameters. It also handles dynamic calculation of flattened size and input channel format detection (CHW vs HWC).
  * Both use `init_weights_xavier` for initialization.
* **`factory.py`**: `create_network(name, observation_space, action_space, params)` function that instantiates networks based on their registered name and parameters from the YAML's network configuration sections.

### 3.6. `env/` Module: The Simulation Worlds

* Contains PySC2 environment wrappers (`MoveToBeaconDiscreteEnv`, `MoveToBeaconEnv`, `DefeatRoachesEnv`) that adhere to the `gym.Env` interface.
* `DefeatRoachesEnv._get_state()` was updated to ensure it returns a base `np.array(..., copy=True)` to prevent potential type issues.
* **`factory.py`**: `create_environment(name, params)` instantiates environments.

## 4. How to Use the Framework

### 4.1. Setup

1. Ensure Python and StarCraft II (with maps) are installed.
2. Create and activate a virtual environment.
3. Install dependencies using your `pyproject.toml` (with `uv sync`) or `requirements.txt` (with `pip install -r requirements.txt`). Key packages: `pysc2`, `gym`, `numpy`, `torch`, `absl-py`, `PyYAML`, `mlflow`, `tensorboard`.

### 4.2. Running an Experiment (Training)

1. **Choose or Create a YAML Config:** Select/create a file in `configs/` (e.g., `configs/dqn_cnn_train_defeat_roaches.yaml`).
2. **Define Parameters:**
    * Set `experiment_name` (this will be your MLflow experiment name).
    * Configure `environment` (`name`, `params`).
    * Configure `agent` (`name`, `params`). For `DQNAgent`, `params` must include `online_network_config` (with `name` like "MLPNetwork" or "CNNNetwork", and its specific `params` like `hidden_layers` or `conv_channels`). Also include `network_config` for loading consistency.
    * Configure `runner` (`total_episodes`, `is_training: true`, logging/saving details).
3. **Execute:**

    ```bash
    python main.py --config=configs/your_config_file.yaml
    ```

4. **Monitor:**
    * Open MLflow UI: `mlflow ui` (in a new terminal from project root), then browse to `http://localhost:5000`.
    * Open TensorBoard: `tensorboard --logdir=./logs`.

### 4.3. Loading a Trained Model and Running Evaluations

1. **Create/Modify an Evaluation Config:**
    * Copy a training config or create a new one (e.g., `configs/dqn_cnn_eval_defeat_roaches.yaml`).
    * Set `runner.is_training: false`.
    * Set `runner.load_model_path` to the *directory* where your trained model was saved locally (e.g., `models/250529_0015_DQNAgent`). The agent's `load_model` method will look for the specific model file(s) inside this directory.
    * For `DQNAgent`, ensure the `agent.params.network_config` in the YAML matches the architecture of the model you are loading. `DQNAgent.load_model` uses this to reconstruct the network structure before loading weights.
    * Set `runner.total_episodes` to the number of evaluation episodes.
    * Optionally, set `environment.params.is_visualize: true`.
2. **Execute:**

    ```bash
    python main.py --config=configs/your_eval_config_file.yaml
    ```

    The `main.py` script will call the agent's `load_model` method. The loaded agent will have exploration turned off (e.g., epsilon set to minimum). Scores will be logged to MLflow under the same `experiment_name` as a new run.

### 4.4. Pausing and Resuming Training

Resuming training exactly where it left off requires saving and loading not just the model weights, but also the state of the optimizer, the replay memory (for DQN), the current episode number, and the current epsilon value.

* **Current State:**
  * **Model Weights:** Saved and loaded.
  * **Optimizer State:** **Not currently saved/loaded.** When a model is loaded, a new optimizer is created.
  * **Replay Memory:** **Not currently saved/loaded.** A fresh, empty replay memory is created.
  * **Episode Number/Epsilon:** The `Runner` starts episodes from 1. Epsilon starts from its initial value.

* **To Implement Full Resumability (Future Enhancement):**
    1. **Saving State:**
        * Modify `DQNAgent.save_model` (and `TableBasedAgent.save_model` if relevant) to save a dictionary containing:
            * `model_state_dict` (already done)
            * `optimizer_state_dict` (`self.optimizer.state_dict()`)
            * `replay_memory_buffer` (convert `self.memory.memory` deque to a list)
            * `current_step_counter` (`self._step_counter`)
            * `current_epsilon` (`self.epsilon`)
        * The `Runner` would need to save its `current_episode_num` and `total_score_runner`. This could be done in a separate checkpoint file associated with the MLflow run or the model save directory.
    2. **Loading State:**
        * Modify `DQNAgent.load_model` to load these components:
            * Load `model_state_dict` (already done).
            * Load `optimizer_state_dict` into `self.optimizer`.
            * Re-populate `self.memory` from the saved buffer.
            * Restore `self._step_counter` and `self.epsilon`.
        * `main.py` or `Runner` would need a way to load the runner's state (start episode, score). A `resume_from_run_id` feature in `main.py` could fetch the last saved episode from MLflow tags/artifacts of a previous run.
    3. **YAML Configuration for Resuming:**
        * Add a `runner.resume_run_id` (MLflow run ID) or `runner.resume_checkpoint_path` to the YAML.
        * `main.py` would use this to load the full training state.

    *This full resumability is a significant feature and is not implemented in the current refactored version based on our discussion so far, but the modular structure makes it easier to add.*

## 5. How to Modify or Extend

### 5.1. Adding a New Agent

1. Create `agents/myNewAgent.py`, subclassing `AbstractAgent`.
2. Implement all required abstract methods.
3. Register in `agents/factory.py`: `AGENT_REGISTRY["MyNewAgent"] = MyNewAgent`.
4. Create a YAML config in `configs/` specifying `agent.name: "MyNewAgent"` and its `params`.

### 5.2. Adding a New Environment

1. Create `env/myNewEnv.py`, subclassing `gym.Env`.
2. Implement `__init__` (defining `observation_space`, `action_space`), `reset`, `step`, etc.
3. Register in `env/factory.py`: `ENV_REGISTRY["MyNewEnv"] = MyNewEnv`.
4. Create a YAML config specifying `environment.name: "MyNewEnv"`.

### 5.3. Adding a New Neural Network Architecture

1. Create your network class in `networks/architectures.py`, subclassing `networks.base.BaseNetwork`.
    * Its `__init__` should take `observation_space`, `action_space`, and `**kwargs` for architecture-specific parameters.
    * Implement the `forward` method.
2. Register in `networks/factory.py`: `NETWORK_REGISTRY["MyCustomNetwork"] = MyCustomNetwork`.
3. In your YAML config, under `agent.params`, set (for example):

    ```yaml
    online_network_config:
      name: "MyCustomNetwork"
      params:
        custom_param1: value1
    network_config: # For loading
      name: "MyCustomNetwork"
      params:
        custom_param1: value1
    ```

This documentation should provide a solid understanding of the refactored project.

# Project Evolution: Implementing CNN-DQN and Usage Guide

This document outlines the key changes made to the project to implement a Deep Q-Network (DQN) agent with a Convolutional Neural Network (CNN) for the `FullMoveToBeaconEnv`. It also provides general instructions on how to use the current project structure and extend it with new algorithms or environments.

## 1. From MLP to CNN: The Need for Visual Processing

The initial DQN agent, as per Assignment 2, was designed with a Multi-Layer Perceptron (MLP) defined in `agents/NN_model.py` (the `Model` class). This MLP is suitable for environments with low-dimensional, vector-based state representations, like the `MoveToBeaconDiscreteEnv`.

However, the `FullMoveToBeaconEnv` (from `env/env_full.py`) provides a much richer, image-like state representation derived from the StarCraft II screen features (e.g., `player_relative` and `unit_density` layers). To effectively learn from this visual input, a CNN is required. CNNs are specifically designed to process spatial data, identify patterns, and extract relevant features from images.

## 2. Key Changes Made to Implement CNN-DQN

The following modifications were made to the codebase to integrate a CNN into the DQN agent and ensure it works correctly with the `FullMoveToBeaconEnv`.

### 2.1. `agents/NN_model.py`: Defining the CNN Architecture

* **New `CNNModel` Class:**
    * A new class, `CNNModel`, was added to define the convolutional neural network.
    * **Architecture:** The final, robust architecture for a 32x32 input with `input_channels` (e.g., 2 for `FullMoveToBeaconEnv`) is:
        * `Conv2d(input_channels, 32, kernel_size=5, padding=2)` -> `ReLU` -> `MaxPool2d(2, 2)` (Output: 16x16x32)
        * `Conv2d(32, 64, kernel_size=5, padding=2)` -> `ReLU` -> `MaxPool2d(2, 2)` (Output: 8x8x64)
        * `Conv2d(64, 64, kernel_size=3, padding=1)` -> `ReLU` -> `MaxPool2d(2, 2)` (Output: 4x4x64)
        * Flatten layer.
        * `Linear(64 * 4 * 4, 512)` -> `ReLU`
        * `Linear(512, num_actions)`
    * This architecture uses padding to maintain feature map sizes within convolutional layers and max pooling for controlled down-sampling.

* **`init_weights` Function:**
    * This existing function, which applies Xavier uniform initialization, is used for both `Model` and `CNNModel` to ensure good starting weights.

* **`CNNModel.forward()` Method:**
    * The `forward` method was carefully designed to:
        1.  Pass the input tensor `x` sequentially through all defined convolutional and pooling layers.
        2.  **Handle Input Dimensions:** Crucially, it now checks if the input `x` is a single 3D state tensor (C, H, W) or a 4D batch of state tensors (B, C, H, W). If it's a single state, an `unsqueeze(0)` operation is performed to add a temporary batch dimension, as PyTorch's `Conv2d` layers expect a 4D input.
        3.  Flatten the output of the convolutional/pooling stack before passing it to the fully connected layers.

### 2.2. `agents/dqnAgent.py`: Adapting the DQN Agent

* **`__init__` Method:**
    * A `use_cnn: bool = False` parameter was added to the constructor. This flag determines whether the agent should use the MLP (`Model`) or the `CNNModel`.
* **`_create_network()` Method:**
    * This helper method was modified to instantiate either `Model` or `CNNModel` based on the `self.use_cnn` flag. It correctly passes `input_channels` (derived from `state_shape`) and `num_actions` to the `CNNModel` constructor.
* **Tensor Handling in `get_action()` and `update()`:**
    * `get_action()`: When selecting an action for a single state, the state tensor is passed to `online_nn.forward()`. The fix in `CNNModel.forward()` ensures this 3D tensor is handled correctly.
    * `update()` (specifically `_get_sample_from_memory` and `_calculate_loss`): When training on a batch of experiences, the state tensors are already in the correct 4D batch format.
* **`save_model()` and `load_model()` Methods:**
    * These methods were fully implemented to save and load the `state_dict` of the `online_nn`. The `load_model` class method now correctly instantiates a new agent and loads the weights into both its `online_nn` and `target_nn`.

### 2.3. `env/env_full.py`: Correcting Action Space

* **`action_shape` Property:**
    * A critical bug was fixed where this property was returning `(num_actions,)` instead of `range(num_actions)`.
    * The corrected version `return range(self.action_space.n)` ensures that `len(env.action_shape)` gives the correct number of actions and `random.choice(env.action_shape)` samples a valid action index. This resolved a `KeyError` during action execution.

### 2.4. New Run Script: `run_full_move_to_beacon.py`

* A new script was created specifically for running experiments with the `FullMoveToBeaconEnv` and the CNN-enabled DQN.
* **Configuration:**
    * It imports `MoveToBeaconEnv` from `env.env_full`.
    * It sets `use_cnn=True` in the agent's configuration dictionary.
    * It includes hyperparameters suitable for training a CNN (e.g., smaller learning rate, larger replay buffer, appropriate epsilon decay for longer training).

### 2.5. Iterative Debugging

The process to arrive at the final working CNN model involved several iterations of debugging:
1.  **Initial `TypeError`:** The `DQNAgent` was missing implementations for `save_model` and `load_model`.
2.  **`KeyError` in `env_full.py`:** The `action_shape` property was incorrect, causing invalid action indices.
3.  **`RuntimeError` (Kernel size > input size):** The initial CNN architecture was too aggressive in down-sampling for a 32x32 input. This was fixed with a more robust CNN design using padding and max pooling.
4.  **`RuntimeError` (mat1 and mat2 shapes cannot be multiplied):** This occurred because the `CNNModel.forward()` method was not correctly passing the input through all layers, and later because it didn't handle single 3D state inputs (from `get_action`) versus 4D batched inputs (from `update`). Both issues were resolved by correcting the `forward` pass logic.

## 3. General Instructions for Using and Modifying the Project

This project is designed to be modular, allowing for easy experimentation and extension.

### 3.1. Running an Experiment

1.  **Choose or Create a Run Script:**
    * Use existing scripts like `run_dqn.py`, `run_ql.py`, or the new `run_full_move_to_beacon.py`.
    * For new experiments, it's best to copy an existing script and modify it.
2.  **Configure the Experiment:**
    * Open your chosen run script (e.g., `run_my_experiment.py`).
    * Modify the `config` dictionary:
        * `config["env"]`: Specify environment parameters (e.g., `screen_size`, `is_visualize`).
        * `config["agent"]`:
            * Set the agent-specific parameters (e.g., `learning_rate`, `epsilon_decay`, `memory_capacity`).
            * For `DQNAgent`, ensure `use_cnn` is set correctly (`True` for visual environments like `FullMoveToBeaconEnv`, `False` for vector environments like `MoveToBeaconDiscreteEnv`).
            * The `state_shape` and `action_shape` in the agent config are typically overwritten by the actual shapes from the initialized environment.
        * `config["runner"]`:
            * `is_training`: Set to `True` for training, `False` for evaluation.
            * `tensorboard_log_dir`: Specify a directory for TensorBoard logs.
            * `save_model_each_episode_num`: Frequency to save the agent's model during training.
            * `model_save_dir`: Directory to store saved models.
        * `config["total_episodes"]`: The number of episodes to run the experiment for.
3.  **Execute the Script:**
    ```bash
    python your_run_script_name.py
    ```

### 3.2. Evaluating a Trained Agent

1.  **Adapt a Run Script:**
    * In the agent configuration part of your run script, instead of creating a new agent directly, use the `load_model` class method:
        ```python
        # Example for DQNAgent
        agent_config = config["agent"]
        agent_config["state_shape"] = env.state_shape # Ensure these are set
        agent_config["action_shape"] = env.action_shape
        agent = DQNAgent.load_model(
            path="path/to/your/saved_model_directory", 
            filename="your_model_file.pt", 
            **agent_config # Pass the original config used for training
        )
        ```
2.  **Set Runner to Evaluation Mode:**
    * In `config["runner"]`, set `is_training: False`.
3.  **Set Low Epsilon for Agent:**
    * Manually set the loaded agent's epsilon to a very small value to ensure it primarily exploits its learned policy:
        ```python
        agent.epsilon = 0.01 # Or lower
        agent.epsilon_min = 0.01 
        ```
4.  Run the script.

### 3.3. Monitoring with TensorBoard

1.  Ensure `tensorboard_log_dir` is set in your runner configuration.
2.  While your experiment is running or after it has finished, open a new terminal.
3.  Navigate to your project's root directory.
4.  Run the command:
    ```bash
    tensorboard --logdir ./logs 
    ```
    (Adjust `./logs` if your `tensorboard_log_dir` points elsewhere).
5.  Open the URL provided by TensorBoard (usually `http://localhost:6006`) in your web browser.

### 3.4. Adding a New Agent

1.  **Create Agent File:** In the `/agents` directory, create a new Python file (e.g., `my_new_agent.py`).
2.  **Define Agent Class:**
    ```python
    from agents.abstractAgent import AbstractAgent
    # Other necessary imports

    class MyNewAgent(AbstractAgent):
        def __init__(self, state_shape: tuple, action_shape: tuple, **kwargs: Any):
            super().__init__(state_shape, action_shape)
            # Initialize your agent's specific parameters, networks, etc.
            # Example: self.my_parameter = kwargs.get('my_parameter', default_value)

        def get_action(self, state: Any) -> Any:
            # Implement logic to select an action based on the state
            pass

        def update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool, **kwargs: Any) -> float | None:
            # Implement the learning/update rule for your algorithm
            # Return new epsilon if applicable, else None
            pass

        def save_model(self, path: str, filename: str = "my_new_agent_model.pt") -> None:
            # Implement logic to save your agent's learned parameters
            pass

        @classmethod
        def load_model(cls, path: str, filename: str = "my_new_agent_model.pt", **kwargs: Any) -> 'MyNewAgent':
            # Implement logic to load parameters and create an agent instance
            # instance = cls(state_shape=kwargs['state_shape'], action_shape=kwargs['action_shape'], ...)
            # Load weights into instance.
            # return instance
            pass
    ```
3.  **Create a Run Script:** Copy an existing `run_*.py` script, import your `MyNewAgent`, and modify the `config["agent"]` section to instantiate and configure your new agent.

### 3.5. Adding a New Environment

1.  **Create Environment File:** In the `/env` directory, create a new Python file (e.g., `my_new_env.py`).
2.  **Define Environment Class:**
    ```python
    import gym
    from pysc2.env import sc2_env
    # Other necessary imports

    class MyNewEnv(gym.Env):
        def __init__(self, **kwargs: Any):
            super().__init__()
            # Initialize your PySC2 environment or other custom environment
            # self._env = sc2_env.SC2Env(...)
            
            # Define action_space and observation_space (must be gym.spaces types)
            # self.action_space = gym.spaces.Discrete(num_actions)
            # self.observation_space = gym.spaces.Box(low=..., high=..., shape=..., dtype=...)

        def reset(self) -> Any:
            # Logic to reset the environment to an initial state
            # Must return the initial observation
            pass

        def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
            # Logic to take an action in the environment
            # Must return: next_observation, reward, done_flag, info_dict
            pass

        def close(self) -> None:
            # Clean up environment resources
            # if hasattr(self, '_env') and self._env is not None:
            #     self._env.close()
            super().close()

        @property
        def state_shape(self) -> tuple:
            return self.observation_space.shape

        @property
        def action_shape(self) -> Any: # e.g., range for discrete, tuple for Box
            if isinstance(self.action_space, gym.spaces.Discrete):
                return range(self.action_space.n)
            # Add handling for other space types if needed
            return self.action_space.shape 
    ```
3.  **Use in Run Script:** Import your `MyNewEnv` in a run script and instantiate it in the `config["env"]` section.

### 3.6. Tips for Further Development

* **Hyperparameter Optimization:** Finding the right hyperparameters is key. Consider systematic approaches like grid search or random search if manual tuning becomes too slow. Tools like Optuna can automate this.
* **Advanced DQN Variants:** Explore implementing extensions to DQN, such as:
    * **Double DQN (DDQN):** Helps mitigate overestimation of Q-values.
    * **Dueling DQN:** Separates the estimation of state values and action advantages.
    * **Prioritized Experience Replay (PER):** Samples more important transitions from the replay buffer more frequently.
* **Other Algorithms:** The `AbstractAgent` interface allows you to implement other RL algorithms (e.g., A2C, A3C, PPO) within the same framework.
* **Different Environments:** Test your agents on other PySC2 mini-games or standard Gym environments (like CartPole, LunarLander) to see how well they generalize.
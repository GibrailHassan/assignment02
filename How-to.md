# Reinforcement Learning with StarCraft II: Project `assignment02`

## 1. Project Overview

### 1.1. Introduction
This project, `assignment02`, focuses on implementing and experimenting with various Reinforcement Learning (RL) agents in the context of Blizzard Entertainment's StarCraft II. Using the PySC2 library, which provides a Python interface to the StarCraft II Learning Environment, the project allows agents to learn policies for accomplishing tasks in simplified mini-games like "MoveToBeacon" and "DefeatRoaches."

The primary goal is to understand, implement, and compare different RL algorithms, from basic Q-Learning and SARSA to more advanced Deep Q-Networks (DQN).

**Key Technologies Used:**
*   **Python:** The primary programming language.
*   **PySC2:** The StarCraft II Learning Environment API by DeepMind.
*   **OpenAI Gym:** The project follows the Gym interface for environment interaction.
*   **PyTorch:** Used for implementing neural networks in the DQN agent.
*   **NumPy:** For numerical operations.
*   **TensorBoard:** For logging and visualizing training progress.

### 1.2. Project Structure

The project is organized into several key directories:

*   `agents/`: Contains the implementations of different RL agents.
    *   `abstractAgent.py`: Defines an abstract base class `AbstractAgent` that all specific agents inherit from. It outlines the common interface (e.g., `get_action`, `update`, `save_model`, `load_model`).
    *   `randomAgent.py`: An agent that selects actions randomly. Useful as a baseline.
    *   `basicAgent.py`: A simple heuristic-based agent designed for the "MoveToBeacon" mini-game. It calculates the move that minimizes the direct distance to the beacon.
    *   `qlAgent.py`: Implements the Q-Learning algorithm using a Q-table.
    *   `sarsaAgent.py`: Implements the SARSA algorithm, also using a Q-table.
    *   `dqnAgent.py`: Implements the Deep Q-Network (DQN) algorithm. This includes the `ReplayMemory` class for experience replay.
    *   `NN_model.py`: Defines the neural network architecture (`Model` class) used by the `DQNAgent`.

*   `env/`: Contains wrappers for the StarCraft II environments, making them compatible with the OpenAI Gym interface.
    *   `env_discrete.py`: Provides `MoveToBeaconDiscreteEnv`. This environment simplifies the "MoveToBeacon" mini-game by discretizing the state (relative distance to beacon) and action space (8 movement directions).
    *   `env_full.py`: Provides `MoveToBeaconEnv`. This is a more complex version for "MoveToBeacon" that can use richer screen features as state and a larger action space.
    *   `dr_env.py`: Provides `DefeatRoachesEnv` for the "DefeatRoaches" mini-game, where the agent controls marines to defeat enemy roaches. The state is typically derived from screen features.
    *   `utils.py`: Contains utility functions for environment interactions, such as calculating target positions, discretizing distances, and preprocessing observation channels.

*   `runner/`:
    *   `runner.py`: The `Runner` class orchestrates the training and evaluation loops. It handles agent-environment interactions, episode management, scoring, logging to TensorBoard, and saving models.

*   `run_*.py` scripts: These are the main executable scripts for training or evaluating specific agents.
    *   `run_dqn.py`: Trains the DQN agent on `MoveToBeaconDiscreteEnv`.
    *   `run_ql.py`: Trains the Q-Learning agent on `MoveToBeaconDiscreteEnv`.
    *   `run_sarsa.py`: Trains the SARSA agent on `MoveToBeaconDiscreteEnv`.
    *   `run_basic.py`: Runs the RandomAgent or BasicAgent (commented out) on `MoveToBeaconDiscreteEnv`.
    *   `run_dr_env.py`: Trains the DQN agent on `DefeatRoachesEnv`.
    *   `run_eval_ql.py`, `run_eval_sarsa.py`: Scripts for evaluating pre-trained Q-Learning and SARSA models, respectively.

*   `models/` (Implicit): This directory is where trained agent models are typically saved by the `Runner`.
*   `logs/` (Implicit): This directory is where TensorBoard logs are saved by the `Runner`.
*   `ignore_me_migrate_model.py`: A utility script, likely used for converting older model formats (e.g., pickle to PyTorch tensor) for Q-learning agents.
*   `__init__.py`: An empty file in the project root that helps Python recognize the `assignment02` directory as a package and modifies `sys.path` to allow imports from the parent directory if needed.

### 1.3. How to Run

1.  **Prerequisites:**
    *   Install StarCraft II (the full game or the free Starter Edition).
    *   Install the mini-game map packs from DeepMind's PySC2 repository.
    *   Install Python and the required libraries: `pysc2`, `gym`, `numpy`, `torch`, `absl-py`, `tensorboard`. You can usually install these via pip:
        ```bash
        pip install pysc2 gym numpy torch absl-py tensorboard
        ```

2.  **Running an Experiment:**
    *   Navigate to the `assignment02` project directory in your terminal.
    *   To train an agent, execute one of the `run_*.py` scripts. For example, to train the DQN agent:
        ```bash
        python run_dqn.py
        ```
    *   To evaluate a pre-trained agent (assuming you have saved models in the `models/` directory):
        ```bash
        python run_eval_ql.py
        ```

3.  **Configuration:**
    *   Most run scripts (`run_*.py`) define environment parameters (like `screen_size`, `step_mul`, `is_visualize`) and agent hyperparameters (like `learning_rate`, `epsilon`, `batch_size`) directly within the script. You can modify these values to experiment.
    *   Setting `is_visualize=True` in an environment's initialization will render the StarCraft II game, allowing you to watch the agent play. This significantly slows down training.

## 2. Core Reinforcement Learning Concepts

Reinforcement Learning is a subfield of machine learning where an **agent** learns to make a sequence of decisions by interacting with an **environment** to maximize a cumulative **reward**.

### 2.1. Agent
The learner and decision-maker.
*   **StarCraft II Example:** In the "MoveToBeacon" mini-game, the agent is the single Terran Marine unit. In "DefeatRoaches," the agent controls a group of Marines. The Python classes like `DQNAgent`, `QLearningAgent` in `agents/` are the software implementations of these agents.

### 2.2. Environment
The external system with which the agent interacts. It presents situations to the agent and gives feedback (rewards and new states) based on the agent's actions.
*   **StarCraft II Example:** The "MoveToBeacon" map itself, including the Marine's starting position and the Beacon's location, constitutes the environment. The `MoveToBeaconDiscreteEnv` class in `env/env_discrete.py` is a Python wrapper around the PySC2 environment.

### 2.3. State (s)
A representation of the environment at a particular time. It's the information the agent uses to make decisions.
*   **StarCraft II Example:**
    *   In `MoveToBeaconDiscreteEnv`, the state is a 2D vector `[dx, dy]` representing the discretized relative distance from the Marine to the Beacon.
    *   In `DefeatRoachesEnv` or `MoveToBeaconEnv` (full version), the state can be a more complex representation, such as a stack of feature layers from the game screen (e.g., `player_relative` showing friendly/enemy units, `unit_density`).

### 2.4. Action (a)
A choice made by the agent that influences the environment.
*   **StarCraft II Example:**
    *   In `MoveToBeaconDiscreteEnv`, an action is one of 8 discrete movement directions (e.g., "up", "up-left").
    *   In `DefeatRoachesEnv`, an action might be an "Attack_screen" command targeting a specific (x,y) coordinate on the game screen.

### 2.5. Reward (r)
A scalar feedback signal from the environment that indicates how good or bad the agent's last action was in the context of the current state. The agent's goal is to maximize the total reward over time.
*   **StarCraft II Example:**
    *   In "MoveToBeacon," the agent receives a reward of +1 when it reaches the beacon, and 0 for other steps.
    *   In "DefeatRoaches," the reward is typically the change in the game's cumulative score, which might increase when an enemy Roach is damaged or killed.

### 2.6. Policy (π)
The agent's strategy or decision-making function. It maps states to actions (deterministic policy) or probabilities of taking each action in a state (stochastic policy).
*   **StarCraft II Example:**
    *   For `QLearningAgent`, the policy is derived from its Q-table: in a given state, choose the action with the highest Q-value (exploitation) or a random action (exploration).
    *   For `DQNAgent`, the policy is represented by its neural network, which outputs Q-values for all actions given a state.

### 2.7. Value Function (V(s) and Q(s,a))
Value functions estimate the expected cumulative future reward.
*   **State-Value Function V(s):** The expected return starting from state *s* and then following policy *π*.
*   **Action-Value Function Q(s,a):** The expected return starting from state *s*, taking action *a*, and thereafter following policy *π*. This is often what agents try to learn.
*   **StarCraft II Example:**
    *   The `QLearningAgent` and `SARSAAgent` maintain a Q-table, where `Q_table[state_x, state_y, action_idx]` stores the estimated Q-value for taking a specific action in a discretized state.
    *   The `DQNAgent`'s neural network approximates the Q(s,a) function. Given a state, it outputs a Q-value for each possible action.

### 2.8. Model (Model-free vs. Model-based)
An agent might have an internal model of the environment, which predicts how the environment will respond to its actions (i.e., P(s',r | s,a) - the probability of transitioning to state s' with reward r, given state s and action a).
*   **Model-free RL:** Agents learn the optimal policy or value function directly from experience, without explicitly building a model of the environment. Most agents in this project (`QLearningAgent`, `SARSAAgent`, `DQNAgent`) are model-free.
*   **Model-based RL:** Agents learn a model of the environment and then use this model for planning (e.g., by simulating experiences).
*   **StarCraft II Example:** The agents in this project don't learn an explicit model of StarCraft II's game dynamics. They learn directly by playing.

### 2.9. Exploration vs. Exploitation
A fundamental trade-off in RL:
*   **Exploration:** Trying out new actions to discover potentially better strategies and gather more information about the environment.
*   **Exploitation:** Using the current best-known strategy to maximize immediate reward.
*   **StarCraft II Example:** The Q-Learning, SARSA, and DQN agents in this project use an **epsilon-greedy** strategy. With probability `epsilon`, the agent chooses a random action (explore). With probability `1-epsilon`, it chooses the action believed to be best based on current Q-value estimates (exploit). `epsilon` typically decays over time, shifting from exploration to exploitation.

### 2.10. Markov Decision Process (MDP)
RL problems are often formalized as MDPs. An MDP is defined by a tuple (S, A, P, R, γ), where S is the set of states, A is the set of actions, P is the state transition probability function, R is the reward function, and γ is the discount factor. A key assumption is the **Markov property**: the future is independent of the past given the present state.
*   **StarCraft II Example:** The "MoveToBeacon" task, when simplified to (Marine_pos, Beacon_pos) as the state, can be approximated as an MDP. The current positions are assumed to contain all necessary information to predict the outcome of the next move.

### 2.11. Learning Algorithms

#### 2.11.1. Q-Learning
An **off-policy temporal difference (TD) control** algorithm.
*   It learns the optimal action-value function Q*(s,a) directly.
*   "Off-policy" means it learns about the optimal policy while possibly following a different (e.g., exploratory) policy.
*   Update rule (simplified): `Q(s,a) <- Q(s,a) + α * [r + γ * max_a'(Q(s',a')) - Q(s,a)]`
    *   `α` is the learning rate.
    *   `γ` is the discount factor.
    *   It updates the Q-value of the current state-action pair `(s,a)` based on the reward `r` received, and the maximum Q-value achievable in the next state `s'`, according to its current estimates.
*   **StarCraft II Example:** Implemented in `agents/qlAgent.py`. The agent updates its Q-table after each step.

#### 2.11.2. SARSA (State-Action-Reward-State-Action)
An **on-policy temporal difference (TD) control** algorithm.
*   It also learns an action-value function Q(s,a).
*   "On-policy" means it learns about the policy it is currently following (including its exploratory actions).
*   Update rule (simplified): `Q(s,a) <- Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]`
    *   The key difference from Q-Learning is that it uses the Q-value of the *actual next action a'* taken in the next state *s'* for the update, rather than the maximum possible Q-value.
*   **StarCraft II Example:** Implemented in `agents/sarsaAgent.py`. The agent updates its Q-table using the (s, a, r, s', a') tuple.

#### 2.11.3. Deep Q-Network (DQN)
Extends Q-Learning to handle large or continuous state spaces by using a deep neural network to approximate the Q(s,a) function. This is crucial for complex environments like StarCraft II where a Q-table would be intractably large.
*   **Key Innovations:**
    *   **Experience Replay:** Transitions `(s, a, r, s', done)` are stored in a replay memory (buffer). The network is trained on mini-batches randomly sampled from this buffer. This breaks temporal correlations in the training data, leading to more stable learning. Implemented via `ReplayMemory` in `agents/dqnAgent.py`.
    *   **Target Network:** A separate neural network (the "target network") is used to generate the target Q-values `(r + γ * max_a'(Q_target(s',a')))`. The weights of this target network are periodically copied from the main "online" network that is being actively trained. This helps stabilize learning by providing more consistent targets.
*   **StarCraft II Example:** Implemented in `agents/dqnAgent.py`, using the neural network defined in `agents/NN_model.py`. It's used for both `MoveToBeaconDiscreteEnv` and the more complex `DefeatRoachesEnv`.

### 2.12. Temporal Difference (TD) Learning
A class of model-free RL methods that learn by bootstrapping from the current estimate of the value function. They update value estimates based on observed rewards and the estimated value of subsequent states, without waiting for the final outcome of an episode. Both Q-Learning and SARSA are TD learning methods.

### 2.13. Discount Factor (γ)
A value between 0 and 1 that determines the present value of future rewards.
*   If `γ` is close to 0, the agent is "myopic" or short-sighted, focusing mainly on immediate rewards.
*   If `γ` is close to 1, the agent is "far-sighted," valuing future rewards highly and aiming for long-term success.
*   **StarCraft II Example:** The `discount_factor` parameter in the agent constructors (`QLearningAgent`, `SARSAAgent`, `DQNAgent`) represents `γ`.

## 3. Agents Implemented

### 3.1. Random Agent (`randomAgent.py`)
*   **Strategy:** Chooses actions uniformly at random from the available action space.
*   **Learning:** Does not learn.
*   **Purpose:** Serves as a simple baseline to compare the performance of learning agents. If a learning agent cannot outperform the random agent, it indicates a problem with the learning setup.
*   **Features:** Keeps a history of states and actions taken, which can be saved.

### 3.2. Basic Agent (`basicAgent.py`)
*   **Strategy:** A heuristic-based, non-learning agent specifically for the "MoveToBeacon" environment. It iterates through all possible actions and simulates the resulting distance to the beacon, choosing the action that minimizes this distance.
*   **Learning:** Does not learn.
*   **Purpose:** Provides a rule-based baseline that should perform reasonably well on the "MoveToBeacon" task.
*   **Note:** Its `get_action` method requires the environment instance itself to call `env.roll_to_next_state(action)`.

### 3.3. Q-Learning Agent (`qlAgent.py`)
*   **Algorithm:** Implements the Q-Learning algorithm.
*   **State/Action Value Representation:** Uses a PyTorch tensor as a Q-table to store `Q(s,a)` values. The state is assumed to be discretizable into indices for this table (e.g., `(x_coord_discrete, y_coord_discrete)`).
*   **Exploration:** Uses an epsilon-greedy strategy. `epsilon` decays over episodes.
*   **Update:** Off-policy TD update.
*   **Key Hyperparameters:** `learning_rate`, `discount_factor`, `epsilon`, `epsilon_decay`, `epsilon_min`.

### 3.4. SARSA Agent (`sarsaAgent.py`)
*   **Algorithm:** Implements the SARSA algorithm.
*   **State/Action Value Representation:** Similar to Q-Learning, uses a PyTorch tensor as a Q-table.
*   **Exploration:** Uses an epsilon-greedy strategy with decaying `epsilon`.
*   **Update:** On-policy TD update, using the (state, action, reward, next_state, next_action) tuple.
*   **Key Hyperparameters:** `learning_rate`, `discount_factor`, `epsilon`, `epsilon_decay`, `epsilon_min`.

### 3.5. Deep Q-Network (DQN) Agent (`dqnAgent.py`)
*   **Algorithm:** Implements the Deep Q-Network algorithm.
*   **State/Action Value Representation:** Uses a neural network (defined in `agents/NN_model.py`) to approximate the `Q(s,a)` function. The input to the network is the state, and the output is a Q-value for each possible action.
*   **Key Features:**
    *   **Experience Replay:** Uses `ReplayMemory` (a deque) to store transitions and sample mini-batches for training.
    *   **Target Network:** Maintains a separate target network whose weights are periodically updated from the online network to stabilize Q-value targets.
    *   **Optimizer:** Uses AdamW (`torch.optim.AdamW`) for training the online network.
    *   **Loss Function:** Mean Squared Error (MSE) between predicted Q-values and target Q-values.
*   **Exploration:** Epsilon-greedy strategy with decaying `epsilon`.
*   **Key Hyperparameters:** `batch_size`, `learning_rate`, `discount_factor`, `epsilon`, `epsilon_decay`, `epsilon_min`, `memory_capacity`. The `target_update_freq` (how often to update the target network) is managed by the `Runner`'s `run_dqn` method via the `c` parameter.

## 4. Environments

### 4.1. `MoveToBeaconDiscreteEnv` (`env/env_discrete.py`)
*   **Mini-game:** "MoveToBeacon" from PySC2.
*   **Goal:** Navigate a single Terran Marine to a randomly placed Beacon.
*   **State Representation:** A 2D NumPy array `[dx, dy]` where `dx` and `dy` are the discretized distances from the Marine to the Beacon along the x and y axes. The `distance_discrete_range` parameter in the constructor controls the level of discretization.
*   **Action Space:** `gym.spaces.Discrete(8)`, representing 8 movement directions (up, down, left, right, and diagonals). The `distance` parameter controls how far the marine attempts to move in one step.
*   **Reward:** +1 for reaching the beacon, 0 otherwise. A new beacon appears when one is reached.
*   **Used by:** `qlAgent.py`, `sarsaAgent.py`, `dqnAgent.py` (in `run_dqn.py`), `randomAgent.py`, `basicAgent.py`.

### 4.2. `MoveToBeaconEnv` (`env/env_full.py`)
*   **Mini-game:** "MoveToBeacon".
*   **State Representation:** More complex than the discrete version. It uses a stack of feature screen layers from PySC2, such as `player_relative` (showing friendly, enemy, neutral units) and `unit_density`. The shape is typically `(num_channels, screen_size, screen_size)`.
*   **Action Space:** `gym.spaces.Discrete(distance_range * num_directions)`. Actions are flattened combinations of movement direction and distance.
*   **Reward:** Similar to `MoveToBeaconDiscreteEnv`.
*   **Purpose:** Allows agents (typically DQN) to learn from richer, more image-like input.

### 4.3. `DefeatRoachesEnv` (`env/dr_env.py`)
*   **Mini-game:** "DefeatRoaches" from PySC2.
*   **Goal:** Control a group of Marines to defeat a group of enemy Roaches.
*   **State Representation:** Typically uses the `player_relative` screen feature, providing a spatial map of friendly and enemy units. Shape is `(screen_size, screen_size, 1)`.
*   **Action Space:** `gym.spaces.Discrete(screen_size * screen_size)`. Each action corresponds to an "Attack_screen" command on one of the `screen_size * screen_size` pixels.
*   **Reward:** Based on the cumulative score provided by PySC2, which reflects damage dealt and units killed.
*   **Used by:** `dqnAgent.py` (in `run_dr_env.py`).

## 5. Running Experiments and Evaluation

### 5.1. Training
*   **Scripts:** Use the `run_*.py` scripts (e.g., `run_dqn.py`, `run_ql.py`, `run_dr_env.py`) to start training sessions.
*   **Configuration:** Agent hyperparameters and environment settings are usually defined at the beginning of these scripts.
    *   `is_visualize=True` in environment setup will render the game.
    *   `episodes` in the runner's call (e.g., `runner.run_dqn(episodes=700, c=250)`) determines the length of training.
*   **Model Saving:** The `Runner` class automatically saves agent models.
    *   Models are saved in a timestamped sub-directory within `models/` (e.g., `models/231026_1030_DQNAgent/`).
    *   The frequency of saving is controlled by `save_model_each_episode_num` in the `Runner`'s constructor.
    *   DQN models are saved as `.pt` files containing state dictionaries and hyperparameters. Q-table agents also save their tables as `.pt` files.
*   **Logging:**
    *   The `Runner` logs metrics like episodic score, total score, mean score (sliding window), and epsilon to TensorBoard.
    *   Logs are saved in a timestamped sub-directory within `logs/`.
    *   To view logs, run `tensorboard --logdir=logs` in your terminal and navigate to the provided URL in a web browser.
    *   Console output also provides per-episode summaries.

### 5.2. Evaluation
*   **Scripts:** Use `run_eval_*.py` scripts (e.g., `run_eval_ql.py`, `run_eval_sarsa.py`).
*   **Loading Models:** These scripts load pre-trained models using the agent's `load_model` class method, specifying the path to the saved model directory.
*   **Exploitation Mode:** For evaluation, `epsilon` is typically set to its minimum value (e.g., `agent.epsilon = agent.epsilon_min`) to ensure the agent exploits its learned knowledge rather than exploring.
*   The `Runner` is initialized with `is_training=False`.

## 6. Further Exploration and Potential Improvements

This project provides a solid foundation for exploring various RL concepts. Here are some potential avenues for further work:

*   **Advanced DQN Variants:**
    *   **Double DQN (DDQN):** To reduce overestimation of Q-values.
    *   **Dueling DQN:** To separately estimate state values and action advantages.
    *   **Prioritized Experience Replay (PER):** To sample important transitions more frequently from the replay buffer. The `ReplayMemory` class has commented-out placeholders for PER.
*   **Policy Gradient Methods:**
    *   Implement REINFORCE, A2C (Advantage Actor-Critic), or A3C (Asynchronous Advantage Actor-Critic). These methods learn the policy directly, which can be beneficial in continuous action spaces or for stochastic policies.
*   **More Complex Environments:**
    *   Tackle more challenging StarCraft II mini-games or even aspects of the full game.
    *   Use the `MoveToBeaconEnv` (full version) or enhance `DefeatRoachesEnv` with more sophisticated state and action representations.
*   **State Representation:**
    *   For DQN, experiment with different neural network architectures (e.g., convolutional layers for processing raw screen pixels directly, as hinted in `NN_model.py`).
    *   Explore different combinations of PySC2 feature layers.
*   **Hyperparameter Tuning:** Systematically tune hyperparameters for all agents and environments to optimize performance.
*   **Code Refinements:**
    *   Centralize hyperparameter configurations (e.g., using config files or command-line arguments via `argparse` or `absl.flags` more extensively).
    *   Enhance model checkpointing and resuming capabilities in the `Runner`.
    *   Add more comprehensive unit tests.


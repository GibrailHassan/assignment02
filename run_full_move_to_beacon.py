# run_full_move_to_beacon.py

"""
Experiment runner for training a DQN agent with a CNN on the
Full Move to Beacon environment.

This script sets up the environment, agent, and runner with appropriate
hyperparameters for this more complex, image-based task. [cite: 55]
"""

import sys
from absl import flags

from agents.dqnAgent import DQNAgent
from env.env_full import MoveToBeaconEnv  # Import the full environment
from runner.runner import Runner

# PySC2 routine
FLAGS = flags.FLAGS
FLAGS(sys.argv)


def main():
    """
    Configures and runs the DQN agent on the FullMoveToBeaconEnv.
    """
    # --- Configuration Block ---
    config = {
        "env": {
            "distance_range": 8,
            "screen_size": 32,
            "step_mul": 8,
            "is_visualize": False,
        },
        "agent": {
            "use_cnn": True,  # CRITICAL: Enable the CNN model
            # Hyperparameters
            "batch_size": 32,
            "learning_rate": 0.00025,
            "discount_factor": 0.99,
            "epsilon": 1.0,
            "epsilon_decay": (1.0 - 0.1) / 100000,  # Linear decay over 100k steps
            "epsilon_min": 0.1,
            "memory_capacity": 100000,
            "target_update_freq": 1000,
        },
        "runner": {
            "is_training": True,
            "tensorboard_log_dir": "./logs/dqn_full_env",
            "save_model_each_episode_num": 100,
            "model_save_dir": "./models",
        },
        "total_episodes": 5000,  # This complex environment will need more training
    }

    # --- Execution Block ---

    # 1. Initialize the Environment using parameters from the assignment [cite: 60]
    env = MoveToBeaconEnv(**config["env"])

    # 2. Initialize the Agent
    config["agent"]["state_shape"] = env.state_shape
    config["agent"]["action_shape"] = env.action_shape
    agent = DQNAgent(**config["agent"])

    # 3. Initialize the Runner
    runner = Runner(agent=agent, env=env, **config["runner"])

    # 4. Start the experiment
    runner.run(config["total_episodes"])


if __name__ == "__main__":
    main()

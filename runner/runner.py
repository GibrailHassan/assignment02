# runner/runner.py

"""
Defines the Runner class, which orchestrates the agent-environment interaction loop.
"""
import datetime
import os
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agents.abstractAgent import AbstractAgent
from env.env_discrete import MoveToBeaconDiscreteEnv


class Runner:
    """
    Orchestrates the training and evaluation loops for an RL agent.

    The Runner handles episode management, agent-environment interaction,
    logging to TensorBoard, and saving model checkpoints. It is designed
    to be generic and work with any agent that conforms to the AbstractAgent
    interface.
    """

    def __init__(
        self,
        agent: AbstractAgent,
        env: MoveToBeaconDiscreteEnv,  # You might want to use a more generic gym.Env type hint
        is_training: bool = True,
        total_episodes: int = 1000,
        save_model_each_episode_num: int = 50,
        tensorboard_log_dir: str = "./logs",
        model_save_dir: str = "./models",
        **kwargs: Any,  # To catch any extra runner params from config
    ) -> None:
        self.agent = agent
        self.env = env
        self.is_training = is_training
        self.total_episodes = total_episodes
        self.save_model_each_episode_num = save_model_each_episode_num

        self.total_score = 0
        self.episode = 1
        self.sliding_window = [-1] * 10
        self.curr_epsilon = 1.0  # Default value

        # --- Model Saving Setup ---
        self.is_saving_model = (
            self.is_training and self.save_model_each_episode_num > 0 and model_save_dir
        )
        if self.is_saving_model:
            self.models_path = os.path.join(
                os.path.abspath(model_save_dir),
                f'{datetime.datetime.now().strftime("%y%m%d_%H%M")}_{type(agent).__name__}',
            )

        # --- TensorBoard Logging Setup ---
        if tensorboard_log_dir:
            train_eval_prefix = "_train_" if self.is_training else "_eval_"
            self.tb_path = os.path.join(
                os.path.abspath(tensorboard_log_dir),
                f'{datetime.datetime.now().strftime("%y%m%d_%H%M")}{train_eval_prefix}{type(agent).__name__}',
            )
            self.writer = SummaryWriter(self.tb_path)
            print(f"TensorBoard logs will be saved to: {self.tb_path}")
        else:
            self.writer = None

    def summarize(self, episodic_score: float) -> None:
        """Logs metrics to the console and TensorBoard at the end of an episode."""
        self.total_score += episodic_score
        print(
            f"Finished episode {self.episode}/{self.total_episodes}, "
            f"Episodic Score: {episodic_score:.2f}, Total Score: {self.total_score:.2f}"
        )

        # Update and log metrics to TensorBoard
        if self.writer:
            self.writer.add_scalar("Episodic_Score", episodic_score, self.episode)
            self.writer.add_scalar("Total_Score", self.total_score, self.episode)

            # Get and log agent-specific info (like epsilon)
            agent_info = self.agent.get_update_info()
            if agent_info:
                for key, value in agent_info.items():
                    self.writer.add_scalar(f"Agent/{key}", value, self.episode)

            # Log mean score over a sliding window
            self.sliding_window.pop(0)
            self.sliding_window.append(episodic_score)
            if self.episode > 10:
                mean_score = np.mean(self.sliding_window)
                self.writer.add_scalar(
                    "Mean_Score_10_Episodes", mean_score, self.episode
                )
                print(f"Mean score over last 10 episodes: {mean_score:.2f}")

        # --- Model Checkpointing ---
        if (
            self.is_saving_model
            and self.episode % self.save_model_each_episode_num == 0
        ):
            self.agent.save_model(self.models_path)

        self.episode += 1

    def run(self) -> None:
        """
        The main generic run loop for training or evaluation.

        This single method replaces all the previous agent-specific run methods.
        It handles the logic for both on-policy (like SARSA) and off-policy
        (like Q-Learning, DQN) agents.
        """
        for _ in range(self.total_episodes):
            state = self.env.reset()
            self.agent.on_episode_start()
            done = False
            episodic_score = 0

            # The main interaction loop for a single episode
            while not done:
                # 1. Agent selects an action
                action = self.agent.get_action(state, is_training=self.is_training)

                # 2. Environment executes the action
                next_state, reward, done, _ = self.env.step(action)

                # 3. Agent learns from the experience
                if self.is_training:
                    # For on-policy methods, the agent might need the next action.
                    # We provide it here. Off-policy agents can simply ignore it.
                    next_action = self.agent.get_action(
                        next_state, is_training=self.is_training
                    )
                    self.agent.update(
                        state, action, reward, next_state, done, next_action=next_action
                    )

                # 4. Update state and score
                state = next_state
                episodic_score += reward

            # --- End of Episode ---
            self.agent.on_episode_end()
            self.summarize(episodic_score)

        # --- End of Experiment ---
        print(f"Total episodes run: {self.episode - 1}")
        if self.is_saving_model:
            self.agent.save_model(self.models_path)
        # Close TensorBoard writer if it was created
        if self.writer:
            self.writer.close()
        self.env.close()
        print("Experiment finished.")

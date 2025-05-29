"""
Defines the Runner class, which orchestrates the agent-environment interaction loop,
now using utility functions for metric logging.
"""

import datetime
import os
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym

# Import from the utility module
from utils.experiment_utils import (
    log_metrics_to_tensorboard,
    log_metrics_to_mlflow,
)

from agents.abstractAgent import AbstractAgent


class Runner:
    """
    Orchestrates the training and evaluation loops for an RL agent.
    Uses utility functions for logging to TensorBoard and MLflow.
    """

    def __init__(
        self,
        agent: AbstractAgent,
        env: gym.Env,
        is_training: bool = True,
        total_episodes: int = 1000,
        save_model_each_episode_num: int = 0,
        tensorboard_log_dir: str = "./logs",
        model_save_dir: str = "./models",
        mlflow_run_id: str = None,
        **kwargs: Any,
    ) -> None:
        self.agent = agent
        self.env = env
        self.is_training = is_training
        self.total_episodes = total_episodes
        self.save_model_each_episode_num = save_model_each_episode_num
        self.mlflow_run_id = mlflow_run_id

        self.total_score_runner = 0.0
        self.current_episode_num = 1
        self.sliding_window = [-1.0] * 10

        self.is_saving_model_locally = (
            self.is_training and self.save_model_each_episode_num > 0 and model_save_dir
        )
        if self.is_saving_model_locally:
            current_time_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            self.local_models_path = os.path.join(
                os.path.abspath(model_save_dir),
                f"{current_time_str}_{type(agent).__name__}",
            )
            os.makedirs(self.local_models_path, exist_ok=True)
            print(
                f"Local models for this run will be saved in: {self.local_models_path}"
            )
        else:
            self.local_models_path = None

        if tensorboard_log_dir:
            train_eval_prefix = "_train_" if self.is_training else "_eval_"
            current_time_str_tb = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            self.tb_path = os.path.join(
                os.path.abspath(tensorboard_log_dir),
                f"{current_time_str_tb}{train_eval_prefix}{type(agent).__name__}",
            )
            self.writer = SummaryWriter(self.tb_path)
            print(f"TensorBoard logs will be saved to: {self.tb_path}")
        else:
            self.writer = None

    def summarize(self, episodic_score: float) -> None:
        """Logs metrics to the console, TensorBoard, and MLflow at the end of an episode."""
        self.total_score_runner += episodic_score

        print(
            f"Finished episode {self.current_episode_num}/{self.total_episodes}, "
            f"Episodic Score: {episodic_score:.2f}, Runner Total Score: {self.total_score_runner:.2f}"
        )

        metrics_to_log = {
            "Episodic_Score": episodic_score,
            "Runner_Total_Score": self.total_score_runner,
        }

        agent_info = self.agent.get_update_info()
        if agent_info:
            for key, value in agent_info.items():
                metrics_to_log[f"Agent_{key}"] = value

        self.sliding_window.pop(0)
        self.sliding_window.append(episodic_score)
        if self.current_episode_num >= 10:
            mean_score = np.mean(self.sliding_window)
            metrics_to_log["Mean_Score_10_Episodes"] = mean_score
            print(f"Mean score over last 10 episodes: {mean_score:.2f}")

        log_metrics_to_tensorboard(
            self.writer, metrics_to_log, self.current_episode_num
        )
        log_metrics_to_mlflow(metrics_to_log, self.current_episode_num)

        if (
            self.is_saving_model_locally
            and self.current_episode_num % self.save_model_each_episode_num == 0
        ):
            checkpoint_filename = f"checkpoint_ep{self.current_episode_num}.pt"
            saved_model_file_path = self.agent.save_model(
                self.local_models_path, filename=checkpoint_filename
            )

            if self.mlflow_run_id and saved_model_file_path and mlflow.active_run():
                try:
                    mlflow.set_tag(
                        f"local_checkpoint_ep_{self.current_episode_num}",
                        saved_model_file_path,
                    )
                except Exception as e:
                    print(
                        f"Warning: Failed to set MLflow tag for local checkpoint: {e}"
                    )

        self.current_episode_num += 1

    def run(self) -> None:
        """
        The main generic run loop for training or evaluation.
        """
        print(
            f"Runner starting. Training: {self.is_training}. Total episodes: {self.total_episodes}."
        )
        try:
            for _ in range(self.total_episodes):
                state = self.env.reset()
                self.agent.on_episode_start()
                done = False
                episodic_score = 0.0

                while not done:
                    action = self.agent.get_action(state, is_training=self.is_training)
                    next_state, reward, done, info = self.env.step(action)

                    if self.is_training:
                        next_action = self.agent.get_action(
                            next_state, is_training=self.is_training
                        )
                        self.agent.update(
                            state,
                            action,
                            reward,
                            next_state,
                            done,
                            next_action=next_action,
                        )

                    state = next_state
                    episodic_score += reward

                self.agent.on_episode_end()
                self.summarize(episodic_score)

        finally:
            if self.writer:
                self.writer.close()
            self.env.close()
            print("Experiment finished by Runner (resources closed).")

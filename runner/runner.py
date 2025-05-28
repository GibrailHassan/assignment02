# runner/runner.py

"""
Defines the Runner class, which orchestrates the agent-environment interaction loop,
now with MLflow metric logging capabilities.
"""
import datetime
import os
from typing import Any

import numpy as np

# import torch # Not directly used in this file anymore
from torch.utils.tensorboard import SummaryWriter  # Keep for TensorBoard

# MLflow import
import mlflow

from agents.abstractAgent import AbstractAgent
import gym  # For type hinting env

# Default environment type hint if not all envs inherit from a common base
# from env.env_discrete import MoveToBeaconDiscreteEnv


class Runner:
    """
    Orchestrates the training and evaluation loops for an RL agent.

    The Runner handles episode management, agent-environment interaction,
    logging to TensorBoard and MLflow, and triggering model saving.
    """

    def __init__(
        self,
        agent: AbstractAgent,
        env: gym.Env,  # Use generic gym.Env
        is_training: bool = True,
        total_episodes: int = 1000,
        save_model_each_episode_num: int = 0,  # Default to 0 (no saving from runner)
        tensorboard_log_dir: str = "./logs",
        model_save_dir: str = "./models",  # Local model saving path
        mlflow_run_id: str = None,  # Added for MLflow integration
        **kwargs: Any,
    ) -> None:
        self.agent = agent
        self.env = env
        self.is_training = is_training
        self.total_episodes = total_episodes
        self.save_model_each_episode_num = save_model_each_episode_num
        self.mlflow_run_id = mlflow_run_id  # Store MLflow run ID

        self.total_score_runner = (
            0.0  # Renamed to avoid conflict if env has total_score
        )
        self.current_episode_num = 1  # Renamed from self.episode
        self.sliding_window = [-1.0] * 10  # Use float for scores

        # --- Local Model Saving Setup ---
        # The agent's save_model method will handle MLflow artifact logging.
        # Runner's responsibility is to call it periodically.
        self.is_saving_model_locally = (
            self.is_training and self.save_model_each_episode_num > 0 and model_save_dir
        )
        if self.is_saving_model_locally:
            # Create a unique directory for this run's locally saved models
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

        # --- TensorBoard Logging Setup ---
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

        # Console Output
        print(
            f"Finished episode {self.current_episode_num}/{self.total_episodes}, "
            f"Episodic Score: {episodic_score:.2f}, Runner Total Score: {self.total_score_runner:.2f}"
        )

        # TensorBoard Logging
        if self.writer:
            self.writer.add_scalar(
                "Episodic_Score", episodic_score, self.current_episode_num
            )
            self.writer.add_scalar(
                "Runner_Total_Score", self.total_score_runner, self.current_episode_num
            )

            agent_info_tb = self.agent.get_update_info()
            if agent_info_tb:
                for key, value in agent_info_tb.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(
                            f"Agent/{key}", value, self.current_episode_num
                        )

            self.sliding_window.pop(0)
            self.sliding_window.append(episodic_score)
            if (
                self.current_episode_num >= 10
            ):  # Start logging mean score after 10 episodes
                mean_score = np.mean(self.sliding_window)
                self.writer.add_scalar(
                    "Mean_Score_10_Episodes", mean_score, self.current_episode_num
                )
                print(f"Mean score over last 10 episodes: {mean_score:.2f}")

        # MLflow Metric Logging (if run_id is available)
        if self.mlflow_run_id:
            try:
                mlflow.log_metric(
                    "Episodic_Score", episodic_score, step=self.current_episode_num
                )
                # Log other relevant metrics
                if self.current_episode_num >= 10:
                    mlflow.log_metric(
                        "Mean_Score_10_Episodes",
                        np.mean(self.sliding_window),
                        step=self.current_episode_num,
                    )

                agent_info_mlflow = self.agent.get_update_info()
                if agent_info_mlflow:
                    for key, value in agent_info_mlflow.items():
                        if isinstance(
                            value, (int, float)
                        ):  # MLflow only logs numeric metrics
                            mlflow.log_metric(
                                f"Agent.{key}", value, step=self.current_episode_num
                            )
            except Exception as e:
                print(f"Warning: MLflow metric logging failed: {e}")

        # --- Local Model Checkpointing & MLflow Tagging ---
        # Agent's save_model method will handle logging the model as an MLflow artifact.
        if (
            self.is_saving_model_locally
            and self.current_episode_num % self.save_model_each_episode_num == 0
        ):
            # Define a unique filename for this checkpoint
            checkpoint_filename = f"checkpoint_ep{self.current_episode_num}.pt"
            # (or agent-specific like q_table_epX.pt, dqn_nn_epX.pt)

            # The agent's save_model method is responsible for the actual saving
            # and for logging to MLflow artifacts.
            # We pass the specific filename for this checkpoint.
            saved_model_file_path = self.agent.save_model(
                self.local_models_path, filename=checkpoint_filename
            )

            if self.mlflow_run_id and saved_model_file_path:
                # Log the local path of this checkpoint as a tag in MLflow for reference
                mlflow.set_tag(
                    f"local_checkpoint_ep_{self.current_episode_num}",
                    saved_model_file_path,
                )

        self.current_episode_num += 1

    def run(self) -> None:
        """
        The main generic run loop for training or evaluation.
        """
        print(
            f"Runner starting. Training: {self.is_training}. Total episodes: {self.total_episodes}."
        )
        for ep_num in range(
            1, self.total_episodes + 1
        ):  # Iterate from 1 to total_episodes
            state = self.env.reset()
            self.agent.on_episode_start()  # Agent specific setup for new episode
            done = False
            episodic_score = 0.0

            step_in_episode = 0
            while not done:
                # 1. Agent selects an action
                action = self.agent.get_action(state, is_training=self.is_training)

                # 2. Environment executes the action
                next_state, reward, done, info = self.env.step(
                    action
                )  # Capture info dict

                # 3. Agent learns from the experience (if training)
                if self.is_training:
                    # For on-policy methods, the agent might need the next action.
                    # We provide it here. Off-policy agents can simply ignore it in their update.
                    next_action = self.agent.get_action(
                        next_state, is_training=self.is_training
                    )
                    self.agent.update(
                        state, action, reward, next_state, done, next_action=next_action
                    )

                # 4. Update state and score
                state = next_state
                episodic_score += reward
                step_in_episode += 1

            # --- End of Episode ---
            self.agent.on_episode_end()  # Agent specific cleanup/decay for episode
            self.summarize(episodic_score)  # Log metrics, save model if needed

        # --- End of Experiment ---
        if self.writer:  # Close TensorBoard writer
            self.writer.close()
        self.env.close()  # Close the game environment
        print("Experiment finished by Runner.")

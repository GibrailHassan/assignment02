# runner/runner.py

"""
Defines the Runner class, which orchestrates the agent-environment interaction loop,
logs metrics, and now automatically generates and logs summary plots using Plotly
to MLflow at the end of each run.
"""
import datetime
import os
from typing import Any, Dict, List  # Added List

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym

import mlflow  # For MLflow tagging and active_run check
import pandas as pd  # For creating DataFrames for plotting
import plotly.express as px  # For interactive plotting

# import tempfile # Not strictly needed if using mlflow.plotly.log_figure directly

from utils.experiment_utils import log_metrics_to_tensorboard, log_metrics_to_mlflow
from agents.abstractAgent import AbstractAgent


class Runner:
    """
    Orchestrates the training and evaluation loops for an RL agent.
    Includes automatic generation and logging of Plotly summary plots to MLflow.
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
        experiment_name: str = "DefaultExperiment",
        **kwargs: Any,
    ) -> None:
        self.agent = agent
        self.env = env
        self.is_training = is_training
        self.total_episodes = total_episodes
        self.save_model_each_episode_num = save_model_each_episode_num
        self.mlflow_run_id = mlflow_run_id
        self.experiment_name = experiment_name

        self.total_score_runner = 0.0
        self.current_episode_num = 1
        self.sliding_window = [-1.0] * 10

        # Initialize history for storing metrics for end-of-run plotting
        self.history: Dict[str, List[Any]] = {
            "episodes": [],
            "episodic_scores": [],
            "mean_scores_10_ep": [],  # Will have Nones for first few episodes
            "agent_epsilon": [],
            # Add other agent-specific metrics you want to plot from get_update_info
            # For example, if get_update_info returns {'loss': value}, add "agent_loss": []
        }

        self._setup_paths(tensorboard_log_dir, model_save_dir)

    def _setup_paths(self, tensorboard_log_dir: str, model_save_dir: str) -> None:
        """Initializes paths for local model saving and TensorBoard logging."""
        current_time_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        # Sanitize experiment_name for use in file/directory paths
        safe_experiment_name = (
            self.experiment_name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace("\\", "_")
        )
        run_signature = (
            f"{current_time_str}_{safe_experiment_name}_{type(self.agent).__name__}"
        )

        self.is_saving_model_locally = (
            self.is_training and self.save_model_each_episode_num > 0 and model_save_dir
        )
        if self.is_saving_model_locally:
            self.local_models_path = os.path.join(
                os.path.abspath(model_save_dir), run_signature
            )
            os.makedirs(self.local_models_path, exist_ok=True)
            print(
                f"Local models for this run will be saved in: {self.local_models_path}"
            )
        else:
            self.local_models_path = None

        if tensorboard_log_dir:
            train_eval_prefix = "_train" if self.is_training else "_eval"
            tb_run_signature = f"{run_signature}{train_eval_prefix}"
            self.tb_path = os.path.join(
                os.path.abspath(tensorboard_log_dir), tb_run_signature
            )
            self.writer = SummaryWriter(self.tb_path)
            print(f"TensorBoard logs will be saved to: {self.tb_path}")
        else:
            self.writer = None

    def _handle_periodic_saving(self) -> None:
        """Handles local model checkpointing and MLflow tagging if conditions are met."""
        if not (
            self.is_saving_model_locally
            and self.current_episode_num > 0
            and self.save_model_each_episode_num > 0
            and self.current_episode_num % self.save_model_each_episode_num == 0
        ):
            return

        checkpoint_filename = f"checkpoint_ep{self.current_episode_num}.pt"

        saved_model_file_path = self.agent.save_model(
            self.local_models_path, filename=checkpoint_filename
        )

        if self.mlflow_run_id and saved_model_file_path and mlflow.active_run():
            try:
                relative_checkpoint_path = os.path.join(
                    os.path.basename(self.local_models_path), checkpoint_filename
                )
                mlflow.set_tag(
                    f"local_checkpoint_ep_{self.current_episode_num}",
                    relative_checkpoint_path,
                )
            except Exception as e:
                print(f"Warning: Failed to set MLflow tag for local checkpoint: {e}")

    def _generate_and_log_summary_plots(self) -> None:
        """
        Generates summary plots from accumulated history using Plotly and logs them to MLflow.
        Called at the end of the run.
        """
        if not self.mlflow_run_id or not mlflow.active_run():
            print("No active MLflow run, skipping summary plot generation for MLflow.")
            return

        if not self.history["episodes"]:  # No data collected
            print("No history data to plot.")
            return

        print("Generating and logging summary plots to MLflow using Plotly...")

        # Create a DataFrame from the history
        # Pad shorter lists with None or handle carefully if lengths differ significantly
        max_len = len(self.history["episodes"])
        history_for_df = {}
        for key, values in self.history.items():
            history_for_df[key] = values + [None] * (max_len - len(values))

        df = pd.DataFrame(history_for_df)
        df = df.dropna(subset=["episodes"])  # Ensure episodes column is not None

        # Plot Episodic and Mean Scores
        metrics_to_plot_y = []
        if "episodic_scores" in df.columns and not df["episodic_scores"].dropna().empty:
            metrics_to_plot_y.append("episodic_scores")
        if (
            "mean_scores_10_ep" in df.columns
            and not df["mean_scores_10_ep"].dropna().empty
        ):
            metrics_to_plot_y.append("mean_scores_10_ep")

        if metrics_to_plot_y:
            try:
                fig_scores = px.line(
                    df.dropna(subset=metrics_to_plot_y),
                    x="episodes",
                    y=metrics_to_plot_y,
                    title=f"Training Scores: {self.experiment_name}",
                    labels={
                        "episodes": "Episode",
                        "value": "Score",
                        "variable": "Metric",
                    },
                )
                fig_scores.update_layout(hovermode="x unified")
                mlflow.plotly.log_figure(
                    fig_scores, artifact_file="plots/interactive_scores_plot.html"
                )
                print("Logged interactive scores plot to MLflow artifacts.")
            except Exception as e:
                print(f"Warning: Failed to generate or log Plotly scores plot: {e}")

        # Plot Agent Epsilon
        if "agent_epsilon" in df.columns and not df["agent_epsilon"].dropna().empty:
            try:
                fig_eps = px.line(
                    df.dropna(subset=["agent_epsilon"]),
                    x="episodes",
                    y="agent_epsilon",
                    title=f"Agent Epsilon Decay: {self.experiment_name}",
                    labels={"episodes": "Episode", "agent_epsilon": "Epsilon"},
                )
                fig_eps.update_layout(hovermode="x unified")
                mlflow.plotly.log_figure(
                    fig_eps, artifact_file="plots/interactive_epsilon_plot.html"
                )
                print("Logged interactive epsilon plot to MLflow artifacts.")
            except Exception as e:
                print(f"Warning: Failed to generate or log Plotly epsilon plot: {e}")

        # Add plots for other accumulated agent_info metrics if desired

    def summarize(self, episodic_score: float) -> None:
        """Logs metrics to console, TensorBoard, and MLflow; accumulates history; handles checkpointing."""
        self.total_score_runner += episodic_score

        print(
            f"Finished episode {self.current_episode_num}/{self.total_episodes}, "
            f"Episodic Score: {episodic_score:.2f}, Runner Total Score: {self.total_score_runner:.2f}"
        )

        # Accumulate history for end-of-run plotting
        self.history["episodes"].append(self.current_episode_num)
        self.history["episodic_scores"].append(episodic_score)

        metrics_to_log_now = {  # For immediate logging per episode
            "Episodic_Score": episodic_score,
            # "Runner_Total_Score": self.total_score_runner, # Total score might be better as a final tag/param
        }

        agent_info = self.agent.get_update_info()
        if agent_info:
            for key, value in agent_info.items():
                # Log to current episode metrics
                metrics_to_log_now[f"Agent_{key.replace('/', '_')}"] = value
                # Accumulate in history if it's a plottable series
                history_key = f"agent_{key.lower().replace('/', '_')}"
                if history_key not in self.history:  # Initialize list if key is new
                    self.history[history_key] = [None] * (
                        self.current_episode_num - 1
                    )  # Pad for previous episodes
                self.history[history_key].append(
                    value if isinstance(value, (int, float)) else None
                )

        self.sliding_window.pop(0)
        self.sliding_window.append(episodic_score)
        current_mean_score = None
        if self.current_episode_num >= 10:
            current_mean_score = np.mean(self.sliding_window)
            metrics_to_log_now["Mean_Score_10_Episodes"] = current_mean_score
            print(f"Mean score over last 10 episodes: {current_mean_score:.2f}")

        self.history["mean_scores_10_ep"].append(current_mean_score)

        log_metrics_to_tensorboard(
            self.writer, metrics_to_log_now, self.current_episode_num
        )
        log_metrics_to_mlflow(metrics_to_log_now, self.current_episode_num)

        self._handle_periodic_saving()

        self.current_episode_num += 1

    def run(self) -> None:
        """The main generic run loop for training or evaluation."""
        print(
            f"Runner starting. Experiment: '{self.experiment_name}'. Training: {self.is_training}. Total episodes: {self.total_episodes}."
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

            # --- Generate and log plots at the end of all episodes ---
            # Only generate plots if MLflow was active for this run
            if self.mlflow_run_id and mlflow.active_run():
                self._generate_and_log_summary_plots()

        finally:
            if self.writer:
                self.writer.close()
            self.env.close()
            print(
                f"Experiment '{self.experiment_name}' finished by Runner (resources closed)."
            )

# runner/runner.py

"""
Defines the Runner class, which orchestrates the agent-environment interaction
loop for training or evaluation of reinforcement learning agents.

The Runner class manages:
- The main episode loop.
- Agent actions and environment steps.
- Agent updates (learning steps).
- Logging of metrics to TensorBoard and MLflow.
- Periodic saving of agent models.
- NEW: Automatic generation and logging of summary plots (using Plotly)
  to MLflow at the end of each run.
- Accumulation of metrics throughout the run for end-of-run summary plotting.
"""
import datetime
import os
from typing import Any, Dict, List, Optional  # Added Optional

import numpy as np
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard logging
import gym  # For gym.Env type hinting

import mlflow  # For MLflow tagging, active_run check, and Plotly figure logging
import pandas as pd  # For creating DataFrames for plotting
import plotly.express as px  # For interactive HTML Plotly plots

# `tempfile` was mentioned as not strictly needed if using mlflow.plotly.log_figure directly.
# import tempfile

# Utility functions for logging metrics (assumed to be in utils.experiment_utils)
from utils.experiment_utils import log_metrics_to_tensorboard, log_metrics_to_mlflow

# Abstract agent class for type hinting
from agents.abstractAgent import AbstractAgent


class Runner:
    """
    Orchestrates the training and evaluation loops for a reinforcement learning agent.

    This class handles the interaction between an agent and an environment over a
    specified number of episodes. It logs metrics, saves models, and at the end
    of a run, generates and logs summary plots of key metrics to MLflow using Plotly.

    Attributes:
        agent (AbstractAgent): The RL agent to be run.
        env (gym.Env): The environment for the agent to interact with.
        is_training (bool): Flag indicating if the agent is in training mode.
        total_episodes (int): The total number of episodes to run.
        save_model_each_episode_num (int): Frequency (in episodes) for saving model checkpoints.
                                           If 0, no intermediate saving occurs.
        mlflow_run_id (Optional[str]): The ID of the active MLflow run, if any.
        experiment_name (str): The name of the MLflow experiment.
        total_score_runner (float): Cumulative score over all episodes run by this Runner instance.
        current_episode_num (int): The current episode number (1-indexed).
        sliding_window (List[float]): A sliding window of recent episodic scores, used for
                                      calculating a moving average score.
        history (Dict[str, List[Any]]): A dictionary to store the history of various metrics
                                        (e.g., scores, epsilon) across episodes for end-of-run plotting.
        local_models_path (Optional[str]): Path to save local model checkpoints.
        writer (Optional[SummaryWriter]): TensorBoard SummaryWriter instance.
        is_saving_model_locally (bool): Flag indicating if local model saving is active.
    """

    def __init__(
        self,
        agent: AbstractAgent,
        env: gym.Env,
        is_training: bool = True,
        total_episodes: int = 1000,
        save_model_each_episode_num: int = 0,  # Set to >0 to enable periodic saving
        tensorboard_log_dir: str = "./logs",  # Base directory for TensorBoard logs
        model_save_dir: str = "./models",  # Base directory for saved models
        mlflow_run_id: Optional[str] = None,  # MLflow run ID passed from main.py
        experiment_name: str = "DefaultExperiment",  # MLflow experiment name
        **kwargs: Any,  # Absorb any additional runner parameters from config
    ) -> None:
        """
        Initializes the Runner.

        Args:
            agent (AbstractAgent): The reinforcement learning agent.
            env (gym.Env): The environment.
            is_training (bool, optional): True if training, False for evaluation. Defaults to True.
            total_episodes (int, optional): Total episodes to run. Defaults to 1000.
            save_model_each_episode_num (int, optional): Frequency of saving models.
                0 disables intermediate saving. Defaults to 0.
            tensorboard_log_dir (str, optional): Directory for TensorBoard logs.
                If empty or None, TensorBoard logging is disabled. Defaults to "./logs".
            model_save_dir (str, optional): Directory for saving models locally.
                If empty or None, local model saving is disabled. Defaults to "./models".
            mlflow_run_id (Optional[str], optional): Active MLflow run ID. Defaults to None.
            experiment_name (str, optional): Name of the MLflow experiment. Defaults to "DefaultExperiment".
            **kwargs (Any): Additional keyword arguments (ignored by this constructor).
        """
        self.agent: AbstractAgent = agent
        self.env: gym.Env = env
        self.is_training: bool = is_training
        self.total_episodes: int = total_episodes
        self.save_model_each_episode_num: int = save_model_each_episode_num
        self.mlflow_run_id: Optional[str] = mlflow_run_id
        self.experiment_name: str = experiment_name  # Used for plot titles, etc.

        # --- Initialize tracking variables ---
        self.total_score_runner: float = 0.0  # Cumulative score for the entire run
        self.current_episode_num: int = 1  # Current episode, 1-indexed

        # Sliding window for mean score calculation (e.g., over last 10 episodes)
        # Initialize with a value indicating no score yet (e.g., -1.0 or NaN if preferred for plotting)
        self.sliding_window_size: int = 10
        self.sliding_window: List[float] = [
            -1.0
        ] * self.sliding_window_size  # Placeholder for scores

        # --- History for end-of-run plotting ---
        # Stores various metrics per episode to generate summary plots at the end.
        self.history: Dict[str, List[Any]] = {
            "episodes": [],  # Episode numbers
            "episodic_scores": [],  # Score for each episode
            "mean_scores_10_ep": [],  # Moving average score (padded with None for initial episodes)
            "agent_epsilon": [],  # Agent's epsilon value (if applicable)
            # Other agent-specific metrics (e.g., loss) can be dynamically added
            # in the `summarize` method based on `agent.get_update_info()`.
        }

        # Setup paths for TensorBoard logs and local model saving
        self._setup_paths_and_logging(tensorboard_log_dir, model_save_dir)

    def _setup_paths_and_logging(
        self, tensorboard_log_dir: str, model_save_dir: str
    ) -> None:
        """
        Initializes paths for local model saving and TensorBoard logging.
        Creates necessary directories if they don't exist.

        Args:
            tensorboard_log_dir (str): Base directory for TensorBoard logs.
            model_save_dir (str): Base directory for saving models locally.
        """
        # Generate a unique signature for this run based on time, experiment, and agent type
        # to create unique subdirectories for logs and models.
        current_time_str: str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

        # Sanitize experiment_name for use in file/directory paths by replacing problematic characters.
        safe_experiment_name: str = (
            self.experiment_name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace("\\", "_")
        )
        # Construct a base run signature
        run_signature_base: str = (
            f"{current_time_str}_{safe_experiment_name}_{type(self.agent).__name__}"
        )

        # --- Local Model Saving Setup ---
        # Determine if models should be saved locally during this run.
        self.is_saving_model_locally: bool = (
            self.is_training  # Only save models during training
            and self.save_model_each_episode_num > 0  # Only if save frequency is set
            and model_save_dir is not None
            and model_save_dir != ""  # Only if a save directory is provided
        )

        if self.is_saving_model_locally:
            # Create a unique subdirectory for this run's models
            self.local_models_path: Optional[str] = os.path.join(
                os.path.abspath(model_save_dir),
                run_signature_base,  # Use the base signature
            )
            os.makedirs(
                self.local_models_path, exist_ok=True
            )  # Create directory if it doesn't exist
            print(
                f"Local models for this run will be saved in: {self.local_models_path}"
            )
        else:
            self.local_models_path: Optional[str] = None
            if (
                self.is_training
            ):  # Only print warning if training but saving is disabled
                print(
                    "Local model saving is disabled for this run (either save_model_each_episode_num is 0 or model_save_dir is not set)."
                )

        # --- TensorBoard Logging Setup ---
        if tensorboard_log_dir is not None and tensorboard_log_dir != "":
            # Add a suffix to distinguish training vs. evaluation logs if desired
            train_eval_suffix: str = "_train" if self.is_training else "_eval"
            tb_run_signature: str = f"{run_signature_base}{train_eval_suffix}"

            self.tb_path: Optional[str] = os.path.join(
                os.path.abspath(tensorboard_log_dir), tb_run_signature
            )
            # Initialize TensorBoard SummaryWriter
            self.writer: Optional[SummaryWriter] = SummaryWriter(self.tb_path)
            print(f"TensorBoard logs for this run will be saved to: {self.tb_path}")
        else:
            self.writer: Optional[SummaryWriter] = None
            print(
                "TensorBoard logging is disabled for this run (tensorboard_log_dir not set)."
            )

    def _handle_periodic_saving(self) -> None:
        """
        Handles local model checkpointing if conditions are met (training, save frequency, etc.).
        Also attempts to log information about the local checkpoint as an MLflow tag.
        """
        # Check if conditions for periodic saving are met
        if not (
            self.is_saving_model_locally  # Local saving must be enabled
            and self.local_models_path is not None  # Path must exist
            and self.current_episode_num > 0
            and self.save_model_each_episode_num > 0  # Save frequency must be positive
            and self.current_episode_num % self.save_model_each_episode_num
            == 0  # Current episode is a save point
        ):
            return  # Conditions not met, do nothing

        # --- Perform Model Saving ---
        # Define a filename for the checkpoint
        checkpoint_filename: str = (
            f"checkpoint_ep{self.current_episode_num}.pt"  # Or agent-specific extension
        )

        # Call the agent's save_model method
        print(
            f"Attempting to save model checkpoint at episode {self.current_episode_num}..."
        )
        saved_model_file_path: Optional[str] = self.agent.save_model(
            self.local_models_path, filename=checkpoint_filename
        )

        # If model saved successfully and MLflow is active, log tag
        if saved_model_file_path and self.mlflow_run_id and mlflow.active_run():
            try:
                # Construct a relative path for the tag to be more concise if `local_models_path` is long
                relative_checkpoint_path: str = os.path.join(
                    os.path.basename(self.local_models_path), checkpoint_filename
                )
                # Log a tag to MLflow indicating where the local checkpoint was saved
                mlflow.set_tag(
                    f"local_checkpoint_ep_{self.current_episode_num}",
                    relative_checkpoint_path,
                )
                print(
                    f"Logged MLflow tag for local checkpoint: local_checkpoint_ep_{self.current_episode_num}"
                )
            except Exception as e_tag:
                print(
                    f"Warning: Failed to set MLflow tag for local checkpoint: {e_tag}"
                )

    def _generate_and_log_summary_plots(self) -> None:
        """
        Generates summary plots from accumulated metric history using Plotly
        and logs these plots as HTML artifacts to the active MLflow run.
        This method is called at the end of the entire run (all episodes).
        """
        if not self.mlflow_run_id or not mlflow.active_run():
            print(
                "No active MLflow run, or MLflow run ID not set. Skipping summary plot generation for MLflow."
            )
            return

        if not self.history["episodes"]:  # No data collected in history
            print(
                "No history data collected during the run. Skipping summary plot generation."
            )
            return

        print("Generating and logging summary plots to MLflow using Plotly...")

        # --- Create a Pandas DataFrame from the history dictionary ---
        # Ensure all lists in history have the same length for DataFrame creation by padding shorter lists.
        # This is important if some metrics (e.g., agent-specific ones) weren't logged every episode.
        max_len: int = len(
            self.history["episodes"]
        )  # Assuming "episodes" is always populated
        history_for_df: Dict[str, List[Any]] = {}
        for key, values in self.history.items():
            # Pad with None if a list is shorter than the number of episodes
            padding_needed = max_len - len(values)
            history_for_df[key] = values + (
                [None] * padding_needed if padding_needed > 0 else []
            )

        try:
            df = pd.DataFrame(history_for_df)
            # Ensure 'episodes' column is not None, as it's the primary x-axis
            df = df.dropna(subset=["episodes"])
        except Exception as e_df:
            print(
                f"Error creating DataFrame from history for plotting: {e_df}. Skipping plots."
            )
            return

        if df.empty:
            print("DataFrame for plotting is empty. Skipping plots.")
            return

        # --- Plot 1: Episodic and Mean Scores ---
        metrics_to_plot_y_scores: List[str] = []
        # Check if score metrics exist and have non-null data
        if "episodic_scores" in df.columns and not df["episodic_scores"].dropna().empty:
            metrics_to_plot_y_scores.append("episodic_scores")
        if (
            "mean_scores_10_ep" in df.columns
            and not df["mean_scores_10_ep"].dropna().empty
        ):
            metrics_to_plot_y_scores.append("mean_scores_10_ep")

        if metrics_to_plot_y_scores:
            try:
                # Create a line plot for scores using Plotly Express
                fig_scores = px.line(
                    df.dropna(
                        subset=metrics_to_plot_y_scores
                    ),  # Drop rows where these specific scores are all NaN
                    x="episodes",
                    y=metrics_to_plot_y_scores,  # Plot multiple score lines if available
                    title=f"Training Scores vs. Episodes: {self.experiment_name}",
                    labels={  # Custom labels for axes and legend
                        "episodes": "Episode Number",
                        "value": "Score",  # Default y-axis label when multiple 'y' columns
                        "variable": "Score Metric",  # Legend title for multiple 'y' columns
                    },
                )
                fig_scores.update_layout(
                    hovermode="x unified"
                )  # Enhances interactivity on hover
                # Log the Plotly figure to MLflow as an HTML artifact
                mlflow.plotly.log_figure(
                    fig_scores, artifact_file="plots/interactive_scores_summary.html"
                )
                print(
                    "Logged interactive scores summary plot to MLflow artifacts (plots/interactive_scores_summary.html)."
                )
            except Exception as e_plot_scores:
                print(
                    f"Warning: Failed to generate or log Plotly scores plot: {e_plot_scores}"
                )

        # --- Plot 2: Agent Epsilon (if available) ---
        if "agent_epsilon" in df.columns and not df["agent_epsilon"].dropna().empty:
            try:
                fig_eps = px.line(
                    df.dropna(
                        subset=["agent_epsilon"]
                    ),  # Drop rows where epsilon is NaN
                    x="episodes",
                    y="agent_epsilon",
                    title=f"Agent Epsilon Decay: {self.experiment_name}",
                    labels={
                        "episodes": "Episode Number",
                        "agent_epsilon": "Epsilon Value",
                    },
                )
                fig_eps.update_layout(hovermode="x unified")
                mlflow.plotly.log_figure(
                    fig_eps, artifact_file="plots/interactive_epsilon_summary.html"
                )
                print(
                    "Logged interactive epsilon summary plot to MLflow artifacts (plots/interactive_epsilon_summary.html)."
                )
            except Exception as e_plot_eps:
                print(
                    f"Warning: Failed to generate or log Plotly epsilon plot: {e_plot_eps}"
                )

        # --- Future: Add plots for other accumulated agent_info metrics ---
        # Example: if 'agent_loss' was consistently logged and added to history:
        # if "agent_loss" in df.columns and not df["agent_loss"].dropna().empty:
        #     try:
        #         fig_loss = px.line(...)
        #         mlflow.plotly.log_figure(fig_loss, artifact_file="plots/interactive_loss_summary.html")
        #     except Exception as e_plot_loss:
        #         print(f"Warning: Failed to plot agent loss: {e_plot_loss}")
        print("Summary plot generation finished.")

    def summarize(self, episodic_score: float) -> None:
        """
        Summarizes the results of a completed episode.

        This includes:
        - Printing episode score and cumulative run score to console.
        - Accumulating metrics in `self.history` for end-of-run plotting.
        - Logging metrics to TensorBoard and MLflow for the current episode.
        - Updating the sliding window for mean score calculation.
        - Triggering periodic model saving if applicable.
        - Incrementing the episode counter.

        Args:
            episodic_score (float): The total score obtained in the just-completed episode.
        """
        self.total_score_runner += episodic_score  # Update cumulative run score

        print(
            f"Episode {self.current_episode_num}/{self.total_episodes} Finished. "
            f"Episodic Score: {episodic_score:.2f}. "
            f"Runner Total Score: {self.total_score_runner:.2f}"
        )

        # --- Accumulate metrics in self.history for end-of-run plotting ---
        self.history["episodes"].append(self.current_episode_num)
        self.history["episodic_scores"].append(episodic_score)

        # Prepare a dictionary of metrics for immediate logging (TensorBoard, MLflow this episode)
        metrics_to_log_this_episode: Dict[str, Any] = {
            "Episodic_Score": episodic_score,
            # "Runner_Total_Score": self.total_score_runner, # This might be better as a final tag/param for the run
        }

        # Get and log agent-specific information (e.g., epsilon, loss from last update)
        agent_update_info: Dict[str, Any] = self.agent.get_update_info()
        if agent_update_info:
            for key, value in agent_update_info.items():
                # Add to metrics for immediate logging (TensorBoard, MLflow)
                # Sanitize key for MLflow (dots are fine, but slashes can be problematic in some UIs/tools)
                # The `log_metrics_to_mlflow` util also does some sanitization.
                metrics_to_log_this_episode[f"Agent.{key.replace('/', '_')}"] = value

                # Add to self.history for end-of-run plotting if it's a plottable series
                history_key_agent: str = (
                    f"agent_{key.lower().replace('/', '_').replace('.', '_')}"
                )
                if history_key_agent not in self.history:
                    # Initialize list for this new metric, padding with None for previous episodes
                    self.history[history_key_agent] = [None] * (
                        len(self.history["episodes"]) - 1
                    )

                # Append current value if numerical, else None (Plotly handles None by breaking lines)
                self.history[history_key_agent].append(
                    value if isinstance(value, (int, float)) else None
                )

        # --- Sliding window for mean score ---
        # Remove the oldest score and add the new one
        self.sliding_window.pop(0)
        self.sliding_window.append(episodic_score)

        current_mean_score: Optional[float] = None
        # Calculate mean score only if the window is "full" (or past initial warm-up)
        # This avoids skewed means at the beginning.
        if self.current_episode_num >= self.sliding_window_size:
            # Filter out placeholder values if any still exist (e.g., initial -1.0s)
            valid_scores_in_window = [
                s for s in self.sliding_window if s != -1.0
            ]  # Assuming -1.0 is placeholder
            if valid_scores_in_window:  # Ensure there are scores to average
                current_mean_score = np.mean(valid_scores_in_window)
                metrics_to_log_this_episode[
                    f"Mean_Score_{self.sliding_window_size}_Episodes"
                ] = current_mean_score
                print(
                    f"Mean score over last {self.sliding_window_size} episodes: {current_mean_score:.2f}"
                )
            # else: current_mean_score remains None if window is full of placeholders

        # Add to history for plotting (will be None if not yet calculated)
        self.history["mean_scores_10_ep"].append(
            current_mean_score
        )  # Ensure key matches history init

        # --- Log metrics to TensorBoard and MLflow for the current episode ---
        log_metrics_to_tensorboard(
            self.writer, metrics_to_log_this_episode, self.current_episode_num
        )
        log_metrics_to_mlflow(
            metrics_to_log_this_episode, self.current_episode_num
        )  # Prefix can be added if needed

        # --- Handle periodic model saving ---
        if self.is_training:  # Only save models if in training mode
            self._handle_periodic_saving()

        # Increment episode counter for the next episode
        self.current_episode_num += 1

    def run(self) -> None:
        """
        The main generic run loop for orchestrating agent-environment interaction
        over `self.total_episodes`.
        """
        print(
            f"\nRunner starting. Experiment: '{self.experiment_name}'. "
            f"Agent: {type(self.agent).__name__}. Environment: {type(self.env).__name__}. "
            f"Mode: {'Training' if self.is_training else 'Evaluation'}. Total episodes: {self.total_episodes}."
        )
        try:
            # Loop for the specified total number of episodes
            for episode_count in range(self.total_episodes):
                # Reset the environment at the start of each episode
                # `state` is the initial observation from the environment
                state: np.ndarray = self.env.reset()

                # Notify the agent that a new episode is starting
                self.agent.on_episode_start()

                done: bool = False  # Flag to indicate if the episode has terminated
                episodic_score: float = (
                    0.0  # Accumulator for the current episode's score
                )

                # Inner loop for steps within an episode
                step_in_episode_counter = 0  # Optional: for max steps per episode
                while not done:
                    # 1. Agent selects an action based on the current state and training mode
                    action: Any = self.agent.get_action(
                        state, is_training=self.is_training
                    )

                    # 2. Environment executes the action
                    # Returns next_state, reward, done flag, and additional info
                    next_state, reward, done, info = self.env.step(action)

                    # 3. If training, agent learns from the experience (updates its model/policy)
                    if self.is_training:
                        # For on-policy algorithms like SARSA, the next action might be needed.
                        # The agent's update method should handle if next_action is None or not provided.
                        next_action_for_update: Optional[Any] = None
                        if (
                            not done
                        ):  # Only get next_action if not terminal, for SARSA-like updates
                            # This is a common pattern for SARSA, but other on-policy methods might differ.
                            # If agent is off-policy (like Q-Learning, DQN), it won't use next_action.
                            # Some DQNAgent implementations might get next_action if DDQN is enabled,
                            # but often they select the best action from target_nn directly.
                            # We pass it, and the agent decides if/how to use it.
                            next_action_for_update = self.agent.get_action(
                                next_state, is_training=self.is_training
                            )

                        self.agent.update(
                            state,
                            action,
                            reward,
                            next_state,
                            done,
                            next_action=next_action_for_update,  # Pass for SARSA-like on-policy agents
                            # Any other info from `info` dict could be passed via **kwargs if agent needs it
                        )

                    # Transition to the next state
                    state = next_state
                    episodic_score += reward  # Accumulate reward
                    step_in_episode_counter += 1

                    # Optional: Add a max step limit per episode to prevent infinite loops
                    # if step_in_episode_counter >= MAX_STEPS_PER_EPISODE:
                    #     done = True
                    #     print(f"Episode {self.current_episode_num} reached max step limit.")

                # End of episode loop

                # Notify the agent that the episode has ended
                self.agent.on_episode_end()

                # Summarize results for this episode (logging, history, model saving)
                self.summarize(episodic_score)

            # --- End of all episodes ---
            # Generate and log summary plots to MLflow if conditions are met
            if self.mlflow_run_id and mlflow.active_run():  # Ensure MLflow was active
                self._generate_and_log_summary_plots()

        except KeyboardInterrupt:
            print("\nRun interrupted by user (KeyboardInterrupt) in Runner main loop.")
            if self.mlflow_run_id and mlflow.active_run():
                mlflow.set_tag("run_status", "interrupted_in_runner_loop")
        except Exception as e_run:
            print(f"\nAn error occurred during the Runner's run loop: {e_run}")
            if self.mlflow_run_id and mlflow.active_run():
                mlflow.set_tag("run_status", "error_in_runner_loop")
                mlflow.log_param("runner_loop_error", str(e_run)[:250])
            raise  # Re-raise the exception to be caught by main.py or terminate
        finally:
            # --- Cleanup ---
            # Close the TensorBoard writer if it was initialized
            if self.writer:
                self.writer.close()
                print("TensorBoard writer closed.")

            # Close the environment to release its resources (e.g., PySC2 process)
            if hasattr(self.env, "close") and callable(self.env.close):
                try:
                    self.env.close()
                    print("Environment closed by Runner.")
                except Exception as e_env_close:
                    print(
                        f"Error closing environment in Runner's finally block: {e_env_close}"
                    )

            print(
                f"\nRunner for experiment '{self.experiment_name}' finished. "
                f"Total episodes run: {self.current_episode_num -1}. Final total score: {self.total_score_runner:.2f}."
            )

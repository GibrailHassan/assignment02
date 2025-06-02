# plot_results.py

"""
A script to fetch experiment data from MLflow, generate plots of key metrics,
and save them locally. Optionally, plots can be logged back to MLflow as artifacts.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict

# Define metrics you want to plot
# Format: {"Metric Name in MLflow": "Plot Y-axis Label"}
DEFAULT_METRICS_TO_PLOT = {
    "Episodic_Score": "Episodic Score",
    "Mean_Score_10_Episodes": "Mean Score (10 Ep. Window)",
    "Agent.epsilon": "Epsilon (Agent)",  # Assuming MLflow logged it as Agent.epsilon
    "Agent_epsilon": "Epsilon (Agent)",  # Alternative if prefix was Agent_
}


def plot_metrics_for_run(
    run_id: str, metrics_to_plot: Dict[str, str], output_dir: str, client: MlflowClient
):
    """
    Fetches metrics for a single MLflow run and generates/saves plots.

    Args:
        run_id (str): The ID of the MLflow run.
        metrics_to_plot (Dict[str, str]): Metrics to plot and their y-axis labels.
        output_dir (str): Directory to save the plot images.
        client (MlflowClient): MLflow tracking client.
    """
    print(f"\nProcessing run ID: {run_id}")
    run_data = client.get_run(run_id)
    run_name = run_data.data.tags.get(
        "mlflow.runName", run_id
    )  # Get run name if available

    # Create a subdirectory for this run's plots
    run_plot_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_plot_dir, exist_ok=True)

    for metric_key, y_label in metrics_to_plot.items():
        metric_history = client.get_metric_history(run_id, metric_key)
        if not metric_history:
            # Try with underscore if dot version not found (due to sanitization)
            metric_key_alt = metric_key.replace(".", "_")
            if metric_key_alt != metric_key:
                metric_history = client.get_metric_history(run_id, metric_key_alt)
                if metric_history:
                    print(
                        f"Found metric as '{metric_key_alt}' instead of '{metric_key}'"
                    )
                    metric_key = metric_key_alt  # Use the found key
                else:
                    print(
                        f"Metric '{metric_key}' (and alt) not found for run {run_id}. Skipping plot."
                    )
                    continue
            else:
                print(
                    f"Metric '{metric_key}' not found for run {run_id}. Skipping plot."
                )
                continue

        # Convert metric history to Pandas DataFrame for easier plotting
        # Metric objects have 'key', 'value', 'timestamp', 'step'
        data = [
            {"step": m.step, "value": m.value, "timestamp": m.timestamp}
            for m in metric_history
        ]
        df = pd.DataFrame(data)
        df = df.sort_values(by="step")  # Ensure data is sorted by step

        if df.empty:
            print(f"No data for metric '{metric_key}' in run {run_id}. Skipping plot.")
            continue

        plt.figure(figsize=(10, 6))
        sns.lineplot(x="step", y="value", data=df)
        plt.title(
            f"{y_label} vs. Episodes/Steps for Run: {run_name[:30]}..."
        )  # Shorten run name for title
        plt.xlabel("Episode / Step")
        plt.ylabel(y_label)
        plt.grid(True)

        plot_filename = f"{metric_key.replace('.', '_').replace('/', '_')}_plot.png"
        plot_filepath = os.path.join(run_plot_dir, plot_filename)

        try:
            plt.savefig(plot_filepath)
            print(f"Saved plot: {plot_filepath}")

            # Optionally log the plot back to MLflow as an artifact of the original run
            # Make sure an MLflow run is not active here if you are logging to a *specific past run*
            # For simplicity, we'll use client.log_artifact if we want to add to specific run
            # Note: mlflow.log_artifact() logs to the *current active run*.
            # To log to a *specific past run*, you'd ideally do this when that run is active,
            # or use client.log_artifact(run_id_to_log_to, local_path, artifact_path)
            # For this script, it's easier to just save locally.
            # If you want to add it to the *original* run, that's more complex from a separate script.
            # This example will log it if an active run is set (e.g. if this script itself is an MLflow run)
            if mlflow.active_run():
                mlflow.log_artifact(
                    plot_filepath, artifact_path=f"run_plots/{run_name}"
                )

        except Exception as e:
            print(f"Error saving/logging plot {plot_filepath}: {e}")
        plt.close()


def generate_experiment_report(
    experiment_name: str,
    output_base_dir: str = "experiment_reports",
    metrics_to_plot: Dict[str, str] = None,
):
    """
    Fetches data for all runs in an experiment, generates plots, and saves them.

    Args:
        experiment_name (str): The name of the MLflow experiment.
        output_base_dir (str): Base directory to save all reports and plots.
        metrics_to_plot (Dict[str, str], optional): Metrics to plot. Defaults to DEFAULT_METRICS_TO_PLOT.
    """
    if metrics_to_plot is None:
        metrics_to_plot = DEFAULT_METRICS_TO_PLOT

    client = MlflowClient()
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"Experiment '{experiment_name}' not found.")
            return
    except Exception as e:
        print(f"Error fetching experiment '{experiment_name}': {e}")
        return

    print(
        f"Fetching runs for experiment ID: {experiment.experiment_id} (Name: {experiment_name})"
    )
    # Search for all runs in the experiment
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    # order_by=["start_time DESC"]) # Optional: order runs

    if not runs:
        print(f"No runs found for experiment '{experiment_name}'.")
        return

    # Create a unique directory for this experiment's report
    experiment_output_dir = os.path.join(
        output_base_dir, experiment_name.replace(" ", "_").replace("/", "_")
    )
    os.makedirs(experiment_output_dir, exist_ok=True)
    print(f"Saving reports and plots to: {experiment_output_dir}")

    for run_info in runs:
        plot_metrics_for_run(
            run_info.info.run_id, metrics_to_plot, experiment_output_dir, client
        )

    print(f"\nFinished generating plots for experiment '{experiment_name}'.")
    print(f"Reports saved in: {experiment_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots from MLflow experiment data."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the MLflow experiment to generate reports for.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiment_reports",
        help="Base directory to save the generated plot reports.",
    )
    # Future: Add --run_id to plot for a specific run only.
    # Future: Add --metrics to specify which metrics to plot from command line.

    args = parser.parse_args()

    # Set Seaborn style for nicer plots
    sns.set_theme(style="whitegrid")

    generate_experiment_report(args.experiment_name, args.output_dir)

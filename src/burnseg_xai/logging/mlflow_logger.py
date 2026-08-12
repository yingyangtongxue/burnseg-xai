import os
from pathlib import Path

import mlflow

from burnseg_xai.config.schema import ProjectConfig


def _to_uri(path: str) -> str:
    """Convert a local filesystem path (including Windows drive paths) to a file:// URI."""
    if path.startswith(("http://", "https://", "file://", "sqlite://", "mlflow://")):
        return path
    return Path(os.path.abspath(path)).as_uri()


class MLflowLogger:
    """
    Thin wrapper around MLflow. Requires ProjectConfig (dataclass, not dict).
    Do not use mlflow.autolog(): this project requires explicit named logging.
    """

    def __init__(self, config: ProjectConfig) -> None:
        uri = _to_uri(config.mlflow_tracking_uri)
        os.makedirs(config.mlflow_tracking_uri, exist_ok=True)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(config.mlflow_experiment)

    def start_run(self, run_name: str | None = None) -> str:
        """Starts a new run and returns the run_id."""
        mlflow.start_run(run_name=run_name)
        return mlflow.active_run().info.run_id

    def resume_run(self, run_id: str) -> None:
        """Resumes an existing MLflow run (for checkpoint-based restarts)."""
        mlflow.start_run(run_id=run_id)

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        mlflow.log_metric(name, value, step=step)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        mlflow.log_artifact(path)

    def set_tag(self, key: str, value: str) -> None:
        mlflow.set_tag(key, value)

    def end_run(self) -> None:
        mlflow.end_run()

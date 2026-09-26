from typing import Any, Dict, Optional
import mlflow
from aid2go.training.trackers import Tracker

class MLFlowTracker(Tracker):
    def __init__(self, experiment_name: str, uri: str, run_name: Optional[str] = None,
                 artifact_location: Optional[str] = None):
        mlflow.set_tracking_uri(uri)
        if artifact_location is not None and mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        mlflow.log_artifact(local_path, artifact_path)

    def end_run(self):
        mlflow.end_run()

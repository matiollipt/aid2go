from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

class Tracker(ABC):
    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        pass

    @abstractmethod
    def end_run(self):
        pass

class NoOpTracker(Tracker):
    def log_params(self, params: Dict[str, Any]):
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        pass
    
    def end_run(self):
        pass

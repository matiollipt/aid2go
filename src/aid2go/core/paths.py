from pathlib import Path
from typing import Optional

class ProjectPaths:
    def __init__(self, root_dir: Optional[str | Path] = None):
        if root_dir is None:
            # Default to repo root if not specified (assuming src/aid2go/core/paths.py location)
            self.root = Path(__file__).resolve().parent.parent.parent.parent.parent
        else:
            self.root = Path(root_dir)

        self.data = self.root / "data"
        self.raw = self.data / "raw"
        self.interim = self.data / "interim"
        self.processed = self.data / "processed"
        
        self.configs = self.root / "configs"
        self.reports = self.root / "reports"
        self.models = self.root / "models"
        self.logs = self.root / "logs"

    def get_processed_dir(self, version: str = "default") -> Path:
        p = self.processed / version
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_raw_dir(self) -> Path:
        self.raw.mkdir(parents=True, exist_ok=True)
        return self.raw

    def get_report_dir(self, report_name: str) -> Path:
        p = self.reports / report_name
        p.mkdir(parents=True, exist_ok=True)
        return p

# Global instance for default usage if needed, but prefer passing config
default_paths = ProjectPaths()

from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
from pydantic import BaseModel, Field

from aid2go.core.types import ModelName, EvidenceType

class PathsConfig(BaseModel):
    root_dir: Optional[str] = None
    data_dir: str = "data"
    output_dir: str = "output"

class DataConfig(BaseModel):
    go_obo_url: str = "https://purl.obolibrary.org/obo/go/go-basic.obo"
    goa_url: str = "http://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_gcrp.gaf.gz" # Example default
    evidence_codes: Dict[str, List[str]] = Field(default_factory=dict)
    
    # Preprocessing
    min_sequence_length: int = 50
    max_sequence_length: int = 4000
    ppi_score_threshold: int = 700
    
    dataset_version: str = "v1"

class ModelConfig(BaseModel):
    name: ModelName = ModelName.BASELINE
    hidden_channels: int = 128
    num_layers: int = 2
    dropout: float = 0.5
    heads: int = 4  # For attention models
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TrainingConfig(BaseModel):
    batch_size: int = 1024
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 100
    device: str = "cuda"
    num_workers: int = 4
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    eval_every: int = 1
    patience: int = 10

class TrackingConfig(BaseModel):
    enabled: bool = True
    mlflow_uri: str = "file:./mlruns"
    experiment_name: str = "aid2go"
    run_name: Optional[str] = None

class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save_yaml(self, path: str | Path):
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f)

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
    goa_url: str = "https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_gcrp.gaf.gz"
    goa_archive_path: Optional[str] = None
    reuse_protein_data_path: Optional[str] = None
    evidence_codes: Dict[str, List[str]] = Field(default_factory=dict)
    
    # Preprocessing
    min_sequence_length: int = 50
    max_sequence_length: int = 4000
    dataset_version: str = "v1"
    
    # Prototyping and graph parameters
    top_n_go_terms: int = 1000
    max_proteins: int = 5000
    gaf_chunk_size: int = 2_000_000
    num_workers: Optional[int] = None
    
    # Protein sequence model and embedding storage
    esm_model_name: str = "facebook/esm2_t33_650M_UR50D"
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_token_budget: int = Field(default=8192, ge=1)
    
    # KNN Graph parameters
    knn_k: int = 10
    strict_cosine_similarity: bool = False
    
    # Protein structural representation
    protein_representation: str = "node" # "node" or "graph"
    c_alpha_distance_threshold: float = 8.0

class ModelConfig(BaseModel):
    name: ModelName = ModelName.BASELINE
    hidden_channels: int = 128
    num_layers: int = 2
    dropout: float = 0.5
    heads: int = 4  # For attention models
    attention_budgets: Dict[str, int] = Field(default_factory=lambda: {
        "similarity": 16, "is_a": 16, "has_function": 32, "function_of": 32,
    })
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TrainingConfig(BaseModel):
    batch_size: int = 1024
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 100
    device: str = "auto"
    num_workers: int = 4
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    eval_every: int = 1
    patience: int = 10
    seed: int = 42
    split_strategy: str = "link"
    loss: str = "auto"

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
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)

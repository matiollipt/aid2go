from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any
import json

@dataclass
class DataArtifacts:
    go_graph_path: str
    go_nodes_path: str
    go_edges_path: str
    protein_embeddings_path: str
    protein_index_path: str
    ppi_edge_index_path: str
    goa_path: str
    hetero_data_path: str
    
    metadata: Dict[str, Any]

    def save_manifest(self, path: Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load_manifest(cls, path: Path) -> "DataArtifacts":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

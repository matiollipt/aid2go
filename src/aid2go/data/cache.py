"""Small input fingerprints for derived preprocessing artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from aid2go.core.paths import ProjectPaths


def fingerprint(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"missing": True}
    stat = path.stat()
    result = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size <= 128 * 1024 * 1024:
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def stamp(inputs: dict[str, Path], parameters: dict) -> dict:
    return {"inputs": {name: fingerprint(path) for name, path in sorted(inputs.items())},
            "parameters": parameters}


def manifest_path(directory: Path, stage: str) -> Path:
    return Path(directory) / f".{stage}_inputs.json"


def record(directory: Path, stage: str, inputs: dict[str, Path], parameters: dict):
    path = manifest_path(directory, stage)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(stamp(inputs, parameters), indent=2) + "\n")
    os.replace(temporary, path)


def current(directory: Path, stage: str, inputs: dict[str, Path], parameters: dict) -> bool:
    path = manifest_path(directory, stage)
    if not path.is_file() or any(not item.is_file() or item.stat().st_size == 0
                                  for item in inputs.values()):
        return False
    try:
        previous = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return previous == stamp(inputs, parameters)


def stage_spec(stage: str, config, directory: Path) -> tuple[dict[str, Path], dict]:
    directory = Path(directory)
    data = config.data
    if stage == "go":
        return ({"obo": directory / "go-basic.obo",
                 "nodes_output": directory / "go_nodes.csv",
                 "edges_output": directory / "go_edges.tsv",
                 "embeddings_output": directory / "go_embeddings.pkl"},
                {"go_obo_url": data.go_obo_url, "top_n_go_terms": data.top_n_go_terms,
                 "go_embedding_model": "bert-base-uncased"})
    if stage == "goa":
        archive = directory / data.goa_url.rsplit("/", 1)[-1]
        if data.goa_archive_path:
            candidate = Path(data.goa_archive_path)
            if not candidate.is_absolute():
                candidate = ProjectPaths(config.paths.root_dir).root / candidate
            if candidate.exists():
                archive = candidate
        return ({"archive": archive, "go_nodes": directory / "go_nodes.csv",
                 "obo": directory / "go-basic.obo",
                 "goa_output": directory / "goa_filtered.tsv"},
                {"evidence_codes": data.evidence_codes,
                 "max_proteins": data.max_proteins, "goa_url": data.goa_url})
    if stage == "proteins":
        return ({"goa": directory / "goa_filtered.tsv",
                 "protein_output": directory / "protein_data.h5",
                 "index_output": directory / "protein_index.pkl",
                 "hashes_output": directory / "sequence_hashes.tsv",
                 "sequences_output": directory / "protein_sequences.tsv",
                 "ia_output": directory / "ia_values.pkl",
                 "ia_tsv_output": directory / "ia_values.tsv",
                 "go_nodes": directory / "go_nodes.csv",
                 "obo": directory / "go-basic.obo"},
                {"esm_model_name": data.esm_model_name,
                 "min_sequence_length": data.min_sequence_length,
                 "max_sequence_length": data.max_sequence_length,
                 "c_alpha_distance_threshold": data.c_alpha_distance_threshold})
    if stage == "ppi":
        return ({"protein_data": directory / "protein_data.h5",
                 "protein_index": directory / "protein_index.pkl",
                 "similarity_output": directory / "ppi_edge_index.pt"},
                {"knn_k": data.knn_k,
                 "strict_cosine_similarity": data.strict_cosine_similarity})
    if stage == "graph":
        return ({name: directory / filename for name, filename in {
            "go_embeddings": "go_embeddings.pkl", "go_edges": "go_edges.tsv",
            "goa": "goa_filtered.tsv", "ia": "ia_values.pkl",
            "protein_data": "protein_data.h5", "protein_index": "protein_index.pkl",
            "ppi": "ppi_edge_index.pt", "graph_output": "hetero_data.pt",
        }.items()}, {"protein_representation": data.protein_representation})
    raise ValueError(f"Unknown preprocessing stage: {stage}")


def record_stage(stage: str, config, directory: Path):
    inputs, parameters = stage_spec(stage, config, directory)
    record(directory, stage, inputs, parameters)


def stage_current(stage: str, config, directory: Path) -> bool:
    inputs, parameters = stage_spec(stage, config, directory)
    return current(directory, stage, inputs, parameters)

"""Versioned repair of derived datasets without modifying the source version."""

from __future__ import annotations

import json
import pickle
import shutil
import hashlib
from pathlib import Path

import h5py
import numpy as np
import obonet
import pandas as pd

from aid2go.core.config import AppConfig
from aid2go.core.paths import ProjectPaths


GO_FILES = ("go-basic.obo", "go_nodes.csv", "go_edges.tsv", "go_embeddings.pkl")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_validated_go(config: AppConfig, source_version: str = "v1") -> Path:
    paths = ProjectPaths(config.paths.root_dir)
    source = paths.processed / source_version
    target = paths.get_processed_dir(config.data.dataset_version)
    if source.resolve() == target.resolve():
        raise ValueError("Source and repaired dataset versions must differ")
    missing = [str(source / name) for name in GO_FILES if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot reuse GO artifacts: {', '.join(missing)}")
    nodes = pd.read_csv(source / "go_nodes.csv")
    if len(nodes) != config.data.top_n_go_terms or nodes.term.nunique() != len(nodes):
        raise ValueError("Cached GO term set does not match the requested config")
    with open(source / "go_embeddings.pkl", "rb") as input_file:
        embeddings = pickle.load(input_file)
    if set(nodes.term) != set(embeddings):
        raise ValueError("Cached GO embeddings do not match GO node IDs")
    if any(np.asarray(vector).shape != (768,) or not np.isfinite(np.asarray(vector)).all()
           for vector in embeddings.values()):
        raise ValueError("Cached GO embeddings have invalid dimensions or values")
    ontology = obonet.read_obo(str(source / "go-basic.obo"), ignore_obsolete=True)
    for row in nodes.itertuples(index=False):
        if row.term not in ontology:
            raise ValueError(f"Cached GO term {row.term} is absent from its source OBO")
        attributes = ontology.nodes[row.term]
        definition = attributes.get("def", "")
        definition = definition.split('"', 2)[1] if definition else ""
        if (row.name != attributes.get("name", "")
                or row.namespace != attributes.get("namespace", "unknown")
                or row.definition != definition
                or int(row.degree) != ontology.degree(row.term)):
            raise ValueError(f"Cached GO input differs from its source OBO for {row.term}")
    edges = pd.read_csv(source / "go_edges.tsv", sep="\t")
    if not set(edges.source).issubset(set(nodes.term)) or not set(edges.target).issubset(set(nodes.term)):
        raise ValueError("Cached GO edges reference missing terms")
    for name in GO_FILES:
        destination = target / name
        if destination.exists() and _sha256(destination) != _sha256(source / name):
            raise ValueError(f"Existing repaired GO artifact differs from validated source: {destination}")
        if not destination.exists():
            shutil.copy2(source / name, destination)
    (target / "source_go_artifacts.json").write_text(json.dumps({
        "source_version": source_version,
        "go_terms": len(nodes),
        "go_embeddings": len(embeddings),
        "source_sha256": {name: _sha256(source / name) for name in GO_FILES},
        "files": list(GO_FILES),
    }, indent=2) + "\n")
    from aid2go.data.cache import record_stage
    record_stage("go", config, target)
    return target


def write_repair_audit(config: AppConfig, source_version: str = "v1") -> Path:
    paths = ProjectPaths(config.paths.root_dir)
    source = paths.processed / source_version
    target = paths.processed / config.data.dataset_version
    old = pd.read_csv(source / "goa_filtered.tsv", sep="\t")
    new = pd.read_csv(target / "goa_filtered.tsv", sep="\t")
    with open(source / "protein_index.pkl", "rb") as input_file:
        old_proteins = set(pickle.load(input_file))
    with open(target / "protein_index.pkl", "rb") as input_file:
        new_proteins = set(pickle.load(input_file))
    archive = Path(config.data.goa_archive_path) if config.data.goa_archive_path else None
    if archive and not archive.is_absolute():
        archive = paths.root / archive
    with h5py.File(target / "protein_data.h5", "r") as protein_archive:
        legacy_reused = int(protein_archive.attrs.get("legacy_reused_count", 0))
    report = {
        "source_version": source_version,
        "repaired_version": config.data.dataset_version,
        "source_goa_rows": len(old),
        "source_unique_pairs": len(old.drop_duplicates(["DB_Object_ID", "GO_ID"])),
        "source_duplicate_evidence_rows": int(old.duplicated(["DB_Object_ID", "GO_ID"]).sum()),
        "repaired_goa_rows": len(new),
        "repaired_unique_pairs": len(new.drop_duplicates(["DB_Object_ID", "GO_ID"])),
        "source_proteins": len(old_proteins),
        "repaired_sequence_valid_proteins": len(new_proteins),
        "overlapping_protein_ids": len(old_proteins & new_proteins),
        "raw_archive_bytes": archive.stat().st_size if archive and archive.exists() else None,
        "legacy_embeddings_reused": legacy_reused,
        "legacy_embedding_caveat": (
            "Old HDF5 lacks sequence hashes; reused entries were checked by accession, ESM dimension, and residue length."
            if legacy_reused else "None: all ESM embeddings were regenerated from fetched sequences."
        ),
    }
    path = target / "repair_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path

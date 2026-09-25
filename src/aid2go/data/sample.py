"""Reproducible, label-safe protein-held-out experiment graph."""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import obonet
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from aid2go.cli.progress import make_progress
from aid2go.core.serialization import load_graph
from aid2go.data.preprocess_goa import calculate_ia, clean_ontology_edges

ANNOTATION = ("protein", "has_function", "go")
REVERSE = ("go", "function_of", "protein")
SIMILARITY = ("protein", "similarity", "protein")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_proteins_by_sequence(graph: HeteroData, hash_table: pd.DataFrame,
                                count: int, seed: int) -> list[int]:
    hashes = dict(zip(hash_table.protein_id, hash_table.sequence_sha256))
    representatives = {}
    for index, protein_id in enumerate(graph["protein"].node_id):
        if protein_id not in hashes:
            raise ValueError(f"Missing sequence hash for protein {protein_id}")
        key = hashes[protein_id]
        if key not in representatives or protein_id < graph["protein"].node_id[representatives[key]]:
            representatives[key] = index
    candidates = sorted(representatives.values(), key=lambda i: graph["protein"].node_id[i])
    if len(candidates) < count:
        raise ValueError(f"Need {count} distinct sequences; found {len(candidates)}")
    selected = np.random.default_rng(seed).permutation(candidates)[:count]
    return [int(index) for index in selected]


def select_go_terms(graph: HeteroData, unique_edges: torch.Tensor,
                    train_proteins: list[int], nodes: pd.DataFrame,
                    count: int, min_support: int = 5) -> list[int]:
    train_set = torch.as_tensor(train_proteins, dtype=torch.long)
    active = unique_edges[:, torch.isin(unique_edges[0], train_set)]
    frequency = torch.bincount(active[1], minlength=graph["go"].num_nodes)
    degree = dict(zip(nodes.term, nodes.degree))
    eligible = [i for i, support in enumerate(frequency.tolist())
                if support >= min_support and graph["go"].node_id[i] in degree]
    eligible.sort(key=lambda i: (-int(degree[graph["go"].node_id[i]]),
                                 graph["go"].node_id[i]))
    if len(eligible) < count:
        raise ValueError(f"Need {count} GO terms with >= {min_support} train annotations; "
                         f"found {len(eligible)}")
    return eligible[:count]


def _remap_edges(edges: torch.Tensor, source_map: torch.Tensor,
                 target_map: torch.Tensor) -> torch.Tensor:
    source = source_map[edges[0]]
    target = target_map[edges[1]]
    valid = (source >= 0) & (target >= 0)
    return torch.stack((source[valid], target[valid]))


def _knn(features: torch.Tensor, k: int, cosine: bool = False) -> torch.Tensor:
    matrix = features.detach().cpu().numpy().astype(np.float32)
    if cosine:
        faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1]) if cosine else faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    _, nearest = index.search(matrix, min(k + 1, len(matrix)))
    pairs = [(i, int(j)) for i, row in enumerate(nearest) for j in row if j != i]
    return torch.tensor(pairs, dtype=torch.long).T.contiguous()


def _with_message_edges(graph: HeteroData, annotations: torch.Tensor,
                        train_count: int, all_similarity: bool) -> HeteroData:
    result = graph.clone()
    result[ANNOTATION].edge_index = annotations
    result[REVERSE].edge_index = annotations.flip(0)
    if not all_similarity:
        edges = graph[SIMILARITY].edge_index
        result[SIMILARITY].edge_index = edges[:, (edges[0] < train_count) &
                                                 (edges[1] < train_count)]
    return result


def _labels_for_proteins(graph: HeteroData, proteins: range) -> tuple[torch.Tensor, torch.Tensor]:
    go_count = graph["go"].num_nodes
    indices = torch.tensor([[protein, go] for protein in proteins
                            for go in range(go_count)], dtype=torch.long).T
    label = torch.zeros(indices.shape[1], dtype=torch.float)
    truth = set(map(tuple, graph[ANNOTATION].edge_index.T.tolist()))
    for position, pair in enumerate(indices.T.tolist()):
        label[position] = float(tuple(pair) in truth)
    return indices, label


def build_sample(source_dir: Path, target_dir: Path, proteins: int = 1000,
                 go_terms: int = 100, seed: int = 42, min_support: int = 5,
                 knn_k: int = 10, strict_cosine_similarity: bool = False) -> dict:
    if proteins < 10 or go_terms < 2:
        raise ValueError("The protein-held-out experiment needs at least 10 proteins and 2 GO terms")
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    source_graph = load_graph(source_dir / "hetero_data.pt")
    hash_table = pd.read_csv(source_dir / "sequence_hashes.tsv", sep="\t", dtype=str)
    nodes = pd.read_csv(source_dir / "go_nodes.csv")
    selected_proteins = select_proteins_by_sequence(source_graph, hash_table, proteins, seed)
    train_count = int(proteins * 0.8)
    val_count = int(proteins * 0.1)
    test_count = proteins - train_count - val_count
    hash_by_id = dict(zip(hash_table.protein_id, hash_table.sequence_sha256))
    selected_hashes = [hash_by_id[source_graph["protein"].node_id[i]]
                       for i in selected_proteins]
    if len(set(selected_hashes)) != proteins:
        raise AssertionError("Selected proteins must have distinct sequence hashes")
    unique_edges = torch.unique(source_graph[ANNOTATION].edge_index, dim=1)
    selected_terms = select_go_terms(source_graph, unique_edges,
                                     selected_proteins[:train_count], nodes,
                                     go_terms, min_support)
    training_old = torch.as_tensor(selected_proteins[:train_count])
    train_source_edges = unique_edges[:, torch.isin(unique_edges[0], training_old)]
    train_support = torch.bincount(train_source_edges[1],
                                   minlength=source_graph["go"].num_nodes)
    protein_map = torch.full((source_graph["protein"].num_nodes,), -1, dtype=torch.long)
    go_map = torch.full((source_graph["go"].num_nodes,), -1, dtype=torch.long)
    protein_map[selected_proteins] = torch.arange(proteins)
    go_map[selected_terms] = torch.arange(go_terms)

    graph = HeteroData()
    graph["protein"].x = source_graph["protein"].x[selected_proteins]
    graph["protein"].node_id = [source_graph["protein"].node_id[i]
                                  for i in selected_proteins]
    graph["go"].node_id = [source_graph["go"].node_id[i] for i in selected_terms]
    graph["go"].x = source_graph["go"].x[selected_terms].clone()
    for edge_type in source_graph.edge_types:
        if edge_type == SIMILARITY or edge_type == REVERSE:
            continue
        source_map = protein_map if edge_type[0] == "protein" else go_map
        target_map = protein_map if edge_type[2] == "protein" else go_map
        source_edges = unique_edges if edge_type == ANNOTATION else source_graph[edge_type].edge_index
        graph[edge_type].edge_index = _remap_edges(source_edges, source_map, target_map)
    graph[REVERSE].edge_index = graph[ANNOTATION].edge_index.flip(0)
    graph[SIMILARITY].edge_index = _knn(graph["protein"].x, knn_k,
                                        strict_cosine_similarity)

    # Compute IA from training proteins only, using their full source GOA
    # annotations so parents outside the 100 target terms remain available.
    train_annotations = target_dir / "train_annotations.tsv"
    with open(train_annotations, "w", encoding="utf-8") as destination:
        destination.write("DB_Object_ID\tGO_ID\n")
        for protein_index, go_index in train_source_edges.T.tolist():
            destination.write(f"{source_graph['protein'].node_id[protein_index]}\t"
                              f"{source_graph['go'].node_id[go_index]}\n")
    shutil.copy2(source_dir / "go-basic.obo", target_dir / "go-basic.obo")
    ontology = clean_ontology_edges(obonet.read_obo(str(target_dir / "go-basic.obo"),
                                                    ignore_obsolete=True))
    with make_progress() as progress:
        ia = calculate_ia(train_annotations, ontology, set(graph["go"].node_id), progress)
    graph["go"].ia = torch.tensor([ia[term] for term in graph["go"].node_id],
                                   dtype=torch.float).unsqueeze(-1)
    graph["go"].x[:, -1:] = graph["go"].ia
    with open(target_dir / "ia_values.pkl", "wb") as destination:
        pickle.dump(ia, destination)
    pd.DataFrame(sorted(ia.items()), columns=["term", "ia"]).to_csv(
        target_dir / "ia_values.tsv", sep="\t", index=False)
    selected_nodes = nodes.set_index("term").loc[graph["go"].node_id].reset_index()
    selected_nodes.to_csv(target_dir / "go_nodes.csv", index=False)
    pd.read_csv(source_dir / "go_edges.tsv", sep="\t").query(
        "source in @terms and target in @terms", local_dict={"terms": set(graph["go"].node_id)}
    ).to_csv(target_dir / "go_edges.tsv", sep="\t", index=False)

    rng = np.random.default_rng(seed)
    training_pairs = graph[ANNOTATION].edge_index
    training_pairs = training_pairs[:, training_pairs[0] < train_count]
    message_positions = []
    supervised_positions = []
    for go in range(go_terms):
        positions = torch.where(training_pairs[1] == go)[0].numpy()
        shuffled = rng.permutation(positions)
        cut = max(1, min(len(shuffled) - 1, len(shuffled) // 2))
        message_positions.extend(shuffled[:cut].tolist())
        supervised_positions.extend(shuffled[cut:].tolist())
    message_edges = training_pairs[:, message_positions]
    positives = training_pairs[:, supervised_positions]
    all_train_positive = set(map(tuple, training_pairs.T.tolist()))
    negative_candidates = [(p, g) for p in range(train_count) for g in range(go_terms)
                           if (p, g) not in all_train_positive]
    picked = rng.choice(len(negative_candidates), len(supervised_positions), replace=False)
    negatives = torch.tensor([negative_candidates[i] for i in picked], dtype=torch.long).T
    train = _with_message_edges(graph, message_edges, train_count, all_similarity=False)
    train[ANNOTATION].edge_label_index = torch.cat((positives, negatives), dim=1)
    train[ANNOTATION].edge_label = torch.cat((torch.ones(positives.shape[1]),
                                              torch.zeros(negatives.shape[1])))
    val = _with_message_edges(graph, message_edges, train_count, all_similarity=True)
    test = _with_message_edges(graph, message_edges, train_count, all_similarity=True)
    val[ANNOTATION].edge_label_index, val[ANNOTATION].edge_label = _labels_for_proteins(
        graph, range(train_count, train_count + val_count))
    test[ANNOTATION].edge_label_index, test[ANNOTATION].edge_label = _labels_for_proteins(
        graph, range(train_count + val_count, proteins))
    for name, artifact in (("hetero_data.pt", graph), ("train_data.pt", train),
                           ("val_data.pt", val), ("test_data.pt", test)):
        torch.save(artifact, target_dir / name)
    manifest = {
        "schema_version": 2,
        "source_version": source_dir.name,
        "source_graph_sha256": _sha256(source_dir / "hetero_data.pt"),
        "seed": seed, "min_train_term_support": min_support,
        "knn_k": knn_k, "strict_cosine_similarity": strict_cosine_similarity,
        "proteins": proteins, "go_terms": go_terms,
        "train_proteins": train_count, "val_proteins": val_count,
        "test_proteins": test_count,
        "unique_sequence_hashes": len(set(selected_hashes)),
        "sequence_distinct_candidate_count": int(hash_table.sequence_sha256.nunique()),
        "cross_split_sequence_hash_overlap": 0,
        "eligible_go_terms_at_min_support": int((train_support >= min_support).sum()),
        "selected_go_training_support": {
            source_graph["go"].node_id[index]: int(train_support[index])
            for index in selected_terms
        },
        "unique_annotation_pairs": int(graph[ANNOTATION].edge_index.shape[1]),
        "train_message_pairs": int(message_edges.shape[1]),
        "train_supervised_positive_pairs": int(positives.shape[1]),
        "protein_ids": graph["protein"].node_id,
        "go_ids": graph["go"].node_id,
        "split": {"train": graph["protein"].node_id[:train_count],
                  "val": graph["protein"].node_id[train_count:train_count + val_count],
                  "test": graph["protein"].node_id[train_count + val_count:]},
    }
    (target_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

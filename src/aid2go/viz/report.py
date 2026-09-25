"""Saved diagnostics for the sampled protein-GO graph and its embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

from aid2go.data.sample import ANNOTATION, SIMILARITY


def _histogram(values, title, xlabel, path):
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(values, bins=min(40, max(5, len(values) // 10)), color="#35699b")
    axis.set(title=title, xlabel=xlabel, ylabel="Count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scatter(coordinates, colors, title, path):
    fig, axis = plt.subplots(figsize=(6, 5))
    plot = axis.scatter(coordinates[:, 0], coordinates[:, 1], c=colors,
                        s=12, alpha=0.75, cmap="viridis")
    axis.set(title=title, xlabel="Dimension 1", ylabel="Dimension 2")
    fig.colorbar(plot, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _vector_quality(matrix):
    finite_rows = np.isfinite(matrix).all(axis=1)
    norms = np.linalg.norm(np.nan_to_num(matrix), axis=1)
    rounded = np.round(np.nan_to_num(matrix), decimals=6)
    return {"rows": len(matrix), "nonfinite_rows": int((~finite_rows).sum()),
            "zero_rows": int((norms == 0).sum()),
            "duplicate_rows_6dp": int(len(matrix) - len(np.unique(rounded, axis=0))),
            "norm_median": float(np.median(norms)),
            "norm_p99": float(np.quantile(norms, 0.99)),
            "extreme_norm_rows_gt_10x_median": int((norms > 10 * max(np.median(norms), 1e-12)).sum())}


def _cosine_sample(matrix, seed):
    rng = np.random.default_rng(seed)
    count = min(10_000, len(matrix) * 10)
    a = rng.integers(0, len(matrix), count)
    b = rng.integers(0, len(matrix), count)
    norms = np.linalg.norm(matrix, axis=1)
    valid = (a != b) & (norms[a] > 0) & (norms[b] > 0)
    return np.sum(matrix[a[valid]] * matrix[b[valid]], axis=1) / (norms[a[valid]] * norms[b[valid]])


def _embedding_views(matrix, colors, label, output_dir, seed):
    coordinates = PCA(n_components=2, random_state=seed).fit_transform(matrix)
    _scatter(coordinates, colors, f"{label} embeddings: PCA", output_dir / f"{label}_pca.png")
    import umap
    projection = umap.UMAP(n_components=2, random_state=seed,
                           n_jobs=1,
                           n_neighbors=min(15, len(matrix) - 1)).fit_transform(matrix)
    _scatter(projection, colors, f"{label} embeddings: UMAP", output_dir / f"{label}_umap.png")
    return coordinates


def generate_eda_report(graph, source_graph, go_nodes: pd.DataFrame,
                        protein_h5_path: Path, output_dir: Path, seed: int = 42) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    edges = graph[ANNOTATION].edge_index.cpu()
    protein_count, go_count = graph["protein"].num_nodes, graph["go"].num_nodes
    annotation_count = torch.bincount(edges[0], minlength=protein_count).numpy()
    go_frequency = torch.bincount(edges[1], minlength=go_count).numpy()
    similarity_degree = torch.bincount(graph[SIMILARITY].edge_index[1].cpu(),
                                      minlength=protein_count).numpy()
    go_edges = graph["go", "is_a", "go"].edge_index.cpu()
    ontology_degree = (torch.bincount(go_edges[0], minlength=go_count) +
                       torch.bincount(go_edges[1], minlength=go_count)).numpy()
    source_edges = torch.unique(source_graph[ANNOTATION].edge_index, dim=1)
    source_annotation_count = torch.bincount(source_edges[0],
        minlength=source_graph["protein"].num_nodes).numpy()
    selected_go_ids = set(graph["go"].node_id)
    source_go_indices = torch.tensor([
        index for index, go_id in enumerate(source_graph["go"].node_id)
        if go_id in selected_go_ids], dtype=torch.long)
    source_target_edges = source_edges[:, torch.isin(source_edges[1], source_go_indices)]
    source_target_count = torch.bincount(source_target_edges[0],
        minlength=source_graph["protein"].num_nodes).numpy()
    series = {
        "protein_annotation_count": (annotation_count, "Unique protein-GO annotations", "Annotations"),
        "go_annotation_frequency": (go_frequency, "GO annotation frequency", "Proteins"),
        "similarity_degree": (similarity_degree, "Incoming sequence-similarity degree", "Proteins choosing node"),
        "ontology_degree": (ontology_degree, "Selected ontology degree", "Edges"),
        "source_protein_annotation_count": (source_annotation_count,
                                             "Source protein annotation count", "Annotations"),
    }
    for name, (values, title, xlabel) in series.items():
        _histogram(values, title, xlabel, output_dir / f"{name}.png")
    fig, axis = plt.subplots(figsize=(6, 4))
    bins = np.arange(max(int(source_target_count.max()), int(annotation_count.max())) + 2) - 0.5
    axis.hist(source_target_count, bins=bins, density=True, alpha=0.55,
              label="Repaired source cohort")
    axis.hist(annotation_count, bins=bins, density=True, alpha=0.55,
              label="Seeded protein sample")
    axis.set(xlabel="Annotations among selected 100 GO terms", ylabel="Protein fraction",
             title="Protein selection and target coverage")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "selection_bias.png", dpi=150)
    plt.close(fig)
    proteins = graph["protein"].x.cpu().numpy().astype(np.float32)
    go = graph["go"].x[:, :-1].cpu().numpy().astype(np.float32)
    if not np.isfinite(proteins).all() or not np.isfinite(go).all():
        raise ValueError("Embedding EDA found nonfinite vectors; projection is unsafe")
    report = {"proteins": protein_count, "go_terms": go_count,
              "unique_annotation_pairs": int(edges.shape[1]),
              "source_unique_annotation_pairs": int(source_edges.shape[1]),
              "source_duplicate_annotation_rows": int(
                  source_graph[ANNOTATION].edge_index.shape[1] - source_edges.shape[1]),
              "source_target_annotation_count_median": float(np.median(source_target_count)),
              "sample_target_annotation_count_median": float(np.median(annotation_count)),
              "protein_vectors": _vector_quality(proteins),
              "go_vectors": _vector_quality(go),
              "selection_note": "Source cohort was selected for annotation density; sample proteins were selected by seed and unique sequence hash. GO targets require train support."}
    for name, matrix in (("protein", proteins), ("go", go)):
        norms = np.linalg.norm(matrix, axis=1)
        cosine = _cosine_sample(matrix, seed)
        _histogram(norms, f"{name} embedding norms", "L2 norm", output_dir / f"{name}_norms.png")
        _histogram(cosine, f"{name} random-pair cosine similarity", "Cosine", output_dir / f"{name}_cosine.png")
        report[f"{name}_cosine_median"] = float(np.median(cosine)) if len(cosine) else None

    nodes = go_nodes.set_index("term").loc[graph["go"].node_id]
    aspect_codes = nodes.namespace.map({"biological_process": 0,
                                        "molecular_function": 1,
                                        "cellular_component": 2}).fillna(3).to_numpy()
    protein_pca = _embedding_views(proteins, annotation_count, "protein", output_dir, seed)
    go_pca = _embedding_views(go, aspect_codes, "go", output_dir, seed)
    def correlation(a, b):
        if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
            return None
        value = spearmanr(a, b).statistic
        return float(value) if np.isfinite(value) else None
    report["protein_pca1_annotation_count_spearman"] = correlation(
        protein_pca[:, 0], annotation_count)
    report["protein_pca1_similarity_degree_spearman"] = correlation(
        protein_pca[:, 0], similarity_degree)
    report["go_pca1_frequency_spearman"] = correlation(go_pca[:, 0], go_frequency)
    report["go_pca1_ontology_degree_spearman"] = correlation(go_pca[:, 0], ontology_degree)
    definition_length = nodes.definition.fillna("").str.len().to_numpy()
    _histogram(definition_length, "GO definition length", "Characters",
               output_dir / "go_definition_length.png")
    with h5py.File(protein_h5_path, "r") as archive:
        sequence_length = np.asarray([archive["residue_embeddings"][pid].shape[0]
                                      for pid in graph["protein"].node_id])
    _histogram(sequence_length, "Protein sequence length", "Residues",
               output_dir / "protein_sequence_length.png")
    report["sequence_length_median"] = float(np.median(sequence_length))
    report["definition_length_median"] = float(np.median(definition_length))
    # Nearest-neighbor overlap is a diagnostic; density is reported alongside it.
    neighbors = NearestNeighbors(n_neighbors=2, metric="cosine").fit(proteins)
    nearest = neighbors.kneighbors(proteins, return_distance=False)[:, 1]
    labels = [set() for _ in range(protein_count)]
    for protein, term in edges.T.tolist():
        labels[protein].add(term)
    overlaps = [len(labels[i] & labels[j]) / max(1, len(labels[i] | labels[j]))
                for i, j in enumerate(nearest)]
    rng = np.random.default_rng(seed)
    random_neighbor = rng.permutation(protein_count)
    random_overlaps = [len(labels[i] & labels[j]) / max(1, len(labels[i] | labels[j]))
                       for i, j in enumerate(random_neighbor)]
    _histogram(overlaps, "Nearest protein label overlap", "Jaccard similarity",
               output_dir / "protein_neighbor_label_overlap.png")
    report["protein_nearest_neighbor_label_jaccard_mean"] = float(np.mean(overlaps))
    report["protein_random_pair_label_jaccard_mean"] = float(np.mean(random_overlaps))
    report["protein_annotation_count_median"] = float(np.median(annotation_count))
    report["protein_similarity_degree_median"] = float(np.median(similarity_degree))
    go_neighbors = NearestNeighbors(n_neighbors=2, metric="cosine").fit(go).kneighbors(
        go, return_distance=False)[:, 1]
    ontology_pairs = set(map(tuple, go_edges.T.tolist()))
    report["go_nearest_neighbor_direct_ontology_edge_fraction"] = float(np.mean([
        (i, int(j)) in ontology_pairs or (int(j), i) in ontology_pairs
        for i, j in enumerate(go_neighbors)]))
    random_go = rng.permutation(go_count)
    report["go_random_pair_direct_ontology_edge_fraction"] = float(np.mean([
        (i, int(j)) in ontology_pairs or (int(j), i) in ontology_pairs
        for i, j in enumerate(random_go)]))
    report["protein_biological_metadata"] = "unavailable in cached data"
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame({"protein_id": graph["protein"].node_id,
                  "annotations": annotation_count,
                  "similarity_degree": similarity_degree,
                  "sequence_length": sequence_length,
                  "nearest_neighbor_jaccard": overlaps}).to_csv(
                      output_dir / "protein_diagnostics.tsv", sep="\t", index=False)
    pd.DataFrame({"go_id": graph["go"].node_id,
                  "frequency": go_frequency,
                  "ontology_degree": ontology_degree,
                  "definition_length": definition_length,
                  "aspect": nodes.namespace.to_numpy()}).to_csv(
                      output_dir / "go_diagnostics.tsv", sep="\t", index=False)
    return report

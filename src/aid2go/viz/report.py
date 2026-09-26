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


def go_domain_structure(ontology):
    """Summarize the full OBO graph; obonet edges point from child to parent."""
    import networkx as nx

    rows, extremes, degrees = [], [], {}
    for short, namespace in (("BP", "biological_process"),
                             ("CC", "cellular_component"),
                             ("MF", "molecular_function")):
        ids = [term for term, attrs in ontology.nodes(data=True)
               if attrs.get("namespace") == namespace]
        graph = ontology.subgraph(ids)
        roots = [term for term in graph if graph.out_degree(term) == 0]
        depths = {}
        reverse = graph.reverse(copy=False)
        for root in roots:
            for term, depth in nx.single_source_shortest_path_length(reverse, root).items():
                depths[term] = min(depths.get(term, depth), depth)
        degree = {term: graph.degree(term) for term in graph}
        degrees[short] = {
            "in": np.fromiter((graph.in_degree(t) for t in graph), dtype=int),
            "out": np.fromiter((graph.out_degree(t) for t in graph), dtype=int),
        }
        branching = [graph.in_degree(t) for t in graph if graph.in_degree(t) > 0]
        rows.append({"domain": short, "nodes": len(ids), "edges": graph.number_of_edges(),
                     "avg_min_root_depth": float(np.mean(list(depths.values()))),
                     "unreachable_from_root": len(ids) - len(depths),
                     "avg_children_per_branching_term": float(np.mean(branching)) if branching else 0.0})
        for end, terms in (("highest", sorted(ids, key=lambda t: (-degree[t], t))[:10]),
                           ("lowest", sorted(ids, key=lambda t: (degree[t], t))[:10])):
            extremes.extend({"domain": short, "rank_group": end, "go_id": term,
                             "name": graph.nodes[term].get("name", ""), "degree": degree[term]}
                            for term in terms)
    return pd.DataFrame(rows), pd.DataFrame(extremes), degrees


def project_umap_tsne(matrix: np.ndarray, seed: int = 42):
    """Project the same finite vectors with fixed-seed UMAP and t-SNE."""
    from sklearn.manifold import TSNE
    import umap

    matrix = np.asarray(matrix, dtype=np.float32)
    if len(matrix) < 4 or not np.isfinite(matrix).all():
        raise ValueError("Projection needs at least four finite embedding vectors")
    compact = PCA(n_components=min(50, matrix.shape[1], len(matrix) - 1),
                  random_state=seed).fit_transform(matrix)
    return (umap.UMAP(n_components=2, random_state=seed, n_jobs=1,
                      n_neighbors=min(15, len(matrix) - 1)).fit_transform(compact),
            TSNE(n_components=2, random_state=seed, init="pca",
                 learning_rate="auto", perplexity=min(30, (len(matrix) - 1) / 3)
                 ).fit_transform(compact))


def go_embedding_cluster_scores(matrix: np.ndarray, domains: np.ndarray, seed: int = 42):
    """Compare raw-space k-means with known aspects; silhouette uses aspect labels."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_samples

    clusters = KMeans(n_clusters=3, random_state=seed, n_init=20).fit_predict(matrix)
    silhouette = silhouette_samples(matrix, domains)
    result = {"kmeans_ari": float(adjusted_rand_score(domains, clusters))}
    result.update({f"{domain}_true_domain_silhouette": float(silhouette[domains == domain].mean())
                   for domain in np.unique(domains)})
    return result


def knn_label_transfer_example(graph, train_count: int, min_nodes: int = 30,
                               max_nodes: int = 80, go_budget: int = 5):
    """Use the saved protein KNN edges to transfer donor GO labels, excluding self."""
    if go_budget < 1:
        raise ValueError("GO prediction budget must be positive")
    annotations = graph[ANNOTATION].edge_index.cpu().numpy()
    similarity = graph[SIMILARITY].edge_index.cpu().numpy()
    labels = [set() for _ in range(graph["protein"].num_nodes)]
    donors = [[] for _ in labels]
    for protein, go in annotations.T:
        labels[int(protein)].add(int(go))
    for source, target in similarity.T:
        if source < train_count and target < train_count and source != target:
            donors[int(source)].append(int(target))
    candidates = []
    for protein in range(train_count):
        neighbors = sorted(set(donors[protein]))
        if not labels[protein] or not neighbors:
            continue
        votes = {}
        for donor in neighbors:
            for term in labels[donor]:
                votes[term] = votes.get(term, 0) + 1
        predicted = set(sorted(votes, key=lambda term: (-votes[term], term))[:go_budget])
        go_context = set(labels[protein]) | predicted | set().union(*(labels[d] for d in neighbors))
        node_count = 1 + len(neighbors) + len(go_context)
        candidates.append((abs(node_count - 50), protein, node_count, neighbors,
                           predicted, go_context))
    if not candidates:
        raise ValueError("No annotated training protein has KNN donors")
    eligible = [row for row in candidates if min_nodes <= row[2] <= max_nodes]
    _, protein, node_count, neighbors, predicted, go_context = min(eligible or candidates)
    truth = labels[protein]
    overlap = len(truth & predicted)
    return {"protein": protein, "donors": neighbors, "truth": truth,
            "predicted": predicted, "go_context": go_context, "labels": labels,
            "nodes": node_count, "go_budget": go_budget,
            "precision": overlap / max(1, len(predicted)),
            "recall": overlap / len(truth),
            "jaccard": overlap / max(1, len(truth | predicted)),
            "within_requested_size": min_nodes <= node_count <= max_nodes}


def plot_knn_label_transfer(graph, example, path: Path):
    """Draw observed and transferred GO edges with identical node positions."""
    import networkx as nx

    focal, donors = example["protein"], example["donors"]
    proteins = [focal] + donors
    terms = sorted(example["go_context"])
    positions = {("p", p): (0, 1 - i / max(1, len(proteins) - 1))
                 for i, p in enumerate(proteins)}
    positions.update({("g", g): (1, 1 - i / max(1, len(terms) - 1))
                      for i, g in enumerate(terms)})
    fig, axes = plt.subplots(1, 2, figsize=(17, 11), sharex=True, sharey=True)
    for axis, predicted in zip(axes, (False, True)):
        network = nx.Graph()
        network.add_nodes_from(positions)
        for donor in donors:
            network.add_edges_from((("p", donor), ("g", term))
                                   for term in example["labels"][donor] if term in example["go_context"])
        selected = example["predicted"] if predicted else example["truth"]
        network.add_edges_from((("p", focal), ("g", term)) for term in selected)
        nx.draw_networkx_edges(network, positions, ax=axis, alpha=0.20, width=0.7)
        nx.draw_networkx_edges(network, positions, ax=axis,
                               edgelist=[(("p", focal), ("g", term)) for term in selected],
                               edge_color="#d35400" if predicted else "#166534", width=2)
        nx.draw_networkx_nodes(network, positions, nodelist=[("p", p) for p in proteins],
                               node_shape="s", node_color="#2769a3", node_size=80, ax=axis)
        nx.draw_networkx_nodes(network, positions, nodelist=[("g", g) for g in terms],
                               node_shape="o", node_color="#d99637", node_size=65, ax=axis)
        for protein in proteins:
            axis.text(-0.03, positions[("p", protein)][1],
                      graph["protein"].node_id[protein], ha="right", va="center",
                      fontsize=7, fontweight="bold" if protein == focal else "normal")
        for term in sorted(example["truth"] | example["predicted"]):
            axis.text(1.03, positions[("g", term)][1], graph["go"].node_id[term],
                      ha="left", va="center", fontsize=7)
        axis.set_title("KNN-transferred GO neighbors" if predicted else "Observed GO neighbors")
        axis.set_xlim(-0.15, 1.28)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig


def plot_hetero_bipartite(graph, example, path: Path):
    """Plot a small HeteroData annotation subgraph with shape, color and degree cues."""
    import networkx as nx

    proteins = [example["protein"]] + example["donors"][:7]
    terms = sorted(set().union(*(example["labels"][p] for p in proteins)))[:32]
    edges = [(p, g) for p in proteins for g in terms if g in example["labels"][p]]
    positions = {("p", p): (0, 1 - i / max(1, len(proteins) - 1))
                 for i, p in enumerate(proteins)}
    positions.update({("g", g): (1, 1 - i / max(1, len(terms) - 1))
                      for i, g in enumerate(terms)})
    network = nx.Graph()
    network.add_nodes_from(positions)
    network.add_edges_from((("p", p), ("g", g)) for p, g in edges)
    fig, axis = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(network, positions, ax=axis, alpha=0.28)
    for kind, nodes, color, shape in (
        ("protein", [("p", p) for p in proteins], "#2769a3", "s"),
        ("GO", [("g", g) for g in terms], "#d99637", "o")):
        nx.draw_networkx_nodes(network, positions, nodelist=nodes, node_shape=shape,
                               node_color=color, node_size=[90 + 55 * network.degree(n) for n in nodes],
                               label=kind, ax=axis)
    for protein in proteins:
        axis.text(-0.03, positions[("p", protein)][1], graph["protein"].node_id[protein],
                  ha="right", va="center", fontsize=8)
    for term in terms:
        axis.text(1.03, positions[("g", term)][1], graph["go"].node_id[term],
                  ha="left", va="center", fontsize=7)
    axis.legend(scatterpoints=1)
    axis.set_title("Protein-GO HeteroData subgraph (node size = local degree)")
    axis.set_xlim(-0.15, 1.30)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig, {"proteins": len(proteins), "go_terms": len(terms), "annotation_edges": len(edges)}

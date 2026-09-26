# %% [markdown]
# # GO, protein, and heterogeneous-graph exploratory analysis
# Run from the repository root: `MPLCONFIGDIR=/tmp/matplotlib-aid2go .venv/bin/python notebooks/eda_go_protein_graph.py`.

# %% [markdown]
# Reuse audit: the seven sections reuse package loaders and metrics; focused new helpers live in the existing viz and training modules.

# %%
"""Percent script for the repaired v2 dataset and fixed-seed protein sample."""

from pathlib import Path
import json
from io import BytesIO

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import obonet
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from aid2go.cli.progress import make_progress
from aid2go.core.config import AppConfig
from aid2go.core.seed import seed_everything
from aid2go.core.serialization import load_graph
from aid2go.data.preprocess_goa import calculate_ia, clean_ontology_edges
from aid2go.data.sample import ANNOTATION
from aid2go.training.metrics import EpochDiagnostics
from aid2go.training.mlflow_tracker import MLFlowTracker
from aid2go.training.trainer import Trainer
from aid2go.viz.report import (go_domain_structure, project_umap_tsne,
                               go_embedding_cluster_scores,
                               knn_label_transfer_example,
                               plot_knn_label_transfer, plot_hetero_bipartite)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/v2_unique_pairs"
SAMPLE = ROOT / "data/processed/v2_unique_pairs_sample_1000x100_seed42"
OUT = ROOT / "outputs/eda"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def show_inline(fig):
    """Display a saved figure in Jupyter while remaining runnable headlessly."""
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
    except ImportError:
        plt.close(fig)
        return
    if get_ipython() is not None:
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        display(Image(data=buffer.getvalue()))
    plt.close(fig)


seed_everything(SEED)
source_graph = load_graph(SOURCE / "hetero_data.pt")
sample_graph = load_graph(SAMPLE / "hetero_data.pt")
manifest = json.loads((SAMPLE / "split_manifest.json").read_text())
go_nodes = pd.read_csv(SOURCE / "go_nodes.csv").set_index("term")
sample_nodes = pd.read_csv(SAMPLE / "go_nodes.csv").set_index("term")
ontology = clean_ontology_edges(obonet.read_obo(str(SOURCE / "go-basic.obo"),
                                               ignore_obsolete=True))
audit = [
    (1, "OBO reading exists in preprocess_go; go_domain_structure added to viz.report for full-domain depth and degree summaries."),
    (2, "GO vectors are already in HeteroData; viz.report adds paired UMAP/t-SNE and raw-space clustering metrics."),
    (3, "Protein vectors and residue lengths are already in HeteroData/HDF5; paired projections and length views use viz.report projection helper."),
    (4, "calculate_ia and cached IA exist in preprocess_goa/sample; only the jointplot and Spearman analysis are new."),
    (5, "The sampled FAISS KNN edge index exists; viz.report adds leave-one-out donor-label transfer and a local comparison plot."),
    (6, "HeteroData and load_graph exist; viz.report adds a small shape-and-color bipartite plot."),
    (7, "Trainer, CAFA-style Fmax/Smin routines, and MLFlowTracker exist; training.metrics adds optional epoch diagnostics; Trainer calls the hook."),
]
print("REUSE AUDIT")
for section, detail in audit:
    print(f"{section}. {detail}")

# %% [markdown]
# 1. Inspect the full GO ontology by domain, including directed degrees, depths, branching, and extreme terms.

# %%
structure, extreme_terms, degree_arrays = go_domain_structure(ontology)
structure.to_csv(OUT / "go_domain_structure.tsv", sep="\t", index=False)
extreme_terms.to_csv(OUT / "go_degree_extremes.tsv", sep="\t", index=False)
print("\nGO domain structure (minimum root distance; branching among terms with children):")
print(structure.to_string(index=False))
print("\nTop and bottom 10 terms by total degree per domain:")
print(extreme_terms.to_string(index=False))
fig, axes = plt.subplots(3, 2, figsize=(11, 11))
for row, domain in enumerate(("BP", "CC", "MF")):
    for column, direction in enumerate(("in", "out")):
        values = degree_arrays[domain][direction]
        axes[row, column].hist(values, bins=np.arange(values.max() + 2) - 0.5,
                               color="#35699b")
        axes[row, column].set_yscale("log")
        axes[row, column].set(xlabel=f"{direction}-degree", ylabel="Terms (log count)",
                              title=f"{domain}: {direction}-degree")
fig.tight_layout()
fig.savefig(OUT / "go_domain_in_out_degree.png", dpi=160)
show_inline(fig)

# %% [markdown]
# 2. Check whether raw GO definition embeddings separate the BP, CC, and MF domains.

# %%
go_ids = list(source_graph["go"].node_id)
go_vectors = source_graph["go"].x[:, :-1].cpu().numpy().astype(np.float32)
go_domain = go_nodes.loc[go_ids, "namespace"].to_numpy()
go_umap, go_tsne = project_umap_tsne(go_vectors, SEED)
cluster_scores = go_embedding_cluster_scores(go_vectors, go_domain, SEED)
print("\nGO embedding clustering on raw, non-projected vectors:")
for key, value in cluster_scores.items():
    print(f"{key}: {value:.4f}")
(OUT / "go_embedding_clustering.json").write_text(json.dumps(cluster_scores, indent=2) + "\n")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for axis, coordinates, title in zip(axes, (go_umap, go_tsne), ("UMAP", "t-SNE")):
    for domain, color in (("biological_process", "#2769a3"),
                          ("cellular_component", "#d99637"),
                          ("molecular_function", "#8d4d9f")):
        selected = go_domain == domain
        axis.scatter(coordinates[selected, 0], coordinates[selected, 1],
                     s=10, alpha=0.7, color=color, label=domain)
    axis.set(title=f"GO definition embeddings: {title}", xlabel="Axis 1", ylabel="Axis 2")
axes[1].legend(markerscale=2, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "go_embeddings_umap_tsne_domains.png", dpi=160)
show_inline(fig)

# %% [markdown]
# 3. Compare sequence length and annotation density with the sampled protein embedding geometry.

# %%
protein_ids = list(sample_graph["protein"].node_id)
protein_vectors = sample_graph["protein"].x.cpu().numpy().astype(np.float32)
with h5py.File(SOURCE / "protein_data.h5", "r") as archive:
    lengths = np.asarray([archive["residue_embeddings"][protein].shape[0]
                          for protein in protein_ids])
edge_index = sample_graph[ANNOTATION].edge_index.cpu()
annotation_count = torch.bincount(edge_index[0], minlength=len(protein_ids)).numpy()
print("\nSampled protein lengths and GO annotation counts:")
print(pd.DataFrame({"sequence_length": lengths,
                    "go_annotations": annotation_count}).describe().to_string())
protein_pc1 = PCA(n_components=1, random_state=SEED).fit_transform(protein_vectors)[:, 0]
print("Spearman length vs protein PC1:", spearmanr(lengths, protein_pc1).statistic)
print("Spearman length vs embedding norm:",
      spearmanr(lengths, np.linalg.norm(protein_vectors, axis=1)).statistic)
pd.DataFrame({"protein_id": protein_ids, "sequence_length": lengths,
              "go_annotations": annotation_count}).to_csv(
                  OUT / "protein_length_annotation_counts.tsv", sep="\t", index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(lengths, bins=40, color="#35699b")
axes[0].set(xlabel="Residues", ylabel="Proteins", title="Sequence lengths")
axes[1].boxplot(lengths, orientation="vertical")
axes[1].set(ylabel="Residues", title="Sequence-length spread")
fig.tight_layout()
fig.savefig(OUT / "protein_sequence_length_hist_box.png", dpi=160)
show_inline(fig)
protein_umap, protein_tsne = project_umap_tsne(protein_vectors, SEED)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for row, (values, label) in enumerate(((annotation_count, "GO annotations"),
                                        (lengths, "Sequence length"))):
    for column, (coordinates, method) in enumerate(((protein_umap, "UMAP"),
                                                     (protein_tsne, "t-SNE"))):
        points = axes[row, column].scatter(coordinates[:, 0], coordinates[:, 1],
                                          c=values, cmap="viridis", s=11, alpha=0.75)
        axes[row, column].set(title=f"Protein {method}: {label}",
                              xlabel="Axis 1", ylabel="Axis 2")
        fig.colorbar(points, ax=axes[row, column], label=label)
fig.tight_layout()
fig.savefig(OUT / "protein_embeddings_umap_tsne_density_length.png", dpi=160)
show_inline(fig)

# %% [markdown]
# 4. Recompute final-cohort IA, compare it with the cached values, and relate IA to ontology degree.

# %%
with make_progress() as progress:
    current_ia = calculate_ia(SOURCE / "goa_filtered.tsv", ontology, set(go_ids), progress)
cached_ia = source_graph["go"].ia.squeeze(-1).cpu().numpy()
ia_values = np.asarray([current_ia[term] for term in go_ids])
print("\nIA cache maximum absolute difference:", float(np.max(np.abs(ia_values - cached_ia))))
sample_ia = sample_graph["go"].ia.squeeze(-1).cpu().numpy()
print("Sample training-only IA: terms =", len(sample_ia),
      "median =", float(np.median(sample_ia)), "max =", float(np.max(sample_ia)))
degree = go_nodes.loc[go_ids, "degree"].to_numpy()
ia_frame = pd.DataFrame({"go_id": go_ids, "domain": go_domain,
                         "degree": degree, "ia": ia_values,
                         "degree_plus_one": degree + 1,
                         "ia_plus_0_01": ia_values + 0.01})
ia_frame.to_csv(OUT / "go_ia_degree.tsv", sep="\t", index=False)
rho, p_value = spearmanr(degree, ia_values)
print(f"IA versus degree Spearman rho = {rho:.4f}, p = {p_value:.3g}")
for domain in np.unique(go_domain):
    part = ia_frame[ia_frame.domain == domain]
    result = spearmanr(part.degree, part.ia)
    print(f"  {domain}: rho = {result.statistic:.4f}, p = {result.pvalue:.3g}")
joint = sns.jointplot(data=ia_frame, x="degree_plus_one", y="ia_plus_0_01",
                      hue="domain", kind="scatter", height=7,
                      joint_kws={"s": 15, "alpha": 0.55})
joint.ax_joint.set_xscale("log")
joint.ax_joint.set_yscale("log")
joint.ax_joint.set(xlabel="Ontology degree + 1 (log)", ylabel="IA + 0.01 (log)")
joint.ax_joint.text(0.03, 0.97, f"Spearman rho = {rho:.3f}",
                    transform=joint.ax_joint.transAxes, va="top",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
joint.figure.suptitle("Information accretion versus GO ontology degree", y=1.02)
joint.figure.savefig(OUT / "go_ia_vs_degree_jointplot.png", dpi=160, bbox_inches="tight")
show_inline(joint.figure)

# %% [markdown]
# 5. Compare observed GO neighbors with leave-one-out GO labels transferred from saved protein KNN neighbors.

# %%
example = knn_label_transfer_example(sample_graph, manifest["train_proteins"])
focal_id = sample_graph["protein"].node_id[example["protein"]]
knn_summary = {"protein_id": focal_id, "local_nodes": example["nodes"],
               "within_30_to_80_nodes": example["within_requested_size"],
               "go_prediction_budget": example["go_budget"],
               "knn_donors": len(example["donors"]),
               "donor_protein_ids": [sample_graph["protein"].node_id[i]
                                     for i in example["donors"]],
               "true_go_neighbors": len(example["truth"]),
               "true_go_ids": [sample_graph["go"].node_id[i]
                               for i in sorted(example["truth"])],
               "predicted_go_neighbors": len(example["predicted"]),
               "predicted_go_ids": [sample_graph["go"].node_id[i]
                                    for i in sorted(example["predicted"])],
               "precision": example["precision"], "recall": example["recall"],
               "jaccard": example["jaccard"]}
print("\nKNN label-transfer example:", json.dumps(knn_summary, indent=2))
(OUT / "knn_label_transfer_example.json").write_text(json.dumps(knn_summary, indent=2) + "\n")
fig = plot_knn_label_transfer(sample_graph, example, OUT / "knn_true_vs_transferred_go_edges.png")
show_inline(fig)

# %% [markdown]
# 6. Visualize a small HeteroData annotation subgraph with distinct protein and GO shapes, colors, and degree sizes.

# %%
fig, subgraph_summary = plot_hetero_bipartite(
    sample_graph, example, OUT / "heterodata_protein_go_bipartite_subgraph.png")
print("\nBipartite HeteroData subgraph:", subgraph_summary)
show_inline(fig)

# %% [markdown]
# 7. Reuse the existing Trainer and MLFlowTracker for optional per-epoch domain metrics and learned-embedding snapshots.

# %%
config = AppConfig.from_yaml(ROOT / "configs/default.yaml")
config.data.dataset_version = SAMPLE.name
config.training.split_strategy = "protein"
config.training.loss = "bce"
config.training.epochs = 5
config.training.seed = SEED
config.training.device = "cpu"
config.tracking.enabled = True
config.tracking.experiment_name = "aid2go_eda_protein_holdout_diagnostics"
config.tracking.run_name = "baseline_seed42"
config.tracking.mlflow_uri = "sqlite:///" + str((OUT / "mlflow.db").resolve())
diagnostic_model_dir = OUT / "diagnostic_model"
diagnostic_model_dir.mkdir(parents=True, exist_ok=True)
config.save_yaml(diagnostic_model_dir / "resolved_config.yaml")
train_data = load_graph(SAMPLE / "train_data.pt")
val_data = load_graph(SAMPLE / "val_data.pt")
diagnostics = EpochDiagnostics(train_data, val_data,
                               sample_nodes.namespace.to_dict(), OUT,
                               manifest["train_proteins"], manifest["val_proteins"],
                               umap_every=2, seed=SEED)
assert np.allclose(diagnostics.ia, sample_ia), "Training diagnostics must use section 4 IA"
tracker = MLFlowTracker(config.tracking.experiment_name, config.tracking.mlflow_uri,
                        config.tracking.run_name,
                        artifact_location=(OUT / "mlflow_artifacts").resolve().as_uri())
tracker.log_params({"dataset_version": SAMPLE.name, "model": config.model.name,
                    "seed": SEED, "loss": "bce", "split": "protein"})
seed_everything(SEED)
try:
    trainer = Trainer(config, train_data, tracker, diagnostic_model_dir,
                      diagnostics=diagnostics)
    trainer.fit(train_data, val_data)
finally:
    tracker.end_run()
history = pd.read_csv(diagnostic_model_dir / "training_history.csv")
print("\nPer-epoch training diagnostics (Smin is min-threshold semantic distance):")
print(history.to_string(index=False))
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for prefix in ("bp", "cc", "mf"):
    axes[0].plot(history.epoch, history[f"train_{prefix}_bce"], marker="o",
                 label=f"{prefix.upper()} train")
    axes[0].plot(history.epoch, history[f"val_{prefix}_bce"], linestyle="--",
                 label=f"{prefix.upper()} validation")
    axes[1].plot(history.epoch, history[f"val_{prefix}_fmax"], marker="o",
                 label=prefix.upper())
    axes[2].plot(history.epoch, history[f"val_{prefix}_smin"], marker="o",
                 label=prefix.upper())
axes[0].set(xlabel="Epoch", ylabel="BCE", title="Per-domain loss")
axes[1].set(xlabel="Epoch", ylabel="Fmax", title="Validation Fmax (higher is better)")
axes[2].set(xlabel="Epoch", ylabel="Smin", title="Validation Smin (lower is better)")
for axis in axes:
    axis.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "training_per_domain_diagnostics.png", dpi=160)
show_inline(fig)

# %% [markdown]
# Summary: report the reusable package paths and the focused additions used by this analysis.

# %%
print("\nREUSE / ADDITIONS SUMMARY")
print("Reused: OBO/IA preprocessing, cached GO and protein vectors, HeteroData loader, "
      "FAISS KNN edges, CAFA-style metrics, Trainer, and MLFlowTracker.")
print("Added: domain/projection/cluster/local-network helpers in aid2go.viz.report; "
      "opt-in EpochDiagnostics in aid2go.training.metrics; Trainer hook and model encode accessors.")
print("Figures and tables:", OUT)

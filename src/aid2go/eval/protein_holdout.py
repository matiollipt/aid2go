"""Internal whole-protein holdout evaluation on a fixed GO target universe."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve

from aid2go.data.sample import ANNOTATION, SIMILARITY
from aid2go.eval.cafa_evaluator import CAFAEvaluator
from aid2go.eval.cafa_metrics import calculate_aupr


@torch.no_grad()
def score_matrix(model, data, protein_start: int, protein_stop: int,
                 device: torch.device) -> np.ndarray:
    model.eval()
    graph = data.to(device)
    go_count = graph["go"].num_nodes
    protein = torch.arange(protein_start, protein_stop, device=device).repeat_interleave(go_count)
    go = torch.arange(go_count, device=device).repeat(protein_stop - protein_start)
    indices = torch.stack((protein, go))
    logits = model(graph.x_dict, graph.edge_index_dict, indices,
                   go_node_ids=graph["go"].get("node_id", None))
    return torch.sigmoid(logits).reshape(protein_stop - protein_start, go_count).cpu().numpy()


def truth_matrix(data, protein_start: int, protein_stop: int) -> np.ndarray:
    truth = np.zeros((protein_stop - protein_start, data["go"].num_nodes), dtype=np.int8)
    indices = data[ANNOTATION].edge_label_index.cpu().numpy()
    labels = data[ANNOTATION].edge_label.cpu().numpy()
    for protein, go in indices[:, labels == 1].T:
        if protein_start <= protein < protein_stop:
            truth[protein - protein_start, go] = 1
    return truth


def _safe_metrics(evaluator: CAFAEvaluator, truth: np.ndarray,
                  scores: np.ndarray, go_ids: list[str]) -> dict:
    if not truth.size or truth.sum() == 0:
        return {"status": "no observed positives", "proteins": int(truth.shape[0]),
                "go_terms": int(truth.shape[1])}
    metrics = evaluator.evaluate(truth, scores, go_terms=go_ids)
    macro = [calculate_aupr(truth[:, j], scores[:, j])
             for j in range(truth.shape[1]) if 0 < truth[:, j].sum() < truth.shape[0]]
    metrics["macro_aupr"] = float(np.mean(macro)) if macro else None
    metrics["macro_aupr_terms"] = len(macro)
    metrics["proteins"] = int(truth.shape[0])
    metrics["go_terms"] = int(truth.shape[1])
    metrics["observed_positive_pairs"] = int(truth.sum())
    return metrics


def _banded(evaluator, truth, scores, go_ids, values, bands, axis=0):
    out = {}
    for name, lower, upper in bands:
        mask = (values >= lower) & (values <= upper)
        if axis == 0:
            out[name] = _safe_metrics(evaluator, truth[mask], scores[mask], go_ids)
        else:
            indices = np.where(mask)[0]
            out[name] = _safe_metrics(evaluator, truth[:, indices],
                                      scores[:, indices], [go_ids[i] for i in indices])
    return out


def _save_figures(truth, scores, go_ids, aspects, output_dir, history_path=None):
    fig, axis = plt.subplots(figsize=(6, 4))
    for name, indices in [("Overall", list(range(len(go_ids))))] + [
        (aspect, [i for i, go_id in enumerate(go_ids) if aspects.get(go_id) == aspect])
        for aspect in ("biological_process", "cellular_component", "molecular_function")
    ]:
        if not indices or not truth[:, indices].any():
            continue
        precision, recall, _ = precision_recall_curve(
            truth[:, indices].ravel(), scores[:, indices].ravel())
        axis.plot(recall, precision, label=name)
    axis.set(xlabel="Recall", ylabel="Precision", title="Internal protein-holdout precision–recall",
             xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall.png", dpi=150)
    plt.close(fig)
    if history_path is None:
        return
    history = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.epoch, history.train_loss, marker="o")
    axes[0].set(xlabel="Epoch", ylabel="BCE", title="Training loss")
    for metric, label in (("val_aupr", "Validation AUPR"),
                          ("val_fmax", "Validation Fmax")):
        axes[1].plot(history.epoch, history[metric], marker="o", label=label)
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation metrics")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def _loss_contributions(model, train_data, go_ids, go_nodes, term_frequency, device):
    graph = train_data.to(device)
    pairs = graph[ANNOTATION].edge_label_index
    target = graph[ANNOTATION].edge_label
    logits = model(graph.x_dict, graph.edge_index_dict, pairs,
                   go_node_ids=graph["go"].get("node_id", None))
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none").cpu().numpy()
    go_indices = pairs[1].cpu().numpy()
    aspects = dict(zip(go_nodes.term, go_nodes.namespace))
    labels = {
        "aspect": np.asarray([aspects.get(go_ids[index], "unknown") for index in go_indices]),
        "go_frequency": np.asarray([
            "1-9" if term_frequency[index] < 10 else
            "10-99" if term_frequency[index] < 100 else "100+"
            for index in go_indices]),
        "observed_label": target.cpu().numpy().astype(int).astype(str),
    }
    total = max(float(losses.sum()), 1e-12)
    return {dimension: {
        str(group): {"pairs": int(mask.sum()), "mean_bce": float(losses[mask].mean()),
                     "loss_share": float(losses[mask].sum() / total)}
        for group in np.unique(values)
        for mask in [values == group]
    } for dimension, values in labels.items()}


def evaluate_holdout(model, val_data, test_data, full_graph, go_nodes: pd.DataFrame,
                     train_count: int, val_count: int, device: torch.device,
                     output_dir: Path, train_data=None, history_path=None) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protein_count = full_graph["protein"].num_nodes
    go_ids = list(full_graph["go"].node_id)
    val_scores = score_matrix(model, val_data, train_count, train_count + val_count, device)
    test_scores = score_matrix(model, test_data, train_count + val_count, protein_count, device)
    val_truth = truth_matrix(val_data, train_count, train_count + val_count)
    test_truth = truth_matrix(test_data, train_count + val_count, protein_count)
    thresholds = np.linspace(0.01, 0.99, 99)
    def flat_f1(threshold):
        predicted = val_scores >= threshold
        tp = np.logical_and(predicted, val_truth == 1).sum()
        fp = np.logical_and(predicted, val_truth == 0).sum()
        fn = np.logical_and(~predicted, val_truth == 1).sum()
        return 2 * tp / max(1, 2 * tp + fp + fn)
    threshold = float(max(thresholds, key=flat_f1))
    ia = full_graph["go"].ia.squeeze(-1).cpu().numpy()
    aspects = dict(zip(go_nodes.term, go_nodes.namespace))
    _save_figures(test_truth, test_scores, go_ids, aspects, output_dir, history_path)
    evaluator = CAFAEvaluator(dict(zip(go_ids, ia)), aspects)
    results = {"protocol": "internal protein-held-out, observed annotations as targets; not CAFA",
               "validation_selected_threshold": threshold,
               "overall": _safe_metrics(evaluator, test_truth, test_scores, go_ids)}

    selected_protein_indices = np.arange(train_count + val_count, protein_count)
    annotation_degree = np.bincount(full_graph[ANNOTATION].edge_index[0].cpu().numpy(),
                                    minlength=protein_count)[selected_protein_indices]
    # Outgoing KNN degree is fixed by k; incoming degree measures how often a
    # protein is selected as another protein's neighbor.
    similarity_degree = np.bincount(full_graph[SIMILARITY].edge_index[1].cpu().numpy(),
                                    minlength=protein_count)[selected_protein_indices]
    training_edges = full_graph[ANNOTATION].edge_index
    training_edges = training_edges[:, training_edges[0] < train_count]
    term_frequency = np.bincount(training_edges[1].cpu().numpy(), minlength=len(go_ids))
    if train_data is not None:
        results["loss_contributions"] = _loss_contributions(
            model, train_data, go_ids, go_nodes, term_frequency, device)
    results["annotation_count_bands"] = _banded(
        evaluator, test_truth, test_scores, go_ids, annotation_degree,
        [("0-1", 0, 1), ("2-4", 2, 4), ("5+", 5, np.inf)])
    results["incoming_similarity_degree_bands"] = _banded(
        evaluator, test_truth, test_scores, go_ids, similarity_degree,
        [("0-4", 0, 4), ("5-9", 5, 9), ("10+", 10, np.inf)])
    results["go_training_frequency_bands"] = _banded(
        evaluator, test_truth, test_scores, go_ids, term_frequency,
        [("1-9", 1, 9), ("10-99", 10, 99), ("100+", 100, np.inf)], axis=1)

    np.savez_compressed(output_dir / "heldout_scores.npz", scores=test_scores,
                        truth=test_truth, val_scores=val_scores, val_truth=val_truth,
                        go_ids=np.asarray(go_ids),
                        protein_ids=np.asarray(full_graph["protein"].node_id[train_count + val_count:]))
    pd.DataFrame({"protein_id": full_graph["protein"].node_id[train_count + val_count:],
                  "observed_annotations": annotation_degree,
                  "similarity_degree": similarity_degree}).to_csv(
                      output_dir / "test_proteins.tsv", sep="\t", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2,
                                                        allow_nan=False) + "\n")
    lines = ["# Internal protein-held-out evaluation", "",
             "Observed annotations are incomplete; absent links are evaluation negatives, not proven nonfunctions.",
             "GO terms and IA were selected from training proteins only. This is not a CAFA benchmark.", "",
             "| Metric | Value |", "|---|---:|"]
    for name, value in results["overall"].items():
        if isinstance(value, (float, int)):
            lines.append(f"| {name} | {value:.4f} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    return results

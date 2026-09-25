"""Save per-prediction sensitivity and attention diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from aid2go.data.sample import ANNOTATION, REVERSE


def _pair_score(model, data, protein: int, go: int, edge_index_dict=None) -> float:
    with torch.no_grad():
        pair = torch.tensor([[protein], [go]], dtype=torch.long,
                            device=data["protein"].x.device)
        score = model(data.x_dict, edge_index_dict or data.edge_index_dict, pair,
                      go_node_ids=data["go"].get("node_id", None))
        return float(torch.sigmoid(score)[0].cpu())


def _feature_attribution(model, data, protein: int, go: int) -> dict:
    features = {name: value.detach().clone().requires_grad_(True)
                for name, value in data.x_dict.items()}
    pair = torch.tensor([[protein], [go]], dtype=torch.long,
                        device=data["protein"].x.device)
    model.zero_grad(set_to_none=True)
    logit = model(features, data.edge_index_dict, pair,
                  go_node_ids=data["go"].get("node_id", None))[0]
    gradients = torch.autograd.grad(logit, (features["protein"], features["go"]))
    output = {}
    for name, index, gradient in (("protein", protein, gradients[0]),
                                  ("go", go, gradients[1])):
        impact = (gradient[index] * features[name][index]).abs().detach().cpu().numpy()
        highest = np.argsort(impact)[::-1][:10]
        output[name] = [{"feature": int(i), "magnitude": float(impact[i])}
                        for i in highest]
    return output


def _neighbor_diagnostics(model, data, protein: int, go: int, score: float) -> list[dict]:
    attention = model.attention_by_relation() if hasattr(model, "attention_by_relation") else []
    last_attention = attention[-1] if attention else {}
    neighbors = []
    for relation, edges in data.edge_index_dict.items():
        target_index = protein if relation[2] == "protein" else go
        positions = torch.where(edges[1] == target_index)[0]
        if len(positions) == 0:
            continue
        alpha_by_pair = {}
        weights = last_attention.get(relation)
        if weights is not None:
            selected, alpha = weights
            for source, target, coeff in zip(selected[0].tolist(), selected[1].tolist(),
                                             alpha.mean(dim=1).tolist()):
                alpha_by_pair[(source, target)] = coeff
        candidates = sorted(positions.tolist(), key=lambda i: (
            -alpha_by_pair.get(tuple(edges[:, i].tolist()), -1),
            int(edges[0, i])))[:5]
        for position in candidates:
            source, target = map(int, edges[:, position].tolist())
            altered = {kind: value for kind, value in data.edge_index_dict.items()}
            mask = torch.ones(edges.shape[1], dtype=torch.bool, device=edges.device)
            mask[position] = False
            altered[relation] = edges[:, mask]
            if relation == ANNOTATION:
                reverse = data[REVERSE].edge_index
                altered[REVERSE] = reverse[:, ~((reverse[0] == target) & (reverse[1] == source))]
            elif relation == REVERSE:
                forward = data[ANNOTATION].edge_index
                altered[ANNOTATION] = forward[:, ~((forward[0] == target) & (forward[1] == source))]
            changed = _pair_score(model, data, protein, go, altered)
            source_ids = data[relation[0]].node_id
            target_ids = data[relation[2]].node_id
            neighbors.append({"relation": relation[1], "source_id": source_ids[source],
                              "target_id": target_ids[target],
                              "attention_mean_last_layer": alpha_by_pair.get((source, target)),
                              "score_change_on_removal": changed - score})
    return neighbors


def _figure(feature, neighbors, title: str, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    protein = feature["protein"][:6]
    go = feature["go"][:6]
    top = [(f"P:{item['feature']}", item["magnitude"]) for item in protein]
    top += [(f"GO:{item['feature']}", item["magnitude"]) for item in go]
    axes[0].barh([label for label, _ in top][::-1], [value for _, value in top][::-1])
    axes[0].set_title("Gradient × input magnitude")
    ranked = sorted(neighbors, key=lambda item: abs(item["score_change_on_removal"]),
                    reverse=True)[:10]
    axes[1].barh([f"{item['relation']}:{item['source_id']}" for item in ranked][::-1],
                 [item["score_change_on_removal"] for item in ranked][::-1])
    axes[1].set_title("Score change after edge removal")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_prediction_explanations(model, data, scores: np.ndarray, truth: np.ndarray,
                                 threshold: float, protein_start: int,
                                 output_dir: Path) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predicted = scores >= threshold
    candidates = {
        "true_positive": np.argwhere((truth == 1) & predicted),
        "false_positive": np.argwhere((truth == 0) & predicted),
        "false_negative": np.argwhere((truth == 1) & ~predicted),
    }
    outputs = []
    model.eval()
    for category, locations in candidates.items():
        if not len(locations):
            outputs.append({"category": category, "status": "no qualifying prediction"})
            continue
        values = scores[locations[:, 0], locations[:, 1]]
        selected = int(np.argmin(values) if category == "false_negative" else np.argmax(values))
        relative_protein, go = map(int, locations[selected])
        protein = protein_start + relative_protein
        score = _pair_score(model, data, protein, go)
        feature = _feature_attribution(model, data, protein, go)
        neighbors = _neighbor_diagnostics(model, data, protein, go, score)
        record = {"category": category, "protein_id": data["protein"].node_id[protein],
                  "go_id": data["go"].node_id[go], "protein_index": protein,
                  "go_index": go, "observed_label": int(truth[relative_protein, go]),
                  "predicted_score": score, "threshold": threshold,
                  "feature_attribution": feature, "relation_neighbors": neighbors,
                  "interpretation": "Attention and perturbation are model diagnostics, not causal biological evidence."}
        (output_dir / f"{category}.json").write_text(json.dumps(record, indent=2) + "\n")
        _figure(feature, neighbors, f"{category}: {record['protein_id']} → {record['go_id']}",
                output_dir / f"{category}.png")
        outputs.append(record)
    (output_dir / "index.json").write_text(json.dumps(outputs, indent=2) + "\n")
    return outputs


def save_link_explanation(model, data, protein: int, go: int,
                          output_dir: Path) -> dict:
    """Save a requested pair's attribution and edge-removal diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    score = _pair_score(model, data, protein, go)
    feature = _feature_attribution(model, data, protein, go)
    neighbors = _neighbor_diagnostics(model, data, protein, go, score)
    record = {"protein_id": data["protein"].node_id[protein],
              "go_id": data["go"].node_id[go], "protein_index": protein,
              "go_index": go, "predicted_score": score,
              "feature_attribution": feature, "relation_neighbors": neighbors,
              "interpretation": "Attention and perturbation are model diagnostics, not causal biological evidence."}
    (output_dir / "explanation.json").write_text(json.dumps(record, indent=2) + "\n")
    _figure(feature, neighbors, f"{record['protein_id']} → {record['go_id']}",
            output_dir / "explanation.png")
    return record

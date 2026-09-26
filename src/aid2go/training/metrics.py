"""Optional per-epoch diagnostics for the existing Trainer and tracker."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from aid2go.eval.cafa_metrics import calculate_fmax, calculate_smax


class EpochDiagnostics:
    """Log domain losses, held-out metrics, imbalance, and learned embeddings."""

    def __init__(self, train_data, val_data, go_aspects: dict[str, str],
                 output_dir: Path, train_count: int, val_count: int,
                 umap_every: int = 2, seed: int = 42):
        self.train_data = train_data
        self.val_data = val_data
        self.go_ids = list(train_data["go"].node_id)
        self.go_aspects = go_aspects
        self.ia = train_data["go"].ia.squeeze(-1).cpu().numpy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_count = train_count
        self.val_count = val_count
        self.umap_every = umap_every
        self.seed = seed

    def start(self, tracker):
        relation = ("protein", "has_function", "go")
        message = self.train_data[relation].edge_index.cpu()
        supervised = self.train_data[relation].edge_label_index.cpu()
        positive = self.train_data[relation].edge_label.cpu() == 1
        all_terms = torch.cat((message[1], supervised[1, positive]))
        counts = torch.bincount(all_terms, minlength=len(self.go_ids)).numpy()
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.hist(counts, bins=min(30, max(5, len(counts) // 3)), color="#35699b")
        axis.set(xlabel="Training proteins per GO term", ylabel="GO terms",
                 title="Training label imbalance")
        fig.tight_layout()
        path = self.output_dir / "training_proteins_per_go_term.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        tracker.log_artifact(str(path), "diagnostics")
        pd.DataFrame({"go_id": self.go_ids, "training_proteins": counts}).to_csv(
            self.output_dir / "training_go_frequency.tsv", sep="\t", index=False)

    @torch.no_grad()
    def _scores(self, model, data, device):
        graph = data.to(device)
        relation = ("protein", "has_function", "go")
        pairs = graph[relation].edge_label_index
        truth = graph[relation].edge_label
        logits = model(graph.x_dict, graph.edge_index_dict, pairs,
                       go_node_ids=graph["go"].get("node_id", None))
        return (logits.float().cpu().numpy(), truth.cpu().numpy(),
                pairs[1].cpu().numpy())

    @torch.no_grad()
    def log_epoch(self, epoch: int, model, device, tracker, total_epochs: int) -> dict:
        model.eval()
        train_logits, train_truth, train_terms = self._scores(model, self.train_data, device)
        val_logits, val_truth, val_terms = self._scores(model, self.val_data, device)
        metrics = {
            "train_bce_post_epoch": float(F.binary_cross_entropy_with_logits(
                torch.from_numpy(train_logits), torch.from_numpy(train_truth)).item()),
            "val_bce": float(F.binary_cross_entropy_with_logits(
                torch.from_numpy(val_logits), torch.from_numpy(val_truth)).item()),
        }
        if len(val_truth) != self.val_count * len(self.go_ids):
            raise ValueError("Per-domain diagnostics require all validation protein-GO pairs")
        val_matrix = val_truth.reshape(self.val_count, len(self.go_ids))
        score_matrix = torch.sigmoid(torch.from_numpy(val_logits)).numpy().reshape(val_matrix.shape)
        for code, aspect in (("bp", "biological_process"),
                             ("cc", "cellular_component"),
                             ("mf", "molecular_function")):
            indices = np.asarray([i for i, go_id in enumerate(self.go_ids)
                                  if self.go_aspects.get(go_id) == aspect], dtype=int)
            if not len(indices):
                continue
            for split, logits, truth, terms in (
                ("train", train_logits, train_truth, train_terms),
                ("val", val_logits, val_truth, val_terms)):
                mask = np.isin(terms, indices)
                if mask.any():
                    metrics[f"{split}_{code}_bce"] = float(F.binary_cross_entropy_with_logits(
                        torch.from_numpy(logits[mask]), torch.from_numpy(truth[mask])).item())
            domain_truth = val_matrix[:, indices]
            if domain_truth.sum() > 0:
                metrics[f"val_{code}_fmax"] = calculate_fmax(domain_truth, score_matrix[:, indices])
                # calculate_smax implements min_threshold sqrt(RU^2 + MI^2), i.e. Smin.
                metrics[f"val_{code}_smin"] = calculate_smax(
                    domain_truth, score_matrix[:, indices], self.ia[indices])
        if self.umap_every > 0 and (epoch % self.umap_every == 0 or epoch == total_epochs):
            self._embedding_snapshot(epoch, model, device, tracker)
        return metrics

    @torch.no_grad()
    def _embedding_snapshot(self, epoch, model, device, tracker):
        if not hasattr(model, "encode"):
            return
        import umap

        graph = self.train_data.to(device)
        learned = model.encode(graph.x_dict, graph.edge_index_dict)["protein"].cpu().numpy()
        coordinates = umap.UMAP(random_state=self.seed, n_jobs=1,
                                n_neighbors=min(15, len(learned) - 1)).fit_transform(learned)
        fig, axis = plt.subplots(figsize=(6, 5))
        for label, start, stop, color in (
            ("train", 0, self.train_count, "#2769a3"),
            ("validation", self.train_count, self.train_count + self.val_count, "#d99637"),
            ("test", self.train_count + self.val_count, len(learned), "#8d4d9f")):
            axis.scatter(coordinates[start:stop, 0], coordinates[start:stop, 1],
                         s=8, alpha=0.65, color=color, label=label)
        axis.set(title=f"Learned protein embeddings, epoch {epoch}",
                 xlabel="UMAP 1", ylabel="UMAP 2")
        axis.legend()
        fig.tight_layout()
        path = self.output_dir / f"learned_protein_umap_epoch_{epoch:02d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        tracker.log_artifact(str(path), "diagnostics")

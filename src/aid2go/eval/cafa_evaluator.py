import logging
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from pathlib import Path

from aid2go.eval.cafa_metrics import calculate_fmax, calculate_smax, calculate_aupr

logger = logging.getLogger(__name__)

class CAFAEvaluator:
    """
    Certified CAFA Evaluator implementing Fmax, Smax (Information Accretion weighted),
    and AUPR metrics across GO subontologies (BPO, CCO, MFO).
    """

    def __init__(self, ia_dict: Optional[Dict[str, float]] = None, go_aspect_map: Optional[Dict[str, str]] = None):
        self.ia_dict = ia_dict or {}
        self.go_aspect_map = go_aspect_map or {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        go_terms: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluates overall and aspect-specific CAFA metrics.

        Args:
            y_true: Binary ground truth array/matrix.
            y_score: Prediction probability array/matrix.
            go_terms: Optional list of GO term IDs corresponding to columns of y_true/y_score.

        Returns:
            Dict containing global & per-aspect metrics.
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        metrics = {
            "fmax": calculate_fmax(y_true, y_score),
            "aupr": calculate_aupr(y_true, y_score),
        }

        if go_terms is not None and y_true.ndim == 2:
            ia_weights = np.array([self.ia_dict.get(term, 1.0) for term in go_terms])
            metrics["smax"] = calculate_smax(y_true, y_score, ia_weights)

            # Aspect-specific evaluation
            aspects = {"BPO": "biological_process", "CCO": "cellular_component", "MFO": "molecular_function"}
            for aspect_code, aspect_name in aspects.items():
                indices = [
                    i for i, t in enumerate(go_terms)
                    if self.go_aspect_map.get(t) == aspect_name or self.go_aspect_map.get(t) == aspect_code
                ]
                if indices:
                    yt_sub = y_true[:, indices]
                    ys_sub = y_score[:, indices]
                    ia_sub = ia_weights[indices]

                    metrics[f"{aspect_code}_fmax"] = calculate_fmax(yt_sub, ys_sub)
                    metrics[f"{aspect_code}_smax"] = calculate_smax(yt_sub, ys_sub, ia_sub)
                    metrics[f"{aspect_code}_aupr"] = calculate_aupr(yt_sub, ys_sub)

        return metrics

    def generate_report(self, metrics: Dict[str, float], output_dir: Path) -> Path:
        """Saves evaluation metrics to JSON and Markdown summary reports."""
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "cafa_evaluation_report.md"

        lines = [
            "# CAFA Certified Evaluation Report",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
        ]
        for k, v in metrics.items():
            lines.append(f"| `{k}` | `{v:.4f}` |")

        with open(report_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(f"Saved evaluation report to {report_path}")
        return report_path

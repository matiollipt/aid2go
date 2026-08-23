import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

def export_cafa_submission(
    predictions: Dict[str, Dict[str, float]],
    output_path: Path,
    min_threshold: float = 0.001,
    top_k: int = 1500,
    include_header: bool = False
) -> Path:
    """
    Exports protein GO predictions into official 3-column CAFA submission TSV.

    Format:
    <Protein_ID>\t<GO_ID>\t<Score>

    Args:
        predictions: Dict mapping protein_id -> dict of {go_id: float_score}.
        output_path: Target output file path.
        min_threshold: Minimum score threshold to include (default: 0.001).
        top_k: Maximum predictions per protein.
        include_header: Whether to include column headers (default: False for official CAFA).

    Returns:
        Path to generated submission file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for pid, term_scores in predictions.items():
        sorted_terms = sorted(
            [(t, s) for t, s in term_scores.items() if s >= min_threshold],
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        for go_id, score in sorted_terms:
            rows.append((pid, go_id, f"{score:.3f}"))

    df = pd.DataFrame(rows, columns=["Protein_ID", "GO_ID", "Score"])
    df.to_csv(output_path, sep="\t", index=False, header=include_header)
    logger.info(f"Saved CAFA submission file with {len(df)} predictions for {len(predictions)} proteins to {output_path}")
    print(f"✓ Saved CAFA submission table ({len(df)} lines) to {output_path}")
    return output_path

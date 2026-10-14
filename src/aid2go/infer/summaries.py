import logging
from pathlib import Path
from typing import Dict, List, Tuple
import os
import tempfile
from aid2go.cli.progress import track_items

logger = logging.getLogger(__name__)

def export_cafa_submission(
    predictions: Dict[str, Dict[str, float]],
    output_path: Path,
    min_threshold: float = 0.001,
    top_k: int = 1500,
    include_header: bool = False,
    term_order: Dict[str, int] | None = None,
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
        term_order: Parent-first DAG ranks to preserve true-path closure at score ties.

    Returns:
        Path to generated submission file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent,
                                         suffix=".tmp", delete=False) as target:
            temporary = Path(target.name)
            if include_header:
                target.write("Protein_ID\tGO_ID\tScore\n")
            for pid in track_items(predictions, "Writing submission", "proteins"):
                sorted_terms = sorted(
                    ((term, score) for term, score in predictions[pid].items()
                     if score >= min_threshold),
                    key=lambda item: (-item[1], (term_order or {}).get(item[0], 0)),
                )[:top_k]
                for go_id, score in sorted_terms:
                    target.write(f"{pid}\t{go_id}\t{score:.3f}\n")
                    count += 1
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    logger.info("Saved %s predictions for %s proteins to %s", count, len(predictions), output_path)
    print(f"✓ Saved CAFA submission table ({count} lines) to {output_path}")
    return output_path

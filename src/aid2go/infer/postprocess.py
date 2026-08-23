import logging
from typing import Dict, List, Set, Tuple
import numpy as np
import networkx as nx
import obonet
from pathlib import Path

logger = logging.getLogger(__name__)

def load_go_dag(obo_path: Path) -> nx.DiGraph:
    """Reads GO OBO file into a clean NetworkX DiGraph."""
    dag = obonet.read_obo(str(obo_path), ignore_obsolete=True)
    # Filter edges to only is_a and part_of
    clean_dag = dag.copy()
    remove_edges = [(u, v, k) for u, v, k in dag.edges if k not in ("is_a", "part_of")]
    clean_dag.remove_edges_from(remove_edges)
    return clean_dag

def enforce_true_path_rule(scores_dict: Dict[str, float], dag: nx.DiGraph) -> Dict[str, float]:
    """
    Enforces the True-Path Rule: P(parent) >= P(child).
    
    Given a mapping of GO_ID -> predicted_score for a single protein,
    propagates scores upwards along edges in topological reverse order so that
    parents obtain at least the maximum score of any of their descendant terms.

    Args:
        scores_dict: Dictionary mapping term_id (str) -> score (float).
        dag: GO DAG networkx DiGraph where edges point from child -> parent or parent -> child.
             Note: obonet edges point from child -> parent (u is_a v means u -> v).

    Returns:
        Propagated scores dictionary with monotonic True-Path guarantee.
    """
    propagated = scores_dict.copy()
    
    # In obonet, u -> v means u is_a v (u is child, v is parent).
    # We want to iterate from children to parents (bottom-up in GO DAG).
    # Simple algorithm: iterate over terms with non-zero scores and push max score to all ancestors.
    for term, score in list(scores_dict.items()):
        if score <= 0:
            continue
        if term in dag:
            # nx.descendants in obonet points up to parents
            ancestors = nx.descendants(dag, term)
            for parent in ancestors:
                if parent in propagated:
                    propagated[parent] = max(propagated[parent], score)
                else:
                    propagated[parent] = score

    return propagated

def filter_by_threshold(scores_dict: Dict[str, float], threshold: float = 0.001, top_k: int = None) -> List[Tuple[str, float]]:
    """
    Filters and sorts term predictions by threshold and top-k limit.
    """
    filtered = [(term, score) for term, score in scores_dict.items() if score >= threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        filtered = filtered[:top_k]
    return filtered

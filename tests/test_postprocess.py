import pytest
import networkx as nx
from aid2go.infer.postprocess import enforce_true_path_rule, filter_by_threshold

def test_true_path_rule_propagation():
    # Build simple DAG: C -> P (obonet format where child points to parent)
    dag = nx.DiGraph()
    dag.add_edge("GO:0000002", "GO:0000001")  # GO:2 is child, GO:1 is parent

    scores = {
        "GO:0000001": 0.3,
        "GO:0000002": 0.8
    }

    propagated = enforce_true_path_rule(scores, dag)

    # Parent score must be updated to max(0.3, 0.8) = 0.8
    assert propagated["GO:0000001"] == 0.8
    assert propagated["GO:0000002"] == 0.8

def test_filter_by_threshold():
    scores = {"GO:001": 0.9, "GO:002": 0.05, "GO:003": 0.0001}
    filtered = filter_by_threshold(scores, threshold=0.01, top_k=2)
    assert len(filtered) == 2
    assert filtered[0] == ("GO:001", 0.9)
    assert filtered[1] == ("GO:002", 0.05)

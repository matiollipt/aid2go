import pytest
import numpy as np
from aid2go.eval.cafa_metrics import calculate_fmax, calculate_smax, calculate_aupr
from aid2go.eval.cafa_evaluator import CAFAEvaluator

def test_cafa_metrics_calculation():
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_score = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.3]])
    ia_weights = np.array([1.5, 2.0, 1.2])

    fmax = calculate_fmax(y_true, y_score)
    smax = calculate_smax(y_true, y_score, ia_weights)
    aupr = calculate_aupr(y_true, y_score)

    assert 0.0 <= fmax <= 1.0
    assert smax >= 0.0
    assert 0.0 <= aupr <= 1.0

def test_cafa_evaluator(tmp_path):
    evaluator = CAFAEvaluator(
        ia_dict={"GO:0000001": 1.5, "GO:0000002": 2.0},
        go_aspect_map={"GO:0000001": "biological_process", "GO:0000002": "cellular_component"}
    )
    y_true = np.array([[1, 0], [0, 1]])
    y_score = np.array([[0.9, 0.2], [0.1, 0.85]])
    go_terms = ["GO:0000001", "GO:0000002"]

    metrics = evaluator.evaluate(y_true, y_score, go_terms=go_terms)
    assert "fmax" in metrics
    assert "smax" in metrics
    assert "aupr" in metrics
    assert "BPO_fmax" in metrics

    report_path = evaluator.generate_report(metrics, tmp_path)
    assert report_path.exists()

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import precision_recall_curve, auc

def calculate_fmax(y_true: np.ndarray, y_score: np.ndarray, thresholds: Optional[np.ndarray] = None) -> float:
    """
    Calculates Fmax (maximum F1-score across decision thresholds).

    Args:
        y_true: (N, C) or 1D array of ground truth targets.
        y_score: (N, C) or 1D array of predicted probability scores.
        thresholds: Array of thresholds to evaluate.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.ndim == 1 or y_true.shape[0] == 1:
        precision, recall, _ = precision_recall_curve(y_true.flatten(), y_score.flatten())
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        return float(np.max(f1_scores))

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    max_f1 = 0.0
    for tau in thresholds:
        pred_binary = (y_score >= tau).astype(int)
        
        # Protein-centric evaluation
        tp = np.sum(pred_binary * y_true, axis=1)
        fp = np.sum(pred_binary * (1 - y_true), axis=1)
        fn = np.sum((1 - pred_binary) * y_true, axis=1)

        pred_count = tp + fp
        true_count = tp + fn

        valid_pred = pred_count > 0
        if not np.any(valid_pred):
            continue

        prec_i = tp[valid_pred] / pred_count[valid_pred]
        rec_i = tp / np.maximum(true_count, 1)

        prec = np.mean(prec_i)
        rec = np.mean(rec_i)

        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > max_f1:
                max_f1 = f1

    return float(max_f1)

def calculate_smax(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ia_weights: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> float:
    """
    Calculates Smax (minimum semantic distance metric weighted by Information Accretion).

    S(tau) = sqrt( RU(tau)^2 + MI(tau)^2 )
    Smax = min_tau S(tau)

    Args:
        y_true: (N, C) binary target matrix.
        y_score: (N, C) continuous prediction matrix.
        ia_weights: (C,) 1D array of Information Accretion (IA) values for each term.
        thresholds: Array of thresholds to evaluate.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    ia_weights = np.asarray(ia_weights)

    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_score = y_score.reshape(1, -1)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    num_proteins = y_true.shape[0]
    min_s = float("inf")

    for tau in thresholds:
        pred_binary = (y_score >= tau).astype(int)

        # Remaining Uncertainty (RU): terms in y_true but not in pred_binary
        ru_matrix = (y_true * (1 - pred_binary)) * ia_weights
        ru = np.mean(np.sum(ru_matrix, axis=1))

        # Misinformation (MI): terms in pred_binary but not in y_true
        mi_matrix = ((1 - y_true) * pred_binary) * ia_weights
        mi = np.mean(np.sum(mi_matrix, axis=1))

        s_val = float(np.sqrt(ru**2 + mi**2))
        if s_val < min_s:
            min_s = s_val

    return float(min_s) if min_s != float("inf") else 0.0

def calculate_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculates Micro-AUPR (Area Under Precision-Recall Curve)."""
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))

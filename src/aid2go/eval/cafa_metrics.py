import numpy as np
from sklearn.metrics import precision_recall_curve, auc

def calculate_fmax(y_true, y_score, step=0.01):
    """
    Calculates Fmax (maximum F1-score across thresholds).
    Args:
        y_true: (N, C) or (N,)
        y_score: (N, C) or (N,)
    """
    # Simplified version for single link prediction (1D arrays)
    # For multi-label (N, C), we compute precision/recall per threshold across all samples
    
    # If inputs are flattened or 1D (link prediction binary classification)
    precision, recall, thresholds = precision_recall_curve(y_true.flatten(), y_score.flatten())
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    return np.max(f1_scores)

def calculate_aupr(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true.flatten(), y_score.flatten())
    return auc(recall, precision)

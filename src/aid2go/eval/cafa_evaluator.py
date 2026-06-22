from typing import Dict
from aid2go.eval.cafa_metrics import calculate_fmax, calculate_aupr

class CAFAEvaluator:
    def evaluate(self, y_true, y_score) -> Dict[str, float]:
        fmax = calculate_fmax(y_true, y_score)
        aupr = calculate_aupr(y_true, y_score)
        return {"fmax": fmax, "aupr": aupr}

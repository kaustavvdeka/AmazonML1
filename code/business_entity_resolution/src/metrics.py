"""
metrics.py - Evaluation metrics for entity resolution.

Implements macro-averaged F0.5, precision, recall following the
challenge specification (singletons included).
"""

from typing import Dict, List, Set, Tuple

import numpy as np


def f05(precision: float, recall: float) -> float:
    """F0.5 = (1.25 * P * R) / (0.25 * P + R)."""
    denom = 0.25 * precision + recall
    return 1.25 * precision * recall / denom if denom > 0 else 0.0


def macro_f05(
    s1_ids: List[str],
    predictions: Dict[str, Set[str]],
    gt: Dict[str, List[str]],
) -> Tuple[float, float, float, float]:
    """
    Compute macro-averaged F0.5, precision, recall, F1.

    Singletons (empty true set):
      - Correctly predicted empty  → score 1.0
      - Incorrectly predicted non-empty → score 0.0

    Returns
    -------
    (macro_f05, macro_precision, macro_recall, macro_f1)
    """
    scores_f05 = []
    scores_prec = []
    scores_rec  = []

    for s1_id in s1_ids:
        true_set = set(gt.get(s1_id, []))
        pred_set = predictions.get(s1_id, set())

        if not true_set and not pred_set:
            scores_f05.append(1.0)
            scores_prec.append(1.0)
            scores_rec.append(1.0)
        elif not true_set and pred_set:
            scores_f05.append(0.0)
            scores_prec.append(0.0)
            scores_rec.append(1.0)
        elif true_set and not pred_set:
            scores_f05.append(0.0)
            scores_prec.append(1.0)
            scores_rec.append(0.0)
        else:
            tp  = len(true_set & pred_set)
            p   = tp / len(pred_set)
            r   = tp / len(true_set)
            scores_f05.append(f05(p, r))
            scores_prec.append(p)
            scores_rec.append(r)

    macro_p  = float(np.mean(scores_prec))
    macro_r  = float(np.mean(scores_rec))
    macro_f  = float(np.mean(scores_f05))
    macro_f1 = float(np.mean([
        2*p*r/(p+r) if (p+r) > 0 else 0.0
        for p, r in zip(scores_prec, scores_rec)
    ]))

    return macro_f, macro_p, macro_r, macro_f1

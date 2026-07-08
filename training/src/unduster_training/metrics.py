"""Component-wise precision/recall per defect type.

Pixel IoU punishes thin scratches unfairly; what matters is whether each
physical defect was found and how many phantoms were invented.
"""

import numpy as np
import scipy.ndimage as ndi

KINDS = {1: "dust", 2: "scratch", 3: "hair"}


def score_masks(pred: np.ndarray, gt_typed: np.ndarray) -> dict:
    labels, n = ndi.label(gt_typed > 0)
    tp_by_kind = {k: 0 for k in KINDS.values()}
    total_by_kind = {k: 0 for k in KINDS.values()}
    tp = fn = 0
    for i in range(1, n + 1):
        comp = labels == i
        vals, counts = np.unique(gt_typed[comp], return_counts=True)
        kind = KINDS[int(vals[np.argmax(counts)])]
        total_by_kind[kind] += 1
        if pred[comp].mean() >= 0.5:
            tp_by_kind[kind] += 1
            tp += 1
        else:
            fn += 1
    pred_labels, m = ndi.label(pred)
    fp = 0
    for i in range(1, m + 1):
        if not (gt_typed[pred_labels == i] > 0).any():
            fp += 1
    out: dict = {"tp": tp, "fn": fn, "fp": fp}
    for kind in KINDS.values():
        total = total_by_kind[kind]
        out[f"recall_{kind}"] = (tp_by_kind[kind] / total) if total else None
    out["precision"] = (m - fp) / m if m else 1.0
    return out

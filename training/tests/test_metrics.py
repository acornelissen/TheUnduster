import numpy as np

from unduster_training.metrics import score_masks


def _gt():
    gt = np.zeros((100, 100), np.uint8)
    gt[10:14, 10:14] = 1  # dust A
    gt[50:54, 50:54] = 1  # dust B
    gt[80, 10:60] = 2  # scratch
    return gt


def test_perfect_prediction():
    gt = _gt()
    s = score_masks(gt > 0, gt)
    assert s["recall_dust"] == 1.0 and s["recall_scratch"] == 1.0
    assert s["recall_hair"] is None
    assert s["precision"] == 1.0 and s["fp"] == 0


def test_miss_and_false_positive():
    gt = _gt()
    pred = np.zeros_like(gt, bool)
    pred[10:14, 10:14] = True  # hit dust A
    pred[30:33, 70:73] = True  # false positive
    s = score_masks(pred, gt)
    assert s["recall_dust"] == 0.5
    assert s["recall_scratch"] == 0.0
    assert s["fp"] == 1
    assert s["precision"] == 0.5  # 1 of 2 predicted components is real


def test_half_coverage_rule():
    gt = _gt()
    pred = np.zeros_like(gt, bool)
    pred[10:14, 10:12] = True  # exactly 50% of dust A
    s = score_masks(pred, gt)
    assert s["recall_dust"] == 0.5  # >= 50% counts

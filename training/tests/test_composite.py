import numpy as np

from unduster_training.composite import synthesize
from unduster_training.harvest import Defect


def _toy_defects():
    m = np.zeros((7, 7), bool)
    m[2:5, 2:5] = True
    d = np.where(m, -0.4, 0.0).astype(np.float32)
    line_m = np.zeros((3, 41), bool)
    line_m[1, :] = True
    line_d = np.where(line_m, -0.3, 0.0).astype(np.float32)
    return [Defect(d, m, "dust"), Defect(line_d, line_m, "scratch")]


def test_synthesize_colour():
    rng = np.random.default_rng(0)
    clean = np.full((256, 256, 3), 0.6, np.float32)
    img, gt = synthesize(clean, _toy_defects(), rng, n_range=(10, 20))
    assert img.shape == (256, 256, 3) and gt.shape == (256, 256)
    assert gt.dtype == bool and gt.any()
    assert img.min() >= 0 and img.max() <= 1


def test_defects_change_pixels_only_under_mask():
    rng = np.random.default_rng(1)
    clean = np.full((256, 256), 0.6, np.float32)
    img, gt = synthesize(clean, _toy_defects(), rng, n_range=(8, 12), grain_strength=(0.0, 0.0))
    base, _ = synthesize(clean, _toy_defects(), rng2 := np.random.default_rng(1), n_range=(0, 0), grain_strength=(0.0, 0.0))
    changed = np.abs(img - base) > 1e-5
    assert changed.any()
    assert not (changed & ~gt).any()  # nothing outside the GT mask moved


def test_bw_variant_is_single_channel():
    rng = np.random.default_rng(2)
    clean = np.full((128, 128, 3), 0.5, np.float32)
    img, gt = synthesize(clean, _toy_defects(), rng, bw=True)
    assert img.ndim == 2

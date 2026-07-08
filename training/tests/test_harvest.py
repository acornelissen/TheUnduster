import numpy as np

from unduster_training.harvest import Defect, harvest, load_library, save_library


def test_harvest_finds_all_defects(blank_scan):
    scan, truth = blank_scan
    defects = harvest(scan)
    assert len(defects) == len(truth)
    kinds = sorted(d.kind for d in defects)
    assert kinds == sorted(k for k, _, _ in truth)


def test_defect_deltas_are_signed_and_local(blank_scan):
    scan, _ = blank_scan
    for d in harvest(scan):
        assert d.delta.shape == d.mask.shape
        assert d.mask.any()
        assert d.delta[d.mask].mean() < -0.1  # painted defects are dark
        assert abs(d.delta[~d.mask]).mean() < 0.02  # background carries ~no signal


def test_library_round_trip(tmp_path, blank_scan):
    scan, _ = blank_scan
    defects = harvest(scan)
    save_library(defects, tmp_path / "lib")
    back = load_library(tmp_path / "lib")
    assert len(back) == len(defects)
    orig = {(d.kind, d.mask.sum()) for d in defects}
    assert {(d.kind, d.mask.sum()) for d in back} == orig


def test_defect_touching_scan_corner_is_harvested():
    rng = np.random.default_rng(5)
    scan = np.full((200, 200), 0.82, np.float32) + rng.normal(0, 0.004, (200, 200)).astype(
        np.float32
    )
    scan[0:4, 0:4] -= 0.35  # dust blob flush against (0,0)
    defects = harvest(np.clip(scan, 0, 1))
    assert len(defects) == 1
    d = defects[0]
    assert d.kind == "dust"
    assert d.mask.any()
    assert d.delta.shape == d.mask.shape  # crop clamped at the border, no crash


def test_sub_min_area_speck_is_excluded():
    rng = np.random.default_rng(6)
    scan = np.full((200, 200), 0.82, np.float32) + rng.normal(0, 0.004, (200, 200)).astype(
        np.float32
    )
    scan[100, 100] -= 0.4  # single pixel < min_area=3
    scan[50:54, 50:54] -= 0.35  # real blob, kept
    defects = harvest(np.clip(scan, 0, 1))
    assert len(defects) == 1
    assert defects[0].mask.sum() >= 3

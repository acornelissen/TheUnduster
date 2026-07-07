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

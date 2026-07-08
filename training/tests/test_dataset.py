import numpy as np
import pytest
import torch

from unduster_training.dataset import SyntheticDefects
from unduster_training.harvest import harvest, save_library
from unduster_training.io import save_image

# pytest puts tests/ itself on sys.path (no __init__.py), so conftest imports flat
from conftest import make_blank_scan


def _setup(tmp_path):
    scan, _ = make_blank_scan()
    save_library(harvest(scan), tmp_path / "lib")
    rng = np.random.default_rng(3)
    for i in range(2):
        save_image(tmp_path / f"clean{i}.png", rng.random((700, 900, 3)).astype(np.float32), bit_depth=8)
    return tmp_path


def test_colour_item_shapes(tmp_path):
    root = _setup(tmp_path)
    ds = SyntheticDefects(root, root / "lib", variant="colour", patch=256, length=8, seed=0)
    assert len(ds) == 8
    x, y = ds[0]
    assert x.shape == (3, 256, 256) and x.dtype == torch.float32
    assert y.shape == (1, 256, 256)
    assert set(torch.unique(y).tolist()) <= {0.0, 1.0}


def test_bw_item_shapes(tmp_path):
    root = _setup(tmp_path)
    ds = SyntheticDefects(root, root / "lib", variant="bw", patch=256, length=8, seed=0)
    x, y = ds[1]
    assert x.shape == (1, 256, 256)


def test_deterministic_per_index(tmp_path):
    root = _setup(tmp_path)
    a = SyntheticDefects(root, root / "lib", patch=256, length=8, seed=7)
    b = SyntheticDefects(root, root / "lib", patch=256, length=8, seed=7)
    xa, ya = a[3]
    xb, yb = b[3]
    assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_small_clean_image_raises(tmp_path):
    root = _setup(tmp_path)
    save_image(root / "tiny.png", np.full((100, 150, 3), 0.5, np.float32), bit_depth=8)
    ds = SyntheticDefects(root, root / "lib", patch=256, length=8, seed=0)
    with pytest.raises(ValueError, match="tiny.png"):
        for i in range(len(ds)):
            ds[i]  # tiny.png will be picked within a few draws

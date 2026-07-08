import numpy as np
import torch

from unduster_training.detectors import OnnxDetector, classical_detect, implied_mask
from unduster_training.export import export_onnx
from unduster_training.model import UNet
from unduster_training.train import save_checkpoint

from conftest import make_blank_scan


def test_classical_finds_painted_defects():
    scan, truth = make_blank_scan()
    mask = classical_detect(scan)
    for _, cy, cx in truth:
        # +-6 window: a 5px-radius blob's detectable edge sits ~5px from centre
        assert mask[cy - 6 : cy + 7, cx - 6 : cx + 7].any(), f"missed defect at {cy},{cx}"
    assert mask.mean() < 0.05  # not screaming everywhere


def test_onnx_detector_runs_tiled(tmp_path):
    torch.manual_seed(0)
    model = UNet(in_ch=1, base=8, depth=2)
    ckpt, onnx_path = tmp_path / "m.pt", tmp_path / "m.onnx"
    save_checkpoint(model, "bw", 1, ckpt)
    export_onnx(ckpt, onnx_path)
    det = OnnxDetector(onnx_path, threshold=0.5)
    img = np.random.default_rng(0).random((700, 900)).astype(np.float32)  # forces tiling
    mask = det(img)
    assert mask.shape == (700, 900) and mask.dtype == bool


def test_onnx_detector_channel_adapt(tmp_path):
    torch.manual_seed(0)
    model = UNet(in_ch=1, base=8, depth=2)
    ckpt, onnx_path = tmp_path / "m.pt", tmp_path / "m.onnx"
    save_checkpoint(model, "bw", 1, ckpt)
    export_onnx(ckpt, onnx_path)
    det = OnnxDetector(onnx_path)
    rgb = np.random.default_rng(0).random((256, 256, 3)).astype(np.float32)
    assert det(rgb).shape == (256, 256)  # RGB in, grey model: converted, not crashed


def test_implied_mask():
    before = np.full((64, 64), 0.5, np.float32)
    after = before.copy()
    after[10:14, 10:14] = 0.6  # the competitor healed here
    m = implied_mask(before, after)
    assert m[11, 11] and not m[40, 40]

"""Train a tiny demo detector on synthetic defects and commit it as a fixture.

The random-weight tiny-detector.onnx proves the inference path but fires on
everything when run against real scans. This script trains the same-size
model for a few hundred steps on synthetic dust/scratch/hair overlays so the
app demo behaves sanely until a real model exists.

Run from training/:  uv run python scripts/make_demo_detector.py
Deterministic seeds; takes a few minutes on MPS/CPU.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from unduster_training.dataset import SyntheticDefects
from unduster_training.export import export_onnx, parity_gap
from unduster_training.harvest import harvest, save_library
from unduster_training.io import save_image
from unduster_training.model import UNet
from unduster_training.train import save_checkpoint, train_steps

OUT = Path(__file__).resolve().parents[2] / "engine" / "fixtures"
STEPS = 600
BATCH = 4
PATCH = 256


def paint_blank_scan(rng: np.random.Generator, h: int = 500, w: int = 500) -> np.ndarray:
    """Synthetic dusty blank film: uniform base plus painted defects."""
    img = np.full((h, w), rng.uniform(0.7, 0.9), np.float32)
    img += rng.normal(0, 0.004, (h, w)).astype(np.float32)
    for _ in range(rng.integers(6, 14)):  # dust
        cy, cx = rng.integers(10, h - 10), rng.integers(10, w - 10)
        r = int(rng.integers(1, 5))
        yy, xx = np.ogrid[:h, :w]
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] -= rng.uniform(0.2, 0.45)
    for _ in range(rng.integers(1, 4)):  # scratches
        y = int(rng.integers(20, h - 20))
        x0 = int(rng.integers(0, w // 2))
        length = int(rng.integers(60, w - x0 - 1))
        img[y : y + 2, x0 : x0 + length] -= rng.uniform(0.2, 0.4)
    return np.clip(img, 0, 1)


def make_clean(rng: np.random.Generator, h: int = 700, w: int = 900) -> np.ndarray:
    """Clean 'photo': smooth low-frequency structure, no defects."""
    small = rng.random((-(-h // 32), -(-w // 32))).astype(np.float32)
    big = np.kron(small, np.ones((32, 32), np.float32))[:h, :w]
    ramp = np.linspace(0.2, 0.8, w, dtype=np.float32)[None, :]
    return np.clip(0.5 * big + 0.5 * ramp, 0, 1)


def main() -> None:
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "clean").mkdir()
        defects = []
        for i in range(8):
            defects.extend(harvest(paint_blank_scan(rng)))
        save_library(defects, tmp / "lib")
        print(f"defect library: {len(defects)}")
        for i in range(16):
            save_image(tmp / "clean" / f"c{i:02d}.png", make_clean(rng), bit_depth=8)

        torch.manual_seed(42)
        ds = SyntheticDefects(
            tmp / "clean", tmp / "lib", variant="bw", patch=PATCH,
            length=STEPS * BATCH, seed=42,
        )
        loader = DataLoader(ds, batch_size=BATCH, num_workers=2)
        model = UNet(in_ch=1, base=8, depth=2)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        losses = train_steps(model, loader, steps=STEPS, lr=3e-3, device=device)
        print(f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")

        ckpt = tmp / "demo.pt"
        save_checkpoint(model.cpu(), "bw", STEPS, ckpt)
        onnx_path = OUT / "demo-detector.onnx"
        export_onnx(ckpt, onnx_path)
        # fixtures must be single self-contained files (same rule as
        # make_engine_fixtures): re-save with embedded weights
        import onnx

        onnx.save(onnx.load(str(onnx_path)), str(onnx_path))
        onnx_path.with_suffix(".onnx.data").unlink(missing_ok=True)
        gap = parity_gap(ckpt, onnx_path, size=256)
        print(f"exported {onnx_path.name}, parity gap {gap:.2e}")

        # sanity: quiet on clean, loud on a defect
        from unduster_training.detectors import OnnxDetector

        det = OnnxDetector(onnx_path)
        clean = make_clean(np.random.default_rng(7), 512, 512)
        dirty = clean.copy()
        dirty[100:104, 100:104] = 0.02
        p_clean = det.probabilities(clean)
        p_defect = det.probabilities(dirty)[100:104, 100:104]
        print(
            f"clean mean prob {p_clean.mean():.4f} (99.9th {np.quantile(p_clean, 0.999):.3f}), "
            f"defect mean prob {p_defect.mean():.3f}"
        )
        assert p_clean.mean() < 0.05, "demo model too noisy on clean images"
        assert p_defect.mean() > 0.5, "demo model misses an obvious speck"


if __name__ == "__main__":
    main()

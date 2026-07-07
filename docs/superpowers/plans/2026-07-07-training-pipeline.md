# Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline pipeline that harvests real film defects, synthesizes training data, trains colour and B&W dust/scratch detectors, exports them to ONNX, and benchmarks them for per-defect-type precision/recall against baselines.

**Architecture:** A Python package (`unduster_training`) under `training/` in the TheUnduster repo. Real defect overlays are harvested from scans of blank film, composited onto clean images with film-look augmentation (grain, tone curves) to make synthetic training pairs. A small U-Net trains on 512px patches, exports to ONNX with a torch/ORT parity check. The benchmark harness scores any detector (ours, a classical baseline, any external ONNX model, or a competitor's implied mask from before/after pairs) on a hand-labelled test roll.

**Tech Stack:** Python 3.12, uv, PyTorch (MPS), numpy, scipy, OpenCV (headless), imageio+tifffile, onnx, onnxruntime, pytest.

## Global Constraints

- Everything lives under `training/` in the repo `/Users/albert/Development/TheUnduster`.
- Python 3.12 pinned via `training/mise.toml`; dependencies managed by uv; run everything as `uv run ...` from `training/`.
- Images in memory are always `float32` in `[0, 1]`, shape `HxW` (grey) or `HxWx3` (RGB). 16-bit files map 65535 → 1.0.
- Detection tile size is 512 with 64px overlap (must match the spec and the future `fd-infer` crate).
- Ground-truth typed masks use palette values: 0 background, 1 dust, 2 scratch, 3 hair.
- Determinism: every random operation takes an explicit `numpy.random.Generator`; datasets derive per-item generators from `(seed, index)`.
- `training/data/` is gitignored — real scans, defect libraries, checkpoints, and the labelled roll never enter git.
- Training and inference must run on Apple Silicon: device pick order is `mps`, `cuda`, `cpu`.
- Tests must not download anything or need real scans; all fixtures are generated in-test.
- Commit after every task. No emoji anywhere. No Co-Authored-By lines.

---

### Task 1: Project scaffolding and image I/O

**Files:**
- Create: `training/mise.toml`
- Create: `training/pyproject.toml`
- Create: `training/.gitignore`
- Create: `training/src/unduster_training/__init__.py`
- Create: `training/src/unduster_training/io.py`
- Test: `training/tests/test_io.py`

**Interfaces:**
- Produces: `load_image(path) -> np.ndarray` (float32 [0,1], HxW or HxWx3), `save_image(path, img, bit_depth=16)`, `to_gray(img) -> np.ndarray` (HxW). All later tasks use these.

- [ ] **Step 1: Scaffold the project**

`training/mise.toml`:

```toml
[tools]
python = "3.12"
uv = "latest"
```

`training/pyproject.toml`:

```toml
[project]
name = "unduster-training"
version = "0.1.0"
description = "Offline training pipeline for TheUnduster defect detector"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.13",
    "opencv-python-headless>=4.9",
    "imageio>=2.34",
    "tifffile>=2024.2.12",
    "torch>=2.3",
    "onnx>=1.16",
    "onnxruntime>=1.18",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/unduster_training"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`training/.gitignore`:

```
data/
.venv/
__pycache__/
*.onnx
*.pt
```

`training/src/unduster_training/__init__.py` is empty.

Run: `cd /Users/albert/Development/TheUnduster/training && mise install && uv sync`
Expected: venv created, all deps resolve.

- [ ] **Step 2: Write the failing test**

`training/tests/test_io.py`:

```python
import numpy as np
import pytest

from unduster_training.io import load_image, save_image, to_gray


@pytest.mark.parametrize("ext,bit_depth", [("tif", 16), ("tif", 8), ("png", 16), ("png", 8), ("jpg", 8)])
def test_round_trip(tmp_path, ext, bit_depth):
    # smooth gradient, not noise: JPEG on noise has unbounded per-pixel error
    ramp = np.linspace(0, 1, 64 * 48, dtype=np.float32).reshape(64, 48)
    img = np.stack([ramp, ramp * 0.5, 1.0 - ramp], axis=-1)
    path = tmp_path / f"x.{ext}"
    save_image(path, img, bit_depth=bit_depth)
    back = load_image(path)
    assert back.dtype == np.float32
    assert back.shape == (64, 48, 3)
    assert back.min() >= 0.0 and back.max() <= 1.0
    tol = 0.05 if ext == "jpg" else (1 / 255 if bit_depth == 8 else 1 / 65535) + 1e-6
    assert np.abs(back - img).max() <= tol


def test_gray_round_trip(tmp_path):
    img = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    path = tmp_path / "g.tif"
    save_image(path, img, bit_depth=16)
    back = load_image(path)
    assert back.shape == (32, 32)
    assert np.abs(back - img).max() <= 1 / 65535 + 1e-6


def test_to_gray():
    img = np.zeros((4, 4, 3), np.float32)
    img[..., 1] = 1.0
    g = to_gray(img)
    assert g.shape == (4, 4)
    assert 0.5 < g[0, 0] < 0.8  # Rec.709 green weight
    assert to_gray(g) is g  # already grey passes through
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_io.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError`.

- [ ] **Step 4: Write the implementation**

`training/src/unduster_training/io.py`:

```python
"""Image I/O. In-memory convention: float32 in [0, 1], HxW or HxWx3."""

from pathlib import Path

import imageio.v3 as iio
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    raw = iio.imread(path)
    if raw.ndim == 3 and raw.shape[2] == 4:
        raw = raw[..., :3]
    if raw.dtype == np.uint8:
        img = raw.astype(np.float32) / 255.0
    elif raw.dtype == np.uint16:
        img = raw.astype(np.float32) / 65535.0
    else:
        img = np.clip(raw.astype(np.float32), 0.0, 1.0)
    return img


def save_image(path: str | Path, img: np.ndarray, bit_depth: int = 16) -> None:
    path = Path(path)
    img = np.clip(img, 0.0, 1.0)
    if path.suffix.lower() in (".jpg", ".jpeg") or bit_depth == 8:
        out = (img * 255.0 + 0.5).astype(np.uint8)
    elif bit_depth == 16:
        out = (img * 65535.0 + 0.5).astype(np.uint16)
    else:
        raise ValueError(f"unsupported bit depth: {bit_depth}")
    iio.imwrite(path, out)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    w = np.array([0.2126, 0.7152, 0.0722], np.float32)  # Rec.709
    return img @ w
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_io.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add training
git commit -m "Scaffold training pipeline and add image I/O"
```

---

### Task 2: Defect harvesting from blank film scans

**Files:**
- Create: `training/src/unduster_training/harvest.py`
- Create: `training/tests/conftest.py`
- Test: `training/tests/test_harvest.py`

**Interfaces:**
- Consumes: `to_gray` from Task 1.
- Produces: `Defect` dataclass (`delta: np.ndarray float32 HxW`, `mask: np.ndarray bool HxW`, `kind: str` in `{"dust","scratch","hair"}`); `harvest(scan_gray, min_area=3, max_area=5000, k_sigma=6.0) -> list[Defect]`; `save_library(defects, dir_path)`; `load_library(dir_path) -> list[Defect]`; CLI `python -m unduster_training.harvest <scan_dir> <library_dir>`.

- [ ] **Step 1: Write the shared fixture and failing test**

`training/tests/conftest.py`:

```python
import numpy as np
import pytest


def make_blank_scan(h=400, w=400, base=0.82, noise=0.004, seed=1):
    """Uniform 'blank film' scan with known painted defects.

    Returns (scan_gray, truth) where truth lists (kind, cy, cx).
    """
    rng = np.random.default_rng(seed)
    img = np.full((h, w), base, np.float32) + rng.normal(0, noise, (h, w)).astype(np.float32)
    truth = []
    # dust: compact dark blobs
    for cy, cx, r in [(60, 60, 3), (150, 300, 5), (320, 80, 2)]:
        yy, xx = np.ogrid[:h, :w]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        img[blob] -= 0.35
        truth.append(("dust", cy, cx))
    # scratch: straight thin dark line
    for x in range(100, 300):
        img[200:202, x] -= 0.3
    truth.append(("scratch", 200, 200))
    # hair: curved thin line
    for t in np.linspace(0, np.pi, 300):
        y = int(300 + 40 * np.sin(t) * np.cos(3 * t))
        x = int(120 + 60 * t / np.pi + 20 * np.sin(2 * t))
        img[y : y + 2, x : x + 2] -= 0.3
    truth.append(("hair", 300, 150))
    return np.clip(img, 0, 1), truth


@pytest.fixture
def blank_scan():
    return make_blank_scan()
```

`training/tests/test_harvest.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_harvest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'unduster_training.harvest'`.

- [ ] **Step 3: Write the implementation**

`training/src/unduster_training/harvest.py`:

```python
"""Harvest real defect overlays from scans of blank/leader film.

A blank scan is (nearly) uniform, so anything that deviates from a smoothed
background estimate is a physical defect. Each connected component becomes a
Defect: a signed brightness delta plus a boolean mask, classified as dust,
scratch, or hair by shape.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

from .io import load_image, to_gray


@dataclass
class Defect:
    delta: np.ndarray  # float32 HxW, signed (defect minus background)
    mask: np.ndarray  # bool HxW
    kind: str  # "dust" | "scratch" | "hair"


def _classify(mask: np.ndarray) -> str:
    coords = np.argwhere(mask).astype(np.float64)
    if len(coords) < 10:
        return "dust"
    centered = coords - coords.mean(axis=0)
    cov = centered.T @ centered / len(coords)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    minor, major = max(evals[0], 1e-6), max(evals[1], 1e-6)
    elongation = np.sqrt(major / minor)
    if elongation < 4.0:
        return "dust"
    # residual from the principal axis separates straight scratches from curly hairs
    axis = evecs[:, 1]
    residual = centered - np.outer(centered @ axis, axis)
    rms = np.sqrt((residual**2).sum(axis=1).mean())
    return "scratch" if rms < 1.5 else "hair"


def harvest(
    scan_gray: np.ndarray,
    min_area: int = 3,
    max_area: int = 5000,
    k_sigma: float = 6.0,
) -> list[Defect]:
    bg = ndi.median_filter(scan_gray, size=21)
    diff = scan_gray - bg
    mad = np.median(np.abs(diff - np.median(diff)))
    sigma = max(1.4826 * mad, 1e-4)
    hot = np.abs(diff) > k_sigma * sigma
    labels, n = ndi.label(hot)
    defects: list[Defect] = []
    for sl in ndi.find_objects(labels):
        comp = labels[sl] > 0
        area = int(comp.sum())
        if not (min_area <= area <= max_area):
            continue
        pad = 2
        y0 = max(sl[0].start - pad, 0)
        y1 = min(sl[0].stop + pad, scan_gray.shape[0])
        x0 = max(sl[1].start - pad, 0)
        x1 = min(sl[1].stop + pad, scan_gray.shape[1])
        mask = np.zeros((y1 - y0, x1 - x0), bool)
        mask[sl[0].start - y0 : sl[0].stop - y0, sl[1].start - x0 : sl[1].stop - x0] = comp
        delta = (diff[y0:y1, x0:x1] * mask).astype(np.float32)
        defects.append(Defect(delta=delta, mask=mask, kind=_classify(mask)))
    return defects


def save_library(defects: list[Defect], dir_path: str | Path) -> None:
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(defects):
        np.savez_compressed(dir_path / f"{i:06d}_{d.kind}.npz", delta=d.delta, mask=d.mask)


def load_library(dir_path: str | Path) -> list[Defect]:
    out = []
    for f in sorted(Path(dir_path).glob("*.npz")):
        kind = f.stem.split("_", 1)[1]
        z = np.load(f)
        out.append(Defect(delta=z["delta"], mask=z["mask"].astype(bool), kind=kind))
    return out


def main() -> None:
    scan_dir, lib_dir = Path(sys.argv[1]), Path(sys.argv[2])
    all_defects: list[Defect] = []
    for f in sorted(scan_dir.iterdir()):
        if f.suffix.lower() not in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            continue
        scan = to_gray(load_image(f))
        found = harvest(scan)
        print(f"{f.name}: {len(found)} defects")
        all_defects.extend(found)
    save_library(all_defects, lib_dir)
    print(f"library: {len(all_defects)} defects -> {lib_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_harvest.py -v`
Expected: all PASS. If `test_harvest_finds_all_defects` finds extra components: noise speckles mean the fixture `noise` is too high relative to `k_sigma`; a split hair (5 components instead of 4) means the fixture's hair curve has a gap — thicken it to 3px in `make_blank_scan`, don't loosen the harvester.

- [ ] **Step 5: Commit**

```bash
git add training
git commit -m "Harvest defect overlays from blank film scans"
```

---

### Task 3: Film-look augmentation — grain and tone curves

**Files:**
- Create: `training/src/unduster_training/filmlook.py`
- Test: `training/tests/test_filmlook.py`

**Interfaces:**
- Consumes: `to_gray` from Task 1.
- Produces: `add_grain(img, rng, strength=0.04, size=0.8) -> np.ndarray`, `apply_random_curve(img, rng) -> np.ndarray`. Used by Task 4's compositor.

- [ ] **Step 1: Write the failing test**

`training/tests/test_filmlook.py`:

```python
import numpy as np

from unduster_training.filmlook import add_grain, apply_random_curve


def test_grain_adds_midtone_noise():
    rng = np.random.default_rng(0)
    img = np.full((256, 256), 0.5, np.float32)
    out = add_grain(img, rng, strength=0.05)
    assert out.shape == img.shape
    assert out.min() >= 0 and out.max() <= 1
    assert 0.02 < out.std() < 0.09
    assert abs(out.mean() - 0.5) < 0.01


def test_grain_spares_extremes():
    rng = np.random.default_rng(0)
    dark = add_grain(np.full((256, 256), 0.02, np.float32), rng, strength=0.05)
    mid = add_grain(np.full((256, 256), 0.5, np.float32), rng, strength=0.05)
    assert dark.std() < mid.std() * 0.4


def test_grain_colour_shape():
    rng = np.random.default_rng(0)
    img = np.full((64, 64, 3), 0.5, np.float32)
    assert add_grain(img, rng).shape == (64, 64, 3)


def test_curve_is_monotonic_and_bounded():
    rng = np.random.default_rng(0)
    ramp = np.linspace(0, 1, 1024, dtype=np.float32)
    for _ in range(20):
        out = apply_random_curve(ramp, rng)
        assert out.min() >= 0 and out.max() <= 1
        assert (np.diff(out) >= -1e-6).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_filmlook.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`training/src/unduster_training/filmlook.py`:

```python
"""Make clean digital images look like film scans: grain and tone curves."""

import cv2
import numpy as np

from .io import to_gray


def add_grain(img: np.ndarray, rng: np.random.Generator, strength: float = 0.04, size: float = 0.8) -> np.ndarray:
    noise = rng.standard_normal(img.shape[:2]).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), size)
    noise /= max(noise.std(), 1e-6)
    lum = to_gray(img)
    amp = strength * 4.0 * lum * (1.0 - lum)  # grain peaks in midtones
    grain = noise * amp
    if img.ndim == 3:
        grain = grain[..., None]
    return np.clip(img + grain, 0.0, 1.0)


def apply_random_curve(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gamma = rng.uniform(0.8, 1.25)
    lift = rng.uniform(0.0, 0.06)
    gain = rng.uniform(0.92, 1.0)
    return np.clip(lift + np.power(img, gamma) * (gain - lift), 0.0, 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_filmlook.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add training
git commit -m "Add film grain and tone curve augmentation"
```

---

### Task 4: Synthetic sample compositor

**Files:**
- Create: `training/src/unduster_training/composite.py`
- Test: `training/tests/test_composite.py`

**Interfaces:**
- Consumes: `Defect` (Task 2), `add_grain`, `apply_random_curve` (Task 3).
- Produces: `synthesize(clean, defects, rng, n_range=(5, 60), grain_strength=(0.01, 0.08), bw=False) -> tuple[np.ndarray, np.ndarray]` returning (image float32, gt_mask bool HxW). Used by Task 5.

- [ ] **Step 1: Write the failing test**

`training/tests/test_composite.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_composite.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`training/src/unduster_training/composite.py`:

```python
"""Composite harvested defects onto clean images to make training pairs.

Note for test determinism: the film-look pass consumes rng draws BEFORE any
defect placement, in a fixed order (curve, grain strength, grain size), so two
calls with equal seeds and different n_range still produce identical bases.
"""

import cv2
import numpy as np
import scipy.ndimage as ndi

from .filmlook import add_grain, apply_random_curve
from .harvest import Defect
from .io import to_gray


def _augment(d: Defect, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    delta = d.delta * rng.uniform(0.6, 1.3)
    if rng.random() < 0.5:
        delta = -delta  # dust reads light on inverted negative scans
    angle = rng.uniform(0.0, 360.0)
    delta = ndi.rotate(delta, angle, order=1, reshape=True)
    mask = ndi.rotate(d.mask.astype(np.float32), angle, order=1, reshape=True) > 0.5
    scale = rng.uniform(0.7, 1.4)
    h = max(int(round(delta.shape[0] * scale)), 1)
    w = max(int(round(delta.shape[1] * scale)), 1)
    delta = cv2.resize(delta, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
    delta = np.where(mask, delta, 0.0).astype(np.float32)
    return delta, mask


def _place(img: np.ndarray, gt: np.ndarray, delta: np.ndarray, mask: np.ndarray, cy: int, cx: int) -> None:
    h, w = delta.shape
    y0, x0 = cy - h // 2, cx - w // 2
    iy0, ix0 = max(y0, 0), max(x0, 0)
    iy1, ix1 = min(y0 + h, img.shape[0]), min(x0 + w, img.shape[1])
    if iy1 <= iy0 or ix1 <= ix0:
        return
    dy0, dx0 = iy0 - y0, ix0 - x0
    sub_d = delta[dy0 : dy0 + iy1 - iy0, dx0 : dx0 + ix1 - ix0]
    sub_m = mask[dy0 : dy0 + iy1 - iy0, dx0 : dx0 + ix1 - ix0]
    if img.ndim == 3:
        img[iy0:iy1, ix0:ix1] += sub_d[..., None]
    else:
        img[iy0:iy1, ix0:ix1] += sub_d
    gt[iy0:iy1, ix0:ix1] |= sub_m


def synthesize(
    clean: np.ndarray,
    defects: list[Defect],
    rng: np.random.Generator,
    n_range: tuple[int, int] = (5, 60),
    grain_strength: tuple[float, float] = (0.01, 0.08),
    bw: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    img = to_gray(clean).copy() if bw else clean.copy()
    img = apply_random_curve(img, rng)
    strength = rng.uniform(*grain_strength)
    size = rng.uniform(0.6, 1.4)
    if strength > 0:
        img = add_grain(img, rng, strength=strength, size=size)
    gt = np.zeros(img.shape[:2], bool)
    n = int(rng.integers(n_range[0], n_range[1] + 1))
    for _ in range(n):
        if not defects:
            break
        d = defects[int(rng.integers(len(defects)))]
        delta, mask = _augment(d, rng)
        cy = int(rng.integers(0, img.shape[0]))
        cx = int(rng.integers(0, img.shape[1]))
        _place(img, gt, delta, mask, cy, cx)
    return np.clip(img, 0.0, 1.0), gt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_composite.py -v`
Expected: all PASS. `test_defects_change_pixels_only_under_mask` works because with `grain_strength=(0.0, 0.0)` the base pass consumes identical rng draws for both calls before placement begins.

- [ ] **Step 5: Commit**

```bash
git add training
git commit -m "Composite harvested defects into synthetic training pairs"
```

---

### Task 5: Torch dataset

**Files:**
- Create: `training/src/unduster_training/dataset.py`
- Test: `training/tests/test_dataset.py`

**Interfaces:**
- Consumes: `load_image`, `synthesize`, `load_library`.
- Produces: `SyntheticDefects(clean_dir, library_dir, variant="colour", patch=512, length=10000, seed=0)` — a `torch.utils.data.Dataset` yielding `(x, y)` with `x: FloatTensor Cx512x512` (C=3 colour, C=1 bw), `y: FloatTensor 1x512x512` in {0,1}. Used by Task 7.

- [ ] **Step 1: Write the failing test**

`training/tests/test_dataset.py`:

```python
import numpy as np
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`training/src/unduster_training/dataset.py`:

```python
"""Streaming synthetic dataset: random clean patch + composited defects."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .composite import synthesize
from .harvest import load_library
from .io import load_image

_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


class SyntheticDefects(Dataset):
    def __init__(
        self,
        clean_dir: str | Path,
        library_dir: str | Path,
        variant: str = "colour",
        patch: int = 512,
        length: int = 10000,
        seed: int = 0,
    ):
        assert variant in ("colour", "bw")
        self.files = sorted(p for p in Path(clean_dir).iterdir() if p.suffix.lower() in _EXTS)
        if not self.files:
            raise ValueError(f"no clean images in {clean_dir}")
        self.defects = load_library(library_dir)
        self.variant = variant
        self.patch = patch
        self.length = length
        self.seed = seed
        # B&W: heavier grain so the model learns grain is not dust
        self.grain = (0.03, 0.12) if variant == "bw" else (0.01, 0.08)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng((self.seed << 32) + i)
        img = load_image(self.files[int(rng.integers(len(self.files)))])
        p = self.patch
        y0 = int(rng.integers(0, max(img.shape[0] - p, 1)))
        x0 = int(rng.integers(0, max(img.shape[1] - p, 1)))
        crop = img[y0 : y0 + p, x0 : x0 + p]
        out, gt = synthesize(crop, self.defects, rng, grain_strength=self.grain, bw=self.variant == "bw")
        if out.ndim == 2:
            x = torch.from_numpy(out[None])
        else:
            x = torch.from_numpy(out.transpose(2, 0, 1).copy())
        y = torch.from_numpy(gt[None].astype(np.float32))
        return x, y
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add training
git commit -m "Add streaming synthetic torch dataset"
```

---

### Task 6: U-Net model

**Files:**
- Create: `training/src/unduster_training/model.py`
- Test: `training/tests/test_model.py`

**Interfaces:**
- Produces: `UNet(in_ch=3, base=32, depth=4)` — `forward(x: Bx C x H x W) -> Bx1xHxW` logits, H and W divisible by `2**depth`. Used by Tasks 7, 8.

- [ ] **Step 1: Write the failing test**

`training/tests/test_model.py`:

```python
import pytest
import torch

from unduster_training.model import UNet


@pytest.mark.parametrize("in_ch", [1, 3])
def test_forward_shape(in_ch):
    m = UNet(in_ch=in_ch)
    x = torch.randn(2, in_ch, 64, 64)
    out = m(x)
    assert out.shape == (2, 1, 64, 64)


def test_param_budget():
    n = sum(p.numel() for p in UNet(in_ch=3).parameters())
    assert n < 8_000_000  # small enough for fast on-device tiled inference


def test_rejects_bad_size():
    with pytest.raises(AssertionError):
        UNet(in_ch=1)(torch.randn(1, 1, 60, 64))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`training/src/unduster_training/model.py`:

```python
"""Small U-Net for per-pixel defect probability."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        chans = [base * 2**i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        c = in_ch
        for ch in chans[:-1]:
            self.downs.append(DoubleConv(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chans[-2], chans[-1])
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for ch in reversed(chans[:-1]):
            self.ups.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
            self.up_convs.append(DoubleConv(ch * 2, ch))
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = 2**self.depth
        assert x.shape[-2] % f == 0 and x.shape[-1] % f == 0, f"H and W must be divisible by {f}"
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = conv(torch.cat([x, skip], dim=1))
        return self.head(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add training
git commit -m "Add small U-Net detector model"
```

---

### Task 7: Training loop and CLI

**Files:**
- Create: `training/src/unduster_training/train.py`
- Test: `training/tests/test_train.py`

**Interfaces:**
- Consumes: `UNet`, `SyntheticDefects`.
- Produces: `dice_loss(logits, target) -> Tensor`, `train_steps(model, loader, steps, lr, device) -> list[float]` (per-step losses), `save_checkpoint(model, variant, step, path)` / `load_checkpoint(path) -> tuple[UNet, dict]`; CLI `python -m unduster_training.train --variant colour --clean-dir D --library-dir L --steps N --batch B --out ckpt.pt`. Checkpoint format: `{"state_dict": ..., "variant": str, "in_ch": int, "base": int, "depth": int, "step": int}`. Used by Task 8.

- [ ] **Step 1: Write the failing test**

`training/tests/test_train.py`:

```python
import torch

from unduster_training.model import UNet
from unduster_training.train import dice_loss, load_checkpoint, save_checkpoint, train_steps


def test_dice_loss_bounds():
    logits = torch.full((2, 1, 16, 16), -10.0)
    target = torch.zeros(2, 1, 16, 16)
    assert dice_loss(logits, target) < 0.1
    target_all = torch.ones(2, 1, 16, 16)
    assert dice_loss(logits, target_all) > 0.9


def test_model_memorizes_tiny_batch():
    torch.manual_seed(0)
    x = torch.rand(2, 1, 32, 32)
    y = (torch.rand(2, 1, 32, 32) > 0.9).float()
    batch = [(x, y)] * 60
    model = UNet(in_ch=1, base=8, depth=2)
    losses = train_steps(model, batch, steps=60, lr=3e-3, device="cpu")
    assert losses[-1] < losses[0] * 0.5


def test_checkpoint_round_trip(tmp_path):
    model = UNet(in_ch=1, base=8, depth=2)
    p = tmp_path / "ck.pt"
    save_checkpoint(model, "bw", 123, p)
    back, meta = load_checkpoint(p)
    assert meta["variant"] == "bw" and meta["step"] == 123
    x = torch.rand(1, 1, 32, 32)
    model.eval(), back.eval()
    assert torch.allclose(model(x), back(x), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`training/src/unduster_training/train.py`:

```python
"""Train the defect detector on synthetic pairs."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import SyntheticDefects
from .model import UNet


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + 1.0
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    return (1.0 - num / den).mean()


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pos_weight = torch.tensor(8.0, device=logits.device)  # defects are rare pixels
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    return bce + dice_loss(logits, target)


def train_steps(model, batches, steps: int, lr: float, device: str) -> list[float]:
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    losses = []
    it = iter(batches)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(batches)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
    return losses


def save_checkpoint(model: UNet, variant: str, step: int, path: str | Path) -> None:
    first = model.downs[0].block[0]
    torch.save(
        {
            "state_dict": model.state_dict(),
            "variant": variant,
            "in_ch": first.in_channels,
            "base": first.out_channels,
            "depth": model.depth,
            "step": step,
        },
        path,
    )


def load_checkpoint(path: str | Path) -> tuple[UNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = UNet(in_ch=ck["in_ch"], base=ck["base"], depth=ck["depth"])
    model.load_state_dict(ck["state_dict"])
    meta = {k: ck[k] for k in ("variant", "in_ch", "base", "depth", "step")}
    return model, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["colour", "bw"], required=True)
    ap.add_argument("--clean-dir", required=True)
    ap.add_argument("--library-dir", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = pick_device()
    ds = SyntheticDefects(
        args.clean_dir, args.library_dir, variant=args.variant, patch=args.patch,
        length=args.steps * args.batch, seed=args.seed,
    )
    loader = DataLoader(ds, batch_size=args.batch, num_workers=4, persistent_workers=True)
    model = UNet(in_ch=1 if args.variant == "bw" else 3)
    print(f"training {args.variant} on {device} for {args.steps} steps")
    losses = train_steps(model, loader, steps=args.steps, lr=args.lr, device=device)
    print(f"final loss {losses[-1]:.4f}")
    save_checkpoint(model.cpu(), args.variant, args.steps, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train.py -v`
Expected: all PASS (memorization test takes under a minute on CPU).

- [ ] **Step 5: Commit**

```bash
git add training
git commit -m "Add training loop, checkpointing, and CLI"
```

---

### Task 8: ONNX export with parity check

**Files:**
- Create: `training/src/unduster_training/export.py`
- Test: `training/tests/test_export.py`

**Interfaces:**
- Consumes: `load_checkpoint`, `save_checkpoint`, `UNet`.
- Produces: `export_onnx(ckpt_path, onnx_path)`, `parity_gap(ckpt_path, onnx_path, size=512) -> float` (max abs sigmoid-prob difference); CLI `python -m unduster_training.export ckpt.pt model.onnx`. ONNX input `"image"` NCHW dynamic in N/H/W, output `"logits"`. Used by Task 9's `OnnxDetector` and later by `fd-infer`.

- [ ] **Step 1: Write the failing test**

`training/tests/test_export.py`:

```python
import torch

from unduster_training.export import export_onnx, parity_gap
from unduster_training.model import UNet
from unduster_training.train import save_checkpoint


def test_export_and_parity(tmp_path):
    torch.manual_seed(0)
    model = UNet(in_ch=1, base=8, depth=2)
    ckpt = tmp_path / "m.pt"
    onnx_path = tmp_path / "m.onnx"
    save_checkpoint(model, "bw", 1, ckpt)
    export_onnx(ckpt, onnx_path)
    assert onnx_path.exists()
    assert parity_gap(ckpt, onnx_path, size=64) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`training/src/unduster_training/export.py`:

```python
"""Export a trained checkpoint to ONNX and verify torch/ORT parity."""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .train import load_checkpoint


def export_onnx(ckpt_path: str | Path, onnx_path: str | Path) -> None:
    model, meta = load_checkpoint(ckpt_path)
    model.eval()
    dummy = torch.zeros(1, meta["in_ch"], 512, 512)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "n", 2: "h", 3: "w"}, "logits": {0: "n", 2: "h", 3: "w"}},
        opset_version=17,
    )


def parity_gap(ckpt_path: str | Path, onnx_path: str | Path, size: int = 512) -> float:
    model, meta = load_checkpoint(ckpt_path)
    model.eval()
    x = torch.rand(1, meta["in_ch"], size, size, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        want = torch.sigmoid(model(x)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"image": x.numpy()})[0]
    got = 1.0 / (1.0 + np.exp(-got))
    return float(np.abs(want - got).max())


def main() -> None:
    ckpt, out = sys.argv[1], sys.argv[2]
    export_onnx(ckpt, out)
    gap = parity_gap(ckpt, out)
    print(f"exported {out}, parity gap {gap:.2e}")
    if gap >= 1e-3:
        raise SystemExit("parity gap too large; do not ship this model")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_export.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add training
git commit -m "Export checkpoints to ONNX with parity check"
```

---

### Task 9: Detectors — classical baseline, tiled ONNX, competitor implied mask

**Files:**
- Create: `training/src/unduster_training/detectors.py`
- Test: `training/tests/test_detectors.py`

**Interfaces:**
- Consumes: `to_gray` (Task 1); ONNX files from Task 8.
- Produces: `classical_detect(img, ksize=5, k_sigma=4.0) -> np.ndarray bool HxW`; `OnnxDetector(path, threshold=0.5)` callable `(img) -> np.ndarray bool HxW` (tiles 512, overlap 64, edge-padded, prob-averaged in overlaps; converts channels to match the model); `implied_mask(before, after, thresh=0.004) -> np.ndarray bool HxW`. Used by Task 11.

- [ ] **Step 1: Write the failing test**

`training/tests/test_detectors.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_detectors.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`training/src/unduster_training/detectors.py`:

```python
"""Detectors the benchmark can score.

- classical_detect: median-residual thresholding, the Photoshop-era floor.
- OnnxDetector: tiled inference for our exported models or any external ONNX
  segmentation model (the published-weights baseline path).
- implied_mask: recover a competitor's effective mask from before/after pairs.

The tiling here (512px, 64 overlap, reflect-padded edges, averaged overlaps)
is the reference behaviour for the Rust fd-infer crate.
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .io import to_gray

TILE = 512
OVERLAP = 64


def classical_detect(img: np.ndarray, ksize: int = 5, k_sigma: float = 4.0) -> np.ndarray:
    g = to_gray(img)
    bg = cv2.medianBlur(g, ksize)
    diff = g - bg
    mad = np.median(np.abs(diff - np.median(diff)))
    sigma = max(1.4826 * mad, 1e-4)
    return np.abs(diff) > k_sigma * sigma


def implied_mask(before: np.ndarray, after: np.ndarray, thresh: float = 0.004) -> np.ndarray:
    return np.abs(to_gray(after) - to_gray(before)) > thresh


class OnnxDetector:
    def __init__(self, path: str | Path, threshold: float = 0.5):
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.in_ch = int(inp.shape[1])
        self.threshold = threshold

    def _prep(self, img: np.ndarray) -> np.ndarray:
        if self.in_ch == 1:
            g = to_gray(img)
            return g[None]  # 1xHxW
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        return img.transpose(2, 0, 1)  # 3xHxW

    def probabilities(self, img: np.ndarray) -> np.ndarray:
        x = self._prep(img).astype(np.float32)
        _, h, w = x.shape
        stride = TILE - OVERLAP
        acc = np.zeros((h, w), np.float32)
        weight = np.zeros((h, w), np.float32)
        for y0 in range(0, max(h - OVERLAP, 1), stride):
            for x0 in range(0, max(w - OVERLAP, 1), stride):
                y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
                tile = x[:, y0:y1, x0:x1]
                py, px = TILE - (y1 - y0), TILE - (x1 - x0)
                if py or px:
                    # edge mode: pad width may exceed axis size (reflect would raise)
                    tile = np.pad(tile, ((0, 0), (0, py), (0, px)), mode="edge")
                logits = self.sess.run(None, {self.input_name: tile[None]})[0][0, 0]
                prob = 1.0 / (1.0 + np.exp(-logits))
                acc[y0:y1, x0:x1] += prob[: y1 - y0, : x1 - x0]
                weight[y0:y1, x0:x1] += 1.0
        return acc / weight

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.probabilities(img) > self.threshold
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_detectors.py -v`
Expected: all PASS. Note `cv2.medianBlur` on float32 requires `ksize <= 5`; the default stays 5.

- [ ] **Step 5: Commit**

```bash
git add training
git commit -m "Add classical, tiled ONNX, and implied-mask detectors"
```

---

### Task 10: Per-defect-type metrics

**Files:**
- Create: `training/src/unduster_training/metrics.py`
- Test: `training/tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing beyond numpy/scipy.
- Produces: `KINDS = {1: "dust", 2: "scratch", 3: "hair"}`; `score_masks(pred: np.ndarray bool, gt_typed: np.ndarray uint8) -> dict` with keys `recall_dust`, `recall_scratch`, `recall_hair` (float or None when no GT of that type), `precision` (float), `tp`, `fn`, `fp` (ints). Matching rule: a GT component counts detected when at least 50% of its pixels are covered by `pred`; a predicted component overlapping zero GT pixels is a false positive. Used by Task 11.

- [ ] **Step 1: Write the failing test**

`training/tests/test_metrics.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`training/src/unduster_training/metrics.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add training
git commit -m "Add component-wise per-type precision and recall"
```

---

### Task 11: Benchmark harness

**Files:**
- Create: `training/src/unduster_training/benchmark.py`
- Test: `training/tests/test_benchmark.py`

**Interfaces:**
- Consumes: `load_image`, `score_masks`, `classical_detect`, `OnnxDetector`, `implied_mask`.
- Produces: `run_benchmark(frames_dir, labels_dir, detectors: dict[str, callable]) -> dict[str, dict]` (mean scores per detector across frames); `write_report(results, md_path, json_path)`; CLI `python -m unduster_training.benchmark --frames F --labels L [--onnx name=path ...] [--competitor name=after_dir ...] --out-dir reports/`. Labelled roll layout: `frames/NNNN.<ext>` and `labels/NNNN.png` (uint8 palette 0/1/2/3, same stem). Competitor dirs contain healed frames with matching stems; their masks are implied from before/after.
- Exit criteria for the whole sub-project: this harness, run on the real labelled roll, is the release gate described in the spec.

- [ ] **Step 1: Write the failing test**

`training/tests/test_benchmark.py`:

```python
import json

import numpy as np

from unduster_training.benchmark import run_benchmark, write_report
from unduster_training.io import save_image


def _make_roll(root):
    (root / "frames").mkdir()
    (root / "labels").mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        img = np.full((200, 200), 0.7, np.float32) + rng.normal(0, 0.003, (200, 200)).astype(np.float32)
        label = np.zeros((200, 200), np.uint8)
        img[50:54, 50:54] -= 0.4
        label[50:54, 50:54] = 1  # dust
        img[120, 20:120] -= 0.35
        label[120, 20:120] = 2  # scratch
        save_image(root / "frames" / f"{i:04d}.tif", np.clip(img, 0, 1))
        # labels saved raw as uint8 palette values
        import imageio.v3 as iio

        iio.imwrite(root / "labels" / f"{i:04d}.png", label)
    return root


def test_benchmark_scores_perfect_oracle(tmp_path):
    root = _make_roll(tmp_path)
    import imageio.v3 as iio

    def oracle(img, _cache={}):
        # cheat by reading the matching label; benchmark passes frames in sorted order
        i = len(_cache)
        _cache[i] = True
        return iio.imread(root / "labels" / f"{i:04d}.png") > 0

    results = run_benchmark(root / "frames", root / "labels", {"oracle": oracle})
    s = results["oracle"]
    assert s["recall_dust"] == 1.0 and s["recall_scratch"] == 1.0 and s["precision"] == 1.0


def test_report_files(tmp_path):
    root = _make_roll(tmp_path)
    from unduster_training.detectors import classical_detect

    results = run_benchmark(root / "frames", root / "labels", {"classical": classical_detect})
    write_report(results, tmp_path / "report.md", tmp_path / "report.json")
    assert "classical" in (tmp_path / "report.md").read_text()
    data = json.loads((tmp_path / "report.json").read_text())
    assert "classical" in data and "precision" in data["classical"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`training/src/unduster_training/benchmark.py`:

```python
"""Score detectors on the hand-labelled benchmark roll. The release gate."""

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from .detectors import OnnxDetector, classical_detect, implied_mask
from .io import load_image
from .metrics import KINDS, score_masks

_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _frames(frames_dir: Path) -> list[Path]:
    return sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in _EXTS)


def run_benchmark(frames_dir, labels_dir, detectors: dict) -> dict:
    frames_dir, labels_dir = Path(frames_dir), Path(labels_dir)
    per_detector: dict[str, list[dict]] = {name: [] for name in detectors}
    for frame_path in _frames(frames_dir):
        img = load_image(frame_path)
        gt = iio.imread(labels_dir / f"{frame_path.stem}.png").astype(np.uint8)
        for name, det in detectors.items():
            per_detector[name].append(score_masks(det(img), gt))
    results = {}
    for name, scores in per_detector.items():
        agg: dict = {"frames": len(scores)}
        for key in ["precision"] + [f"recall_{k}" for k in KINDS.values()]:
            vals = [s[key] for s in scores if s[key] is not None]
            agg[key] = float(np.mean(vals)) if vals else None
        for key in ("tp", "fn", "fp"):
            agg[key] = int(sum(s[key] for s in scores))
        results[name] = agg
    return results


def write_report(results: dict, md_path, json_path) -> None:
    Path(json_path).write_text(json.dumps(results, indent=2))
    cols = ["precision", "recall_dust", "recall_scratch", "recall_hair", "tp", "fn", "fp"]
    lines = ["| detector | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for name, s in results.items():
        cells = [f"{s[c]:.3f}" if isinstance(s[c], float) else str(s[c]) for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    Path(md_path).write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--onnx", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--competitor", action="append", default=[], metavar="NAME=AFTER_DIR")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    detectors: dict = {"classical": classical_detect}
    for spec in args.onnx:
        name, path = spec.split("=", 1)
        detectors[name] = OnnxDetector(path)
    for spec in args.competitor:
        name, after_dir = spec.split("=", 1)
        after = Path(after_dir)

        def comp_det(img, _after=after, _frames=iter(_frames(Path(args.frames)))):
            frame_path = next(_frames)
            healed = load_image(_after / frame_path.name)
            return implied_mask(img, healed)

        detectors[name] = comp_det

    results = run_benchmark(args.frames, args.labels, detectors)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_report(results, out / "benchmark.md", out / "benchmark.json")
    print((out / "benchmark.md").read_text())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -v`
Expected: every test in the package passes.

- [ ] **Step 6: Commit**

```bash
git add training
git commit -m "Add benchmark harness with markdown and JSON reports"
```

---

### Task 12: Data playbook and pipeline runbook

**Files:**
- Create: `training/README.md`
- Create: `training/DATA.md`
- Create: `.github/workflows/training.yml`

**Interfaces:**
- Consumes: every CLI from Tasks 2, 7, 8, 11.
- Produces: the human instructions for the physical work (scanning, labelling), the exact commands to run the real pipeline end to end, and CI that runs the test suite on every push touching `training/`. (The benchmark on the real labelled roll cannot run in hosted CI — the data is local and gitignored — so it is a documented local release gate, enforced by the report being committed alongside any model release.)

- [ ] **Step 1: Write the runbook**

`training/README.md`:

```markdown
# TheUnduster training pipeline

Offline pipeline: harvest real defects, synthesize training data, train the
detector, export ONNX, benchmark. Nothing here ships in the app except the
exported .onnx files.

## Setup

    mise install
    uv sync

## Pipeline (run from training/)

1. Harvest defects from blank film scans (see DATA.md for capture):

       uv run python -m unduster_training.harvest data/blank_scans data/library

2. Train both variants (clean images: any large, defect-free photo set;
   1000+ images recommended, mixed subjects):

       uv run python -m unduster_training.train --variant colour \
           --clean-dir data/clean --library-dir data/library \
           --steps 20000 --batch 8 --out data/ckpt/colour.pt
       uv run python -m unduster_training.train --variant bw \
           --clean-dir data/clean --library-dir data/library \
           --steps 20000 --batch 8 --out data/ckpt/bw.pt

3. Export to ONNX (fails loudly if torch/ORT disagree):

       uv run python -m unduster_training.export data/ckpt/colour.pt data/onnx/colour.onnx
       uv run python -m unduster_training.export data/ckpt/bw.pt data/onnx/bw.onnx

4. Benchmark against the labelled roll (see DATA.md for labelling):

       uv run python -m unduster_training.benchmark \
           --frames data/benchmark/frames --labels data/benchmark/labels \
           --onnx ours-colour=data/onnx/colour.onnx --onnx ours-bw=data/onnx/bw.onnx \
           --competitor retouch4me=data/benchmark/retouch4me \
           --out-dir reports

## Release gate

A model ships only if, on the labelled roll, it beats the classical baseline
and the competitor columns on precision AND per-type recall is no worse than
the previous shipped model. The benchmark report is the record.

## Tests

    uv run pytest
```

`training/DATA.md`:

```markdown
# Data capture and labelling

All of this lives under training/data/ (gitignored).

## Blank film scans -> data/blank_scans/

Purpose: harvest real dust, scratches, and hairs.

- Scan unexposed-but-developed film, blank leaders, and film edges.
  Do NOT clean them first. Dusty is the point.
- At least 20 strips, colour and B&W stocks mixed.
- 3200 dpi or higher, 16-bit TIFF, no scanner dust removal (ICE OFF),
  no sharpening, flat/linear profile if the software allows it.
- Name files freely; the harvester walks the whole directory.

## Clean images -> data/clean/

Any large set of defect-free photographs (your own digital photos work).
1000+ images, mixed subjects, some with fine detail (branches, birds,
stars, fabric) precisely because those are the false-positive traps.

## Benchmark roll -> data/benchmark/

- frames/: 20+ real scanned frames WITH their real defects, colour and
  B&W mixed, varied subjects including night skies and fine detail.
  Named 0001.tif, 0002.tif, ...
- labels/: for each frame, 0001.png etc (same stem): uint8, single
  channel, pixel values 0 background, 1 dust, 2 scratch, 3 hair.
  Paint in any editor over the frame; exact edges matter less than
  covering each defect's core (detection is scored at 50% coverage).
- retouch4me/ (optional): the same frames healed by the competitor,
  same filenames. The harness recovers their mask by before/after diff.

The labelled roll is sacred: never train on it, never tune on it by eye.
```

`.github/workflows/training.yml`:

```yaml
name: training

on:
  push:
    paths:
      - "training/**"
      - ".github/workflows/training.yml"
  pull_request:
    paths:
      - "training/**"

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: training
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      - run: uv sync
      - run: uv run pytest -v
```

- [ ] **Step 2: Verify the documented commands match the code**

Run: `uv run python -m unduster_training.harvest --help 2>&1 | head -2; uv run python -m unduster_training.train --help | head -3; uv run python -m unduster_training.export --help 2>&1 | head -2; uv run python -m unduster_training.benchmark --help | head -3`
Expected: harvest and export print usage errors or run (they take positional args); train and benchmark print argparse help. No ImportError anywhere.

- [ ] **Step 3: Commit**

```bash
git add training
git commit -m "Document data capture, labelling, and pipeline runbook"
```

---

## Definition of done for this sub-project

- `uv run pytest` green in `training/`.
- All CLIs run end to end on fixture-scale data.
- The real-data steps (scanning, labelling, first real training run) are human work guided by DATA.md — they are not blockers for starting sub-project 2 (core engine), which consumes the ONNX contract defined in Task 8 and the tiling contract defined in Task 9.

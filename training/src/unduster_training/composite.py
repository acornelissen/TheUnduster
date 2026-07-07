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

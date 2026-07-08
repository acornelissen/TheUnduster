"""Detectors the benchmark can score.

- classical_detect: median-residual thresholding, the Photoshop-era floor.
- OnnxDetector: tiled inference for our exported models or any external ONNX
  segmentation model (the published-weights baseline path).
- implied_mask: recover a competitor's effective mask from before/after pairs.

The tiling here (512px, 64 overlap, edge-padded borders, averaged overlaps)
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
        ch = inp.shape[1]
        if not isinstance(ch, int) or ch not in (1, 3):
            raise ValueError(
                f"model {path} has channel dimension {ch!r}; detectors need a "
                f"fixed 1- or 3-channel input"
            )
        self.in_ch = ch
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

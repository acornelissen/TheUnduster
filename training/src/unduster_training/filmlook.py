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

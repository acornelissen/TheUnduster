"""Image I/O. In-memory convention: float32 in [0, 1], HxW or HxWx3."""

from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".png":
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw.ndim == 3 and raw.shape[2] >= 3:
            raw = raw[..., :3][..., ::-1]
    else:
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
        iio.imwrite(path, out)
    elif bit_depth == 16:
        out = (img * 65535.0 + 0.5).astype(np.uint16)
        if path.suffix.lower() == ".png":
            if out.ndim == 3:
                out = out[..., ::-1]
            cv2.imwrite(str(path), out)
        else:
            iio.imwrite(path, out)
    else:
        raise ValueError(f"unsupported bit depth: {bit_depth}")


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    w = np.array([0.2126, 0.7152, 0.0722], np.float32)  # Rec.709
    return img @ w

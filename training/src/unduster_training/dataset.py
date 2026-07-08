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

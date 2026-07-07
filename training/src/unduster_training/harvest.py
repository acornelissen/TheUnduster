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

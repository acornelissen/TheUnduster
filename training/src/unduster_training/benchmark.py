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


def competitor_detector(frames_dir, after_dir):
    """Detector that recovers a competitor's mask from their healed output.

    Pairs frames by stem: for frame 0001.tif it loads after_dir/0001.* and
    diffs. Stateless per call except an internal position counter matched to
    _frames() ordering, so it must be used with run_benchmark over the same
    frames_dir.
    """
    frames = _frames(Path(frames_dir))
    state = {"i": 0}

    def detect(img):
        frame_path = frames[state["i"]]
        state["i"] += 1
        matches = sorted(Path(after_dir).glob(f"{frame_path.stem}.*"))
        if not matches:
            raise FileNotFoundError(
                f"no healed frame for {frame_path.stem} in {after_dir}"
            )
        healed = load_image(matches[0])
        return implied_mask(img, healed)

    return detect


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
        detectors[name] = competitor_detector(args.frames, after_dir)

    results = run_benchmark(args.frames, args.labels, detectors)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_report(results, out / "benchmark.md", out / "benchmark.json")
    print((out / "benchmark.md").read_text())


if __name__ == "__main__":
    main()

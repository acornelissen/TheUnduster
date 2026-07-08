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


def test_report_renders_none_cells(tmp_path):
    root = _make_roll(tmp_path)
    from unduster_training.detectors import classical_detect

    results = run_benchmark(root / "frames", root / "labels", {"classical": classical_detect})
    write_report(results, tmp_path / "report.md", tmp_path / "report.json")
    md = (tmp_path / "report.md").read_text()
    row = next(line for line in md.splitlines() if line.startswith("| classical"))
    assert "None" in row  # recall_hair: fixture has no hair labels
    assert row.count("|") == 9  # header has 8 columns + edges: table stays well-formed
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["classical"]["recall_hair"] is None


def test_competitor_detector_scores_partial_heal(tmp_path):
    root = _make_roll(tmp_path)
    after = root / "competitor"
    after.mkdir()
    import imageio.v3 as iio
    from unduster_training.benchmark import competitor_detector
    from unduster_training.io import load_image, save_image

    for f in sorted((root / "frames").iterdir()):
        img = load_image(f)
        label = iio.imread(root / "labels" / f"{f.stem}.png")
        healed = img.copy()
        healed[label == 1] = 0.7  # competitor heals dust only, misses scratches
        save_image(after / f.name, healed)

    det = competitor_detector(root / "frames", after)
    results = run_benchmark(root / "frames", root / "labels", {"comp": det})
    assert results["comp"]["recall_dust"] == 1.0
    assert results["comp"]["recall_scratch"] == 0.0

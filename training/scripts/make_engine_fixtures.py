"""Generate committed fixtures the Rust engine tests use.

Run from training/:  uv run python scripts/make_engine_fixtures.py
Deterministic: fixed seeds, no timestamps. Overwrites engine/fixtures/.
"""

import json
from pathlib import Path

import numpy as np
import onnx
import torch

from unduster_training.detectors import OnnxDetector

OUT = Path(__file__).resolve().parents[2] / "engine" / "fixtures"
WIDTH, HEIGHT = 600, 540  # both > 512: forces 2x2 tile grid with overlaps


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.c1 = torch.nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = torch.nn.Conv2d(8, 1, 3, padding=1)

    def forward(self, x):
        return self.c2(torch.relu(self.c1(x)))


class TinyInpaint(torch.nn.Module):
    def forward(self, image, mask):
        mean = image.mean(dim=(2, 3), keepdim=True)
        return image * (1.0 - mask) + mean * mask


def export_detector() -> None:
    model = TinyDetector().eval()
    torch.onnx.export(
        model,
        (torch.zeros(1, 1, 512, 512),),
        str(OUT / "tiny-detector.onnx"),
        input_names=["image"],
        output_names=["logits"],
        dynamic_shapes={"x": {2: "h", 3: "w"}},
        opset_version=17,
        dynamo=True,
    )


def export_inpaint() -> None:
    model = TinyInpaint().eval()
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, 64, 64), torch.zeros(1, 1, 64, 64)),
        str(OUT / "tiny-inpaint.onnx"),
        input_names=["image", "mask"],
        output_names=["output"],
        dynamic_shapes={"image": {2: "h", 3: "w"}, "mask": {2: "h", 3: "w"}},
        opset_version=17,
        dynamo=True,
    )


def embed_weights(path: Path) -> None:
    """The dynamo exporter writes weights to a .data sidecar; fixtures must
    be single self-contained files, so re-save with embedded tensors."""
    model = onnx.load(str(path))
    onnx.save(model, str(path))
    path.with_suffix(".onnx.data").unlink(missing_ok=True)


def make_parity() -> None:
    rng = np.random.default_rng(11)
    img_u16 = (rng.random((HEIGHT, WIDTH)) * 65535.0).astype("<u2")
    (OUT / "parity-input.bin").write_bytes(img_u16.tobytes())
    img_f32 = img_u16.astype(np.float32) / 65535.0
    det = OnnxDetector(OUT / "tiny-detector.onnx")
    probs = det.probabilities(img_f32)
    probs_u16 = np.round(probs * 65535.0).astype("<u2")
    (OUT / "parity-expected.bin").write_bytes(probs_u16.tobytes())
    (OUT / "parity-meta.json").write_text(
        json.dumps({"width": WIDTH, "height": HEIGHT, "tolerance": 0.002}, indent=2)
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    export_detector()
    export_inpaint()
    embed_weights(OUT / "tiny-detector.onnx")
    embed_weights(OUT / "tiny-inpaint.onnx")
    make_parity()
    for f in sorted(OUT.iterdir()):
        print(f"{f.name}: {f.stat().st_size} bytes")


if __name__ == "__main__":
    main()

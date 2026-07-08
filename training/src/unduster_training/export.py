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
        dynamo=False,
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

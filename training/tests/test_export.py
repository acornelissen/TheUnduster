import torch

from unduster_training.export import export_onnx, parity_gap
from unduster_training.model import UNet
from unduster_training.train import save_checkpoint


def test_export_and_parity(tmp_path):
    torch.manual_seed(0)
    model = UNet(in_ch=1, base=8, depth=2)
    ckpt = tmp_path / "m.pt"
    onnx_path = tmp_path / "m.onnx"
    save_checkpoint(model, "bw", 1, ckpt)
    export_onnx(ckpt, onnx_path)
    assert onnx_path.exists()
    assert parity_gap(ckpt, onnx_path, size=64) < 1e-3

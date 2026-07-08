import onnx
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


def test_export_graph_contract(tmp_path):
    """Freezes the ONNX I/O contract the Rust fd-infer crate depends on."""
    torch.manual_seed(0)
    model = UNet(in_ch=1, base=8, depth=2)
    ckpt = tmp_path / "m.pt"
    onnx_path = tmp_path / "m.onnx"
    save_checkpoint(model, "bw", 1, ckpt)
    export_onnx(ckpt, onnx_path)

    graph = onnx.load(str(onnx_path))

    assert graph.graph.input[0].name == "image"
    assert graph.graph.output[0].name == "logits"

    dims = graph.graph.input[0].type.tensor_type.shape.dim
    assert dims[0].HasField("dim_param")
    assert dims[2].HasField("dim_param")
    assert dims[3].HasField("dim_param")
    assert dims[1].HasField("dim_value")
    assert dims[1].dim_value == 1

    opset_versions = {
        imp.version for imp in graph.opset_import if imp.domain in ("", "ai.onnx")
    }
    assert 17 in opset_versions

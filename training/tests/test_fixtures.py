"""Tests for the engine fixture generator (training/scripts/make_engine_fixtures.py).

The generator lives under scripts/, not the installed package, so it is
loaded here by file path.
"""

import importlib.util
from pathlib import Path

import numpy as np
import onnxruntime

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "make_engine_fixtures.py"
)


def _load_make_engine_fixtures():
    spec = importlib.util.spec_from_file_location(
        "make_engine_fixtures", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_engine_fixtures = _load_make_engine_fixtures()
make_fixed_inpaint_fixture = make_engine_fixtures.make_fixed_inpaint_fixture


def test_fixed_inpaint_fixture_contract(tmp_path):
    path = tmp_path / "tiny-inpaint-fixed.onnx"
    make_fixed_inpaint_fixture(path, size=64)
    sess = onnxruntime.InferenceSession(str(path))
    (image_in, mask_in) = sess.get_inputs()
    assert image_in.name == "image"
    assert image_in.shape == [1, 3, 64, 64]  # static, no symbolic dims
    assert mask_in.shape == [1, 1, 64, 64]
    rng = np.random.default_rng(7)
    image = rng.random((1, 3, 64, 64), dtype=np.float32)
    mask = np.zeros((1, 1, 64, 64), dtype=np.float32)
    mask[0, 0, 20:40, 20:40] = 1.0
    (out,) = sess.run(None, {"image": image, "mask": mask})
    assert out.shape == (1, 3, 64, 64)
    # output is 0-255 scaled (LaMa contract): unmasked pixels = 255 * input
    np.testing.assert_allclose(
        out[0, :, 0, 0], 255.0 * image[0, :, 0, 0], rtol=1e-4
    )
    # masked pixels are the per-channel mean, scaled
    np.testing.assert_allclose(
        out[0, 0, 30, 30], 255.0 * image[0, 0].mean(), rtol=1e-3
    )

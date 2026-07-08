import numpy as np
import pytest

from unduster_training.io import load_image, save_image, to_gray


@pytest.mark.parametrize("ext,bit_depth", [("tif", 16), ("tif", 8), ("png", 16), ("png", 8), ("jpg", 8)])
def test_round_trip(tmp_path, ext, bit_depth):
    # smooth gradient, not noise: JPEG on noise has unbounded per-pixel error
    ramp = np.linspace(0, 1, 64 * 48, dtype=np.float32).reshape(64, 48)
    img = np.stack([ramp, ramp * 0.5, 1.0 - ramp], axis=-1)
    path = tmp_path / f"x.{ext}"
    save_image(path, img, bit_depth=bit_depth)
    back = load_image(path)
    assert back.dtype == np.float32
    assert back.shape == (64, 48, 3)
    assert back.min() >= 0.0 and back.max() <= 1.0
    tol = 0.05 if ext == "jpg" else (1 / 255 if bit_depth == 8 else 1 / 65535) + 1e-6
    assert np.abs(back - img).max() <= tol


def test_png_files_are_real_pngs(tmp_path):
    img = np.linspace(0, 1, 32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3)
    for bit_depth in (8, 16):
        path = tmp_path / f"x{bit_depth}.png"
        save_image(path, img, bit_depth=bit_depth)
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        back = load_image(path)
        assert back.shape == (32, 32, 3)
        tol = (1 / 255 if bit_depth == 8 else 1 / 65535) + 1e-6
        assert np.abs(back - img).max() <= tol


def test_gray_round_trip(tmp_path):
    img = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    path = tmp_path / "g.tif"
    save_image(path, img, bit_depth=16)
    back = load_image(path)
    assert back.shape == (32, 32)
    assert np.abs(back - img).max() <= 1 / 65535 + 1e-6


@pytest.mark.parametrize("bit_depth", [8, 16])
def test_gray_png_round_trip(tmp_path, bit_depth):
    # pins the cv2 grayscale PNG path at both depths
    img = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    path = tmp_path / "g.png"
    save_image(path, img, bit_depth=bit_depth)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    back = load_image(path)
    assert back.shape == (32, 32)
    tol = (1 / 255 if bit_depth == 8 else 1 / 65535) + 1e-6
    assert np.abs(back - img).max() <= tol


def test_load_image_rejects_corrupt_png(tmp_path):
    path = tmp_path / "x.png"
    path.write_bytes(b"not a real png")
    with pytest.raises(ValueError, match="x.png"):
        load_image(path)


def test_to_gray():
    img = np.zeros((4, 4, 3), np.float32)
    img[..., 1] = 1.0
    g = to_gray(img)
    assert g.shape == (4, 4)
    assert 0.5 < g[0, 0] < 0.8  # Rec.709 green weight
    assert to_gray(g) is g  # already grey passes through

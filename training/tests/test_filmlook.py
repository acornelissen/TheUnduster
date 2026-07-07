import numpy as np

from unduster_training.filmlook import add_grain, apply_random_curve


def test_grain_adds_midtone_noise():
    rng = np.random.default_rng(0)
    img = np.full((256, 256), 0.5, np.float32)
    out = add_grain(img, rng, strength=0.05)
    assert out.shape == img.shape
    assert out.min() >= 0 and out.max() <= 1
    assert 0.02 < out.std() < 0.09
    assert abs(out.mean() - 0.5) < 0.01


def test_grain_spares_extremes():
    rng = np.random.default_rng(0)
    dark = add_grain(np.full((256, 256), 0.02, np.float32), rng, strength=0.05)
    mid = add_grain(np.full((256, 256), 0.5, np.float32), rng, strength=0.05)
    assert dark.std() < mid.std() * 0.4


def test_grain_colour_shape():
    rng = np.random.default_rng(0)
    img = np.full((64, 64, 3), 0.5, np.float32)
    assert add_grain(img, rng).shape == (64, 64, 3)


def test_curve_is_monotonic_and_bounded():
    rng = np.random.default_rng(0)
    ramp = np.linspace(0, 1, 1024, dtype=np.float32)
    for _ in range(20):
        out = apply_random_curve(ramp, rng)
        assert out.min() >= 0 and out.max() <= 1
        assert (np.diff(out) >= -1e-6).all()

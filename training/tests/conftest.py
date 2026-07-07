import numpy as np
import pytest


def make_blank_scan(h=400, w=400, base=0.82, noise=0.004, seed=1):
    """Uniform 'blank film' scan with known painted defects.

    Returns (scan_gray, truth) where truth lists (kind, cy, cx).
    """
    rng = np.random.default_rng(seed)
    img = np.full((h, w), base, np.float32) + rng.normal(0, noise, (h, w)).astype(np.float32)
    truth = []
    # dust: compact dark blobs
    for cy, cx, r in [(60, 60, 3), (150, 300, 5), (320, 80, 2)]:
        yy, xx = np.ogrid[:h, :w]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        img[blob] -= 0.35
        truth.append(("dust", cy, cx))
    # scratch: straight thin dark line
    for x in range(100, 300):
        img[200:202, x] -= 0.3
    truth.append(("scratch", 200, 200))
    # hair: curved thin line
    for t in np.linspace(0, 2 * np.pi, 800):
        y = int(300 + 25 * np.sin(t) * np.cos(3 * t))
        x = int(50 + 180 * t / (2 * np.pi) + 25 * np.sin(2 * t))
        img[y : y + 2, x : x + 2] -= 0.3
    truth.append(("hair", 300, 150))
    return np.clip(img, 0, 1), truth


@pytest.fixture
def blank_scan():
    return make_blank_scan()

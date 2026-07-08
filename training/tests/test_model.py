import pytest
import torch

from unduster_training.model import UNet


@pytest.mark.parametrize("in_ch", [1, 3])
def test_forward_shape(in_ch):
    m = UNet(in_ch=in_ch)
    x = torch.randn(2, in_ch, 64, 64)
    out = m(x)
    assert out.shape == (2, 1, 64, 64)


def test_param_budget():
    n = sum(p.numel() for p in UNet(in_ch=3).parameters())
    assert n < 8_000_000  # small enough for fast on-device tiled inference


def test_rejects_bad_size():
    with pytest.raises(AssertionError):
        UNet(in_ch=1)(torch.randn(1, 1, 60, 64))

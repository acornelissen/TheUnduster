import torch

from unduster_training.model import UNet
from unduster_training.train import dice_loss, load_checkpoint, save_checkpoint, train_steps


def test_dice_loss_bounds():
    logits = torch.full((2, 1, 16, 16), -10.0)
    target = torch.zeros(2, 1, 16, 16)
    assert dice_loss(logits, target) < 0.1
    target_all = torch.ones(2, 1, 16, 16)
    assert dice_loss(logits, target_all) > 0.9


def test_model_memorizes_tiny_batch():
    torch.manual_seed(0)
    x = torch.rand(2, 1, 32, 32)
    y = (torch.rand(2, 1, 32, 32) > 0.9).float()
    batch = [(x, y)] * 60
    model = UNet(in_ch=1, base=8, depth=2)
    losses = train_steps(model, batch, steps=60, lr=3e-3, device="cpu")
    assert losses[-1] < losses[0] * 0.5


def test_checkpoint_round_trip(tmp_path):
    model = UNet(in_ch=1, base=8, depth=2)
    p = tmp_path / "ck.pt"
    save_checkpoint(model, "bw", 123, p)
    back, meta = load_checkpoint(p)
    assert meta["variant"] == "bw" and meta["step"] == 123
    x = torch.rand(1, 1, 32, 32)
    model.eval(), back.eval()
    assert torch.allclose(model(x), back(x), atol=1e-6)

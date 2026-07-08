"""Train the defect detector on synthetic pairs."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import SyntheticDefects
from .model import UNet


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + 1.0
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    return (1.0 - num / den).mean()


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pos_weight = torch.tensor(8.0, device=logits.device)  # defects are rare pixels
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    return bce + dice_loss(logits, target)


def train_steps(model, batches, steps: int, lr: float, device: str) -> list[float]:
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    losses = []
    it = iter(batches)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(batches)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
    return losses


def save_checkpoint(model: UNet, variant: str, step: int, path: str | Path) -> None:
    first = model.downs[0].block[0]
    torch.save(
        {
            "state_dict": model.state_dict(),
            "variant": variant,
            "in_ch": first.in_channels,
            "base": first.out_channels,
            "depth": model.depth,
            "step": step,
        },
        path,
    )


def load_checkpoint(path: str | Path) -> tuple[UNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = UNet(in_ch=ck["in_ch"], base=ck["base"], depth=ck["depth"])
    model.load_state_dict(ck["state_dict"])
    meta = {k: ck[k] for k in ("variant", "in_ch", "base", "depth", "step")}
    return model, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["colour", "bw"], required=True)
    ap.add_argument("--clean-dir", required=True)
    ap.add_argument("--library-dir", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = pick_device()
    ds = SyntheticDefects(
        args.clean_dir, args.library_dir, variant=args.variant, patch=args.patch,
        length=args.steps * args.batch, seed=args.seed,
    )
    loader = DataLoader(ds, batch_size=args.batch, num_workers=4, persistent_workers=True)
    model = UNet(in_ch=1 if args.variant == "bw" else 3)
    print(f"training {args.variant} on {device} for {args.steps} steps")
    losses = train_steps(model, loader, steps=args.steps, lr=args.lr, device=device)
    print(f"final loss {losses[-1]:.4f}")
    save_checkpoint(model.cpu(), args.variant, args.steps, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

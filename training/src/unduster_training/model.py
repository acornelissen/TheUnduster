"""Small U-Net for per-pixel defect probability."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        chans = [base * 2**i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        c = in_ch
        for ch in chans[:-1]:
            self.downs.append(DoubleConv(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chans[-2], chans[-1])
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for ch in reversed(chans[:-1]):
            self.ups.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
            self.up_convs.append(DoubleConv(ch * 2, ch))
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = 2**self.depth
        assert x.shape[-2] % f == 0 and x.shape[-1] % f == 0, f"H and W must be divisible by {f}"
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = conv(torch.cat([x, skip], dim=1))
        return self.head(x)

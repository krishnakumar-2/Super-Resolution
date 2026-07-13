import torch
import torch.nn as nn
import torch.nn.functional as F


def velocity_from_vorticity(omega):
    B, _, H, W = omega.shape
    w = omega[:, 0].float()
    kx = torch.fft.fftfreq(H, d=1.0).view(1, H, 1).to(w.device) * 2 * torch.pi
    ky = (
        torch.fft.rfftfreq(W, d=1.0).view(1, 1, -1).to(w.device) * 2 * torch.pi
    )
    k2 = (kx**2 + ky**2).clone()
    k2[..., 0, 0] = 1.0
    w_ft = torch.fft.rfft2(w)
    psi_ft = w_ft / k2
    u = torch.fft.irfft2(1j * ky * psi_ft, s=(H, W))
    v = torch.fft.irfft2(-1j * kx * psi_ft, s=(H, W))
    return torch.stack([u, v], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, padding_mode="circular"),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1, padding_mode="circular"),
        )

    def forward(self, x):
        return x + self.body(x)


class VortSR(nn.Module):
    def __init__(self, hr_res=128, channels=64, n_blocks=8, lr_res=32):
        super().__init__()
        assert hr_res % lr_res == 0, "hr_res must be a multiple of lr_res"
        self.hr_res = hr_res
        self.head = nn.Conv2d(
            1, channels, 3, padding=1, padding_mode="circular"
        )
        self.body = nn.Sequential(
            *[ResBlock(channels) for _ in range(n_blocks)]
        )

        up, s = [], hr_res // lr_res
        while s > 1:
            f = 2 if s % 2 == 0 else s
            up += [
                nn.Conv2d(
                    channels,
                    channels * f * f,
                    3,
                    padding=1,
                    padding_mode="circular",
                ),
                nn.PixelShuffle(f),
            ]
            s //= f
        self.upsample = nn.Sequential(*up)

        self.tail = nn.Conv2d(
            channels, 1, 3, padding=1, padding_mode="circular"
        )
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, lr):
        up = F.interpolate(
            lr,
            size=(self.hr_res, self.hr_res),
            mode="bicubic",
            align_corners=False,
        )
        z = self.head(lr)
        z = self.body(z)
        z = self.upsample(z)
        omega = up + self.tail(z)
        vel = velocity_from_vorticity(omega)
        return omega, vel

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

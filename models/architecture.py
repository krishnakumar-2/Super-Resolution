import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LiftedEncoder(nn.Module):
    def __init__(self, in_channels=2, latent_channels=64, target_res=64):
        super().__init__()
        self.target_res = target_res

        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.GELU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.GELU(),
        )

    def forward(self, x):
        z = self.lift(x)
        z = F.interpolate(z, size=(self.target_res, self.target_res), mode='bicubic', align_corners=False)
        return z


class SpectralTransportBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.radial_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        self.weight = nn.Parameter(
            torch.randn(channels, channels, dtype=torch.cfloat) * 0.02
        )

        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        x_ft = torch.fft.rfft2(x)
        _, _, Hf, Wf = x_ft.shape

        freq_x = torch.fft.fftfreq(H, d=1.0).to(x.device)
        freq_y = torch.fft.rfftfreq(W, d=1.0).to(x.device)
        KX, KY = torch.meshgrid(freq_x, freq_y, indexing='ij')
        k_rad = torch.sqrt(KX ** 2 + KY ** 2)

        k_max = k_rad.max().clamp(min=1e-8)
        k_norm = (k_rad / k_max).unsqueeze(-1)

        gain = self.radial_mlp(k_norm).squeeze(-1)
        gain = gain.unsqueeze(0).unsqueeze(0)

        x_ft_mod = x_ft * gain
        x_ft_mixed = torch.einsum('bchw,dc->bdhw', x_ft_mod, self.weight)
        x_spectral = torch.fft.irfft2(x_ft_mixed, s=(H, W))

        return x_spectral + self.residual(x)


class DeRhamProjectionHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.to_stream = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, z):
        B, C, H, W = z.shape

        psi = self.to_stream(z).squeeze(1)
        psi_ft = torch.fft.rfft2(psi)

        kx = torch.fft.fftfreq(H, d=1.0).to(z.device)
        ky = torch.fft.rfftfreq(W, d=1.0).to(z.device)
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')

        u_ft = (2j * torch.pi * KY.unsqueeze(0)) * psi_ft
        v_ft = (-2j * torch.pi * KX.unsqueeze(0)) * psi_ft

        u = torch.fft.irfft2(u_ft, s=(H, W))
        v = torch.fft.irfft2(v_ft, s=(H, W))

        return torch.stack([u, v], dim=-1)


class DR_STO(nn.Module):
    def __init__(self, in_channels=2, latent_channels=64, lr_res=16, hr_res=64):
        super().__init__()
        self.encoder  = LiftedEncoder(in_channels, latent_channels, hr_res)
        self.sotb     = SpectralTransportBlock(latent_channels)
        self.dec_head = DeRhamProjectionHead(latent_channels)

    def forward(self, x):
        z   = self.encoder(x)
        z   = self.sotb(z)
        out = self.dec_head(z)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


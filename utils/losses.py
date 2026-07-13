import torch
import torch.nn as nn


def rel_l2(pred, target):
    B = pred.shape[0]
    d = (pred - target).reshape(B, -1)
    t = target.reshape(B, -1)
    return torch.mean(torch.norm(d, dim=1) / (torch.norm(t, dim=1) + 1e-8))


def _radial_E(field):
    f = field.permute(0, 3, 1, 2) if field.shape[-1] in (1, 2) else field
    H, W = f.shape[-2], f.shape[-1]
    e = (torch.fft.fft2(f.float()).abs() ** 2).mean(0).sum(0)
    kx = torch.fft.fftfreq(H, d=1.0 / H).to(field.device)
    ky = torch.fft.fftfreq(W, d=1.0 / W).to(field.device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    k = torch.sqrt(KX**2 + KY**2).round().long().flatten()
    shells = torch.zeros(int(k.max()) + 1, device=field.device)
    shells.scatter_add_(0, k, e.flatten())
    return shells


def spectral_loss(pred_v, target_v):
    return torch.mean(
        (
            torch.log(_radial_E(pred_v) + 1e-8)
            - torch.log(_radial_E(target_v) + 1e-8)
        )
        ** 2
    )


class SRLoss(nn.Module):
    def __init__(self, a_spec=0.1, g_vel=1.0):
        super().__init__()
        self.a = a_spec
        self.g = g_vel

    def forward(self, omega, vel, v_hr):
        lv = rel_l2(vel, v_hr)
        ls = spectral_loss(vel, v_hr)
        total = self.g * lv + self.a * ls
        return total, {"v": lv.item(), "spec": ls.item()}

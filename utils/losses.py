import torch
import torch.nn as nn
import numpy as np


def relative_l2_loss(pred, target):
    diff = (pred - target).reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    return torch.mean(
        torch.norm(diff, dim=1) / (torch.norm(target_flat, dim=1) + 1e-8)
    )


def wasserstein1_spectral_loss(pred, target):
    B, H, W, _ = pred.shape

    pred_f   = pred.permute(0, 3, 1, 2)
    target_f = target.permute(0, 3, 1, 2)

    pred_psd   = torch.mean(torch.abs(torch.fft.rfft2(pred_f))   ** 2, dim=(0, 1))
    target_psd = torch.mean(torch.abs(torch.fft.rfft2(target_f)) ** 2, dim=(0, 1))

    pred_psd   = pred_psd.flatten()
    target_psd = target_psd.flatten()

    pred_psd   = pred_psd   / (pred_psd.sum()   + 1e-8)
    target_psd = target_psd / (target_psd.sum() + 1e-8)

    pred_cdf   = torch.cumsum(pred_psd,   dim=0)
    target_cdf = torch.cumsum(target_psd, dim=0)

    return torch.mean(torch.abs(pred_cdf - target_cdf))


class DRSTOLoss(nn.Module):
    def __init__(self, lambda_sot=0.1):
        super().__init__()
        self.lambda_sot = lambda_sot

    def forward(self, pred, target):
        l2_loss       = relative_l2_loss(pred, target)
        spectral_loss = wasserstein1_spectral_loss(pred, target)
        total         = l2_loss + self.lambda_sot * spectral_loss

        return total, {'l2': l2_loss.item(), 'spectral': spectral_loss.item()}

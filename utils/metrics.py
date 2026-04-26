import numpy as np
import torch


@torch.no_grad()
def relative_l2(pred, target):
    diff = (pred - target).reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    return (
        torch.norm(diff, dim=1) / (torch.norm(target_flat, dim=1) + 1e-8)
    ).mean().item()


@torch.no_grad()
def compute_divergence(uv):
    B, H, W, _ = uv.shape
    u = uv[..., 0]
    v = uv[..., 1]

    u_ft = torch.fft.rfft2(u)
    v_ft = torch.fft.rfft2(v)

    kx = torch.fft.fftfreq(H, d=1.0).to(uv.device)
    ky = torch.fft.rfftfreq(W, d=1.0).to(uv.device)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')

    div_ft = (2j * torch.pi * KX.unsqueeze(0) * u_ft
            + 2j * torch.pi * KY.unsqueeze(0) * v_ft)

    div = torch.fft.irfft2(div_ft, s=(H, W))
    return div


@torch.no_grad()
def max_divergence(uv):
    div = compute_divergence(uv)
    return div.abs().max().item()


@torch.no_grad()
def radial_energy_spectrum(uv, n_bins=32):
    B, H, W, _ = uv.shape
    uv_f = uv.permute(0, 3, 1, 2)

    psd = 0.5 * torch.mean(
        torch.abs(torch.fft.fft2(uv_f)) ** 2,
        dim=(0, 1)
    )

    kx = torch.fft.fftfreq(H, d=1.0 / H).numpy()
    ky = torch.fft.fftfreq(W, d=1.0 / W).numpy()
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX ** 2 + KY ** 2)

    psd_np = psd.cpu().numpy()
    k_max  = K.max()
    bins   = np.linspace(0, k_max, n_bins + 1)
    k_bins = 0.5 * (bins[:-1] + bins[1:])

    E_k = np.zeros(n_bins)
    for i in range(n_bins):
        mask   = (K >= bins[i]) & (K < bins[i + 1])
        E_k[i] = psd_np[mask].mean() if mask.any() else 0.0

    return k_bins, E_k


@torch.no_grad()
def vorticity_pdf(uv, n_bins=100):
    B, H, W, _ = uv.shape
    u = uv[..., 0]
    v = uv[..., 1]

    u_ft = torch.fft.rfft2(u)
    v_ft = torch.fft.rfft2(v)

    kx = torch.fft.fftfreq(H, d=1.0).to(uv.device)
    ky = torch.fft.rfftfreq(W, d=1.0).to(uv.device)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')

    omega_ft = 2j * torch.pi * (KX.unsqueeze(0) * v_ft - KY.unsqueeze(0) * u_ft)
    omega    = torch.fft.irfft2(omega_ft, s=(H, W))

    omega_np = omega.cpu().numpy().flatten()
    counts, edges = np.histogram(omega_np, bins=n_bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, counts


@torch.no_grad()
def evaluate_loader(model, loader, device, n_spectral_samples=64):
    model.eval()
    all_rel_l2  = []
    all_max_div = []
    pred_list, target_list = [], []

    for u_lr, u_hr in loader:
        u_lr = u_lr.to(device)
        u_hr = u_hr.to(device)
        pred = model(u_lr)

        all_rel_l2.append(relative_l2(pred, u_hr))
        all_max_div.append(max_divergence(pred))

        if len(pred_list) * u_lr.shape[0] < n_spectral_samples:
            pred_list.append(pred.cpu())
            target_list.append(u_hr.cpu())

    pred_cat   = torch.cat(pred_list,   dim=0)[:n_spectral_samples]
    target_cat = torch.cat(target_list, dim=0)[:n_spectral_samples]

    k_bins, E_pred    = radial_energy_spectrum(pred_cat)
    _,      E_target  = radial_energy_spectrum(target_cat)
    omega_bins, pdf_pred   = vorticity_pdf(pred_cat)
    _,          pdf_target = vorticity_pdf(target_cat)

    return {
        'rel_l2'    : float(np.mean(all_rel_l2)),
        'max_div'   : float(np.max(all_max_div)),
        'mean_div'  : float(np.mean(all_max_div)),
        'k_bins'    : k_bins,
        'E_pred'    : E_pred,
        'E_target'  : E_target,
        'omega_bins': omega_bins,
        'pdf_pred'  : pdf_pred,
        'pdf_target': pdf_target,
    }

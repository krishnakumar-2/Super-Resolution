import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.io
import h5py

log = logging.getLogger(__name__)


def spectral_coarsen(u_hr, lr_res):
    B, C, H, W = u_hr.shape
    assert H == W

    u_ft = torch.fft.rfft2(u_hr)

    kmax = lr_res // 2
    u_ft_trunc = torch.zeros(B, C, lr_res, kmax + 1,
                             dtype=u_ft.dtype, device=u_ft.device)

    u_ft_trunc[:, :, :kmax, :] = u_ft[:, :, :kmax, :kmax + 1]
    u_ft_trunc[:, :, kmax:, :] = u_ft[:, :, H - kmax:, :kmax + 1]

    scale = (lr_res / H) ** 2
    u_lr = torch.fft.irfft2(u_ft_trunc, s=(lr_res, lr_res)) * scale

    return u_lr


def vorticity_to_velocity(w):
    B, H, W = w.shape
    w_ft = torch.fft.fft2(w)

    kx = torch.fft.fftfreq(H, d=1.0 / H).to(w.device)
    ky = torch.fft.fftfreq(W, d=1.0 / W).to(w.device)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')

    k2 = KX ** 2 + KY ** 2
    k2[0, 0] = 1.0

    psi_ft = -w_ft / (4.0 * np.pi ** 2 * k2.unsqueeze(0))
    psi_ft[:, 0, 0] = 0.0

    u_ft = 2j * np.pi * KY.unsqueeze(0) * psi_ft
    v_ft = -2j * np.pi * KX.unsqueeze(0) * psi_ft

    u = torch.fft.ifft2(u_ft).real
    v = torch.fft.ifft2(v_ft).real
    return torch.stack([u, v], dim=-1)


def generate_synthetic_turbulence(n, res, seed=42):
    rng = np.random.default_rng(seed)
    x = torch.linspace(0, 2 * np.pi, res + 1)[:-1]
    y = torch.linspace(0, 2 * np.pi, res + 1)[:-1]
    X, Y = torch.meshgrid(x, y, indexing='ij')

    fields = []
    for _ in range(n):
        phases = rng.uniform(0, 2 * np.pi, 8)
        amps   = rng.uniform(0.3, 1.0, 4)
        psi = (amps[0] * torch.sin(3 * X + phases[0]) * torch.cos(4 * Y + phases[1])
             + amps[1] * torch.sin(8 * X + phases[2]) * torch.cos(7 * Y + phases[3])
             + amps[2] * torch.sin(5 * X + phases[4]) * torch.cos(2 * Y + phases[5])
             + amps[3] * torch.sin(2 * X + phases[6]) * torch.cos(9 * Y + phases[7]))

        psi_ft = torch.fft.fft2(psi.unsqueeze(0))
        kx = torch.fft.fftfreq(res, d=1.0 / res)
        ky = torch.fft.fftfreq(res, d=1.0 / res)
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        u_ft = 2j * np.pi / (2 * np.pi) * KY.unsqueeze(0) * psi_ft
        v_ft = -2j * np.pi / (2 * np.pi) * KX.unsqueeze(0) * psi_ft
        u = torch.fft.ifft2(u_ft).real.squeeze(0)
        v = torch.fft.ifft2(v_ft).real.squeeze(0)
        fields.append(torch.stack([u, v], dim=-1))

    return torch.stack(fields)


class NavierStokesDataset(Dataset):
    def __init__(self, u_hr, mean, std, lr_res=16):
        self.mean   = mean
        self.std    = std
        self.lr_res = lr_res

        u_hr_norm = (u_hr - mean) / (std + 1e-8)

        u_hr_bchw = u_hr_norm.permute(0, 3, 1, 2).contiguous()
        u_lr_bchw = spectral_coarsen(u_hr_bchw, lr_res)

        self.u_hr = u_hr_norm
        self.u_lr = u_lr_bchw

    def __len__(self):
        return len(self.u_hr)

    def __getitem__(self, idx):
        return self.u_lr[idx], self.u_hr[idx]


def load_raw_dataset(file_path):
    if not os.path.exists(file_path):
        log.warning("Dataset file not found. Using synthetic turbulence (debug only).")
        return generate_synthetic_turbulence(500, 64)

    log.info(f"Loading dataset: {file_path}")

    try:
        mat = scipy.io.loadmat(file_path)
        key = 'u' if 'u' in mat else 'a'
        raw = np.array(mat[key], dtype=np.float32)
        if raw.ndim == 4:
            raw = raw.transpose(3, 2, 0, 1)
        data = torch.from_numpy(raw)

    except NotImplementedError:
        log.info("HDF5 (MATLAB v7.3) format detected.")
        with h5py.File(file_path, 'r') as f:
            key = 'u' if 'u' in f else 'a'
            raw = np.array(f[key], dtype=np.float32)
        if raw.ndim == 4:
            raw = raw.transpose(0, 1, 3, 2)
        data = torch.from_numpy(raw)

    log.info(f"Raw tensor shape: {tuple(data.shape)}")

    N, T, H, W = data.shape
    w_last = data[:, -1, :, :]

    log.info(f"Computing velocity from vorticity for {N} samples ...")
    batch  = 256
    u_list = []
    for i in range(0, N, batch):
        u_list.append(vorticity_to_velocity(w_last[i:i+batch]))
    u_hr = torch.cat(u_list, dim=0)

    log.info(f"Velocity tensor shape: {tuple(u_hr.shape)}")
    return u_hr


def build_dataloaders(file_path, lr_res=16, batch_size=32,
                      n_train=4000, n_val=500, n_test=500):
    u_hr  = load_raw_dataset(file_path)
    total = len(u_hr)

    n_train = min(n_train, total)
    n_val   = min(n_val,   total - n_train)
    n_test  = min(n_test,  total - n_train - n_val)

    u_train = u_hr[:n_train]
    u_val   = u_hr[n_train : n_train + n_val]
    u_test  = u_hr[n_train + n_val : n_train + n_val + n_test]

    mean = u_train.mean(dim=(0, 1, 2), keepdim=True)
    std  = u_train.std(dim=(0, 1, 2),  keepdim=True)

    log.info(f"Train: {len(u_train)} | Val: {len(u_val)} | Test: {len(u_test)}")
    log.info(f"Normalization: mean={mean.squeeze().tolist()}, std={std.squeeze().tolist()}")

    train_ds = NavierStokesDataset(u_train, mean, std, lr_res)
    val_ds   = NavierStokesDataset(u_val,   mean, std, lr_res)
    test_ds  = NavierStokesDataset(u_test,  mean, std, lr_res)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, (mean, std)


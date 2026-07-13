import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader

SCALE = 4


def _find(path):
    if os.path.isdir(path):
        h = sorted(glob.glob(os.path.join(path, "**", "*.pt"), recursive=True))
        if not h:
            raise FileNotFoundError(path)
        return h[0]
    return path


def _load(path):
    raw = torch.load(_find(path), map_location="cpu")
    if isinstance(raw, dict):
        raw = next(iter(raw.values()))
    d = torch.as_tensor(raw).float()
    return d


def _to_videos_uv(d):
    if d.dim() == 5:
        if d.shape[1] in (2, 3):
            d = d.permute(0, 2, 3, 4, 1)
        elif d.shape[2] in (2, 3):
            d = d.permute(0, 1, 3, 4, 2)
        d = d.reshape(-1, d.shape[2], d.shape[3], d.shape[4])
    elif d.dim() == 4:
        if d.shape[1] in (2, 3):
            d = d.permute(0, 2, 3, 1)
    return d[..., :2].contiguous()


def _center(uv):
    u, v = uv[..., 0], uv[..., 1]
    u = 0.5 * (u + torch.roll(u, 1, dims=-2))
    v = 0.5 * (v + torch.roll(v, 1, dims=-1))
    return u, v


def _vorticity(uv, chunk=1024):
    out = []
    for i in range(0, uv.shape[0], chunk):
        u, v = _center(uv[i : i + chunk])
        H, W = u.shape[-2], u.shape[-1]
        kx = torch.fft.fftfreq(H, d=1.0).view(1, H, 1) * 2 * torch.pi
        ky = torch.fft.rfftfreq(W, d=1.0).view(1, 1, -1) * 2 * torch.pi
        w_ft = 1j * kx * torch.fft.rfft2(v) - 1j * ky * torch.fft.rfft2(u)
        out.append(torch.fft.irfft2(w_ft, s=(H, W)))
    return torch.cat(out, dim=0)


def _downsample(w, out, chunk=1024):
    if w.shape[0] > chunk:
        return torch.cat(
            [
                _downsample(w[i : i + chunk], out, chunk)
                for i in range(0, w.shape[0], chunk)
            ],
            dim=0,
        )
    N, H, W = w.shape
    f = torch.fft.fftshift(torch.fft.fft2(w), dim=(-2, -1))
    c, half = H // 2, out // 2
    f = f[:, c - half : c + half, c - half : c + half]
    f = torch.fft.ifftshift(f, dim=(-2, -1))
    return (torch.fft.ifft2(f).real * (out * out) / (H * W)).contiguous()


class VortDataset(Dataset):
    def __init__(self, w_hr, w_lr, length, mean, std, train):
        self.hr = w_hr
        self.lr = w_lr
        self.length = length
        self.mean = mean
        self.std = std
        self.train = train
        self.N = w_hr.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = random if self.train else random.Random(i)
        idx = rng.randrange(self.N)
        hr = (self.hr[idx] - self.mean) / self.std
        lr = (self.lr[idx] - self.mean) / self.std
        return lr.unsqueeze(0), hr.unsqueeze(0)


def build_dataloaders(file_path, lr_res, batch_size, n_train, n_val, n_test):
    uv = _to_videos_uv(_load(file_path))
    cap = n_train + n_val + n_test
    if uv.shape[0] > cap:
        sel = torch.linspace(0, uv.shape[0] - 1, cap).long()
        uv = uv[sel]
    w = _vorticity(uv)
    n = w.shape[0]
    a, b = int(0.8 * n), int(0.9 * n)
    tr, va, te = w[:a], w[a:b], w[b:]
    mean, std = tr.mean(), tr.std() + 1e-8

    def prep(x):
        return x, _downsample(x, lr_res)

    hr_tr, lr_tr = prep(tr)
    hr_va, lr_va = prep(va)
    hr_te, lr_te = prep(te)

    sets = [
        VortDataset(hr_tr, lr_tr, n_train, mean, std, True),
        VortDataset(hr_va, lr_va, n_val, mean, std, False),
        VortDataset(hr_te, lr_te, n_test, mean, std, False),
    ]

    def ld(ds, sh):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=sh,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            drop_last=sh,
        )

    return (
        ld(sets[0], True),
        ld(sets[1], False),
        ld(sets[2], False),
        (mean, std),
    )

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.architecture import VortSR, velocity_from_vorticity
from utils.dataset import build_dataloaders

MODEL_NAME = "DR-STO"


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "tests" else _HERE
_CANDIDATES = [
    os.path.join(_ROOT, "models", "checkpoints"),
    os.path.join(os.getcwd(), "models", "checkpoints"),
    "/content/drive/MyDrive/VortSR/checkpoints",
]
CKPT_DIR = next(
    (
        d
        for d in _CANDIDATES
        if os.path.exists(os.path.join(d, "best_model.pt"))
    ),
    _CANDIDATES[0],
)
CHECKPOINT = os.path.join(CKPT_DIR, "best_model.pt")


def _resolve_data(p):
    for c in (
        p,
        os.path.join(_ROOT, p),
        os.path.join(_ROOT, "data", os.path.basename(p)),
    ):
        if os.path.exists(c):
            return c
    return p


N_SPECTRUM = 256


def rel_l2(pred, tgt):
    d = (pred - tgt).reshape(len(pred), -1)
    t = tgt.reshape(len(tgt), -1)
    return d.norm(dim=1) / (t.norm(dim=1) + 1e-12)


def max_div(vel):
    B, H, W, _ = vel.shape
    kx = (
        torch.fft.fftfreq(H, d=1.0).view(1, H, 1).to(vel.device) * 2 * torch.pi
    )
    ky = (
        torch.fft.rfftfreq(W, d=1.0).view(1, 1, -1).to(vel.device)
        * 2
        * torch.pi
    )
    d = torch.fft.irfft2(
        1j * kx * torch.fft.rfft2(vel[..., 0])
        + 1j * ky * torch.fft.rfft2(vel[..., 1]),
        s=(H, W),
    )
    return d.abs().reshape(B, -1).max(dim=1).values


def radial_spectrum(vel):
    u, w = vel[..., 0], vel[..., 1]
    N, H, W = u.shape
    e = np.zeros((H, W))
    for i in range(0, N, 64):
        e += 0.5 * (
            np.abs(np.fft.fft2(u[i : i + 64])) ** 2
            + np.abs(np.fft.fft2(w[i : i + 64])) ** 2
        ).sum(0)
    e /= N
    kx = np.fft.fftfreq(H) * H
    ky = np.fft.fftfreq(W) * W
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    kr = np.rint(np.hypot(KX, KY)).astype(int)
    kmax = H // 2
    E = np.bincount(kr.ravel(), weights=e.ravel(), minlength=kmax)[:kmax]
    return np.arange(1, kmax), E[1:]


def fit_slope(k, E, k_lo, k_hi):
    m = (k >= k_lo) & (k <= k_hi) & (E > 0)
    if m.sum() < 3:
        return None
    return np.polyfit(np.log(k[m]), np.log(E[m]), 1)[0]


def stats(name, x, unit="%", scale=100.0):
    x = np.asarray(x) * scale
    print(f"\n  {name}")
    print(f"    Mean ± Std : {x.mean():.2f}{unit} ± {x.std():.2f}{unit}")
    print(f"    Median     : {np.median(x):.2f}{unit}")
    print(f"    Best       : {x.min():.2f}{unit}")
    print(f"    Worst      : {x.max():.2f}{unit}")


@torch.no_grad()
def test_unseen():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"No checkpoint at {CHECKPOINT}. Run train.py first."
        )
    ck = torch.load(CHECKPOINT, map_location=device)
    cfg = ck["cfg"]
    model = VortSR(
        cfg["hr_res"], cfg["channels"], cfg["n_blocks"], cfg["lr_res"]
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f'Checkpoint : epoch {ck["epoch"]}  ({CHECKPOINT})')
    print(f"Parameters : {model.count_parameters():,}")

    _, _, te, _ = build_dataloaders(
        _resolve_data(cfg["data_path"]),
        cfg["lr_res"],
        32,
        cfg["n_train"],
        cfg["n_val"],
        cfg["n_test"],
    )
    print(
        f"Test samples: {len(te.dataset)}  "
        f"(drawn from the held-out final 10% of snapshots — never trained on)"
    )

    l2_v, l2_w, l2_bi, div_m, div_bi = [], [], [], [], []
    Vp, Vg, Vb = [], [], []

    print("\nEvaluating ...")
    for lr, w_hr in te:
        lr, w_hr = lr.to(device), w_hr.to(device)
        omega, vel = model(lr)
        v_gt = velocity_from_vorticity(w_hr)

        w_bi = F.interpolate(
            lr,
            size=(cfg["hr_res"], cfg["hr_res"]),
            mode="bicubic",
            align_corners=False,
        )
        v_bi = velocity_from_vorticity(w_bi)

        l2_v.append(rel_l2(vel, v_gt).cpu())
        l2_w.append(rel_l2(omega, w_hr).cpu())
        l2_bi.append(rel_l2(v_bi, v_gt).cpu())
        div_m.append(max_div(vel).cpu())
        div_bi.append(max_div(v_bi).cpu())

        if sum(len(v) for v in Vp) < N_SPECTRUM:
            Vp.append(vel.float().cpu().numpy())
            Vg.append(v_gt.float().cpu().numpy())
            Vb.append(v_bi.float().cpu().numpy())

    l2_v = torch.cat(l2_v).numpy()
    l2_w = torch.cat(l2_w).numpy()
    l2_bi = torch.cat(l2_bi).numpy()
    div_m = torch.cat(div_m).numpy()
    div_bi = torch.cat(div_bi).numpy()
    Vp = np.concatenate(Vp)[:N_SPECTRUM]
    Vg = np.concatenate(Vg)[:N_SPECTRUM]
    Vb = np.concatenate(Vb)[:N_SPECTRUM]

    print("\n" + "=" * 64)
    print(f"  {MODEL_NAME} — Unseen Test Set Results  ({len(l2_v)} samples)")
    print("=" * 64)

    stats("Velocity relative L2 error", l2_v)
    stats("Vorticity relative L2 error", l2_w)
    stats("Bicubic baseline (velocity rel. L2)", l2_bi)
    print(
        f"\n  Improvement over bicubic : "
        f"{l2_bi.mean() / max(l2_v.mean(), 1e-12):.2f}x lower error"
    )

    print(f"\n  Maximum divergence |div u|")
    print(f"    Mean ± Std : {div_m.mean():.2e} ± {div_m.std():.2e}")
    print(f"    Worst case : {div_m.max():.2e}")
    for thr in (1e-4, 1e-5):
        print(
            f"    Samples with max|div u| < {thr:.0e} : "
            f"{(div_m < thr).mean() * 100:.1f}%"
        )
    print(
        "    (divergence-free by construction; residual is fp32 FFT round-off)"
    )

    nyq = cfg["lr_res"] // 2
    k, Eg = radial_spectrum(Vg)
    _, Ep = radial_spectrum(Vp)
    _, Eb = radial_spectrum(Vb)
    k_lo, k_hi = 4, min(2 * nyq, int(k[-1] * 0.75))
    sg, sp, sb = (fit_slope(k, E, k_lo, k_hi) for E in (Eg, Ep, Eb))
    if sg is not None and sp is not None:
        print(
            f"\n  Energy-spectrum slope (log-log fit, k in [{k_lo}, {k_hi}])"
        )
        print(
            f"    Ground truth : k^{sg:.2f}   (2-D enstrophy-cascade theory: "
            f"k^-3)"
        )
        print(
            f"    {MODEL_NAME:<12} : k^{sp:.2f}   (deviation "
            f"{abs(sp - sg):.3f})"
        )
        if sb is not None:
            print(
                f"    Bicubic      : k^{sb:.2f}   (deviation "
                f"{abs(sb - sg):.3f})"
            )

    print("\n" + "=" * 64)
    print("  LaTeX table rows (method & rel-L2 & mean div & worst div):")
    print("=" * 64)
    print(
        f"  Bicubic & ${l2_bi.mean()*100:.2f}\\pm{l2_bi.std()*100:.2f}$ "
        f"& ${div_bi.mean():.1e}$ & ${div_bi.max():.1e}$ \\\\"
    )
    print(
        f"  {MODEL_NAME} & ${l2_v.mean()*100:.2f}\\pm{l2_v.std()*100:.2f}$ "
        f"& ${div_m.mean():.1e}$ & ${div_m.max():.1e}$ \\\\"
    )
    print("=" * 64)
    print("\nDone. Run make_figures.py for the publication figures.")


if __name__ == "__main__":
    test_unseen()

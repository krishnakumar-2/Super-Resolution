import os
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from models.architecture import DR_STO
from utils.dataset import build_dataloaders
from utils.metrics import (
    compute_divergence,
    radial_energy_spectrum,
    vorticity_pdf,
    relative_l2,
    max_divergence,
)

CHECKPOINT = "models/checkpoints/best_model.pt"
DATA_PATH = "data/ns_V1e-3_N5000_T50.mat"
RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)


W1 = 3.5
W2 = 7.0


C_GT = "#1B2A4A"
C_PRED = "#C0392B"
C_REF = "#2471A3"
C_ACC = "#1A7A64"
CMAP_V = "RdBu_r"
CMAP_E = "afmhot"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "DejaVu Serif",
            "Times New Roman",
            "Computer Modern Roman",
        ],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.titlepad": 5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "legend.borderpad": 0.4,
        "legend.labelspacing": 0.3,
        "legend.handlelength": 1.8,
        "lines.linewidth": 1.6,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "image.interpolation": "nearest",
        "image.origin": "lower",
    }
)


def save(name: str):
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(RESULTS, f"{name}.{ext}"))
    plt.close()
    print(f"  saved  {name}.png + .pdf")


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def strip_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def colorbar_right(ax, im, label="", fs=7.5, pad=0.04, size="5%"):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=fs - 0.5, length=2.5, width=0.5, pad=2)
    cb.outline.set_linewidth(0.5)
    if label:
        cb.set_label(label, fontsize=fs, labelpad=4)
    return cb


def load_model(batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    model = DR_STO(
        in_channels=2,
        latent_channels=cfg["latent_channels"],
        lr_res=cfg["lr_res"],
        hr_res=cfg["hr_res"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(
        f'  Checkpoint: epoch {ckpt["epoch"]},  '
        f"params = {model.count_parameters():,}"
    )

    _, _, test_loader, _ = build_dataloaders(
        file_path=DATA_PATH,
        lr_res=cfg["lr_res"],
        batch_size=batch_size,
        n_train=cfg["n_train"],
        n_val=cfg["n_val"],
        n_test=cfg["n_test"],
    )
    return model, test_loader, device, cfg


@torch.no_grad()
def collect_results(model, loader, device):
    pred_l, gt_l, lr_l, l2_l, div_l = [], [], [], [], []

    for u_lr, u_hr in loader:
        pred = model(u_lr.to(device)).cpu()
        pred_l.append(pred)
        gt_l.append(u_hr)
        lr_l.append(u_lr)
        for i in range(pred.shape[0]):
            l2_l.append(relative_l2(pred[i : i + 1], u_hr[i : i + 1]))
            div_l.append(max_divergence(pred[i : i + 1]))

    pred_all = torch.cat(pred_l)
    gt_all = torch.cat(gt_l)
    lr_all = torch.cat(lr_l)
    k, E_p = radial_energy_spectrum(pred_all, n_bins=40)
    _, E_t = radial_energy_spectrum(gt_all, n_bins=40)
    ob, pp = vorticity_pdf(pred_all, n_bins=150)
    _, pt = vorticity_pdf(gt_all, n_bins=150)

    return dict(
        pred=pred_all,
        gt=gt_all,
        lr=lr_all,
        l2=np.array(l2_l),
        div=np.array(div_l),
        k=k,
        E_pred=E_p,
        E_gt=E_t,
        omega=ob,
        pdf_pred=pp,
        pdf_gt=pt,
    )


def fig1_qualitative(R, cfg):
    print("Fig 1: qualitative comparison")
    hr, lr_r = cfg["hr_res"], cfg["lr_res"]
    idx = 0

    gt_np = R["gt"][idx].numpy()
    pred_np = R["pred"][idx].numpy()

    lr_raw = R["lr"][idx].numpy()
    lr_np = lr_raw[0] if lr_raw.ndim == 3 else lr_raw
    err_np = np.abs(pred_np[..., 0] - gt_np[..., 0])
    div_np = compute_divergence(R["pred"][idx : idx + 1]).squeeze(0).numpy()
    max_div = float(np.abs(div_np).max())
    l2_val = float(R["l2"][idx]) * 100
    vmax = float(
        max(np.abs(gt_np[..., 0]).max(), np.abs(pred_np[..., 0]).max())
    )

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(W2 + 0.6, (W2 + 0.6) / 5 + 0.9),
    )
    fig.subplots_adjust(
        left=0.02, right=0.98, bottom=0.16, top=0.90, wspace=0.55
    )

    kv = dict(cmap=CMAP_V, vmin=-vmax, vmax=vmax)
    ke = dict(cmap=CMAP_E, vmin=0)

    panels = [
        (lr_np, kv, r"$u$", f"LR input  ({lr_r}\u00d7{lr_r})"),
        (gt_np[..., 0], kv, r"$u$", f"Ground truth  ({hr}\u00d7{hr})"),
        (pred_np[..., 0], kv, r"$u$", "DR-STO prediction"),
        (
            err_np,
            ke,
            r"$|\hat{u}-u|$",
            rf"$|$Error$|$   $\ell_2={l2_val:.2f}\%$",
        ),
        (
            np.abs(div_np),
            ke,
            r"$|\nabla\!\cdot\!\mathbf{u}|$",
            rf"$|\nabla\!\cdot\!\mathbf{{u}}|$   max ${max_div:.1e}$",
        ),
    ]

    for ax, (data, kw, cblabel, caption) in zip(axes, panels):
        im = ax.imshow(data, extent=[0, 1, 0, 1], **kw)
        strip_ticks(ax)

        ax.set_xlabel(caption, fontsize=7.5, labelpad=6)
        colorbar_right(ax, im, label=cblabel, fs=7, pad=0.05, size="7%")

    fig.suptitle(
        "Qualitative comparison — DR-STO vs ground truth", y=0.97, fontsize=9
    )
    save("fig1_qualitative")


def fig2_spectrum(R):
    print("Fig 2: radial energy spectrum")
    k, E_p, E_t = R["k"], R["E_pred"], R["E_gt"]
    pos = (k > 0) & (E_p > 0) & (E_t > 0)
    kp, Ep, Et = k[pos], E_p[pos], E_t[pos]

    noise = Et[Et > 0].min() * 100
    inert = (kp >= 2.0) & (kp <= 8.0) & (Et > noise)

    fig, ax = plt.subplots(figsize=(W1, W1 * 0.90))

    ax.loglog(kp, Et, "-", color=C_GT, lw=1.8, zorder=4, label="Ground truth")
    ax.loglog(kp, Ep, "--", color=C_PRED, lw=1.8, zorder=4, label="DR-STO")

    slope_t = slope_p = None
    if inert.sum() > 3:
        ki, Eti, Epi = kp[inert], Et[inert], Ep[inert]
        k_ref = np.array([ki[0], ki[-1]])
        E_ref = Eti[0] * (k_ref / ki[0]) ** (-3)
        ax.loglog(
            k_ref, E_ref, ":", color=C_REF, lw=1.3, label=r"$k^{-3}$ reference"
        )
        slope_t = np.polyfit(np.log(ki), np.log(Eti), 1)[0]
        slope_p = np.polyfit(np.log(ki), np.log(Epi), 1)[0]

        ann = (
            r"Inertial range $k\in[2,8]$:"
            + "\n"
            + rf"GT $\sim k^{{{slope_t:.2f}}}$,  "
            + rf"DR-STO $\sim k^{{{slope_p:.2f}}}$"
        )

        ax.text(
            0.97,
            0.97,
            ann,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            linespacing=1.4,
            bbox=dict(
                boxstyle="round,pad=0.32",
                fc="white",
                ec="#bbbbbb",
                lw=0.6,
                alpha=0.96,
            ),
            zorder=10,
        )

    nyq = 8
    ax.axvline(nyq, color="#aaaaaa", ls=":", lw=0.9, zorder=2)

    ax.text(
        nyq * 0.82,
        1.0,
        r"$k_\mathrm{Nyq}^\mathrm{LR}$",
        transform=ax.get_xaxis_transform(),
        fontsize=7,
        color="#888888",
        va="top",
        ha="right",
    )

    ax.set_xlim(kp.min() * 0.9, kp.max() * 1.1)
    ax.set_xlabel(r"Wavenumber $k$")
    ax.set_ylabel(r"Energy $E(k)$")
    ax.set_title("Radial Kinetic Energy Spectrum")
    ax.legend(loc="lower left")
    ax.grid(True, which="major", alpha=0.12, lw=0.4)
    ax.grid(True, which="minor", alpha=0.05, lw=0.3)
    despine(ax)
    fig.tight_layout()
    fig.savefig(
        os.path.join(RESULTS, "fig2_spectrum.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    fig.savefig(
        os.path.join(RESULTS, "fig2_spectrum.pdf"),
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close()
    print("  saved  fig2_spectrum.png + .pdf")
    return slope_t, slope_p


def fig3_vorticity_pdf(R):
    print("Fig 3: vorticity PDF")
    omega, pp, pt = R["omega"], R["pdf_pred"], R["pdf_gt"]

    def excess_kurtosis(x, p):
        p = np.maximum(p, 0.0)
        p /= p.sum() + 1e-30
        mu = (x * p).sum()
        s2 = ((x - mu) ** 2 * p).sum()
        k4 = ((x - mu) ** 4 * p).sum()
        return float(k4 / (s2**2 + 1e-30)) - 3.0

    kt = excess_kurtosis(omega, pt)
    kp = excess_kurtosis(omega, pp)
    pt_n = np.maximum(pt, 0.0)
    pt_n /= pt_n.sum() + 1e-30
    sig = float(np.sqrt(np.sum(omega**2 * pt_n)))
    gauss = np.exp(-0.5 * (omega / (sig + 1e-30)) ** 2) / (
        sig * np.sqrt(2 * np.pi) + 1e-30
    )

    fig, ax = plt.subplots(figsize=(W1, W1 * 0.88))

    ax.semilogy(
        omega,
        pt + 1e-10,
        "-",
        color=C_GT,
        lw=1.8,
        zorder=4,
        label=rf"GT  ($\kappa_e={kt:.2f}$)",
    )
    ax.semilogy(
        omega,
        pp + 1e-10,
        "--",
        color=C_PRED,
        lw=1.8,
        zorder=4,
        label=rf"DR-STO  ($\kappa_e={kp:.2f}$)",
    )
    ax.semilogy(
        omega,
        gauss + 1e-10,
        ":",
        color=C_REF,
        lw=1.3,
        zorder=3,
        label=r"Gaussian  ($\kappa_e=0$)",
    )

    ax.set_xlabel(r"Vorticity $\omega = \partial_x v - \partial_y u$")
    ax.set_ylabel("Probability density")
    ax.set_title("Vorticity PDF")

    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(
        mticker.LogFormatterSciNotation(labelOnlyBase=False)
    )
    ax.grid(True, which="major", alpha=0.12, lw=0.4)
    ax.grid(True, which="minor", alpha=0.04, lw=0.3)
    despine(ax)
    fig.tight_layout()
    fig.savefig(
        os.path.join(RESULTS, "fig3_vorticity_pdf.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    fig.savefig(
        os.path.join(RESULTS, "fig3_vorticity_pdf.pdf"),
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close()
    print("  saved  fig3_vorticity_pdf.png + .pdf")


def fig4_error_maps(R):
    print("Fig 4: pointwise error maps")
    pred = R["pred"][:4]
    gt = R["gt"][:4]
    err = (pred - gt).abs().numpy()
    vmax = float(err.max())

    fig = plt.figure(figsize=(W2, W2 * 0.42))
    gs = GridSpec(
        2,
        5,
        figure=fig,
        width_ratios=[1, 1, 1, 1, 0.06],
        hspace=0.14,
        wspace=0.12,
        left=0.07,
        right=0.91,
        top=0.83,
        bottom=0.10,
    )

    im_ref = None
    for col in range(4):
        l2 = relative_l2(pred[col : col + 1], gt[col : col + 1]) * 100
        for row, comp in enumerate([r"$u$-component", r"$v$-component"]):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(err[col, :, :, row], cmap=CMAP_E, vmin=0, vmax=vmax)
            im_ref = im
            strip_ticks(ax)
            if row == 0:
                ax.set_title(
                    f"Sample {col+1} — $\ell_2={l2:.2f}\%$",
                    fontsize=7.5,
                    pad=3,
                )
            if col == 0:
                ax.set_ylabel(comp, fontsize=7.5, labelpad=4)

    cax = fig.add_subplot(gs[:, 4])
    cb = fig.colorbar(im_ref, cax=cax)
    cb.set_label(r"$|\hat{u}-u|$", fontsize=8, labelpad=4)
    cb.ax.tick_params(labelsize=7, length=2.5, width=0.55)
    cb.outline.set_linewidth(0.5)

    fig.suptitle(
        "Pointwise Absolute Error — Unseen Test Set",
        y=0.96,
        fontsize=9,
        fontweight="semibold",
    )
    save("fig4_error_maps")


def fig5_unseen_gallery(R, cfg):
    print("Fig 5: unseen test gallery")
    n = 4
    pred = R["pred"][:n]
    gt = R["gt"][:n]
    lr_f = R["lr"][:n]
    err = (pred - gt).abs()

    fig = plt.figure(figsize=(W2, W2 * 0.72))
    gs = GridSpec(
        3,
        6,
        figure=fig,
        width_ratios=[1, 1, 1, 1, 0.055, 0.055],
        hspace=0.09,
        wspace=0.10,
        left=0.08,
        right=0.935,
        top=0.91,
        bottom=0.04,
    )

    row_labels = ["GT", "DR-STO", r"$|$Error$|$"]
    im_vel = im_err = None

    for col in range(n):
        gt_np = gt[col].numpy()
        pr_np = pred[col].numpy()
        er_np = err[col].numpy()
        lr_np = (
            lr_f[col].numpy()[0]
            if lr_f[col].numpy().ndim == 3
            else lr_f[col].numpy()
        )
        l2 = float(R["l2"][col]) * 100
        vmax = float(
            max(np.abs(gt_np[..., 0]).max(), np.abs(pr_np[..., 0]).max())
        )
        emax = float(er_np[..., 0].max())
        kv = dict(cmap=CMAP_V, vmin=-vmax, vmax=vmax)

        ax0 = fig.add_subplot(gs[0, col])
        im_vel = ax0.imshow(gt_np[..., 0], **kv)
        strip_ticks(ax0)
        ax0.set_title(f"Test {col+1}", fontsize=8, pad=3)
        if col == 0:
            ax0.set_ylabel(row_labels[0], fontsize=8, labelpad=4)

        ai = ax0.inset_axes([0.02, 0.02, 0.24, 0.24])
        ai.imshow(lr_np, cmap=CMAP_V, vmin=-vmax, vmax=vmax)
        ai.set_xticks([])
        ai.set_yticks([])
        for sp in ai.spines.values():
            sp.set_edgecolor("white")
            sp.set_linewidth(1.0)

        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(pr_np[..., 0], **kv)
        strip_ticks(ax1)
        if col == 0:
            ax1.set_ylabel(row_labels[1], fontsize=8, labelpad=4)

        ax2 = fig.add_subplot(gs[2, col])
        im_err = ax2.imshow(er_np[..., 0], cmap=CMAP_E, vmin=0, vmax=emax)
        strip_ticks(ax2)
        if col == 0:
            ax2.set_ylabel(row_labels[2], fontsize=8, labelpad=4)
        ax2.set_xlabel(rf"$\ell_2={l2:.2f}\%$", fontsize=7.5, labelpad=4)

    cax_v = fig.add_subplot(gs[0:2, 4])
    cb_v = fig.colorbar(im_vel, cax=cax_v)
    cb_v.set_label(r"$u$", fontsize=7.5, labelpad=3)
    cb_v.ax.tick_params(labelsize=7, length=2, width=0.5)
    cb_v.outline.set_linewidth(0.5)

    cax_e = fig.add_subplot(gs[2, 4])
    cb_e = fig.colorbar(im_err, cax=cax_e)
    cb_e.set_label(r"$|\hat{u}-u|$", fontsize=7.5, labelpad=3)
    cb_e.ax.tick_params(labelsize=7, length=2, width=0.5)
    cb_e.outline.set_linewidth(0.5)

    nt = cfg["n_train"]
    nv = cfg["n_val"]
    fig.suptitle(
        rf"DR-STO — Unseen test set  "
        rf'(samples {nt+nv}–{nt+nv+cfg["n_test"]-1})',
        y=0.985,
        fontsize=9,
        fontweight="semibold",
    )
    save("fig5_unseen_gallery")


def fig6_divergence_cdf(R):
    print("Fig 6: divergence CDF")
    div = R["div"]
    ds = np.sort(div)
    cdf = np.arange(1, len(ds) + 1) / len(ds) * 100

    x_hi = float(np.percentile(div, 99)) * 4.0
    x_lo = float(ds[ds > 0].min()) * 0.5

    fig, ax = plt.subplots(figsize=(W1, W1 * 0.86))
    ax.semilogx(ds, cdf, "-", color=C_PRED, lw=1.8, zorder=4)

    for thr, col in [(1e-6, "#aaaaaa"), (1e-5, "#aaaaaa")]:
        if x_lo <= thr <= x_hi:
            pct = float((div < thr).mean() * 100)
            ax.axvline(thr, color=col, ls=":", lw=0.9, zorder=1)

            ax.text(
                thr * 1.15,
                pct + 4,
                rf"$10^{{{int(round(np.log10(thr)))}}}$: {pct:.0f}%",
                fontsize=7.5,
                color="#555555",
                va="bottom",
            )

    mu = float(div.mean())
    if x_lo <= mu <= x_hi:
        mu_pct = float((div < mu).mean() * 100)
        ax.axvline(mu, color=C_PRED, ls="--", lw=1.1, alpha=0.70, zorder=3)

        ax.text(
            mu * 1.8,
            75,
            f"mean = ${mu:.1e}$",
            color=C_PRED,
            fontsize=7.5,
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85
            ),
        )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-2, 105)
    ax.set_xlabel(r"$\max|\nabla\!\cdot\!\mathbf{u}|$")
    ax.set_ylabel("Cumulative fraction (%)")
    ax.set_title("Divergence Constraint — 500 Test Samples")
    ax.grid(True, which="major", alpha=0.12, lw=0.4)
    ax.grid(True, which="minor", alpha=0.05, lw=0.3)
    despine(ax)
    fig.tight_layout()
    fig.savefig(
        os.path.join(RESULTS, "fig6_divergence_cdf.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    fig.savefig(
        os.path.join(RESULTS, "fig6_divergence_cdf.pdf"),
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close()
    print("  saved  fig6_divergence_cdf.png + .pdf")


def fig7_scatter(R):
    print("Fig 7: predicted vs ground-truth scatter")
    rng = np.random.default_rng(42)
    N = R["pred"].shape[0] * 64 * 64
    idx = rng.choice(N, size=min(8000, N), replace=False)
    xp = R["gt"][..., 0].reshape(-1).numpy()[idx]
    yp = R["pred"][..., 0].reshape(-1).numpy()[idx]
    lim = max(abs(xp).max(), abs(yp).max()) * 1.05

    fig, ax = plt.subplots(figsize=(W1, W1))
    h, xe, ye = np.histogram2d(
        xp, yp, bins=100, range=[[-lim, lim], [-lim, lim]]
    )
    h_masked = np.ma.masked_where(h == 0, h)
    pcm = ax.pcolormesh(
        xe, ye, h_masked.T, cmap="plasma", rasterized=True, zorder=1
    )
    ax.plot(
        [-lim, lim],
        [-lim, lim],
        "-",
        color="white",
        lw=1.2,
        zorder=5,
        label=r"$y=x$",
    )

    ss_res = float(np.sum((xp - yp) ** 2))
    ss_tot = float(np.sum((xp - xp.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-30)
    mae = float(np.mean(np.abs(xp - yp)))

    ax.text(
        0.04,
        0.96,
        f"$R^2 = {r2:.5f}$\n$\\mathrm{{MAE}} = {mae:.4f}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.30", fc="#222222", ec="none", alpha=0.75
        ),
    )

    colorbar_right(ax, pcm, label="Count", fs=7, pad=0.06, size="5%")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Ground truth $u$")
    ax.set_ylabel(r"Predicted $\hat{u}$")
    ax.set_title("Predicted vs. Ground Truth Velocity")
    ax.legend(loc="lower right", fontsize=8)
    despine(ax)
    fig.tight_layout()
    fig.savefig(
        os.path.join(RESULTS, "fig7_scatter.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    fig.savefig(
        os.path.join(RESULTS, "fig7_scatter.pdf"),
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close()
    print("  saved  fig7_scatter.png + .pdf")


def fig8_summary(R, slope_t, slope_p):
    print("Fig 8: summary metrics")
    l2 = R["l2"] * 100
    div = R["div"]

    fig, axes = plt.subplots(
        1, 3, figsize=(W2, W2 * 0.30), constrained_layout=True
    )

    ax = axes[0]
    ax.hist(
        l2, bins=30, color=C_PRED, alpha=0.82, edgecolor="white", linewidth=0.4
    )
    ax.axvline(
        l2.mean(),
        color=C_GT,
        lw=1.5,
        ls="--",
        label=rf"$\mu = {l2.mean():.2f}$%",
    )
    ax.set_xlabel(r"Rel. $\ell_2$ error (%)", fontsize=8.5)
    ax.set_ylabel("Count", fontsize=8.5)
    ax.set_title("(a) Reconstruction accuracy", fontsize=9, pad=4)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.15, lw=0.4)
    despine(ax)

    ax = axes[1]
    log_div = np.log10(np.maximum(div, 1e-20))
    ax.hist(
        log_div,
        bins=30,
        color=C_ACC,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(
        np.log10(div.mean() + 1e-20),
        color=C_GT,
        lw=1.5,
        ls="--",
        label=rf"$\mu = {div.mean():.1e}$",
    )
    ax.set_xlabel(
        r"$\log_{10}\,\max|\nabla\!\cdot\!\mathbf{u}|$", fontsize=8.5
    )
    ax.set_ylabel("Count", fontsize=8.5)
    ax.set_title("(b) Divergence constraint", fontsize=9, pad=4)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.15, lw=0.4)
    despine(ax)

    ax = axes[2]
    st = slope_t if slope_t is not None else -3.0
    sp = slope_p if slope_p is not None else -3.0

    names = ["Ground\ntruth", "DR-STO", "Theory\n($k^{-3}$)"]
    vals = [abs(st), abs(sp), 3.0]
    raw = [st, sp, -3.0]
    cols = [C_GT, C_PRED, C_REF]

    bars = ax.bar(
        names,
        vals,
        color=cols,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.5,
        width=0.48,
    )
    y_top = max(vals)
    for bar, rv in zip(bars, raw):
        bar_h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar_h + y_top * 0.025,
            f"{rv:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_ylim(0, y_top * 1.22)
    ax.set_ylabel(r"Spectral slope $|p|$", fontsize=8.5)
    ax.set_title(r"(c) Inertial-range slope ($k=2$–$8$)", fontsize=9, pad=4)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="y", alpha=0.15, lw=0.4)
    despine(ax)

    save("fig8_summary")


def print_table(R, slope_t, slope_p):
    l2 = R["l2"]
    div = R["div"]
    print()
    print("=" * 66)
    print("  DR-STO  —  Test Set Summary  (500 unseen trajectories)")
    print("=" * 66)
    print(f"  Relative L2 error")
    print(f"    Mean ± Std  : {l2.mean()*100:.3f}% ± {l2.std()*100:.3f}%")
    print(f"    Best / Worst: {l2.min()*100:.3f}% / {l2.max()*100:.3f}%")
    print(f"  Max divergence  |∇·u|")
    print(f"    Mean ± Std  : {div.mean():.2e} ± {div.std():.2e}")
    print(f"    All < 1e-5  : {(div < 1e-5).mean()*100:.1f}%")
    print(f"    All < 1e-4  : {(div < 1e-4).mean()*100:.1f}%")
    if slope_t is not None and slope_p is not None:
        print(f"  Spectral slope  (inertial range k = 2–8)")
        print(f"    Ground truth  : k^{slope_t:.3f}  (theory: k^-3)")
        print(f"    DR-STO        : k^{slope_p:.3f}")
        print(f"    Deviation     : {abs(slope_p - slope_t):.3f}")
    print("=" * 66)
    print("  LaTeX table row:")
    print(
        f"  DR-STO"
        f" & ${l2.mean()*100:.2f}\\pm{l2.std()*100:.2f}$"
        f" & ${div.mean():.1e}$"
        f" & ${(div < 1e-5).mean()*100:.0f}\\%$ \\\\"
    )
    print("=" * 66)


def main():
    print("Loading model ...")
    model, test_loader, device, cfg = load_model(batch_size=32)

    print("Collecting predictions on all 500 unseen test samples ...")
    R = collect_results(model, test_loader, device)

    print("\nGenerating figures:")
    fig1_qualitative(R, cfg)
    slope_t, slope_p = fig2_spectrum(R)
    fig3_vorticity_pdf(R)
    fig4_error_maps(R)
    fig5_unseen_gallery(R, cfg)
    fig6_divergence_cdf(R)
    fig7_scatter(R)
    fig8_summary(R, slope_t, slope_p)

    print_table(R, slope_t, slope_p)
    print(f"\nAll figures saved to ./{RESULTS}/  (PNG 300 dpi + PDF vector)")


if __name__ == "__main__":
    main()

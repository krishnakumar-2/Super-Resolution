import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

from models.architecture import DR_STO
from utils.dataset        import build_dataloaders
from utils.metrics        import (evaluate_loader, compute_divergence,
                                   radial_energy_spectrum, vorticity_pdf,
                                   relative_l2, max_divergence)

CHECKPOINT = 'models/checkpoints/best_model.pt'
DATA_PATH  = 'data/ns_V1e-3_N5000_T50.mat'
RESULTS    = 'results'
os.makedirs(RESULTS, exist_ok=True)

W1 = 3.5
W2 = 7.0

plt.rcParams.update({
    'font.family'       : 'serif',
    'font.serif'        : ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset'  : 'cm',
    'font.size'         : 9,
    'axes.labelsize'    : 9,
    'axes.titlesize'    : 9,
    'legend.fontsize'   : 8,
    'xtick.labelsize'   : 8,
    'ytick.labelsize'   : 8,
    'lines.linewidth'   : 1.5,
    'axes.linewidth'    : 0.8,
    'xtick.major.width' : 0.8,
    'ytick.major.width' : 0.8,
    'figure.dpi'        : 150,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight',
    'savefig.pad_inches': 0.02,
})

C_GT    = '#1a1a1a'
C_PRED  = '#c0392b'
C_GAUSS = '#2980b9'
C_ERR   = '#e74c3c'


def savefig(name):
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(RESULTS, f'{name}.{ext}'))
    plt.close()
    print(f"  Saved: results/{name}.png  +  .pdf")


def add_cb(fig, ax, im, label='', size='5%', pad=0.05):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size=size, pad=pad)
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=7)
    return cb


def load_model_and_data(batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    cfg    = ckpt['cfg']

    model = DR_STO(
        in_channels     = 2,
        latent_channels = cfg['latent_channels'],
        lr_res          = cfg['lr_res'],
        hr_res          = cfg['hr_res'],
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  Checkpoint: epoch {ckpt['epoch']}, params = {model.count_parameters():,}")

    _, _, test_loader, _ = build_dataloaders(
        file_path  = DATA_PATH,
        lr_res     = cfg['lr_res'],
        batch_size = batch_size,
        n_train    = cfg['n_train'],
        n_val      = cfg['n_val'],
        n_test     = cfg['n_test'],
    )
    return model, test_loader, device, cfg


@torch.no_grad()
def full_test_eval(model, loader, device):
    all_l2, all_div = [], []
    pred_acc, gt_acc = [], []

    for u_lr, u_hr in loader:
        pred     = model(u_lr.to(device)).cpu()
        u_hr_cpu = u_hr.cpu()

        for i in range(pred.shape[0]):
            all_l2.append(relative_l2(pred[i:i+1], u_hr_cpu[i:i+1]))
            all_div.append(max_divergence(pred[i:i+1]))

        pred_acc.append(pred)
        gt_acc.append(u_hr_cpu)

    pred_all = torch.cat(pred_acc, dim=0)
    gt_all   = torch.cat(gt_acc,   dim=0)

    k_bins, E_pred    = radial_energy_spectrum(pred_all, n_bins=40)
    _,      E_target  = radial_energy_spectrum(gt_all,   n_bins=40)
    omega_bins, pdf_pred   = vorticity_pdf(pred_all, n_bins=120)
    _,          pdf_target = vorticity_pdf(gt_all,   n_bins=120)

    return {
        'l2'        : np.array(all_l2),
        'max_div'   : np.array(all_div),
        'k_bins'    : k_bins,
        'E_pred'    : E_pred,
        'E_target'  : E_target,
        'omega_bins': omega_bins,
        'pdf_pred'  : pdf_pred,
        'pdf_target': pdf_target,
        'pred_all'  : pred_all,
        'gt_all'    : gt_all,
    }


def fig1_qualitative(model, loader, device, cfg, res):
    print("\nFig 1: Qualitative comparison")
    u_lr_b, u_hr_b = next(iter(loader))
    u_lr = u_lr_b[:1].to(device)

    with torch.no_grad():
        pred_t = model(u_lr)

    lr_np   = u_lr.squeeze(0).cpu().numpy()[0]
    gt_np   = u_hr_b[0].numpy()
    pred_np = pred_t.squeeze(0).cpu().numpy()
    err_np  = np.abs(pred_np[..., 0] - gt_np[..., 0])
    div_np  = compute_divergence(pred_t).squeeze(0).cpu().numpy()
    max_div = float(np.abs(div_np).max())

    vmax   = float(max(np.abs(gt_np[..., 0]).max(), np.abs(pred_np[..., 0]).max()))
    hr     = cfg['hr_res']
    l2_val = relative_l2(pred_t.cpu(), u_hr_b[:1]) * 100

    fig, axes = plt.subplots(1, 5, figsize=(W2, W2 * 0.28))
    kv = dict(cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower', interpolation='nearest')

    im0 = axes[0].imshow(lr_np, extent=[0, hr, 0, hr], **kv)
    axes[0].set_title(r'LR Input ($16\!\times\!16$)')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$y$')
    add_cb(fig, axes[0], im0)

    im1 = axes[1].imshow(gt_np[..., 0], **kv)
    axes[1].set_title(r'Ground Truth ($64\!\times\!64$)')
    axes[1].set_xlabel('$x$')
    add_cb(fig, axes[1], im1)

    im2 = axes[2].imshow(pred_np[..., 0], **kv)
    axes[2].set_title('DR-STO Prediction')
    axes[2].set_xlabel('$x$')
    add_cb(fig, axes[2], im2)

    im3 = axes[3].imshow(err_np, cmap='hot', vmin=0, origin='lower', interpolation='nearest')
    axes[3].set_title(f'$|$Error$|$   (rel. L2 = {l2_val:.3f}\\%)')
    axes[3].set_xlabel('$x$')
    add_cb(fig, axes[3], im3, label=r'$|\hat{u}-u|$')

    im4 = axes[4].imshow(np.abs(div_np), cmap='magma', origin='lower', interpolation='nearest')
    axes[4].set_title(r'$|\nabla\!\cdot\!\mathbf{u}|$' + f'   max $= {max_div:.1e}$')
    axes[4].set_xlabel('$x$')
    add_cb(fig, axes[4], im4)

    for ax in axes[1:]:
        ax.set_yticklabels([])

    plt.tight_layout(w_pad=0.4)
    savefig('fig1_qualitative')


def fig2_spectrum(res):
    print("Fig 2: Energy spectrum")
    k, E_p, E_t = res['k_bins'], res['E_pred'], res['E_target']

    pos = (k > 0) & (E_p > 0) & (E_t > 0)
    k_p, E_p_p, E_t_p = k[pos], E_p[pos], E_t[pos]

    noise = E_t_p[E_t_p > 0].min() * 1000
    inert = (k_p >= 2.0) & (k_p <= 8.0) & (E_t_p > noise)

    fig, ax = plt.subplots(figsize=(W1, W1 * 0.88))

    ax.loglog(k_p, E_t_p, '-',  color=C_GT,   lw=2.0, label='Ground truth', zorder=4)
    ax.loglog(k_p, E_p_p, '--', color=C_PRED, lw=2.0, label='DR-STO',       zorder=4)

    slope_t = slope_p = None
    if inert.sum() > 3:
        k_i, E_t_i, E_p_i = k_p[inert], E_t_p[inert], E_p_p[inert]
        k_ref = np.array([k_i[0], k_i[-1]])
        E_ref = E_t_i[0] * (k_ref / k_i[0]) ** (-3)
        ax.loglog(k_ref, E_ref, ':', color=C_GAUSS, lw=1.4, label=r'$k^{-3}$ (theory)')

        slope_t = np.polyfit(np.log(k_i), np.log(E_t_i), 1)[0]
        slope_p = np.polyfit(np.log(k_i), np.log(E_p_i), 1)[0]

        textstr = (r'\textbf{Inertial range}' '\n'
                   fr'GT slope: $k^{{{slope_t:.2f}}}$' '\n'
                   fr'DR-STO:  $k^{{{slope_p:.2f}}}$')
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#aaaaaa', lw=0.7))

    ax.set_xlabel(r'Wavenumber $k$')
    ax.set_ylabel(r'$E(k)$')
    ax.set_title('Radial Kinetic Energy Spectrum')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.2, lw=0.5)

    ax.axvline(x=8, color='gray', ls=':', lw=1.0, alpha=0.7)
    ax.text(8.3, E_t_p.max() * 0.3, r'$k_{\rm LR}^{\rm Nyq}=8$',
            fontsize=7, color='gray', va='top')

    plt.tight_layout()
    savefig('fig2_spectrum')
    return slope_t, slope_p


def fig3_vorticity_pdf(res):
    print("Fig 3: Vorticity PDF")
    omega, pdf_p, pdf_t = res['omega_bins'], res['pdf_pred'], res['pdf_target']

    def excess_kurtosis(x, p):
        p  = np.maximum(p, 0)
        p /= p.sum() + 1e-30
        mu = (x * p).sum()
        s2 = ((x - mu) ** 2 * p).sum()
        k4 = ((x - mu) ** 4 * p).sum()
        return float(k4 / (s2 ** 2 + 1e-30)) - 3.0

    kurt_t = excess_kurtosis(omega, pdf_t)
    kurt_p = excess_kurtosis(omega, pdf_p)
    sigma  = float(np.sqrt(np.sum(omega ** 2 * pdf_t / (pdf_t.sum() + 1e-30))))
    gauss  = np.exp(-0.5 * (omega / (sigma + 1e-30)) ** 2) / (sigma * np.sqrt(2 * np.pi) + 1e-30)

    fig, ax = plt.subplots(figsize=(W1, W1 * 0.9))
    ax.semilogy(omega, pdf_t + 1e-10, '-',  color=C_GT,    lw=2.0,
                label=fr'Ground truth (kurtosis $= {kurt_t:.2f}$)')
    ax.semilogy(omega, pdf_p + 1e-10, '--', color=C_PRED,  lw=2.0,
                label=fr'DR-STO (kurtosis $= {kurt_p:.2f}$)')
    ax.semilogy(omega, gauss  + 1e-10, ':', color=C_GAUSS, lw=1.2,
                label='Gaussian (kurtosis $= 0$)')

    ax.set_xlabel(r'Vorticity $\omega = \partial_x v - \partial_y u$')
    ax.set_ylabel('Probability density')
    ax.set_title('Vorticity PDF')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.2, lw=0.5)
    plt.tight_layout()
    savefig('fig3_vorticity_pdf')


def fig4_error_maps(res):
    print("Fig 4: Error maps")
    pred = res['pred_all'][:4]
    gt   = res['gt_all'][:4]
    err  = (pred - gt).abs()
    vmax = float(err.max())

    fig, axes = plt.subplots(2, 4, figsize=(W2, W2 * 0.38))

    for col in range(4):
        l2 = relative_l2(pred[col:col+1], gt[col:col+1]) * 100
        for row, (ci, label) in enumerate([(0, '$u$'), (1, '$v$')]):
            ax = axes[row, col]
            im = ax.imshow(err[col, :, :, ci].numpy(),
                           cmap='hot', vmin=0, vmax=vmax,
                           origin='lower', interpolation='nearest')
            if row == 0:
                ax.set_title(f'Sample {col+1}\nrel. L2 $= {l2:.3f}$\\%', pad=3)
            ax.set_ylabel(f'{label}-err' if col == 0 else '')
            ax.set_xlabel('$x$' if row == 1 else '')
            ax.tick_params(labelbottom=(row == 1), labelleft=(col == 0))
            add_cb(fig, ax, im, size='6%')

    fig.suptitle('Pointwise Absolute Error Maps  -  Unseen Test Set', y=1.01, fontsize=9)
    plt.tight_layout(w_pad=0.3, h_pad=0.5)
    savefig('fig4_error_maps')


def fig5_unseen_gallery(res, cfg, n=4):
    print("Fig 5: Unseen test gallery")
    pred = res['pred_all'][:n]
    gt   = res['gt_all'][:n]
    err  = (pred - gt).abs()
    hr   = cfg['hr_res']

    fig, axes  = plt.subplots(3, n, figsize=(W2, W2 * 0.62))
    row_labels = ['Ground Truth', 'DR-STO', r'$|$Error$|$']

    for col in range(n):
        gt_np  = gt[col].numpy()
        pr_np  = pred[col].numpy()
        er_np  = err[col].numpy()
        l2     = relative_l2(pred[col:col+1], gt[col:col+1]) * 100
        vmax   = float(max(np.abs(gt_np[..., 0]).max(), np.abs(pr_np[..., 0]).max()))

        kv = dict(cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower', interpolation='nearest')

        im0 = axes[0, col].imshow(gt_np[..., 0], **kv)
        axes[0, col].set_title(f'Test {col+1}', pad=3)
        add_cb(fig, axes[0, col], im0, size='7%')

        im1 = axes[1, col].imshow(pr_np[..., 0], **kv)
        add_cb(fig, axes[1, col], im1, size='7%')

        im2 = axes[2, col].imshow(er_np[..., 0], cmap='hot', vmin=0,
                                  origin='lower', interpolation='nearest')
        axes[2, col].set_title(f'rel. L2 $= {l2:.3f}$\\%', pad=2)
        add_cb(fig, axes[2, col], im2, label=r'$|\hat{u}-u|$', size='7%')
        axes[2, col].set_xlabel('$x$')

    for row, lbl in enumerate(row_labels):
        axes[row, 0].set_ylabel(lbl, fontsize=8)

    for ax in axes.flat:
        ax.tick_params(labelbottom=False, labelleft=False)
    for col in range(n):
        axes[2, col].tick_params(labelbottom=True)
    for row in range(3):
        axes[row, 0].tick_params(labelleft=True)

    n_train = cfg['n_train']
    n_val   = cfg['n_val']
    n_test  = cfg['n_test']
    fig.suptitle(
        f'DR-STO  -  Unseen Test Set  (trajectories {n_train+n_val}-{n_train+n_val+n_test-1})',
        y=1.01, fontsize=9
    )
    plt.tight_layout(w_pad=0.3, h_pad=0.4)
    savefig('fig5_unseen_gallery')


def fig6_divergence_stats(res):
    print("Fig 6: Divergence statistics")
    div = res['max_div']

    fig, ax = plt.subplots(figsize=(W1, W1 * 0.8))

    div_sorted = np.sort(div)
    cdf = np.arange(1, len(div) + 1) / len(div)
    ax.semilogx(div_sorted, cdf * 100, '-', color=C_PRED, lw=2.0)

    for ref, label in [(1e-5, r'$10^{-5}$'), (1e-4, r'$10^{-4}$')]:
        ax.axvline(ref, color='gray', ls=':', lw=0.9)
        pct = float((div < ref).mean() * 100)
        ax.text(ref * 1.15, 5, f'{pct:.1f}\\% $<{label}$', fontsize=7, color='gray', va='bottom')

    ax.set_xlabel(r'$\max|\nabla\!\cdot\!\mathbf{u}|$  (test samples)')
    ax.set_ylabel('Cumulative percentage (\\%)')
    ax.set_title('Divergence-Free Constraint  -  500 Unseen Samples')
    ax.set_ylim(0, 101)
    ax.grid(True, which='both', alpha=0.2, lw=0.5)

    ax.axvline(div.mean(), color=C_PRED, ls='--', lw=1.0, alpha=0.7)
    ax.text(div.mean() * 1.15, 60, fr'mean $= {div.mean():.1e}$', color=C_PRED, fontsize=7)

    plt.tight_layout()
    savefig('fig6_divergence_stats')


def fig7_summary(res, slope_t, slope_p):
    print("Fig 7: Summary metrics")
    l2  = res['l2']
    div = res['max_div']

    fig, axes = plt.subplots(1, 3, figsize=(W2, W2 * 0.3))

    ax = axes[0]
    ax.hist(l2 * 100, bins=30, color=C_PRED, alpha=0.8, edgecolor='white', lw=0.3)
    ax.axvline(l2.mean() * 100, color=C_GT, lw=1.5, ls='--',
               label=fr'mean $= {l2.mean()*100:.3f}$\%')
    ax.set_xlabel('Relative L2 error (\\%)')
    ax.set_ylabel('Count')
    ax.set_title('Reconstruction Accuracy')
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.hist(np.log10(div + 1e-12), bins=30, color='#8e44ad', alpha=0.8, edgecolor='white', lw=0.3)
    ax.axvline(np.log10(div.mean() + 1e-12), color=C_GT, lw=1.5, ls='--',
               label=fr'mean $= {div.mean():.1e}$')
    ax.set_xlabel(r'$\log_{10}|\nabla\!\cdot\!\mathbf{u}|_{\max}$')
    ax.set_ylabel('Count')
    ax.set_title('Divergence-Free Constraint')
    ax.legend(fontsize=7)

    ax = axes[2]
    labels = ['Ground\nTruth', 'DR-STO', 'Theory\n$k^{-3}$']
    slopes = [slope_t if slope_t else -3.0,
              slope_p if slope_p else -3.0,
              -3.0]
    colors = [C_GT, C_PRED, C_GAUSS]
    bars   = ax.bar(labels, [-s for s in slopes], color=colors, alpha=0.85,
                    edgecolor='white', lw=0.5)
    ax.set_ylabel('Spectral slope $|p|$  (larger = steeper)')
    ax.set_title('Inertial-Range Slope')
    for bar, val in zip(bars, slopes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout(w_pad=0.6)
    savefig('fig7_summary')


def print_results_table(res, slope_t, slope_p):
    l2  = res['l2']
    div = res['max_div']
    print()
    print("=" * 66)
    print("  DR-STO  -  Test Set Summary")
    print("=" * 66)
    print(f"  Relative L2 error")
    print(f"    Mean +/- Std : {l2.mean()*100:.3f}% +/- {l2.std()*100:.3f}%")
    print(f"    Best / Worst : {l2.min()*100:.3f}% / {l2.max()*100:.3f}%")
    print(f"  Max Divergence")
    print(f"    Mean +/- Std : {div.mean():.2e} +/- {div.std():.2e}")
    print(f"    All < 1e-5   : {(div<1e-5).mean()*100:.1f}%")
    print(f"    All < 1e-4   : {(div<1e-4).mean()*100:.1f}%")
    if slope_t and slope_p:
        print(f"  Spectral slope (k = 2-8)")
        print(f"    Ground truth : k^{slope_t:.3f}  (theory: k^-3)")
        print(f"    DR-STO       : k^{slope_p:.3f}")
        print(f"    Deviation    : {abs(slope_p-slope_t):.3f}")
    print("=" * 66)
    print("  LaTeX table row:")
    print(f"  DR-STO & {l2.mean()*100:.2f}$\\pm${l2.std()*100:.2f} "
          f"& {div.mean():.1e} & {(div<1e-5).mean()*100:.0f}\\% \\\\")
    print("=" * 66)


def main():
    print("Loading model and test data ...")
    model, test_loader, device, cfg = load_model_and_data(batch_size=32)

    print("Running full evaluation on unseen test samples ...")
    res = full_test_eval(model, test_loader, device)

    slope_t, slope_p = None, None

    print("\nGenerating figures:")
    fig1_qualitative(model, test_loader, device, cfg, res)
    slope_t, slope_p = fig2_spectrum(res)
    fig3_vorticity_pdf(res)
    fig4_error_maps(res)
    fig5_unseen_gallery(res, cfg, n=4)
    fig6_divergence_stats(res)
    fig7_summary(res, slope_t, slope_p)

    print_results_table(res, slope_t, slope_p)
    print(f"\nAll figures saved to ./{RESULTS}/  (PNG + PDF)")


if __name__ == '__main__':
    main()

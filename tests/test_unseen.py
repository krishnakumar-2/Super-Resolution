import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.architecture import DR_STO
from utils.dataset        import build_dataloaders
from utils.metrics        import relative_l2, max_divergence, evaluate_loader

CHECKPOINT = 'models/checkpoints/best_model.pt'
DATA_PATH  = 'data/ns_V1e-3_N5000_T50.mat'


def test_unseen():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"No checkpoint at {CHECKPOINT}. Run train.py first.")

    ckpt = torch.load(CHECKPOINT, map_location=device)
    cfg  = ckpt['cfg']

    model = DR_STO(
        in_channels     = 2,
        latent_channels = cfg['latent_channels'],
        lr_res          = cfg['lr_res'],
        hr_res          = cfg['hr_res'],
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"Parameters: {model.count_parameters():,}")

    _, _, test_loader, _ = build_dataloaders(
        file_path  = DATA_PATH,
        lr_res     = cfg['lr_res'],
        batch_size = 16,
        n_train    = cfg['n_train'],
        n_val      = cfg['n_val'],
        n_test     = cfg['n_test'],
    )

    n_test = len(test_loader.dataset)
    print(f"Test samples: {n_test}  (trajectories {cfg['n_train']+cfg['n_val']} "
          f"- {cfg['n_train']+cfg['n_val']+n_test-1})")

    all_l2  = []
    all_div = []

    print("\nEvaluating ...")
    with torch.no_grad():
        for u_lr, u_hr in test_loader:
            u_lr = u_lr.to(device)
            u_hr = u_hr.to(device)
            pred = model(u_lr)

            for i in range(pred.shape[0]):
                l2  = relative_l2(pred[i:i+1], u_hr[i:i+1])
                div = max_divergence(pred[i:i+1])
                all_l2.append(l2)
                all_div.append(div)

    all_l2  = np.array(all_l2)
    all_div = np.array(all_div)

    print("\n" + "=" * 60)
    print("  DR-STO  -  Unseen Test Set Results")
    print("=" * 60)
    print(f"\n  Relative L2 Error")
    print(f"    Mean  +/- Std : {all_l2.mean()*100:.2f}% +/- {all_l2.std()*100:.2f}%")
    print(f"    Best case     : {all_l2.min()*100:.2f}%")
    print(f"    Worst case    : {all_l2.max()*100:.2f}%")
    print(f"    Median        : {np.median(all_l2)*100:.2f}%")

    print(f"\n  Maximum Divergence |div u|")
    print(f"    Mean  +/- Std : {all_div.mean():.2e} +/- {all_div.std():.2e}")
    print(f"    Best case     : {all_div.min():.2e}")
    print(f"    Worst case    : {all_div.max():.2e}")
    print(f"    Median        : {np.median(all_div):.2e}")

    print(f"\n  Physical Conservation")
    pct_below_1e5 = (all_div < 1e-5).mean() * 100
    pct_below_1e4 = (all_div < 1e-4).mean() * 100
    print(f"    Samples with |div u| < 1e-5 : {pct_below_1e5:.1f}%")
    print(f"    Samples with |div u| < 1e-4 : {pct_below_1e4:.1f}%")

    print("\n" + "=" * 60)
    print("  LaTeX table row:")
    print("=" * 60)
    print(f"  DR-STO & {all_l2.mean()*100:.2f}\\% $\\pm$ {all_l2.std()*100:.2f}\\%"
          f" & {all_div.mean():.1e} & {all_div.max():.1e} \\\\")
    print("=" * 60)

    print("\nComputing spectral metrics ...")
    metrics = evaluate_loader(model, test_loader, device, n_spectral_samples=n_test)

    k   = metrics['k_bins']
    E_p = metrics['E_pred']
    E_t = metrics['E_target']

    valid = (k > k.max() * 0.1) & (E_p > 0) & (E_t > 0)
    if valid.sum() > 3:
        slope_pred   = np.polyfit(np.log(k[valid]), np.log(E_p[valid]), 1)[0]
        slope_target = np.polyfit(np.log(k[valid]), np.log(E_t[valid]), 1)[0]
        print(f"\n  Energy Spectrum Slope (log-log fit, mid-to-high k)")
        print(f"    Ground truth : k^{slope_target:.2f}  (theory: k^-3)")
        print(f"    DR-STO       : k^{slope_pred:.2f}")
        print(f"    Deviation    : {abs(slope_pred - slope_target):.3f}")


if __name__ == '__main__':
    test_unseen()


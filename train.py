import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.architecture import DR_STO
from utils.dataset       import build_dataloaders
from utils.losses        import DRSTOLoss
from utils.metrics       import evaluate_loader


CFG = {
    'data_path'      : 'data/ns_V1e-3_N5000_T50.mat',
    'lr_res'         : 16,
    'hr_res'         : 64,
    'n_train'        : 4000,
    'n_val'          : 500,
    'n_test'         : 500,
    'latent_channels': 64,
    'batch_size'     : 32,
    'epochs'         : 200,
    'lr'             : 1e-3,
    'weight_decay'   : 1e-4,
    'lambda_sot'     : 0.1,
    'checkpoint_dir' : 'models/checkpoints',
    'results_dir'    : 'results',
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def save_training_curves(train_losses, val_losses, val_rel_l2, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label='Train loss', color='steelblue')
    axes[0].plot(val_losses,   label='Val loss',   color='coral')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Curves')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_rel_l2, color='forestgreen')
    axes[1].set_xlabel('Epoch (every 5)')
    axes[1].set_ylabel('Relative L2 error')
    axes[1].set_title('Validation Relative L2')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()


def train():
    os.makedirs(CFG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CFG['results_dir'],    exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    train_loader, val_loader, test_loader, (mean, std) = build_dataloaders(
        file_path  = CFG['data_path'],
        lr_res     = CFG['lr_res'],
        batch_size = CFG['batch_size'],
        n_train    = CFG['n_train'],
        n_val      = CFG['n_val'],
        n_test     = CFG['n_test'],
    )

    torch.save({'mean': mean, 'std': std},
               os.path.join(CFG['checkpoint_dir'], 'norm_stats.pt'))

    model = DR_STO(
        in_channels     = 2,
        latent_channels = CFG['latent_channels'],
        lr_res          = CFG['lr_res'],
        hr_res          = CFG['hr_res'],
    ).to(device)

    log.info(f"Model parameters: {model.count_parameters():,}")

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr           = CFG['lr'],
        weight_decay = CFG['weight_decay'],
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimiser, start_factor=0.1, end_factor=1.0, total_iters=10
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=CFG['epochs'] - 10
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimiser,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[10],
    )

    criterion = DRSTOLoss(lambda_sot=CFG['lambda_sot'])

    best_val_loss = float('inf')
    train_losses, val_losses, val_rel_l2_history = [], [], []

    log.info("Starting training ...")
    for epoch in range(CFG['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_l2   = 0.0

        for u_lr, u_hr in train_loader:
            u_lr = u_lr.to(device)
            u_hr = u_hr.to(device)

            optimiser.zero_grad(set_to_none=True)
            pred = model(u_lr)
            loss, loss_parts = criterion(pred, u_hr)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_loss += loss.item()
            epoch_l2   += loss_parts['l2']

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for u_lr, u_hr in val_loader:
                    u_lr = u_lr.to(device)
                    u_hr = u_hr.to(device)
                    pred = model(u_lr)
                    loss, _ = criterion(pred, u_hr)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            metrics = evaluate_loader(model, val_loader, device, n_spectral_samples=32)
            val_rel_l2_history.append(metrics['rel_l2'])

            log.info(
                f"Epoch [{epoch+1:3d}/{CFG['epochs']}] "
                f"| Train: {avg_train_loss:.5f} "
                f"| Val: {avg_val_loss:.5f} "
                f"| Rel L2: {metrics['rel_l2']:.4f} "
                f"| Max Div: {metrics['max_div']:.2e} "
                f"| LR: {scheduler.get_last_lr()[0]:.2e}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    {
                        'epoch'      : epoch + 1,
                        'model_state': model.state_dict(),
                        'cfg'        : CFG,
                    },
                    os.path.join(CFG['checkpoint_dir'], 'best_model.pt'),
                )
                log.info(f"  Best model saved (val_loss={best_val_loss:.5f})")

    log.info("\nLoading best checkpoint for final test evaluation ...")
    ckpt = torch.load(os.path.join(CFG['checkpoint_dir'], 'best_model.pt'),
                      map_location=device)
    model.load_state_dict(ckpt['model_state'])

    test_metrics = evaluate_loader(model, test_loader, device, n_spectral_samples=128)

    log.info("=" * 60)
    log.info("TEST SET RESULTS")
    log.info(f"  Relative L2 error : {test_metrics['rel_l2']:.4f}  ({test_metrics['rel_l2']*100:.2f}%)")
    log.info(f"  Max divergence    : {test_metrics['max_div']:.2e}")
    log.info(f"  Mean max div      : {test_metrics['mean_div']:.2e}")
    log.info("=" * 60)

    save_training_curves(train_losses, val_losses, val_rel_l2_history, CFG['results_dir'])
    log.info(f"Training curves saved to {CFG['results_dir']}/training_curves.png")


if __name__ == '__main__':
    train()

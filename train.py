import os
import copy
import time
import logging
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from models.architecture import VortSR, velocity_from_vorticity
from utils.dataset import build_dataloaders
from utils.losses import SRLoss, rel_l2

CFG = {
    "data_path": "data/kolmogorov.pt",
    "lr_res": 32,
    "hr_res": 128,
    "n_train": 6000,
    "n_val": 1000,
    "n_test": 1000,
    "channels": 64,
    "n_blocks": 8,
    "batch_size": 64,
    "epochs": 100,
    "stop_epoch": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "ema_decay": 0.999,
    "amp": True,
    "channels_last": True,
    "checkpoint_dir": "models/checkpoints",
    "results_dir": "results",
}


_ROOT = os.path.dirname(os.path.abspath(__file__))
for _k in ("data_path", "checkpoint_dir", "results_dir"):
    if not os.path.isabs(CFG[_k]):
        CFG[_k] = os.path.join(_ROOT, CFG[_k])

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

RESUME = os.path.join(CFG["checkpoint_dir"], "last_state.pt")
BEST = os.path.join(CFG["checkpoint_dir"], "best_model.pt")

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


class EMA:
    def __init__(self, m, d):
        self.d = d
        self.s = {k: v.detach().clone() for k, v in m.state_dict().items()}
        self.fk = [k for k, v in self.s.items() if v.is_floating_point()]

    @torch.no_grad()
    def update(self, m):
        sd = m.state_dict()
        dst = [self.s[k] for k in self.fk]
        torch._foreach_mul_(dst, self.d)
        torch._foreach_add_(dst, [sd[k] for k in self.fk], alpha=1 - self.d)
        for k, v in sd.items():
            if not v.is_floating_point():
                self.s[k].copy_(v)

    def state_dict(self):
        return self.s

    def load(self, sd):
        self.s = {k: v.clone() for k, v in sd.items()}


def _prep(lr, v_hr, device, ch_last):
    lr = lr.to(device, non_blocking=True)
    if ch_last:
        lr = lr.contiguous(memory_format=torch.channels_last)
    v_hr = v_hr.to(device, non_blocking=True)
    if v_hr.dim() == 4 and v_hr.shape[1] == 1:
        v_hr = velocity_from_vorticity(v_hr)
    return lr, v_hr


@torch.no_grad()
def evaluate(model, loader, device, amp=False, ch_last=False):
    model.eval()
    rv = []
    md = 0.0
    for lr, v_hr in loader:
        lr, v_hr = _prep(lr, v_hr, device, ch_last)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            _, vel = model(lr)
        vel = vel.float()
        rv.append(rel_l2(vel, v_hr).item())
        H, W = vel.shape[1], vel.shape[2]
        kx = (
            torch.fft.fftfreq(H, d=1.0).view(1, H, 1).to(device) * 2 * torch.pi
        )
        ky = (
            torch.fft.rfftfreq(W, d=1.0).view(1, 1, -1).to(device)
            * 2
            * torch.pi
        )
        div = torch.fft.irfft2(
            1j * kx * torch.fft.rfft2(vel[..., 0])
            + 1j * ky * torch.fft.rfft2(vel[..., 1]),
            s=(H, W),
        )
        md = max(md, div.abs().max().item())
    return sum(rv) / len(rv), md


def train():
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CFG["results_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = CFG["amp"] and device.type == "cuda"
    ch_last = CFG["channels_last"] and device.type == "cuda"
    log.info(f"Device: {device} | AMP: {amp} | channels_last: {ch_last}")

    tl, vl, te, (mean, std) = build_dataloaders(
        CFG["data_path"],
        CFG["lr_res"],
        CFG["batch_size"],
        CFG["n_train"],
        CFG["n_val"],
        CFG["n_test"],
    )
    torch.save(
        {"mean": mean, "std": std},
        os.path.join(CFG["checkpoint_dir"], "norm_stats.pt"),
    )

    model = VortSR(
        CFG["hr_res"], CFG["channels"], CFG["n_blocks"], CFG["lr_res"]
    ).to(device)
    if ch_last:
        model = model.to(memory_format=torch.channels_last)
    log.info(f"Parameters: {model.count_parameters():,}")

    ema = EMA(model, CFG["ema_decay"])
    opt = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    warm = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, total_iters=5
    )
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=CFG["epochs"] - 5
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [warm, cos], milestones=[5]
    )
    crit = SRLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_ep = 0
    best = float("inf")
    tr_h, rv_h = [], []
    if os.path.exists(RESUME):
        st = torch.load(RESUME, map_location=device)
        msd = model.state_dict()
        compatible = (
            "model" in st
            and set(st["model"]) == set(msd)
            and all(st["model"][k].shape == v.shape for k, v in msd.items())
        )
        if compatible:
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            ema.load(st["ema"])
            if "scaler" in st:
                scaler.load_state_dict(st["scaler"])
            start_ep = st["epoch"]
            best = st["best"]
            tr_h = st["tr_h"]
            rv_h = st["rv_h"]
            log.info(
                f"Resumed from epoch {start_ep} (best vel rel-L2 {best:.4f})"
            )
        else:
            log.warning(
                f"{RESUME} is from an older architecture - starting fresh "
                f"(delete models/checkpoints/*.pt to silence this)."
            )

    end_ep = min(CFG["stop_epoch"] or CFG["epochs"], CFG["epochs"])
    log.info(f"Checkpoints -> {os.path.abspath(CFG['checkpoint_dir'])}")
    if start_ep >= end_ep:
        log.info(
            f"Schedule already at epoch {start_ep} - nothing to train "
            f"(increase CFG['epochs'] to continue). Evaluating best "
            f"checkpoint ..."
        )
    else:
        log.info(
            f"Training epochs {start_ep+1}-{end_ep} of the "
            f"{CFG['epochs']}-epoch schedule ..."
        )
    for ep in range(start_ep, end_ep):
        model.train()
        t0 = time.time()
        run = torch.zeros((), device=device)
        loop = tqdm(tl, desc=f"Epoch {ep+1}/{CFG['epochs']}", leave=False)
        for it, (lr, v_hr) in enumerate(loop):
            lr, v_hr = _prep(lr, v_hr, device, ch_last)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                omega, vel = model(lr)
                loss, _ = crit(omega, vel, v_hr)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            run += loss.detach()
            if it % 25 == 0:
                loop.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()
        tr_h.append((run / len(tl)).item())
        log.info(
            f"Epoch {ep+1}/{CFG['epochs']} | train {tr_h[-1]:.4f} | "
            f"{time.time()-t0:.1f}s"
        )

        if (ep + 1) % 5 == 0 or ep == 0:
            bak = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema.state_dict())
            rv, md = evaluate(model, vl, device, amp, ch_last)
            rv_h.append(rv)
            log.info(
                f"Epoch [{ep+1:3d}/{CFG['epochs']}] | vel rel-L2 {rv:.4f} | "
                f"max div {md:.2e} "
                f"| LR {sched.get_last_lr()[0]:.2e}"
            )
            if rv < best:
                best = rv
                torch.save(
                    {
                        "epoch": ep + 1,
                        "model_state": ema.state_dict(),
                        "cfg": CFG,
                    },
                    BEST,
                )
                log.info(f"  Best saved (vel rel-L2 {best:.4f})")
            model.load_state_dict(bak)

        torch.save(
            {
                "epoch": ep + 1,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "ema": ema.state_dict(),
                "scaler": scaler.state_dict(),
                "best": best,
                "tr_h": tr_h,
                "rv_h": rv_h,
            },
            RESUME,
        )

    if end_ep < CFG["epochs"]:
        log.info(
            f"Paused at epoch {end_ep} as configured. To continue from epoch "
            f"{end_ep+1}, "
            f"set CFG['stop_epoch'] = None and rerun train.py - it resumes "
            f"automatically."
        )

    ck = torch.load(BEST, map_location=device)
    model.load_state_dict(ck["model_state"])
    rv, md = evaluate(model, te, device, amp, ch_last)
    log.info("=" * 60)
    log.info(f"TEST | vel rel-L2 {rv:.4f} ({rv*100:.2f}%) | max div {md:.2e}")
    log.info("=" * 60)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(tr_h)
    ax[0].set_yscale("log")
    ax[0].set_title("Train loss")
    ax[0].grid(alpha=0.3)
    ax[1].plot(rv_h, color="forestgreen")
    ax[1].set_title("Val vel rel-L2")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(CFG["results_dir"], "training_curves.png"), dpi=150
    )
    plt.close()
    log.info("Run make_figures.py to generate the publication figures.")


if __name__ == "__main__":
    train()

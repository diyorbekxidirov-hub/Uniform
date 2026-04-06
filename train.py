"""
train.py  —  Uniform3 training script.
EfficientNet-B0 + CBAM + GeM + Triplet Loss.

Phase 1 (epochs 1-5)  : backbone frozen, head only
Phase 2 (epochs 6-50) : full fine-tune, backbone at 10x lower LR

Run:
    python3 train.py
    python3 train.py --resume checkpoints/epoch_010.pt
"""

import argparse
import logging
import random
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from configs.config import (
    TOTAL_EPOCHS, FREEZE_EPOCHS, WARMUP_EPOCHS,
    LR, LR_BACKBONE, WEIGHT_DECAY, TRIPLET_MARGIN,
    BATCH_SIZE, NUM_WORKERS, VIRTUAL_EPOCH,
    CHECKPOINT_DIR, LOG_DIR, SAVE_EVERY_N, SEED,
)
from data.dataloader import build_dataloader
from models.model import build_model
from utils.metrics import AverageMeter, compute_triplet_accuracy


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("uniform3")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def triplet_loss(emb_a, emb_p, emb_n, margin=TRIPLET_MARGIN):
    d_ap = F.pairwise_distance(emb_a, emb_p, p=2)
    d_an = F.pairwise_distance(emb_a, emb_n, p=2)
    return F.relu(d_ap - d_an + margin).mean()


def build_optimizer(model, phase: int) -> AdamW:
    if phase == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        return AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    return AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": model.cbam.parameters(),     "lr": LR},
        {"params": model.head.parameters(),     "lr": LR},
    ], weight_decay=WEIGHT_DECAY)


def build_scheduler(optimizer, n_epochs: int, warmup: int):
    def warmup_fn(ep):
        return (ep + 1) / warmup if ep < warmup else 1.0
    w = LambdaLR(optimizer, lr_lambda=warmup_fn)
    c = CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - warmup), eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[w, c], milestones=[warmup])


def train_one_epoch(model, loader, optimizer, device, epoch, logger):
    model.train()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()
    t0     = time.time()

    for step, (anchor, positive, negative) in enumerate(loader):
        anchor   = anchor.to(device, non_blocking=True)
        positive = positive.to(device, non_blocking=True)
        negative = negative.to(device, non_blocking=True)
        B = anchor.size(0)

        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        loss = triplet_loss(emb_a, emb_p, emb_n)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        acc = compute_triplet_accuracy(emb_a, emb_p, emb_n)
        loss_m.update(loss.item(), B)
        acc_m.update(acc, B)

        if (step + 1) % 200 == 0:
            logger.info(
                f"Epoch [{epoch:3d}/{TOTAL_EPOCHS}]  "
                f"Step [{step+1:4d}/{len(loader)}]  "
                f"Loss={loss_m.avg:.4f}  Acc={acc_m.avg:.3f}  "
                f"Time={time.time()-t0:.1f}s"
            )

    return {"loss": loss_m.avg, "acc": acc_m.avg}


@torch.no_grad()
def validate(model, loader, device, logger, epoch) -> float:
    model.eval()
    meter = AverageMeter()
    for anchor, positive, negative in loader:
        emb_a = model(anchor.to(device))
        emb_p = model(positive.to(device))
        emb_n = model(negative.to(device))
        meter.update(compute_triplet_accuracy(emb_a, emb_p, emb_n), anchor.size(0))
    logger.info(f"[Val] Epoch {epoch}  Triplet Acc: {meter.avg:.4f}")
    return meter.avg


def save_checkpoint(model, optimizer, scheduler, epoch, metric,
                    ckpt_dir: Path, is_best: bool = False):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metric":          metric,
        "model_version":   "uniform3",
    }
    torch.save(state, ckpt_dir / f"epoch_{epoch:03d}.pt")
    if is_best:
        torch.save(state, ckpt_dir / "best.pt")
        print(f"  [Best] epoch={epoch}  val_acc={metric:.4f}")


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    if optimizer and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler and "scheduler_state" in state:
        scheduler.load_state_dict(state["scheduler_state"])
    print(f"[Resume] epoch={state['epoch']}  metric={state.get('metric','?'):.4f}")
    return state["epoch"]


def main(resume: Optional[str] = None):
    set_seed(SEED)
    logger = setup_logging(LOG_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_loader, _ = build_dataloader(
        train=True, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, length=VIRTUAL_EPOCH,
    )
    val_loader, _ = build_dataloader(
        train=False, batch_size=BATCH_SIZE,
        num_workers=4, length=3_000,
    )

    model     = build_model(device=str(device), freeze=True)
    optimizer = build_optimizer(model, phase=1)
    scheduler = build_scheduler(optimizer, TOTAL_EPOCHS, WARMUP_EPOCHS)

    start_epoch = 0
    best_metric = 0.0

    if resume:
        start_epoch = load_checkpoint(Path(resume), model, optimizer, scheduler)

    logger.info(
        f"Training Uniform3: {TOTAL_EPOCHS} epochs  "
        f"freeze={FREEZE_EPOCHS}  warmup={WARMUP_EPOCHS}  "
        f"batch={BATCH_SIZE}  lr={LR}  margin={TRIPLET_MARGIN}"
    )

    for epoch in range(start_epoch + 1, TOTAL_EPOCHS + 1):

        if epoch == FREEZE_EPOCHS + 1:
            model.unfreeze_backbone()
            optimizer = build_optimizer(model, phase=2)
            remaining = TOTAL_EPOCHS - FREEZE_EPOCHS
            scheduler = build_scheduler(optimizer, remaining,
                                         max(1, WARMUP_EPOCHS - FREEZE_EPOCHS))
            logger.info(f"[Epoch {epoch}] Phase 2 — backbone unfrozen.")

        train_m = train_one_epoch(model, train_loader, optimizer, device, epoch, logger)
        scheduler.step()
        val_acc = validate(model, val_loader, device, logger, epoch)

        logger.info(
            f"[Epoch {epoch:3d}]  "
            f"loss={train_m['loss']:.4f}  "
            f"train_acc={train_m['acc']:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        is_best = val_acc > best_metric
        if is_best:
            best_metric = val_acc

        if epoch % SAVE_EVERY_N == 0 or is_best:
            save_checkpoint(model, optimizer, scheduler, epoch,
                            val_acc, CHECKPOINT_DIR, is_best=is_best)

    logger.info(f"Done. Best val_acc={best_metric:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(resume=args.resume)

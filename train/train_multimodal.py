"""
train_multimodal.py — Training Loop for Multi-Modal MLP (Phase 2)
===================================================================
Trains the MultiModalMLP model on preprocessed MoA data.

Key differences from train_mlp.py:
  - Uses MultiModalMLP (3 encoder branches) instead of flat MLPBaseline
  - Uses LabelSmoothedFocalLoss instead of plain FocalLoss
  - Saves checkpoint as: checkpoints/multimodal_mlp_best.pth
  - Supports --dry-run flag for quick testing

USAGE:
    python train/train_multimodal.py             # Full training
    python train/train_multimodal.py --dry-run   # 1 epoch, 200 samples
"""

import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    PROCESSED_DIR, CHECKPOINTS_DIR,
    BATCH_SIZE, MAX_EPOCHS, EARLY_STOP_PATIENCE,
    LEARNING_RATE, WEIGHT_DECAY, LR_T_0, LR_T_MULT,
    RANDOM_SEED,
)
from utils.seed import set_seed
from data.dataset import MoADataset
from models.multimodal_mlp import MultiModalMLP, LabelSmoothedFocalLoss
from train.evaluate import evaluate_model

set_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {DEVICE}")


def load_data(dry_run=False):
    """
    Load preprocessed numpy arrays. If dry_run, use first 200 samples only.

    Args:
        dry_run (bool): True = use 200 samples for a quick sanity check
    Returns:
        tuple: (X_train, y_train)
    """
    X = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))

    if dry_run:
        X, y = X[:200], y[:200]
        print(f"[DryRun] Using {len(X)} samples")
    else:
        print(f"[Data] {len(X)} samples | {X.shape[1]} features | {y.shape[1]} labels")

    return X, y


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    One full pass through the training data.

    Args:
        model     (nn.Module): The MultiModalMLP model
        loader    (DataLoader): Training data loader
        optimizer: AdamW optimizer
        criterion: LabelSmoothedFocalLoss
        device    (torch.device): cpu or cuda
    Returns:
        float: Mean training loss for this epoch
    """
    model.train()   # Enable Dropout and batch-statistics BatchNorm
    total_loss = 0.0

    for features, labels in loader:
        features = features.to(device)
        labels   = labels.to(device)

        optimizer.zero_grad()           # Clear gradients from previous batch
        logits = model(features)        # Forward pass through all 3 branches
        loss   = criterion(logits, labels)  # Label-smoothed focal loss
        loss.backward()                 # Backpropagation: compute gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip exploding grads
        optimizer.step()                # Update weights

        total_loss += loss.item() * len(features)

    return total_loss / len(loader.dataset)


def train(dry_run=False):
    """
    Full training pipeline for the MultiModalMLP.

    Steps:
      1. Load data
      2. 80/20 train/val split
      3. Build DataLoaders
      4. Init model, loss, optimizer, scheduler
      5. Train with early stopping
      6. Save best checkpoint

    Args:
        dry_run (bool): True = 1 epoch on 200 samples
    Returns:
        MultiModalMLP: best trained model
    """
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    X, y = load_data(dry_run=dry_run)
    dataset = MoADataset(X, y)

    n_val   = int(0.2 * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    print(f"[Split] Train: {n_train} | Val: {n_val}")

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MultiModalMLP().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] MultiModalMLP | Parameters: {n_params:,}")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    criterion = LabelSmoothedFocalLoss(eps=0.05)   # Focal loss + label smoothing
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=LR_T_0, T_mult=LR_T_MULT,
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_auroc    = 0.0
    patience_cnt  = 0
    n_epochs      = 1 if dry_run else MAX_EPOCHS
    ckpt_path     = os.path.join(CHECKPOINTS_DIR, "multimodal_mlp_best.pth")

    print(f"\n{'='*65}")
    print(f"Training MultiModalMLP: {n_epochs} epoch(s)")
    print(f"{'='*65}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_loss  = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate_model(model, val_loader, DEVICE)
        val_auroc   = val_metrics["auroc"]
        val_auprc   = val_metrics["auprc"]
        val_f1      = val_metrics["f1"]

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:3d}/{n_epochs}] "
            f"Loss: {train_loss:.4f} | "
            f"AUROC: {val_auroc:.4f} | "
            f"AUPRC: {val_auprc:.4f} | "
            f"F1: {val_f1:.4f} | "
            f"LR: {lr:.6f} | "
            f"Time: {time.time()-t0:.1f}s"
        )

        if val_auroc > best_auroc:
            best_auroc  = val_auroc
            patience_cnt = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_auroc":   val_auroc,
                "val_auprc":   val_auprc,
                "val_f1":      val_f1,
            }, ckpt_path)
            print(f"  [Checkpoint] Saved (AUROC improved to {val_auroc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOP_PATIENCE:
                print(f"\n[EarlyStop] No improvement for {EARLY_STOP_PATIENCE} epochs. Stopping.")
                break

    print(f"\n{'='*65}")
    print(f"Training complete!")
    print(f"Best Val AUROC: {best_auroc:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*65}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MultiModal MLP for MoA")
    parser.add_argument("--dry-run", action="store_true",
                        help="1 epoch on 200 samples for quick testing")
    args = parser.parse_args()
    train(dry_run=args.dry_run)

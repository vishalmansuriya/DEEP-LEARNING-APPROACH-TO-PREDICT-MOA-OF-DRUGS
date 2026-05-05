"""
train_fused.py -- End-to-End Training of the Fused Model (Phase 3)
====================================================================
Trains the FusedModel that combines:
    - MultiModalMLP feature extractor (416-dim latent)
    - GAT graph embeddings            (256-dim, pre-computed)
    - RotatE KG embeddings            (128-dim, pre-computed)
    => Fused 800-dim -> 206 MoA logits

PREREQUISITES:
    1. python data/build_kg.py          -> data/processed/kg_graph.pt
    2. python train/train_gat.py        -> checkpoints/gat_best.pth
                                           outputs/gat_compound_embeddings.npy
    3. python train/train_rotate.py     -> checkpoints/rotate_best.pth
                                           outputs/rotate_embeddings.npy

TRAINING STRATEGY:
    - Freeze the MultiModalMLP sub-encoder for the first FREEZE_EPOCHS epochs
      (let the new fusion head stabilize first)
    - Unfreeze all parameters after FREEZE_EPOCHS for end-to-end fine-tuning
    - Load pre-trained MultiModalMLP weights from checkpoints/multimodal_mlp_best.pth

USAGE:
    python train/train_fused.py             # Full training
    python train/train_fused.py --dry-run   # 1 epoch sanity check
"""

import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    PROCESSED_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR,
    BATCH_SIZE, MAX_EPOCHS, EARLY_STOP_PATIENCE,
    FUSED_LR, WEIGHT_DECAY, LR_T_0, LR_T_MULT,
    ROTATE_EMBEDDINGS_FILE, RANDOM_SEED,
)
from utils.seed import set_seed
from models.multimodal_mlp import FusedModel, LabelSmoothedFocalLoss
from train.evaluate import compute_auroc, compute_auprc, compute_f1

set_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {DEVICE}")

FREEZE_EPOCHS = 10  # Freeze MultiModalMLP for first N epochs


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(dry_run=False):
    """
    Load:
        X_train.npy       -- compound features (23814, 879)
        y_train.npy       -- MoA labels (23814, 206)
        gat_compound_embeddings.npy  -- GAT embeddings (n_compounds, 256)
        rotate_embeddings.npy        -- RotatE embeddings (n_compounds, 128)

    Aligns all arrays to the smallest n_compounds (in case of dry-run).

    Returns:
        X  (np.ndarray): (n, 879)
        y  (np.ndarray): (n, 206)
        gat_embs   (np.ndarray): (n, 256)
        rot_embs   (np.ndarray): (n, 128)
    """
    X = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))

    gat_path = os.path.join(OUTPUTS_DIR, "gat_compound_embeddings.npy")
    rot_path = os.path.join(OUTPUTS_DIR, ROTATE_EMBEDDINGS_FILE)

    if not os.path.exists(gat_path):
        raise FileNotFoundError(
            f"Missing: {gat_path}\n"
            "Run 'python train/train_gat.py' first."
        )
    if not os.path.exists(rot_path):
        raise FileNotFoundError(
            f"Missing: {rot_path}\n"
            "Run 'python train/train_rotate.py' first."
        )

    gat_embs = np.load(gat_path)   # (n_compounds, 256)
    rot_embs = np.load(rot_path)   # (n_compounds, 128)

    # Align to minimum length (all should be 23814, but guard for dry-run residues)
    n = min(len(X), len(gat_embs), len(rot_embs))
    X, y, gat_embs, rot_embs = X[:n], y[:n], gat_embs[:n], rot_embs[:n]

    if dry_run:
        n = 500
        X, y, gat_embs, rot_embs = X[:n], y[:n], gat_embs[:n], rot_embs[:n]
        print(f"[DryRun] Using {n} samples")
    else:
        print(f"[Data] {n} samples | X:{X.shape} | GAT:{gat_embs.shape} | RotatE:{rot_embs.shape}")

    return X.astype(np.float32), y.astype(np.float32), gat_embs.astype(np.float32), rot_embs.astype(np.float32)


# =============================================================================
# TRAINING
# =============================================================================

def train(dry_run=False):
    """
    Full training pipeline for the FusedModel.
    """
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # -- Load data ------------------------------------------------------------
    X, y, gat_embs, rot_embs = load_all_data(dry_run=dry_run)
    n = len(X)

    # Convert to tensors
    X_t   = torch.tensor(X)
    y_t   = torch.tensor(y)
    gat_t = torch.tensor(gat_embs)
    rot_t = torch.tensor(rot_embs)

    # -- 80/20 train/val split ------------------------------------------------
    n_val   = int(0.2 * n)
    n_train = n - n_val
    dataset = TensorDataset(X_t, y_t, gat_t, rot_t)
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    print(f"[Split] Train: {n_train} | Val: {n_val}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,   shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)

    # -- Model ----------------------------------------------------------------
    model = FusedModel(freeze_multimodal=True).to(DEVICE)

    # Load pre-trained MultiModalMLP weights
    mm_ckpt = os.path.join(CHECKPOINTS_DIR, "multimodal_mlp_best.pth")
    if os.path.exists(mm_ckpt):
        ckpt = torch.load(mm_ckpt, map_location=DEVICE, weights_only=False)
        model.multimodal.load_state_dict(ckpt["model_state"])
        print(f"[FusedModel] Loaded MultiModalMLP weights from {mm_ckpt}")
    else:
        print(f"[FusedModel] Warning: {mm_ckpt} not found, initializing from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] FusedModel | Total: {n_params:,} | Trainable: {n_trainable:,}")

    # -- Loss, Optimizer, Scheduler -------------------------------------------
    criterion = LabelSmoothedFocalLoss(eps=0.05)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FUSED_LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=LR_T_0, T_mult=LR_T_MULT,
    )

    # -- Training loop --------------------------------------------------------
    best_auroc   = 0.0
    patience_cnt = 0
    n_epochs     = 1 if dry_run else MAX_EPOCHS
    ckpt_path    = os.path.join(CHECKPOINTS_DIR, "fused_best.pth")
    unfrozen     = False

    print(f"\n{'='*65}")
    print(f"Training FusedModel: {n_epochs} epoch(s)")
    print(f"  First {FREEZE_EPOCHS} epochs: MultiModalMLP frozen")
    print(f"  After: full end-to-end fine-tuning")
    print(f"{'='*65}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # Unfreeze MultiModalMLP after FREEZE_EPOCHS
        if epoch > FREEZE_EPOCHS and not unfrozen:
            for p in model.multimodal.parameters():
                p.requires_grad = True
            # Re-create optimizer to include newly unfrozen params
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=FUSED_LR / 5, weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=LR_T_0, T_mult=LR_T_MULT,
            )
            unfrozen = True
            print(f"  [Epoch {epoch}] MultiModalMLP UNFROZEN — end-to-end fine-tuning")

        # Training pass
        model.train()
        total_loss = 0.0
        for x_b, y_b, gat_b, rot_b in train_loader:
            x_b   = x_b.to(DEVICE)
            y_b   = y_b.to(DEVICE)
            gat_b = gat_b.to(DEVICE)
            rot_b = rot_b.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x_b, gat_b, rot_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(x_b)

        train_loss = total_loss / n_train

        # Validation
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x_b, y_b, gat_b, rot_b in val_loader:
                logits = model(x_b.to(DEVICE), gat_b.to(DEVICE), rot_b.to(DEVICE))
                probs  = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(y_b.numpy())

        y_val_np  = np.concatenate(all_labels, axis=0)
        probs_np  = np.concatenate(all_probs,  axis=0)
        val_auroc = compute_auroc(y_val_np, probs_np)
        val_auprc = compute_auprc(y_val_np, probs_np)
        val_f1    = compute_f1(y_val_np, probs_np)

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
            best_auroc   = val_auroc
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
    print(f"Training complete! Best Val AUROC: {best_auroc:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*65}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fused Model for MoA (Phase 3)")
    parser.add_argument("--dry-run", action="store_true", help="1 epoch on 500 samples")
    args = parser.parse_args()
    train(dry_run=args.dry_run)

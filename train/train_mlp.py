"""
train_mlp.py — Full Training Loop for MLP Baseline
=====================================================
Trains the MLP baseline model on the preprocessed MoA dataset.
Supports:
  - Full training (all epochs, all data)
  - --dry-run flag: 1 epoch on 100 samples (quick sanity check)
  - Early stopping (patience=15 by default)
  - AdamW optimizer with cosine annealing learning rate schedule
  - Model checkpoint saving (best validation AUROC)

USAGE:
    python train/train_mlp.py             # Full training
    python train/train_mlp.py --dry-run   # Quick 1-epoch test on 100 samples
"""

import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ── Add project root to Python path ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    PROCESSED_DIR, CHECKPOINTS_DIR,
    BATCH_SIZE, MAX_EPOCHS, EARLY_STOP_PATIENCE,
    LEARNING_RATE, WEIGHT_DECAY, LR_T_0, LR_T_MULT,
    RANDOM_SEED, N_TOTAL_FEATURES, N_MOA_CLASSES,
)
from utils.seed import set_seed
from data.dataset import MoADataset
from models.mlp_baseline import MLPBaseline, FocalLoss
from train.evaluate import evaluate_model

# Ensure results are reproducible
set_seed(RANDOM_SEED)

# ── Detect device (GPU if available, else CPU) ────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {DEVICE}")


def load_processed_data(dry_run=False):
    """
    Load the preprocessed numpy arrays from disk.
    
    If dry_run=True, returns only the first 100 samples for a quick test.
    This lets team members verify the training loop works without waiting
    for the full training run.
    
    Args:
        dry_run (bool): If True, use only 100 samples. Default: False
    
    Returns:
        tuple: (X_train, y_train) — feature and label numpy arrays
    """
    X_path = os.path.join(PROCESSED_DIR, "X_train.npy")  # Path to feature matrix
    y_path = os.path.join(PROCESSED_DIR, "y_train.npy")  # Path to label matrix
    
    # Check that the preprocessed files exist
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            "[ERROR] Preprocessed data not found.\n"
            "Please run: python data/preprocess.py\n"
            "to generate X_train.npy and y_train.npy first."
        )
    
    X = np.load(X_path)  # Load feature matrix — shape: (23814, 184)
    y = np.load(y_path)  # Load label matrix — shape: (23814, 206)
    
    if dry_run:
        # Take only the first 100 samples for a quick sanity check
        X = X[:100]
        y = y[:100]
        print(f"[DryRun] Using {len(X)} samples for quick test")
    else:
        print(f"[Data] Loaded {len(X)} training samples, {X.shape[1]} features, {y.shape[1]} labels")
    
    return X, y


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    Run one complete pass (epoch) through the training data.
    
    An 'epoch' means the model sees every training sample exactly once.
    We process data in mini-batches (256 samples at a time), compute the
    loss for each batch, and update the model's weights via backpropagation.
    
    Args:
        model       (nn.Module): The neural network to train
        train_loader(DataLoader): Provides shuffled mini-batches of training data
        optimizer   (Optimizer): The optimization algorithm (AdamW) that updates weights
        criterion   (nn.Module): The loss function (FocalLoss)
        device      (torch.device): CPU or GPU
    
    Returns:
        float: Average training loss for this epoch
    """
    model.train()  # Set model to training mode — enables Dropout, uses batch BatchNorm
    total_loss = 0.0  # Accumulate loss across all batches
    
    for features, labels in train_loader:
        # ── Move data to GPU/CPU ──────────────────────────────────────────────
        features = features.to(device)  # Move feature tensor to the computation device
        labels   = labels.to(device)    # Move label tensor to the computation device
        
        # ── Zero gradients from previous batch ───────────────────────────────
        # PyTorch accumulates gradients by default — we must clear them each step
        optimizer.zero_grad()
        
        # ── Forward pass: compute predictions ────────────────────────────────
        logits = model(features)  # Shape: (batch_size, 206) — raw model outputs
        
        # ── Compute loss ──────────────────────────────────────────────────────
        loss = criterion(logits, labels)  # Focal loss between predictions and truth
        
        # ── Backward pass: compute gradients ─────────────────────────────────
        # Gradient = how much each weight contributed to the loss
        # PyTorch computes these automatically using the chain rule (autograd)
        loss.backward()
        
        # ── Gradient clipping ─────────────────────────────────────────────────
        # Prevents "exploding gradients" — if any gradient is too large,
        # clip it to a maximum magnitude of 1.0
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # ── Update weights ────────────────────────────────────────────────────
        # AdamW uses the computed gradients to adjust each weight slightly
        optimizer.step()
        
        # Accumulate the batch loss for reporting
        total_loss += loss.item() * len(features)  # .item() converts tensor to Python float
    
    # Return average loss per sample
    return total_loss / len(train_loader.dataset)


def train(dry_run=False):
    """
    Complete training pipeline for the MLP baseline model.
    
    STEPS:
    1. Load preprocessed data
    2. Split into train/validation (80%/20%)
    3. Create DataLoaders
    4. Initialize model, optimizer, scheduler, loss function
    5. Train for up to MAX_EPOCHS with early stopping
    6. Save best model checkpoint
    7. Report final metrics
    
    Args:
        dry_run (bool): If True, run only 1 epoch on 100 samples. Default: False
    
    Returns:
        MLPBaseline: The trained model (best checkpoint)
    """
    
    # ── Create output directories ─────────────────────────────────────────────
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)  # Save model checkpoints here
    
    # ── Load data ─────────────────────────────────────────────────────────────
    X, y = load_processed_data(dry_run=dry_run)
    
    # ── Create PyTorch Dataset ────────────────────────────────────────────────
    dataset = MoADataset(X, y)  # Wrap numpy arrays in our custom Dataset class
    
    # ── Train/Validation Split (80/20) ────────────────────────────────────────
    n_total = len(dataset)
    n_val   = int(0.2 * n_total)  # 20% for validation
    n_train = n_total - n_val     # 80% for training
    
    # random_split splits randomly but reproducibly (because we set the seed)
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED)  # Reproducible split
    )
    
    print(f"[Split] Train: {len(train_set)} samples | Validation: {len(val_set)} samples")
    
    # ── Create DataLoaders ────────────────────────────────────────────────────
    # DataLoader batches the data and provides an iterator for training
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,  # 256 samples per batch
        shuffle=True,           # Shuffle training data every epoch (prevents learning order)
        num_workers=0,          # 0 = load data on the main process (Windows-safe)
        pin_memory=torch.cuda.is_available(),  # Faster GPU transfer if using CUDA
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE * 2,  # Larger batch for validation (no backprop → less memory)
        shuffle=False,              # No shuffling for validation — consistent results
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    # ── Initialize Model ──────────────────────────────────────────────────────
    model = MLPBaseline().to(DEVICE)  # Create model and move it to GPU/CPU
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] MLPBaseline | Trainable parameters: {n_params:,}")
    
    # ── Initialize Loss Function ──────────────────────────────────────────────
    criterion = FocalLoss()  # Focal loss to handle class imbalance (γ=2.0, α=1.0 from config)
    
    # ── Initialize Optimizer ──────────────────────────────────────────────────
    # AdamW = Adam optimizer + decoupled weight decay
    # Adam: adapts learning rate per parameter based on gradient history
    # Weight decay: penalizes large weights (prevents overfitting)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,     # 0.001 — initial step size for weight updates
        weight_decay=WEIGHT_DECAY,  # 0.0001 — L2 regularization strength
    )
    
    # ── Initialize Learning Rate Scheduler ────────────────────────────────────
    # Cosine annealing with warm restarts:
    # LR starts at LEARNING_RATE, decreases following a cosine curve to near 0,
    # then restarts. Each restart cycle is T_MULT times longer than the previous.
    # This helps escape local minima and find better solutions.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=LR_T_0,       # 10 — first cycle length in epochs
        T_mult=LR_T_MULT, # 2  — each cycle is 2× longer than previous
    )
    
    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_auroc  = 0.0    # Track the best validation AUROC seen so far
    patience_counter = 0     # Count how many epochs without improvement
    
    # Determine how many epochs to run
    n_epochs = 1 if dry_run else MAX_EPOCHS
    
    print(f"\n{'='*60}")
    print(f"Starting training: {n_epochs} epoch(s)")
    print(f"{'='*60}")
    
    for epoch in range(1, n_epochs + 1):
        start_time = time.time()  # Record epoch start time
        
        # ── Train one epoch ───────────────────────────────────────────────────
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # ── Validate ──────────────────────────────────────────────────────────
        val_metrics = evaluate_model(model, val_loader, DEVICE)
        val_auroc   = val_metrics["auroc"]   # Primary metric for model selection
        val_auprc   = val_metrics["auprc"]
        val_f1      = val_metrics["f1"]
        
        # ── Update learning rate scheduler ────────────────────────────────────
        scheduler.step()  # Update LR according to cosine annealing schedule
        current_lr = optimizer.param_groups[0]["lr"]  # Get current LR for logging
        
        # ── Log epoch results ─────────────────────────────────────────────────
        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch:3d}/{n_epochs}] "
            f"Loss: {train_loss:.4f} | "
            f"Val AUROC: {val_auroc:.4f} | "
            f"Val AUPRC: {val_auprc:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed:.1f}s"
        )
        
        # ── Save best checkpoint ──────────────────────────────────────────────
        if val_auroc > best_val_auroc:
            best_val_auroc   = val_auroc
            patience_counter = 0  # Reset patience counter — we improved!
            
            # Save model state dict (weights and biases, NOT the optimizer state)
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, "mlp_baseline_best.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),   # All learnable parameters
                "val_auroc":   val_auroc,
                "val_auprc":   val_auprc,
                "val_f1":      val_f1,
            }, checkpoint_path)
            print(f"  [Checkpoint] Saved (AUROC improved to {val_auroc:.4f})")
        
        else:
            patience_counter += 1  # No improvement this epoch
            
            # ── Early stopping check ──────────────────────────────────────────
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n[EarlyStop] No improvement for {EARLY_STOP_PATIENCE} epochs. Stopping.")
                break  # Exit the training loop early
    
    # ── Final Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best Validation AUROC: {best_val_auroc:.4f}")
    print(f"Best checkpoint saved to: {checkpoint_path}")
    print(f"{'='*60}")
    
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT — parse command line arguments and run training
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ArgumentParser lets users pass command-line flags like --dry-run
    parser = argparse.ArgumentParser(
        description="Train the MLP Baseline model for MoA prediction"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",  # True if flag is present, False if not
        help="Run 1 epoch on 100 samples for quick testing"
    )
    
    args = parser.parse_args()  # Parse the command-line arguments
    
    # Run training with the dry-run flag
    train(dry_run=args.dry_run)

"""
train_gat.py -- Training Loop for GAT Model (Phase 3)
======================================================
Trains the GATEncoder on the bipartite compound-MoA knowledge graph.

The GAT is trained as a node CLASSIFIER:
    - Input: compound feature vectors as node features
    - Target: 206 binary MoA labels per compound
    - Loss: LabelSmoothedFocalLoss (same as Phase 2)

The graph structure (edge_index) is used for message passing, not just
as auxiliary features. This is the key difference from the MLP models.

Since the full graph (23,814 compounds + 206 labels = 24,020 nodes) fits
in CPU memory, we do full-graph (full-batch) training — no mini-batching.

USAGE:
    python train/train_gat.py             # Full training
    python train/train_gat.py --dry-run   # 1 epoch sanity check
"""

import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    PROCESSED_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR,
    KG_GRAPH_FILE, GAT_OUT_DIM,
    GAT_LR, GAT_MAX_EPOCHS, EARLY_STOP_PATIENCE,
    WEIGHT_DECAY, RANDOM_SEED,
)
from utils.seed import set_seed
from data.dataset import MoADataset
from models.gat_model import GATEncoder
from models.multimodal_mlp import LabelSmoothedFocalLoss
from train.evaluate import compute_auroc, compute_auprc, compute_f1

set_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {DEVICE}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_graph():
    """
    Load the pre-built knowledge graph from data/processed/kg_graph.pt.

    Returns:
        graph_data (dict): Contains edge_index, node_features, n_compounds, n_labels
    """
    graph_path = os.path.join(PROCESSED_DIR, KG_GRAPH_FILE)
    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"Missing: {graph_path}\n"
            "Run 'python data/build_kg.py' first."
        )
    graph_data = torch.load(graph_path, map_location=DEVICE, weights_only=False)
    n_compounds = graph_data["n_compounds"]
    n_labels    = graph_data["n_labels"]
    n_edges     = graph_data["edge_index"].shape[1]
    print(f"[GAT] Graph loaded: {n_compounds+n_labels} nodes | {n_edges:,} edges")
    return graph_data


def load_labels(dry_run=False):
    """Load training labels y_train.npy."""
    y = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    if dry_run:
        y = y[:500]
        print(f"[DryRun] Using {len(y)} samples")
    return y


# =============================================================================
# GAT CLASSIFIER WRAPPER
# =============================================================================

class GATClassifier(nn.Module):
    """
    Wraps GATEncoder + a linear classification head for MoA prediction.

    The GAT encodes ALL nodes in the graph (compounds + MoA label nodes).
    We then extract only the compound node embeddings and pass them through
    a linear head to predict 206 MoA labels.

    Args:
        n_compounds (int): Number of compound nodes in the graph
        gat_out_dim (int): GAT output embedding dimension
        n_classes   (int): Number of MoA classes to predict (206)
    """

    def __init__(self, n_compounds, gat_out_dim=GAT_OUT_DIM, n_classes=206):
        super().__init__()
        self.n_compounds = n_compounds
        self.gat = GATEncoder(in_dim=879, out_dim=gat_out_dim)
        self.head = nn.Sequential(
            nn.Linear(gat_out_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, node_features, edge_index, compound_indices):
        """
        Args:
            node_features   (Tensor): All node features (n_total_nodes, 879)
            edge_index      (Tensor): All edges (2, n_edges)
            compound_indices(Tensor): Indices of compounds in this batch (batch_size,)

        Returns:
            Tensor: Logits (batch_size, 206)
        """
        # Full graph forward pass through GAT
        all_embeddings = self.gat(node_features, edge_index)  # (n_total_nodes, out_dim)
        # Extract compound embeddings for this batch
        compound_embs = all_embeddings[compound_indices]       # (batch_size, out_dim)
        # Classify
        return self.head(compound_embs)                        # (batch_size, 206)


# =============================================================================
# TRAINING
# =============================================================================

def train(dry_run=False):
    """
    Full training pipeline for the GAT model.

    Uses full-graph inference (all nodes pass through GAT) but mini-batch
    gradient updates on compound subsets for memory efficiency.
    """
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # -- Load graph and labels ------------------------------------------------
    graph_data  = load_graph()
    y           = load_labels(dry_run=dry_run)
    n_compounds = graph_data["n_compounds"]
    n_labels    = graph_data["n_labels"]

    # Align: y might be smaller than n_compounds in dry-run mode
    n_use = min(len(y), n_compounds)
    y     = y[:n_use]

    node_features = graph_data["node_features"].to(DEVICE)  # (n_total_nodes, 879)
    edge_index    = graph_data["edge_index"].to(DEVICE)     # (2, n_edges)

    # -- Train/val split (80/20) on compound indices --------------------------
    all_indices = torch.arange(n_use)
    n_val   = int(0.2 * n_use)
    n_train = n_use - n_val

    perm    = torch.randperm(n_use, generator=torch.Generator().manual_seed(RANDOM_SEED))
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    y_train = torch.tensor(y[train_idx], dtype=torch.float32)
    y_val   = torch.tensor(y[val_idx],   dtype=torch.float32)

    print(f"[Split] Train: {n_train} | Val: {n_val}")

    # -- Model, loss, optimizer -----------------------------------------------
    model = GATClassifier(n_compounds=n_compounds).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] GATClassifier | Parameters: {n_params:,}")

    criterion = LabelSmoothedFocalLoss(eps=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=GAT_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # -- Training loop --------------------------------------------------------
    best_auroc   = 0.0
    patience_cnt = 0
    n_epochs     = 1 if dry_run else GAT_MAX_EPOCHS
    ckpt_path    = os.path.join(CHECKPOINTS_DIR, "gat_best.pth")
    batch_size   = 256

    print(f"\n{'='*65}")
    print(f"Training GATClassifier: {n_epochs} epoch(s)")
    print(f"{'='*65}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()

        # Mini-batch loop over training compounds
        perm_e = train_idx[torch.randperm(len(train_idx))]
        total_loss = 0.0
        n_batches  = 0

        for i in range(0, len(perm_e), batch_size):
            batch_idx = perm_e[i : i + batch_size].to(DEVICE)
            labels    = y_train[i : i + batch_size].to(DEVICE)

            optimizer.zero_grad()
            logits = model(node_features, edge_index, batch_idx)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        train_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_idx_dev = val_idx.to(DEVICE)
            val_logits  = model(node_features, edge_index, val_idx_dev)
            val_probs   = torch.sigmoid(val_logits).cpu().numpy()

        y_val_np = y_val.numpy()

        from train.evaluate import compute_auroc, compute_auprc, compute_f1
        val_auroc = compute_auroc(y_val_np, val_probs)
        val_auprc = compute_auprc(y_val_np, val_probs)
        val_f1    = compute_f1(y_val_np, val_probs)

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
                "n_compounds": n_compounds,
                "n_labels":    n_labels,
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

    # -- Save compound embeddings for FusedModel ------------------------------
    print("[GAT] Extracting compound embeddings for FusedModel...")
    model.eval()
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state"])
    with torch.no_grad():
        all_embs = model.gat(node_features, edge_index)         # (n_total_nodes, 256)
        compound_embs = all_embs[:n_compounds].cpu().numpy()    # (n_compounds, 256)

    emb_path = os.path.join(OUTPUTS_DIR, "gat_compound_embeddings.npy")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    np.save(emb_path, compound_embs)
    print(f"[GAT] Compound embeddings saved: {emb_path}  shape: {compound_embs.shape}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAT for MoA (Phase 3)")
    parser.add_argument("--dry-run", action="store_true", help="1 epoch on 500 samples")
    args = parser.parse_args()
    train(dry_run=args.dry_run)

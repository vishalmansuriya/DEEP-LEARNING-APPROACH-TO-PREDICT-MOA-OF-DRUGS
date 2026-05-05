"""
train_rotate.py -- RotatE Training Loop (Phase 3)
==================================================
Trains the RotatE KG embedding model on the compound-MoA knowledge graph.

RotatE is trained SELF-SUPERVISED:
    - Positive triples: (compound_i, has_moa, moa_label_j) for each edge in KG
    - Negative triples: corrupted versions (random compound or random label)
    - Loss: self-adversarial negative sampling loss

After training, compound embeddings (the first n_compounds rows of the
entity embedding matrix) are saved to:
    outputs/rotate_embeddings.npy  shape: (n_compounds, ROTATE_EMBED_DIM)

These are loaded by train_fused.py as an additional input feature.

USAGE:
    python train/train_rotate.py              # Full training (50 epochs)
    python train/train_rotate.py --epochs 3   # Quick test
"""

import os, sys, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    PROCESSED_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR,
    KG_GRAPH_FILE, ROTATE_EMBED_DIM, ROTATE_LR,
    ROTATE_MAX_EPOCHS, ROTATE_NEG_SAMPLES, ROTATE_EMBEDDINGS_FILE,
    RANDOM_SEED,
)
from utils.seed import set_seed
from models.rotate import RotatE, NegativeSampler

set_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {DEVICE}")


def load_graph():
    """Load KG graph dict from disk."""
    path = os.path.join(PROCESSED_DIR, KG_GRAPH_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}\nRun 'python data/build_kg.py' first.")
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_positive_triples(edge_index, n_compounds):
    """
    Extract compound->MoA (forward direction only) as positive triples.

    The KG is stored as undirected (both directions). We only want
    forward edges (compound -> MoA label) for training.

    Args:
        edge_index  (Tensor): (2, n_edges) -- undirected edge list
        n_compounds (int): Compound nodes are 0..n_compounds-1

    Returns:
        pos_heads (Tensor): Compound node IDs  (n_positive_triples,)
        pos_tails (Tensor): MoA label node IDs (n_positive_triples,)
    """
    src = edge_index[0]
    dst = edge_index[1]
    # Forward direction: src is a compound (< n_compounds), dst is a label (>= n_compounds)
    mask     = (src < n_compounds) & (dst >= n_compounds)
    pos_heads = src[mask]  # compound IDs
    pos_tails = dst[mask]  # MoA label IDs (already offset by n_compounds)
    print(f"[RotatE] Positive triples: {len(pos_heads):,}")
    return pos_heads, pos_tails


def train(n_epochs=None):
    """
    RotatE training loop.

    Args:
        n_epochs (int|None): Number of training epochs. Default: ROTATE_MAX_EPOCHS.
    """
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    n_epochs = n_epochs or ROTATE_MAX_EPOCHS

    # -- Load graph -----------------------------------------------------------
    graph_data  = load_graph()
    edge_index  = graph_data["edge_index"]
    n_compounds = graph_data["n_compounds"]
    n_labels    = graph_data["n_labels"]
    n_entities  = n_compounds + n_labels

    # Extract positive training triples (compound -> MoA)
    pos_heads, pos_tails = extract_positive_triples(edge_index, n_compounds)
    n_triples = len(pos_heads)

    print(f"[RotatE] Entities: {n_entities:,}  ({n_compounds:,} compounds + {n_labels} labels)")
    print(f"[RotatE] Positive triples: {n_triples:,}")

    # -- Model, optimizer -----------------------------------------------------
    model   = RotatE(n_entities=n_entities, embed_dim=ROTATE_EMBED_DIM).to(DEVICE)
    sampler = NegativeSampler(n_compounds, n_labels, n_neg=ROTATE_NEG_SAMPLES)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[RotatE] Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=ROTATE_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    pos_heads = pos_heads.to(DEVICE)
    pos_tails = pos_tails.to(DEVICE)

    # -- Training loop --------------------------------------------------------
    best_loss = float("inf")
    batch_size = 1024
    ckpt_path  = os.path.join(CHECKPOINTS_DIR, "rotate_best.pth")

    print(f"\n{'='*65}")
    print(f"Training RotatE: {n_epochs} epoch(s) | "
          f"embed_dim={ROTATE_EMBED_DIM} | neg_samples={ROTATE_NEG_SAMPLES}")
    print(f"{'='*65}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()

        # Shuffle positive triples
        perm = torch.randperm(n_triples, device=DEVICE)
        ph   = pos_heads[perm]
        pt   = pos_tails[perm]

        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, n_triples, batch_size):
            bh = ph[i : i + batch_size]
            bt = pt[i : i + batch_size]
            neg_h, neg_t = sampler.sample(bh, bt, device=DEVICE)

            optimizer.zero_grad()
            loss = model(bh, bt, neg_h, neg_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:3d}/{n_epochs}] "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {lr:.6f} | "
            f"Time: {time.time()-t0:.1f}s"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "loss":        avg_loss,
                "n_entities":  n_entities,
                "n_compounds": n_compounds,
                "embed_dim":   ROTATE_EMBED_DIM,
            }, ckpt_path)

    print(f"\n{'='*65}")
    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"{'='*65}")

    # -- Save compound embeddings for FusedModel ------------------------------
    print("[RotatE] Extracting compound embeddings...")
    model.eval()
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state"])
    with torch.no_grad():
        compound_embs = model.get_compound_embeddings(n_compounds, device=DEVICE).cpu().numpy()

    emb_path = os.path.join(OUTPUTS_DIR, ROTATE_EMBEDDINGS_FILE)
    np.save(emb_path, compound_embs)
    print(f"[RotatE] Embeddings saved: {emb_path}  shape: {compound_embs.shape}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RotatE KG embeddings (Phase 3)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    args = parser.parse_args()
    train(n_epochs=args.epochs)

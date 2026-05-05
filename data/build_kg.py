"""
build_kg.py -- Knowledge Graph Construction (Phase 3)
=======================================================
Builds a bipartite Knowledge Graph from the MoA training labels:

    Compound nodes  (23,814 nodes, each with a 879-dim feature vector)
    MoA label nodes (206 nodes, each with a small random init embedding)
    Edges           compound -> MoA_label  if label == 1 (positive MoA)
                    + reverse edges (undirected graph)

Saves to:
    data/processed/kg_graph.pt          -- torch dict with edge_index, features, stats
    data/processed/kg_node_features.npy -- node feature matrix (n_nodes, 879)

USAGE:
    python data/build_kg.py
"""

import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    PROCESSED_DIR, N_TOTAL_FEATURES, KG_GRAPH_FILE,
    KG_EMBEDDINGS_FILE, RANDOM_SEED,
)
from utils.seed import set_seed
set_seed(RANDOM_SEED)


def load_arrays():
    """Load X_train.npy, y_train.npy, moa_columns.npy from processed dir."""
    X  = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y  = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    moa_names = list(np.load(os.path.join(PROCESSED_DIR, "moa_columns.npy"), allow_pickle=True))
    print(f"[KG] Loaded  X:{X.shape}  y:{y.shape}  MoA classes:{len(moa_names)}")
    return X, y, moa_names


def build_edge_list(y):
    """
    Build undirected bipartite edge list.

    Node IDs:
        Compound i  -> i
        MoA label j -> n_compounds + j
    """
    n_compounds, n_labels = y.shape
    cmp_ids, lab_ids = np.where(y == 1)
    src = cmp_ids.astype(np.int64)
    dst = (n_compounds + lab_ids).astype(np.int64)
    # Undirected: add reverse edges
    all_src = np.concatenate([src, dst])
    all_dst = np.concatenate([dst, src])
    edge_index = np.stack([all_src, all_dst], axis=0)
    print(f"[KG] Edges: {edge_index.shape[1]:,}  ({edge_index.shape[1]//2:,} unique MoA pairs x2 undirected)")
    return edge_index


def build_node_features(X, n_compounds, n_labels):
    """
    Compound nodes: real 879-dim feature vectors.
    MoA label nodes: small random init (same dim, padded).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    label_feats = rng.normal(0, 0.01, (n_labels, N_TOTAL_FEATURES)).astype(np.float32)
    node_features = np.concatenate([X.astype(np.float32), label_feats], axis=0)
    print(f"[KG] Node feature matrix: {node_features.shape}")
    return node_features


def compute_stats(y, edge_index, n_compounds, n_labels):
    """Print and return KG statistics dict."""
    n_pos = int(y.sum())
    density = n_pos / y.size * 100
    label_counts = y.sum(axis=0)
    avg_moa = y.sum(axis=1).mean()

    print(f"\n{'='*60}")
    print(f"  KNOWLEDGE GRAPH STATISTICS")
    print(f"{'='*60}")
    print(f"  Total nodes      : {n_compounds + n_labels:,}  ({n_compounds:,} compounds + {n_labels} MoA labels)")
    print(f"  Total edges      : {edge_index.shape[1]:,}")
    print(f"  Positive labels  : {n_pos:,}  ({density:.2f}% density)")
    print(f"  Avg MoA / drug   : {avg_moa:.2f}")
    print(f"  Max label freq   : {int(label_counts.max())}")
    print(f"  Min label freq   : {int(label_counts.min())}")
    print(f"  Labels with <5   : {int((label_counts < 5).sum())} / {n_labels}")
    print(f"{'='*60}\n")

    return {
        "n_nodes":          n_compounds + n_labels,
        "n_compound_nodes": n_compounds,
        "n_label_nodes":    n_labels,
        "n_edges":          int(edge_index.shape[1]),
        "n_positive_edges": n_pos,
        "graph_density":    round(density, 4),
        "avg_moa_per_drug": round(float(avg_moa), 4),
    }


def build_knowledge_graph():
    """Full pipeline: load -> edges -> features -> stats -> save."""
    t0 = time.time()
    print(f"\n{'='*60}\n  Building MoA Knowledge Graph\n{'='*60}")

    X, y, moa_names = load_arrays()
    n_compounds, n_labels = y.shape

    edge_index    = build_edge_list(y)
    node_features = build_node_features(X, n_compounds, n_labels)
    stats         = compute_stats(y, edge_index, n_compounds, n_labels)

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    graph_path = os.path.join(PROCESSED_DIR, KG_GRAPH_FILE)
    feat_path  = os.path.join(PROCESSED_DIR, KG_EMBEDDINGS_FILE)

    torch.save({
        "edge_index":    torch.tensor(edge_index,    dtype=torch.long),
        "node_features": torch.tensor(node_features, dtype=torch.float32),
        "moa_names":     moa_names,
        "n_compounds":   n_compounds,
        "n_labels":      n_labels,
        "stats":         stats,
    }, graph_path)
    np.save(feat_path, node_features)

    print(f"[KG] Saved: {graph_path}")
    print(f"[KG] Node features: {feat_path}")
    print(f"[KG] Done in {time.time()-t0:.1f}s")
    return stats


if __name__ == "__main__":
    build_knowledge_graph()

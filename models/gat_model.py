"""
gat_model.py -- Graph Attention Network for MoA Prediction (Phase 3)
=====================================================================
Implements a 2-layer Graph Attention Network (GAT) that operates on the
bipartite compound-MoA knowledge graph built by data/build_kg.py.

WHAT IS A GAT?
A Graph Attention Network (Velickovic et al., 2018) is a neural network
that learns representations of nodes in a graph by aggregating information
from their neighbors. Unlike a basic GCN that averages neighbor features
equally, GAT learns ATTENTION WEIGHTS: which neighbors are most important
for computing each node's embedding.

HOW IT FITS INTO OUR SYSTEM:
    KG Graph (compound <-> MoA edges)
         |
    GATEncoder (2 layers, 4 heads)
         |
    256-dim compound embeddings
         |
    FusedModel (concatenated with MultiModalMLP + RotatE features)

IMPORTANT NOTE ON TORCH-GEOMETRIC:
This GAT is implemented WITHOUT requiring torch_geometric, using only
standard PyTorch sparse operations. This avoids complex installation
on Windows and makes the code fully self-contained.

Architecture:
    Layer 1: 879-dim -> (128 * 4 heads) = 512-dim  (multi-head, concat)
    Layer 2: 512-dim -> 256-dim                      (single head, mean)

Checkpoint: checkpoints/gat_best.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    N_MOA_CLASSES, GAT_HIDDEN_DIM, GAT_OUT_DIM,
    GAT_N_HEADS, GAT_DROPOUT,
)


# =============================================================================
# SINGLE GAT LAYER (custom, no torch_geometric dependency)
# =============================================================================

class GATLayer(nn.Module):
    """
    A single Graph Attention Layer.

    For each node u, computes:
        h_u' = sigma( sum_{v in N(u)} alpha_{uv} * W * h_v )

    where alpha_{uv} are learned attention coefficients (softmax over neighbors).

    Multi-head: run H independent attention heads, then concatenate (or average).

    Args:
        in_dim   (int): Input feature dimension per node
        out_dim  (int): Output feature dimension per attention head
        n_heads  (int): Number of parallel attention heads
        dropout  (float): Dropout on attention coefficients
        concat   (bool): If True, concatenate head outputs (out = n_heads * out_dim)
                         If False, average head outputs (out = out_dim)
    """

    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.3, concat=True):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat  = concat
        self.dropout = dropout

        # Linear projection: maps node features to key/query/value space
        # One projection per head
        self.W = nn.Linear(in_dim, n_heads * out_dim, bias=False)

        # Attention coefficients: learnable vectors a_l and a_r per head
        # alpha_{uv} = LeakyReLU( a_l * W*h_u + a_r * W*h_v )
        self.a_src = nn.Parameter(torch.zeros(n_heads, out_dim))  # Source (left) attn
        self.a_dst = nn.Parameter(torch.zeros(n_heads, out_dim))  # Dest  (right) attn

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.attn_drop  = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Glorot uniform (recommended for GAT)."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, x, edge_index):
        """
        Args:
            x          (Tensor): Node feature matrix, shape (n_nodes, in_dim)
            edge_index (Tensor): Edge list, shape (2, n_edges). edge_index[0]=src, edge_index[1]=dst

        Returns:
            Tensor: Updated node embeddings.
                If concat=True:  shape (n_nodes, n_heads * out_dim)
                If concat=False: shape (n_nodes, out_dim)
        """
        n_nodes = x.size(0)
        src_ids = edge_index[0]  # Source node indices
        dst_ids = edge_index[1]  # Destination node indices

        # Step 1: Project all node features
        # Wh: (n_nodes, n_heads * out_dim) -> reshape to (n_nodes, n_heads, out_dim)
        Wh = self.W(x).view(n_nodes, self.n_heads, self.out_dim)

        # Step 2: Compute attention scores for each edge
        # For edge (u -> v): e_{uv} = LeakyReLU(a_src . Wh_u + a_dst . Wh_v)
        # a_src: (n_heads, out_dim), Wh[src_ids]: (n_edges, n_heads, out_dim)
        Wh_src = Wh[src_ids]  # (n_edges, n_heads, out_dim)
        Wh_dst = Wh[dst_ids]  # (n_edges, n_heads, out_dim)

        # Dot with attention vectors -> (n_edges, n_heads)
        e_src = (Wh_src * self.a_src).sum(dim=-1)  # (n_edges, n_heads)
        e_dst = (Wh_dst * self.a_dst).sum(dim=-1)  # (n_edges, n_heads)
        e     = self.leaky_relu(e_src + e_dst)       # (n_edges, n_heads)

        # Step 3: Softmax over in-neighbors for each destination node
        # We need: alpha_{uv} = exp(e_{uv}) / sum_{k in N(v)} exp(e_{kv})
        # We compute this using scatter softmax over dst_ids
        alpha = self._sparse_softmax(e, dst_ids, n_nodes)  # (n_edges, n_heads)
        alpha = self.attn_drop(alpha)

        # Step 4: Weighted aggregation
        # For each destination v: h_v' = sum_{u} alpha_{uv} * Wh_u
        # Expand alpha to (n_edges, n_heads, out_dim) and multiply
        alpha_expanded = alpha.unsqueeze(-1)              # (n_edges, n_heads, 1)
        weighted       = Wh_src * alpha_expanded          # (n_edges, n_heads, out_dim)

        # Scatter-add to destination nodes
        out = torch.zeros(n_nodes, self.n_heads, self.out_dim, device=x.device)
        idx = dst_ids.unsqueeze(-1).unsqueeze(-1).expand_as(weighted)
        out.scatter_add_(0, idx, weighted)  # (n_nodes, n_heads, out_dim)

        # Step 5: Combine heads
        if self.concat:
            # Concatenate: (n_nodes, n_heads * out_dim)
            out = out.view(n_nodes, self.n_heads * self.out_dim)
        else:
            # Average: (n_nodes, out_dim)
            out = out.mean(dim=1)

        return F.elu(out)

    @staticmethod
    def _sparse_softmax(e, dst_ids, n_nodes):
        """
        Compute softmax over neighbors for each destination node.

        For each destination node v, normalizes attention scores of all
        incoming edges using softmax.

        Args:
            e       (Tensor): Raw scores (n_edges, n_heads)
            dst_ids (Tensor): Destination node indices (n_edges,)
            n_nodes (int): Total number of nodes

        Returns:
            Tensor: Normalized attention weights (n_edges, n_heads)
        """
        # Subtract max for numerical stability (per-node max)
        # Expand dst_ids to (n_edges, n_heads)
        n_heads   = e.size(1)
        idx_exp   = dst_ids.unsqueeze(-1).expand_as(e)  # (n_edges, n_heads)

        # Compute per-node max for stability
        e_max = torch.zeros(n_nodes, n_heads, device=e.device)
        e_max.scatter_reduce_(0, idx_exp, e, reduce="amax", include_self=True)
        e_shifted = e - e_max[dst_ids]  # (n_edges, n_heads)

        # Exp and sum
        exp_e  = torch.exp(e_shifted)
        exp_sum = torch.zeros(n_nodes, n_heads, device=e.device)
        exp_sum.scatter_add_(0, idx_exp, exp_e)

        # Normalize
        alpha = exp_e / (exp_sum[dst_ids] + 1e-16)
        return alpha


# =============================================================================
# 2-LAYER GAT ENCODER
# =============================================================================

class GATEncoder(nn.Module):
    """
    2-layer GAT encoder for the bipartite compound-MoA graph.

    Architecture:
        Layer 1: in_dim -> GAT_HIDDEN_DIM * GAT_N_HEADS  (multi-head concat)
                 = 879 -> 128 * 4 = 512
        BatchNorm + Dropout
        Layer 2: 512 -> GAT_OUT_DIM                       (single head, mean)
                 = 512 -> 256

    Only the COMPOUND node embeddings (first n_compounds rows) are used
    as output features downstream.

    Args:
        in_dim      (int): Node feature dimension (879 for our graph)
        hidden_dim  (int): Per-head hidden dim in layer 1 (default: GAT_HIDDEN_DIM=128)
        out_dim     (int): Output embedding dim (default: GAT_OUT_DIM=256)
        n_heads     (int): Attention heads in layer 1 (default: GAT_N_HEADS=4)
        dropout     (float): Dropout probability (default: GAT_DROPOUT=0.3)
    """

    def __init__(
        self,
        in_dim=879,
        hidden_dim=GAT_HIDDEN_DIM,
        out_dim=GAT_OUT_DIM,
        n_heads=GAT_N_HEADS,
        dropout=GAT_DROPOUT,
    ):
        super().__init__()
        self.dropout = dropout

        # Layer 1: multi-head concat
        self.gat1 = GATLayer(
            in_dim=in_dim,
            out_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            concat=True,
        )
        l1_out_dim = hidden_dim * n_heads  # 128 * 4 = 512

        self.bn1 = nn.BatchNorm1d(l1_out_dim)

        # Layer 2: single head, mean aggregation -> out_dim
        self.gat2 = GATLayer(
            in_dim=l1_out_dim,
            out_dim=out_dim,
            n_heads=1,
            dropout=dropout,
            concat=False,
        )
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.out_dim = out_dim

    def forward(self, x, edge_index):
        """
        Full forward pass through the 2-layer GAT.

        Args:
            x          (Tensor): All node features, shape (n_nodes, in_dim)
            edge_index (Tensor): Edge list, shape (2, n_edges)

        Returns:
            Tensor: All node embeddings, shape (n_nodes, out_dim)
        """
        # Layer 1
        h = self.gat1(x, edge_index)             # (n_nodes, n_heads * hidden_dim)
        h = self.bn1(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.gat2(h, edge_index)             # (n_nodes, out_dim)
        h = self.bn2(h)

        return h  # (n_nodes, out_dim=256)


# =============================================================================
# STANDALONE SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("[GATModel] Running smoke test...")
    torch.manual_seed(42)

    # Simulate a small graph: 100 compound nodes + 10 MoA label nodes
    n_nodes    = 110
    in_dim     = 879
    n_edges    = 300

    x  = torch.randn(n_nodes, in_dim)
    ei = torch.randint(0, n_nodes, (2, n_edges))

    model = GATEncoder(in_dim=in_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  GATEncoder | Parameters: {n_params:,}")

    out = model(x, ei)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}  (expected: [{n_nodes}, {GAT_OUT_DIM}])")
    assert out.shape == (n_nodes, GAT_OUT_DIM), "Shape mismatch!"
    print("[GATModel] Smoke test PASSED.")

"""
rotate.py -- RotatE Knowledge Graph Embedding Model (Phase 3)
=============================================================
Implements RotatE (Sun et al., 2019) for the compound-MoA knowledge graph.

WHAT IS ROTATE?
RotatE models entities as complex-valued vectors and relations as rotations
in complex space. For a triple (head, relation, tail):
    score(h, r, t) = || h * r - t ||  (element-wise complex multiplication)

For our single-relation graph (compound -[has_moa]-> MoA_label), RotatE
learns:
    - Compound embeddings encoding biological "position" in MoA space
    - MoA label embeddings encoding the biological cluster
    - A single rotation vector r (shared across all has_moa edges)

WHY THIS HELPS:
RotatE is trained SELF-SUPERVISED on the KG edges (no MoA labels needed
during embedding training). The resulting compound embeddings capture
co-occurrence patterns: compounds sharing many MoA labels will have
similar embeddings. These are concatenated with MultiModalMLP + GAT features
in the FusedModel, providing a complementary "relational" signal.

TRAINING OBJECTIVE (Self-adversarial negative sampling):
    L = -log(sigma(gamma - d_pos)) - sum_i p_i * log(sigma(d_neg_i - gamma))

Architecture:
    Entities: n_compounds + n_labels nodes, each embedded as a d-dim complex vector
    Relation: single rotation vector r, same dim

Checkpoint: checkpoints/rotate_best.pth
Outputs:    outputs/rotate_embeddings.npy  (compound embeddings only, shape n_compounds x d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    ROTATE_EMBED_DIM, ROTATE_MARGIN, ROTATE_NEG_SAMPLES, RANDOM_SEED,
)


class RotatE(nn.Module):
    """
    RotatE Knowledge Graph Embedding model.

    Embeds all KG entities (compounds + MoA labels) in complex space.
    The single 'has_moa' relation is represented as a rotation vector.

    Scoring function for triple (h, r, t):
        score = gamma - || h * r - t ||_2

    where h, r, t are complex vectors (stored as 2 * embed_dim real floats).

    Args:
        n_entities  (int): Total number of KG nodes (compounds + MoA labels)
        embed_dim   (int): Complex embedding dimension (default: ROTATE_EMBED_DIM=128)
                           Internally stored as 2*embed_dim reals (real + imaginary parts)
        margin      (float): Fixed margin gamma (default: ROTATE_MARGIN=6.0)
    """

    def __init__(self, n_entities, embed_dim=ROTATE_EMBED_DIM, margin=ROTATE_MARGIN):
        super().__init__()
        self.n_entities = n_entities
        self.embed_dim  = embed_dim
        self.margin     = margin

        # Entity embeddings: store as real vectors of size 2*embed_dim
        # First embed_dim dims = real part; second embed_dim dims = imaginary part
        self.entity_emb = nn.Embedding(n_entities, embed_dim * 2)

        # Single relation (has_moa): phase angle in [-pi, pi]
        # Stored as embed_dim angles; complex rotation = exp(i * angle) = cos + i*sin
        self.relation_phase = nn.Parameter(torch.zeros(1, embed_dim))

        self._init_weights()

    def _init_weights(self):
        """Initialize entity embeddings with uniform distribution."""
        embedding_range = (self.margin + 2.0) / self.embed_dim
        nn.init.uniform_(self.entity_emb.weight, -embedding_range, embedding_range)
        nn.init.uniform_(self.relation_phase, -math.pi, math.pi)

    def _complex_mult(self, h_re, h_im, r_re, r_im):
        """
        Element-wise complex multiplication: (h_re + i*h_im) * (r_re + i*r_im)

        Returns:
            (re, im) tuple of the product
        """
        re = h_re * r_re - h_im * r_im
        im = h_re * r_im + h_im * r_re
        return re, im

    def score(self, head_ids, tail_ids):
        """
        Compute RotatE score for a batch of (head, tail) pairs.

        Score = gamma - || h * r - t ||_2
        Higher score = more likely to be a true triple.

        Args:
            head_ids (Tensor): Compound node IDs, shape (batch,)
            tail_ids (Tensor): MoA label node IDs, shape (batch,)

        Returns:
            Tensor: Scores, shape (batch,)
        """
        d = self.embed_dim

        # Get entity embeddings and split into real/imaginary parts
        h = self.entity_emb(head_ids)   # (batch, 2*d)
        t = self.entity_emb(tail_ids)   # (batch, 2*d)

        h_re, h_im = h[:, :d], h[:, d:]  # (batch, d) each
        t_re, t_im = t[:, :d], t[:, d:]

        # Compute relation rotation: r = exp(i * phase) = (cos, sin)
        phase = self.relation_phase  # (1, d)
        r_re  = torch.cos(phase)     # (1, d)
        r_im  = torch.sin(phase)     # (1, d)

        # Rotate head by relation: hr = h * r
        hr_re, hr_im = self._complex_mult(h_re, h_im, r_re, r_im)

        # Distance: || hr - t ||_2
        diff_re = hr_re - t_re  # (batch, d)
        diff_im = hr_im - t_im  # (batch, d)
        dist = torch.sqrt(diff_re ** 2 + diff_im ** 2 + 1e-9)  # (batch, d)
        dist = dist.sum(dim=-1)  # (batch,)

        return self.margin - dist

    def forward(self, pos_heads, pos_tails, neg_heads, neg_tails):
        """
        Compute self-adversarial negative sampling loss.

        Loss = -log(sigma(score_pos)) - mean(log(sigma(-score_neg)))

        Args:
            pos_heads (Tensor): Positive triple head IDs, (batch,)
            pos_tails (Tensor): Positive triple tail IDs, (batch,)
            neg_heads (Tensor): Negative triple head IDs, (batch * neg_samples,)
            neg_tails (Tensor): Negative triple tail IDs, (batch * neg_samples,)

        Returns:
            Tensor: Scalar loss
        """
        pos_score = self.score(pos_heads, pos_tails)  # (batch,)
        neg_score = self.score(neg_heads, neg_tails)  # (batch * neg,)

        pos_loss = -F.logsigmoid(pos_score).mean()
        neg_loss = -F.logsigmoid(-neg_score).mean()

        return pos_loss + neg_loss

    def get_compound_embeddings(self, n_compounds, device="cpu"):
        """
        Extract the real-valued compound embeddings for use in FusedModel.

        Concatenates real and imaginary parts -> 2*embed_dim vector per compound.
        Then projects to embed_dim via mean (real + imaginary average).

        Args:
            n_compounds (int): Number of compound nodes (0..n_compounds-1)
            device: torch device

        Returns:
            Tensor: Compound embeddings, shape (n_compounds, embed_dim)
        """
        ids  = torch.arange(n_compounds, device=device)
        embs = self.entity_emb(ids)  # (n_compounds, 2*embed_dim)
        d    = self.embed_dim
        # Average real and imaginary parts -> embed_dim
        re   = embs[:, :d]
        im   = embs[:, d:]
        return (re + im) / 2.0  # (n_compounds, embed_dim)


# =============================================================================
# NEGATIVE SAMPLER
# =============================================================================

class NegativeSampler:
    """
    Generates negative triples for RotatE training.

    Strategy: corrupt either head or tail (50% each) with a random entity.
    For a compound -> MoA edge:
        - Corrupt head: replace compound with a random compound
        - Corrupt tail: replace MoA label with a random MoA label

    Args:
        n_compounds (int): Number of compound nodes
        n_labels    (int): Number of MoA label nodes
        n_neg       (int): Number of negatives per positive
    """

    def __init__(self, n_compounds, n_labels, n_neg=ROTATE_NEG_SAMPLES):
        self.n_compounds = n_compounds
        self.n_labels    = n_labels
        self.n_neg       = n_neg

    def sample(self, pos_heads, pos_tails, device="cpu"):
        """
        Generate negative triples by corrupting positive triples.

        Args:
            pos_heads (Tensor): (batch,) compound node IDs
            pos_tails (Tensor): (batch,) MoA label node IDs
            device: torch device

        Returns:
            neg_heads (Tensor): (batch * n_neg,)
            neg_tails (Tensor): (batch * n_neg,)
        """
        batch = pos_heads.size(0)
        neg_h_list, neg_t_list = [], []

        for _ in range(self.n_neg):
            corrupt_head = torch.rand(batch) < 0.5  # 50% chance to corrupt head

            # Corrupt head: random compound (0..n_compounds-1)
            rand_compounds = torch.randint(0, self.n_compounds, (batch,), device=device)
            # Corrupt tail: random MoA label (n_compounds..n_compounds+n_labels-1)
            rand_labels = torch.randint(
                self.n_compounds, self.n_compounds + self.n_labels, (batch,), device=device
            )

            neg_h = torch.where(corrupt_head.to(device), rand_compounds, pos_heads)
            neg_t = torch.where(corrupt_head.to(device), pos_tails,      rand_labels)

            neg_h_list.append(neg_h)
            neg_t_list.append(neg_t)

        return torch.cat(neg_h_list), torch.cat(neg_t_list)


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("[RotatE] Running smoke test...")
    torch.manual_seed(42)

    n_compounds = 100
    n_labels    = 20
    n_entities  = n_compounds + n_labels

    model   = RotatE(n_entities=n_entities)
    sampler = NegativeSampler(n_compounds, n_labels, n_neg=4)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  RotatE | Parameters: {n_params:,}")

    # Simulate a batch of positive triples
    pos_h = torch.randint(0, n_compounds, (32,))
    pos_t = torch.randint(n_compounds, n_entities, (32,))
    neg_h, neg_t = sampler.sample(pos_h, pos_t)

    loss = model(pos_h, pos_t, neg_h, neg_t)
    print(f"  Loss: {loss.item():.4f}  (expected: non-negative scalar)")

    compound_embs = model.get_compound_embeddings(n_compounds)
    print(f"  Compound embeddings: {compound_embs.shape}  (expected: [{n_compounds}, {ROTATE_EMBED_DIM}])")

    assert compound_embs.shape == (n_compounds, ROTATE_EMBED_DIM)
    print("[RotatE] Smoke test PASSED.")

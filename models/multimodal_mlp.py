"""
multimodal_mlp.py — Multi-Branch Multi-Modal MLP (Phase 2)
============================================================
Improves over the MLP Baseline by splitting the 879-dim input into
three SEPARATE encoders — one per data modality — then fusing them.

WHY THIS IS BETTER THAN A FLAT MLP:
A flat MLP treats gene expression, cell viability, and experimental
metadata as a single undifferentiated vector. In reality, these three
feature groups have very different statistical properties and biological
roles. Giving each group its own dedicated encoder allows the model to
learn modality-specific representations before combining them.

  Gene expression  (772 dims) -> GeneEncoder   -> 256-dim embedding
  Cell viability   (100 dims) -> CellEncoder   -> 128-dim embedding
  Experimental meta (7 dims)  -> MetaEncoder   -> 32-dim embedding
                                               -> Fusion (416) -> 206 MoA probs

ADDITIONAL IMPROVEMENTS OVER BASELINE:
  - Residual (skip) connections inside each encoder branch
  - Label smoothing in the focal loss (eps=0.05) to reduce overconfidence
  - Heavier dropout (0.4) in the fusion head
  - BatchNorm after fusion concat

TARGET: AUROC 0.810-0.830 (vs baseline 0.7525)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import N_MOA_CLASSES, FOCAL_ALPHA, FOCAL_GAMMA


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SPLIT INDICES (positions inside the 879-dim feature vector)
# ─────────────────────────────────────────────────────────────────────────────
# After preprocessing, feature_columns.npy has the order:
#   [gene cols (0..771), cell cols (772..871), meta cols (872..878)]
# These slices let us split the flat tensor into 3 modality sub-tensors.

GENE_SLICE = slice(0,   772)   # Indices 0-771   → 772 gene expression features
CELL_SLICE = slice(772, 872)   # Indices 772-871  → 100 cell viability features
META_SLICE = slice(872, 879)   # Indices 872-878  → 7 one-hot metadata features


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Build a single encoder block
# ─────────────────────────────────────────────────────────────────────────────

def _make_block(in_dim, out_dim, dropout=0.3):
    """
    Create one Linear -> BatchNorm -> GELU -> Dropout block.

    This is the basic building unit used in all three encoder branches.
    Wrapping it in a function avoids repeating the same 4-line pattern.

    Args:
        in_dim  (int): Input feature dimension
        out_dim (int): Output feature dimension
        dropout (float): Dropout probability

    Returns:
        nn.Sequential: The four-layer block
    """
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),   # Learnable weight matrix W*x + b
        nn.BatchNorm1d(out_dim),       # Normalize batch to mean=0, std=1
        nn.GELU(),                     # Smooth non-linearity (better than ReLU)
        nn.Dropout(p=dropout),         # Randomly zero out neurons during training
    )


# ─────────────────────────────────────────────────────────────────────────────
# GENE ENCODER BRANCH
# ─────────────────────────────────────────────────────────────────────────────

class GeneEncoder(nn.Module):
    """
    Encodes the 772 gene expression features into a 256-dim representation.

    Architecture: 772 -> 1024 -> 512 -> 256 (with residual skip from 512->256)

    RESIDUAL CONNECTION: Instead of just passing output of one layer to the
    next, we ADD the input to the output: output = F(input) + input.
    This creates a "shortcut" that lets gradients flow more easily during
    backpropagation, preventing vanishing gradients in deeper networks.
    It's the key idea from ResNet (which revolutionized image recognition in 2015).

    Here we use a PROJECTED residual: since 512 != 256, we project with a
    1x1 Linear before adding.
    """

    def __init__(self, in_dim=772, out_dim=256, dropout=0.3):
        super().__init__()

        # First two layers — expand then compress
        self.layer1 = _make_block(in_dim, 1024, dropout)    # 772 -> 1024
        self.layer2 = _make_block(1024, 512, dropout)       # 1024 -> 512
        self.layer3 = _make_block(512, out_dim, dropout)    # 512 -> 256

        # Residual projection: maps 512 -> 256 so we can add layer2 output
        # to layer3 output (they must be same dimension to add)
        self.residual_proj = nn.Linear(512, out_dim, bias=False)

        # Kaiming weight initialization for stable training start
        nn.init.kaiming_uniform_(self.residual_proj.weight, nonlinearity='relu')

    def forward(self, x):
        """
        Args:
            x: Gene expression tensor, shape (batch_size, 772)
        Returns:
            256-dim embedding tensor, shape (batch_size, 256)
        """
        x = self.layer1(x)          # (batch, 1024)
        h = self.layer2(x)          # (batch, 512) — save for residual
        out = self.layer3(h)        # (batch, 256)
        out = out + self.residual_proj(h)  # Add projected residual: (batch, 256)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# CELL VIABILITY ENCODER BRANCH
# ─────────────────────────────────────────────────────────────────────────────

class CellEncoder(nn.Module):
    """
    Encodes the 100 cell viability features into a 128-dim representation.

    Architecture: 100 -> 256 -> 128 (with residual skip from input to output)

    Cell viability features capture HOW ALIVE cells are after drug treatment.
    This is complementary to gene expression (WHAT GENES ARE ACTIVE).
    Giving it a separate encoder helps the model learn cell-death patterns
    independently before combining them with gene-level patterns.
    """

    def __init__(self, in_dim=100, out_dim=128, dropout=0.3):
        super().__init__()

        self.layer1 = _make_block(in_dim, 256, dropout)      # 100 -> 256
        self.layer2 = _make_block(256, out_dim, dropout)     # 256 -> 128

        # Residual from input (100) to output (128) — needs projection
        self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.kaiming_uniform_(self.residual_proj.weight, nonlinearity='relu')

    def forward(self, x):
        """
        Args:
            x: Cell viability tensor, shape (batch_size, 100)
        Returns:
            128-dim embedding, shape (batch_size, 128)
        """
        residual = self.residual_proj(x)   # Project input for the skip connection
        x = self.layer1(x)                 # (batch, 256)
        x = self.layer2(x)                 # (batch, 128)
        return x + residual                # Residual addition: (batch, 128)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTAL METADATA ENCODER BRANCH
# ─────────────────────────────────────────────────────────────────────────────

class MetaEncoder(nn.Module):
    """
    Encodes the 7 experimental metadata features into a 32-dim representation.

    The 7 features are one-hot encoded from:
      - cp_type (trt_cp vs ctl_vehicle): Is this a real drug or a control?
      - cp_dose (D1 vs D2): Low dose or high dose?
      - cp_time (24h, 48h, 72h): How long was the treatment?

    These are EXPERIMENTAL DESIGN variables, not drug properties.
    Keeping them separate avoids contaminating the biological signal
    with the experimental setup signal.
    """

    def __init__(self, in_dim=7, out_dim=32, dropout=0.2):
        super().__init__()

        # Small network — only 7 input features, so we don't need many layers
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),   # 7 -> 64 — expand first to learn interactions
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, out_dim),  # 64 -> 32 — compress to embedding
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        """
        Args:
            x: Metadata tensor, shape (batch_size, 7)
        Returns:
            32-dim embedding, shape (batch_size, 32)
        """
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# FUSION HEAD
# ─────────────────────────────────────────────────────────────────────────────

class FusionHead(nn.Module):
    """
    Takes the concatenated embeddings from all 3 branches and maps to 206 MoA logits.

    Fusion input dimension = 256 (gene) + 128 (cell) + 32 (meta) = 416

    Architecture: 416 -> 512 -> 256 -> 206

    The fusion head learns HOW TO COMBINE the three embeddings.
    For example, it might learn: "If the gene embedding looks like a kinase inhibitor
    AND the cell embedding shows moderate toxicity AND the treatment was at high dose,
    then predict a higher probability for kinase_inhibitor MoA."
    """

    def __init__(self, in_dim=416, out_dim=N_MOA_CLASSES, dropout=0.4):
        super().__init__()

        self.net = nn.Sequential(
            # First fusion layer: expand to understand cross-modal interactions
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=dropout),   # Higher dropout in fusion = more regularization

            # Second fusion layer: compress
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=dropout),

            # Output layer: 256 -> 206 (no activation — sigmoid applied in loss)
            nn.Linear(256, out_dim),
        )

    def forward(self, fused):
        """
        Args:
            fused: Concatenated modality embeddings, shape (batch_size, 416)
        Returns:
            Raw logits, shape (batch_size, 206)
        """
        return self.net(fused)


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-MODAL MLP — FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class MultiModalMLP(nn.Module):
    """
    Multi-branch Multi-Modal MLP for MoA prediction.

    Architecture overview:
      Input (879) -> split into 3 branches
        Branch 1: GeneEncoder(772)  -> 256-dim
        Branch 2: CellEncoder(100)  -> 128-dim
        Branch 3: MetaEncoder(7)    -> 32-dim
      Concatenate -> FusionHead(416) -> 206 logits

    Improvements over MLP Baseline:
      - Per-modality specialized encoders
      - Residual connections (prevent gradient vanishing)
      - Label smoothing focal loss (prevent overconfidence)
      - Separate dropout rates per branch (tuned to branch complexity)

    Args:
        gene_out  (int): Output dim of GeneEncoder. Default 256.
        cell_out  (int): Output dim of CellEncoder. Default 128.
        meta_out  (int): Output dim of MetaEncoder. Default 32.
        dropout   (float): Base dropout rate. Default 0.3.
        out_dim   (int): Number of MoA classes. Default 206 (from config).
    """

    def __init__(
        self,
        gene_out=256,
        cell_out=128,
        meta_out=32,
        dropout=0.3,
        out_dim=N_MOA_CLASSES,
    ):
        super().__init__()

        # Three specialized encoder branches
        self.gene_encoder = GeneEncoder(in_dim=772, out_dim=gene_out, dropout=dropout)
        self.cell_encoder = CellEncoder(in_dim=100, out_dim=cell_out, dropout=dropout)
        self.meta_encoder = MetaEncoder(in_dim=7,   out_dim=meta_out, dropout=max(0.1, dropout - 0.1))

        # Compute fusion input dim = sum of all encoder output dims
        fusion_in = gene_out + cell_out + meta_out  # 256 + 128 + 32 = 416

        # Fusion head that learns cross-modal interactions
        self.fusion = FusionHead(in_dim=fusion_in, out_dim=out_dim, dropout=dropout + 0.1)

        # Store the slice indices for splitting the input tensor
        self.gene_slice = GENE_SLICE
        self.cell_slice = CELL_SLICE
        self.meta_slice = META_SLICE

    def forward(self, x):
        """
        Forward pass: split input by modality, encode each, fuse, predict.

        Args:
            x (torch.Tensor): Full feature tensor, shape (batch_size, 879)
        Returns:
            torch.Tensor: Raw logits, shape (batch_size, 206)
        """
        # Step 1: Split the flat 879-dim vector into 3 modality sub-tensors
        x_gene = x[:, self.gene_slice]   # (batch, 772) — gene expression
        x_cell = x[:, self.cell_slice]   # (batch, 100) — cell viability
        x_meta = x[:, self.meta_slice]   # (batch, 7)   — experimental metadata

        # Step 2: Pass each modality through its dedicated encoder
        e_gene = self.gene_encoder(x_gene)   # (batch, 256)
        e_cell = self.cell_encoder(x_cell)   # (batch, 128)
        e_meta = self.meta_encoder(x_meta)   # (batch, 32)

        # Step 3: Concatenate all embeddings into a single fused representation
        # torch.cat along dim=1 stacks columns: [gene_emb | cell_emb | meta_emb]
        fused = torch.cat([e_gene, e_cell, e_meta], dim=1)  # (batch, 416)

        # Step 4: Pass fused representation through the prediction head
        logits = self.fusion(fused)   # (batch, 206)

        return logits

    def predict_proba(self, x):
        """
        Get probability predictions (sigmoid of logits) for inference.

        Args:
            x (torch.Tensor): Input features, shape (batch_size, 879)
        Returns:
            torch.Tensor: Probabilities in [0, 1], shape (batch_size, 206)
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)


# ─────────────────────────────────────────────────────────────────────────────
# LABEL-SMOOTHED FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────

class LabelSmoothedFocalLoss(nn.Module):
    """
    Focal Loss with Label Smoothing — improved version for Phase 2.

    WHAT IS LABEL SMOOTHING?
    Instead of training with hard targets (0.0 or 1.0), we "soften" them:
      positive label: 1.0 -> (1 - eps) = 0.95
      negative label: 0.0 -> eps / 2   = 0.025

    WHY: Hard targets (exactly 0 or 1) push the model to predict extreme
    probabilities (0.0 or 1.0). This causes overconfidence and hurts
    generalization. Smoothed labels keep the model slightly uncertain,
    acting as a regularizer.

    COMBINED WITH FOCAL LOSS:
    The focal (1-p_t)^gamma term down-weights easy/confident predictions,
    while label smoothing prevents the model from becoming overconfident.
    Together they provide strong regularization for multi-label learning.

    Args:
        alpha  (float): Focal loss alpha weight. Default 1.0.
        gamma  (float): Focal loss focusing parameter. Default 2.0.
        eps    (float): Label smoothing factor. Default 0.05.
    """

    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, eps=0.05):
        super().__init__()
        self.alpha = alpha   # Weight for the positive class
        self.gamma = gamma   # How much to down-weight easy/confident predictions
        self.eps   = eps     # Label smoothing strength (0 = no smoothing)

    def forward(self, logits, targets):
        """
        Compute the label-smoothed focal loss.

        Args:
            logits  (torch.Tensor): Raw model outputs before sigmoid, (batch, 206)
            targets (torch.Tensor): Binary ground truth labels, (batch, 206)
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Step 1: Apply label smoothing to targets
        # Positive labels: 1.0 -> 1.0 - eps = 0.95
        # Negative labels: 0.0 -> eps / N_classes ≈ 0.025 (we use eps/2 for simplicity)
        smooth_targets = targets * (1.0 - self.eps) + (1.0 - targets) * (self.eps / 2.0)

        # Step 2: Compute binary cross-entropy with smoothed targets
        bce = F.binary_cross_entropy_with_logits(
            logits, smooth_targets, reduction="none"
        )   # Shape: (batch, 206)

        # Step 3: Compute probabilities for focal weight calculation
        p = torch.sigmoid(logits)   # Predicted probabilities

        # Step 4: Compute p_t using ORIGINAL (unsmoothed) targets for the focal weight
        # p_t = p if target=1, (1-p) if target=0
        p_t = p * targets + (1 - p) * (1 - targets)

        # Step 5: Focal weight — reduces loss for easy/confident predictions
        focal_weight = (1.0 - p_t) ** self.gamma

        # Step 6: Combine
        focal_loss = self.alpha * focal_weight * bce

        return focal_loss.mean()   # Average over batch and all 206 classes


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — FUSED MODEL (MultiModalMLP + GAT + RotatE)
# ─────────────────────────────────────────────────────────────────────────────

class FusedModel(nn.Module):
    """
    Phase 3 Fused Model: combines three complementary representations.

    Input sources (per compound):
        1. MultiModalMLP feature extractor  → 416-dim latent
        2. GAT graph embeddings             → 256-dim (pre-computed or live)
        3. RotatE KG embeddings             → 128-dim (pre-computed)

    Fusion: concatenate → 800-dim → 2-layer head → 206 MoA logits

    WHY THREE SOURCES?
        MultiModalMLP: captures individual compound biology (gene, cell, meta)
        GAT:           captures relational structure (which MoAs co-occur)
        RotatE:        captures KG positional embedding (biological neighborhood)
    Each source provides a COMPLEMENTARY signal that the others lack.

    Args:
        gat_dim    (int): GAT embedding dimension (default 256 from config)
        rotate_dim (int): RotatE embedding dimension (default 128 from config)
        out_dim    (int): Number of MoA classes (default 206)
        dropout    (float): Dropout in the fusion head (default 0.4)
        freeze_multimodal (bool): If True, freeze MultiModalMLP during training.
                                  Useful for the first few fine-tuning epochs.
    """

    def __init__(
        self,
        gat_dim=256,
        rotate_dim=128,
        out_dim=N_MOA_CLASSES,
        dropout=0.4,
        freeze_multimodal=False,
    ):
        super().__init__()

        # ── Sub-model 1: MultiModalMLP (produces 416-dim latent) ──────────────
        self.multimodal = MultiModalMLP()
        # Remove the original classification head — we only need the encoder
        # We'll re-route the output through the fusion head instead.
        # The MultiModalMLP.forward() returns 206-dim logits; we need 416-dim latent.
        # Solution: expose an intermediate hook via get_latent()

        if freeze_multimodal:
            for p in self.multimodal.parameters():
                p.requires_grad = False

        # ── Fusion head ────────────────────────────────────────────────────────
        # 416 (multimodal) + gat_dim (256) + rotate_dim (128) = 800
        fusion_in = 416 + gat_dim + rotate_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=dropout),

            nn.Linear(256, out_dim),
        )

        self.gat_dim    = gat_dim
        self.rotate_dim = rotate_dim
        self.fusion_in  = fusion_in

    def get_multimodal_latent(self, x):
        """
        Extract 416-dim fusion latent from MultiModalMLP (before the head).

        We bypass the FusionHead of MultiModalMLP and return the
        concatenated [gene_emb | cell_emb | meta_emb] directly.

        Args:
            x (Tensor): Full compound features (batch, 879)

        Returns:
            Tensor: 416-dim latent (batch, 416)
        """
        mm = self.multimodal
        x_gene = x[:, mm.gene_slice]
        x_cell = x[:, mm.cell_slice]
        x_meta = x[:, mm.meta_slice]

        e_gene = mm.gene_encoder(x_gene)   # (batch, 256)
        e_cell = mm.cell_encoder(x_cell)   # (batch, 128)
        e_meta = mm.meta_encoder(x_meta)   # (batch, 32)

        return torch.cat([e_gene, e_cell, e_meta], dim=1)  # (batch, 416)

    def forward(self, x, gat_emb, rotate_emb):
        """
        Full forward pass through the fused model.

        Args:
            x          (Tensor): Compound feature matrix (batch, 879)
            gat_emb    (Tensor): GAT graph embeddings   (batch, 256)
            rotate_emb (Tensor): RotatE KG embeddings   (batch, 128)

        Returns:
            Tensor: Raw MoA logits (batch, 206)
        """
        # Get multimodal latent (bypasses multimodal's own head)
        latent = self.get_multimodal_latent(x)        # (batch, 416)

        # Concatenate all three representations
        fused  = torch.cat([latent, gat_emb, rotate_emb], dim=1)  # (batch, 800)

        # Predict
        return self.fusion_head(fused)                # (batch, 206)

    def predict_proba(self, x, gat_emb, rotate_emb):
        """Get sigmoid probabilities for inference."""
        return torch.sigmoid(self.forward(x, gat_emb, rotate_emb))


# ─────────────────────────────────────────────────────────────────────────────
# FUSED MODEL SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[FusedModel] Running smoke test...")
    torch.manual_seed(42)

    batch = 32
    x          = torch.randn(batch, 879)
    gat_emb    = torch.randn(batch, 256)
    rotate_emb = torch.randn(batch, 128)

    model = FusedModel()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  FusedModel | Parameters: {n_params:,}")

    logits = model(x, gat_emb, rotate_emb)
    print(f"  Output: {logits.shape}  (expected: [{batch}, 206])")
    assert logits.shape == (batch, 206)
    print("[FusedModel] Smoke test PASSED.")

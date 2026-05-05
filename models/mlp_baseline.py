"""
mlp_baseline.py — MLP Baseline Model + Focal Loss
==================================================
Defines the 6-layer Multi-Layer Perceptron (MLP) baseline model and the
custom Focal Loss function for handling class imbalance.

ARCHITECTURE: 184 → 2048 → 1024 → 512 → 256 → 128 → 206
Each hidden layer has: Linear → BatchNorm → GELU → Dropout(0.3)

FOCAL LOSS:
Standard binary cross-entropy treats all samples equally. In our dataset,
most MoA labels are 0 (the drug does NOT have that MoA). The model quickly
learns to predict "0" always because it's right 99%+ of the time.
Focal loss adds a (1-p)^gamma factor that DOWN-WEIGHTS easy/confident
predictions and FOCUSES learning on the hard/uncertain cases.
"""

import torch             # Core PyTorch library for tensor operations
import torch.nn as nn    # Neural network layers (Linear, BatchNorm, etc.)
import torch.nn.functional as F  # Functions like sigmoid, binary_cross_entropy

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    N_TOTAL_FEATURES,  # 184 — input dimension
    N_MOA_CLASSES,     # 206 — output dimension (one per MoA class)
    MLP_HIDDEN_DIMS,   # [2048, 1024, 512, 256, 128] — sizes of hidden layers
    MLP_DROPOUT,       # 0.3 — dropout probability
    FOCAL_ALPHA,       # 1.0 — focal loss alpha (positive/negative balance)
    FOCAL_GAMMA,       # 2.0 — focal loss gamma (focus on hard examples)
)


# ─────────────────────────────────────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label binary classification with class imbalance.
    
    Original paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Formula:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Where:
        p_t = predicted probability of the TRUE class
        alpha = weight for positive samples
        gamma = focusing parameter (how much to down-weight easy examples)
    
    With gamma=2: if the model is 90% confident and correct, the loss is
    reduced by (1-0.9)^2 = 0.01x compared to standard cross-entropy.
    But if the model is only 50% confident, the reduction is (1-0.5)^2 = 0.25x.
    So hard examples (low confidence) dominate the learning signal.
    
    Args:
        alpha (float): Weighting factor for the positive class. Default 1.0.
        gamma (float): Focusing parameter. Higher = more focus on hard cases.
                       0 = equivalent to standard binary cross-entropy.
    """
    
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super(FocalLoss, self).__init__()  # Initialize the parent nn.Module class
        self.alpha = alpha  # Store alpha for use in forward()
        self.gamma = gamma  # Store gamma for use in forward()
    
    def forward(self, logits, targets):
        """
        Compute the focal loss between predictions and ground truth labels.
        
        Args:
            logits  (torch.Tensor): Raw model outputs BEFORE sigmoid, shape (batch, 206)
                                    Values can be any real number (-∞ to +∞)
            targets (torch.Tensor): Ground truth binary labels, shape (batch, 206)
                                    Values are 0.0 or 1.0
        
        Returns:
            torch.Tensor: Scalar loss value (single number) averaged over the batch
        """
        
        # Step 1: Apply sigmoid to convert logits to probabilities [0, 1]
        # sigmoid(x) = 1 / (1 + e^(-x))
        # We use F.binary_cross_entropy_with_logits for numerical stability,
        # which applies sigmoid internally
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"  # "none" = return per-element loss, not mean
        )
        # bce_loss shape: (batch_size, 206) — one loss value per sample per MoA class
        
        # Step 2: Convert logits to probabilities for the focal term
        p = torch.sigmoid(logits)  # p[i,j] = predicted probability that sample i has MoA j
        
        # Step 3: Calculate p_t (predicted probability of the TRUE class)
        # If target=1: p_t = p (we want p to be HIGH → confident it's 1)
        # If target=0: p_t = 1-p (we want p to be LOW → confident it's 0)
        p_t = p * targets + (1 - p) * (1 - targets)  # Elegant formula for p_t
        
        # Step 4: Apply the focal term: (1 - p_t)^gamma
        # When p_t is close to 1 (easy/correct prediction): (1-1)^2 ≈ 0 → near-zero weight
        # When p_t is close to 0 (hard/wrong prediction):   (1-0)^2 = 1 → full weight
        focal_weight = (1 - p_t) ** self.gamma  # Shape: (batch_size, 206)
        
        # Step 5: Apply alpha weighting (balance positive vs negative samples)
        focal_loss = self.alpha * focal_weight * bce_loss  # Element-wise multiplication
        
        # Step 6: Average the loss over all samples and all 206 MoA classes
        return focal_loss.mean()  # Returns a single scalar — the batch's average focal loss


# ─────────────────────────────────────────────────────────────────────────────
# MLP BASELINE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class MLPBaseline(nn.Module):
    """
    A 6-layer Multi-Layer Perceptron (MLP) for MoA prediction.
    
    This is the BASELINE model — simple but effective. Each drug's feature
    vector (184 numbers) is passed through 6 transformation layers, and the
    final layer outputs 206 probability scores (one per MoA class).
    
    Think of each layer as a "filter" that looks for different patterns:
    - Early layers: detect simple patterns (high gene g-5, low cell c-12)
    - Middle layers: detect combinations (high g-5 AND low c-12)
    - Late layers: combine combinations into MoA-specific signatures
    
    Architecture: 184 → 2048 → 1024 → 512 → 256 → 128 → 206
    
    Args:
        input_dim  (int): Number of input features. Default: 184 (from config)
        output_dim (int): Number of MoA classes to predict. Default: 206 (from config)
        hidden_dims (list): Sizes of hidden layers. Default: [2048,1024,512,256,128]
        dropout    (float): Dropout rate. Default: 0.3 (30% of neurons dropped per step)
    """
    
    def __init__(
        self,
        input_dim  = N_TOTAL_FEATURES,   # 184 features going in
        output_dim = N_MOA_CLASSES,       # 206 MoA classes coming out
        hidden_dims = MLP_HIDDEN_DIMS,    # [2048, 1024, 512, 256, 128]
        dropout     = MLP_DROPOUT,        # 0.3
    ):
        super(MLPBaseline, self).__init__()  # Must call parent class __init__
        
        # ── Build the network layer by layer ─────────────────────────────────
        # We dynamically build layers based on hidden_dims, so the architecture
        # is easy to change (just modify MLP_HIDDEN_DIMS in config.py)
        
        layers = []  # Empty list to accumulate layer objects
        
        # Track the input size to each layer (starts with the raw feature count)
        prev_dim = input_dim  # For the first layer, input is 184
        
        # Loop through each hidden layer size in [2048, 1024, 512, 256, 128]
        for hidden_dim in hidden_dims:
            
            # LINEAR LAYER: y = W*x + b
            # Learnable weight matrix W (shape: hidden_dim × prev_dim) + bias b
            # This is the "thinking" part — maps features to new representations
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # BATCH NORMALIZATION: normalizes the output of the linear layer
            # After each mini-batch, adjusts outputs to have mean≈0, std≈1
            # Prevents "internal covariate shift" — keeps training stable
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            # GELU ACTIVATION: Gaussian Error Linear Unit
            # Adds non-linearity — without this, stacking linear layers is still
            # just a linear transformation (no more expressive than 1 layer).
            # GELU: f(x) = x * Φ(x) where Φ is the Gaussian CDF
            # Smoother than ReLU, works well for biological data
            layers.append(nn.GELU())
            
            # DROPOUT: randomly sets 30% of neurons to 0 during training
            # This forces the network to learn redundant representations —
            # no single neuron can be relied upon, making the model more robust.
            # Dropout is DISABLED during evaluation/inference automatically.
            layers.append(nn.Dropout(p=dropout))
            
            prev_dim = hidden_dim  # Next layer's input = this layer's output
        
        # FINAL OUTPUT LAYER: maps 128 → 206 (one score per MoA class)
        # NO activation here — we apply sigmoid separately in the loss function
        # for better numerical stability
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Wrap all layers in nn.Sequential — allows us to call them in order
        # with a single forward pass: output = self.network(input)
        self.network = nn.Sequential(*layers)  # *layers unpacks the list
        
        # Initialize weights using a principled scheme (helps training start well)
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize the weights of all linear layers using He (Kaiming) initialization.
        
        WHY: Default random initialization can cause gradients to vanish or explode
        in deep networks. Kaiming initialization scales weights by sqrt(2/fan_in),
        which keeps the variance of activations stable through all layers.
        
        This is particularly important for networks with many layers (6 in our case).
        """
        for module in self.modules():  # Iterate over all sub-modules (layers)
            if isinstance(module, nn.Linear):  # Only apply to Linear layers
                # Kaiming (He) uniform initialization — best for GELU/ReLU activations
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:  # Initialize bias to 0
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass: compute predictions from input features.
        
        This method is called automatically when you do: output = model(input)
        
        Args:
            x (torch.Tensor): Input feature batch, shape (batch_size, 184)
                              Each row is one drug's feature vector
        
        Returns:
            torch.Tensor: Raw logits (before sigmoid), shape (batch_size, 206)
                          Higher value = higher probability of that MoA.
                          Convert to probabilities with torch.sigmoid(output).
        """
        return self.network(x)  # Pass input through all layers in sequence
    
    def predict_proba(self, x):
        """
        Get probability predictions (after sigmoid) for a batch of inputs.
        
        Use this during inference/evaluation (not during training, where we
        use logits directly for numerical stability in the loss function).
        
        Args:
            x (torch.Tensor): Input features, shape (batch_size, 184)
        
        Returns:
            torch.Tensor: Probability scores in [0, 1], shape (batch_size, 206)
        """
        logits = self.forward(x)           # Get raw scores from the network
        return torch.sigmoid(logits)       # Convert to probabilities: p = 1/(1+e^(-logit))

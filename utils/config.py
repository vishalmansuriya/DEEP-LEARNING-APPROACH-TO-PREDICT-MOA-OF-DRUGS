"""
config.py — Centralized Configuration
======================================
All hyperparameters, file paths, and global constants are kept HERE.
This means you only need to change one file if a setting needs to change —
you never need to hunt through multiple scripts.

WHY THIS MATTERS:
If random_seed is set to 42 here, every script that imports this file
uses the same seed → reproducible experiments. If you change it once
here, all scripts see the change automatically.
"""

import os  # Standard Python library for working with file paths and directories

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42  # Fixed seed for ALL random number generators — ensures that
                  # running the same code twice gives EXACTLY the same results.
                  # This is essential for academic reproducibility (others can
                  # verify your results).

# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORY PATHS
# ─────────────────────────────────────────────────────────────────────────────

# Get the directory where THIS config.py file lives, then go up one level
# to reach the project root (the folder containing data/, models/, etc.)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root

# Sub-directories for organizing the project
DATA_DIR        = os.path.join(BASE_DIR, "data")              # Raw and processed data files
RAW_DATA_DIR    = os.path.join(DATA_DIR, "raw")               # Original Kaggle CSV files (unmodified)
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")         # Cleaned, normalized data ready for training
MODELS_DIR      = os.path.join(BASE_DIR, "models")            # Python files defining neural network architectures
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")       # Saved model weights (.pth files)
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")           # Evaluation results, predictions
FIGURES_DIR     = os.path.join(OUTPUTS_DIR, "figures")        # Plots and visualizations
DOCS_DIR        = os.path.join(BASE_DIR, "docs")              # Documentation files

# ─────────────────────────────────────────────────────────────────────────────
# DATASET FILE NAMES (relative to RAW_DATA_DIR)
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_FEATURES_FILE       = "train_features.csv"        # 23,814 rows × 876 columns of input features
TRAIN_TARGETS_SCORED_FILE = "train_targets_scored.csv"  # 23,814 rows × 207 columns (sig_id + 206 MoA labels)
TRAIN_TARGETS_NON_SCORED  = "train_targets_nonscored.csv"  # Extra labels, not part of competition scoring
TEST_FEATURES_FILE        = "test_features.csv"         # 3,983 rows — the drugs we need to predict on
SAMPLE_SUBMISSION_FILE    = "sample_submission.csv"     # Shows the correct output format

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMN PREFIXES
# ─────────────────────────────────────────────────────────────────────────────
# The Kaggle dataset uses these prefixes to distinguish feature types:
# g-0 to g-99  → gene expression (how active each gene is)
# c-0 to c-76  → cell viability (how alive different cell types are)
# Categorical   → cp_type (drug type), cp_dose (dose level), cp_time (duration)

GENE_COLS_PREFIX  = "g-"    # Prefix that identifies gene expression features
CELL_COLS_PREFIX  = "c-"    # Prefix that identifies cell viability features
N_GENE_FEATURES   = 772     # Actual gene expression features in the Kaggle dataset
N_CELL_FEATURES   = 100     # Actual cell viability features in the Kaggle dataset
N_CAT_FEATURES    = 3       # Categorical features: cp_type, cp_dose, cp_time
N_TOTAL_FEATURES  = 879     # After one-hot encoding: 772 + 100 + 7 (one-hot expanded) = 879
                             # cp_type: 2 values -> 2 bits
                             # cp_dose: 2 values -> 2 bits
                             # cp_time: 3 values -> 3 bits  -> 7 extra cols total
N_MOA_CLASSES     = 206     # Number of MoA labels we predict simultaneously

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (MLP Baseline)
# ─────────────────────────────────────────────────────────────────────────────
# These numbers define the SIZE of each layer in the 6-layer MLP.
# Bigger layers = more parameters = more capacity to learn, but slower training.

MLP_HIDDEN_DIMS = [4096, 2048, 1024, 512, 256]  # Layer sizes between input (879) and output (206)
                                                  # Wider first layer to handle the 879-dim input
                                                  # Then gradually compress down to the final answer
MLP_DROPOUT     = 0.3    # Drop 30% of neurons randomly during training to prevent overfitting
                          # Overfitting = the model memorizes training data but fails on new data
MLP_ACTIVATION  = "gelu" # Activation function -- adds non-linearity so the model can learn curves
                          # GELU is smoother than ReLU and tends to work better for biological data

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE         = 256    # How many drug samples to process at once before updating weights.
                             # Larger batches → more stable gradients but need more memory.
MAX_EPOCHS         = 200    # Maximum number of times we loop through the entire training set.
                             # Early stopping usually kicks in before this limit is reached.
EARLY_STOP_PATIENCE = 15   # If validation AUROC doesn't improve for 15 consecutive epochs,
                             # stop training early to save time and prevent overfitting.
LEARNING_RATE      = 1e-3   # How big each weight-update step is. Too large → unstable training.
                             # Too small → very slow training. 0.001 is a good starting point.
WEIGHT_DECAY       = 1e-4   # L2 regularization coefficient in AdamW. Penalizes very large weights,
                             # which helps prevent the model from becoming overly complex.
LR_T_0             = 10     # Number of epochs for the first cosine annealing cycle.
                             # After 10 epochs, the learning rate returns to its starting value.
LR_T_MULT          = 2      # Each successive cycle is 2× longer than the previous one.
                             # This allows the model to settle into progressively better optima.

# ─────────────────────────────────────────────────────────────────────────────
# FOCAL LOSS PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# Focal loss was designed for IMBALANCED datasets.
# In our case: most drugs have 0 or 1 active MoAs out of 206 → 99.5%+ zeros.
# With regular binary cross-entropy, the model learns to predict "0" always.
# Focal loss adds a (1-p)^γ term that reduces the weight of easy (confident) predictions
# and focuses the model's attention on the hard, uncertain cases.

FOCAL_ALPHA = 1.0  # Balancing factor between positive and negative samples.
                   # 1.0 = no extra balancing beyond the focal term itself.
FOCAL_GAMMA = 2.0  # Focusing parameter. γ=2 is the value from the original paper.
                   # Higher γ = more focus on hard cases. 2.0 is the recommended default.

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

N_FOLDS       = 5     # Use 5-fold cross-validation:
                       # Split data into 5 parts, train on 4, validate on 1, rotate.
                       # This gives a more reliable estimate of true performance than a single split.
SCAFFOLD_SPLIT = True  # Use scaffold-based splitting (True) or random splitting (False).
                        # Scaffold splitting is MANDATORY for drug discovery to avoid data leakage.
                        # Leakage = similar molecules in both train and test → artificially good scores.

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

PREDICTION_THRESHOLD = 0.5  # Sigmoid output > 0.5 → predict "1" (this drug has the MoA)
                              # Sigmoid output ≤ 0.5 → predict "0" (this drug does NOT have the MoA)

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

DASH_PORT = 8050   # Port number for the Dash dashboard web server
                   # Access at: http://localhost:8050
DASH_DEBUG = True  # True = shows detailed error messages in the browser (useful during development)
                   # False = production mode (no internal error details shown to users)
TOP_K_PREDICTIONS = 10  # Show the top 10 predicted MoA classes in the prediction bar chart

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — KNOWLEDGE GRAPH SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# The Knowledge Graph (KG) is a bipartite graph:
#   - Compound nodes (23,814 compounds from training data)
#   - MoA label nodes (206 MoA classes)
#   - Edges: compound -> MoA if that compound's label = 1 (positive association)
# This graph structure encodes BIOLOGICAL CO-OCCURRENCE: drugs that share MoA
# labels are likely to have similar biological mechanisms.

KG_GRAPH_FILE      = "kg_graph.pt"         # Saved PyTorch Geometric Data object
KG_EMBEDDINGS_FILE = "kg_node_features.npy" # Node feature matrix

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — GAT (GRAPH ATTENTION NETWORK) SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# GAT uses ATTENTION MECHANISMS on the graph to weight the importance of
# neighboring nodes. Instead of treating all neighbors equally (like GCN),
# GAT learns which neighbors are most informative for each node.
# This is analogous to the attention mechanism in Transformers.

GAT_HIDDEN_DIM   = 128   # Hidden dim inside each GAT attention head
GAT_OUT_DIM      = 256   # Final compound embedding dimension from GAT
GAT_N_HEADS      = 4     # Number of parallel attention heads (multi-head attention)
GAT_DROPOUT      = 0.3   # Dropout on attention coefficients
GAT_LR           = 5e-4  # Separate learning rate for GAT (lower than MLP)
GAT_MAX_EPOCHS   = 100   # GAT converges faster than the MLP
GAT_BATCH_SIZE   = 64    # Number of compound subgraphs per batch (mini-batch GNN)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — ROTATE (KNOWLEDGE GRAPH EMBEDDINGS) SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# RotatE (Sun et al., 2019) embeds KG entities as vectors in complex space.
# Relations are modeled as ROTATIONS in this space. For our single-relation
# graph (compound -[has_moa]-> label), RotatE learns:
#   - Compound embeddings that encode biological "position" in MoA space
#   - MoA label embeddings that encode the biological cluster
# The compound embeddings are then used as extra features in the fused model.

ROTATE_EMBED_DIM    = 128   # Embedding dimension (complex: 64 real + 64 imaginary)
ROTATE_MARGIN       = 6.0   # Margin gamma in the loss: higher = harder negatives
ROTATE_NEG_SAMPLES  = 64    # Number of negative triples per positive triple
ROTATE_LR           = 0.001 # Learning rate for RotatE optimizer
ROTATE_MAX_EPOCHS   = 50    # RotatE converges quickly on dense small graphs
ROTATE_EMBEDDINGS_FILE = "rotate_embeddings.npy"  # Saved in outputs/

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — FUSED MODEL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# The fused model concatenates outputs from all 3 sources:
#   MultiModalMLP latent (416-dim) + GAT embedding (256-dim) + RotatE (128-dim)
#   Total fusion input = 800-dim -> 2-layer head -> 206 MoA logits

FUSED_INPUT_DIM  = 800   # 416 (multimodal) + 256 (GAT) + 128 (RotatE)
FUSED_HIDDEN_DIM = 512   # Intermediate fusion layer
FUSED_DROPOUT    = 0.4   # Dropout in the fusion head
FUSED_LR         = 3e-4  # Lower LR for fine-tuning with pre-trained sub-models

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTION: Create directories if they don't exist
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs():
    """
    Create all necessary project directories if they do not already exist.
    Call this function at the start of any training or data processing script
    to make sure the required folder structure is in place.
    
    Args:
        None
    Returns:
        None — creates folders as a side effect
    """
    # List all directories that must exist for the project to work correctly
    dirs_to_create = [
        RAW_DATA_DIR,    # Where Kaggle CSV files are placed
        PROCESSED_DIR,   # Where preprocessed numpy arrays are saved
        CHECKPOINTS_DIR, # Where trained model weights are saved
        FIGURES_DIR,     # Where plots are saved
    ]
    
    # Loop through each directory and create it if it doesn't exist
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)  # exist_ok=True means no error if folder already exists

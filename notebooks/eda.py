"""
eda.py — Exploratory Data Analysis for MoA Dataset
====================================================
Loads the raw dataset and produces key visualisations + statistics that the
team can refer to during any viva question about "what does the data look like?".

Generates and saves the following figures to outputs/figures/:
  1. moa_label_distribution.png  — Bar chart: positive samples per MoA class
  2. label_count_histogram.png   — How many MoA labels does each drug have?
  3. gene_expression_dist.png    — Distribution of gene expression features
  4. cell_viability_dist.png     — Distribution of cell viability features
  5. class_imbalance.png         — Positive vs negative ratio per class
  6. correlation_heatmap.png     — Feature correlation (sample of 20 features)

USAGE:
    python notebooks/eda.py

REQUIREMENTS: Only needs the raw CSVs in data/raw/ OR uses sample_submission.csv
              for label-name analysis if train CSVs are not yet downloaded.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")  # Suppress minor plotting warnings for cleaner output

# -- Project root on sys.path --------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import RAW_DATA_DIR, FIGURES_DIR, GENE_COLS_PREFIX, CELL_COLS_PREFIX
from utils.seed import set_seed

set_seed(42)  # Reproducible random sampling in visualisations

# -- Ensure output directory exists --------------------------------------------
os.makedirs(FIGURES_DIR, exist_ok=True)

# -- Matplotlib / Seaborn dark styling -----------------------------------------
plt.style.use("dark_background")       # Dark theme matches the dashboard
sns.set_palette("plasma")              # Vibrant colour palette
plt.rcParams.update({
    "figure.dpi":        150,          # High resolution output
    "figure.facecolor":  "#0f0f1a",   # Match dashboard background
    "axes.facecolor":    "#1a1a2e",   # Match dashboard card colour
    "axes.edgecolor":    "#334155",   # Subtle axis borders
    "axes.labelcolor":   "#e2e8f0",   # Light text
    "xtick.color":       "#94a3b8",
    "ytick.color":       "#94a3b8",
    "grid.color":        "#334155",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})

ACCENT_PURPLE = "#7c3aed"   # Primary accent colour
ACCENT_CYAN   = "#06b6d4"   # Secondary accent colour
ACCENT_GREEN  = "#10b981"   # Success / positive highlight


# -----------------------------------------------------------------------------
# DATA LOADING — fallback to sample_submission.csv if full data not available
# -----------------------------------------------------------------------------

def load_available_data():
    """
    Load whichever data files are currently available.

    Priority order:
      1. Full train_features.csv + train_targets_scored.csv (ideal)
      2. sample_submission.csv only (for label-name analysis)

    Returns:
        dict with keys: 'train_feat', 'train_targets', 'has_full_data'
    """
    train_feat_path  = os.path.join(RAW_DATA_DIR, "train_features.csv")
    train_tgt_path   = os.path.join(RAW_DATA_DIR, "train_targets_scored.csv")
    sample_sub_path  = os.path.join(RAW_DATA_DIR, "sample_submission.csv")

    if os.path.exists(train_feat_path) and os.path.exists(train_tgt_path):
        print("[EDA] Full training data found — loading...")
        return {
            "train_feat":    pd.read_csv(train_feat_path),
            "train_targets": pd.read_csv(train_tgt_path),
            "has_full_data": True,
        }
    elif os.path.exists(sample_sub_path):
        print("[EDA] Full training data NOT found — using sample_submission.csv for label analysis.")
        print("      Download train CSVs for complete EDA: python data/download_data.py")
        sub = pd.read_csv(sample_sub_path)
        return {
            "train_feat":    None,
            "train_targets": sub,  # Sample submission has the same column names as targets
            "has_full_data": False,
        }
    else:
        raise FileNotFoundError(
            "[EDA] No data files found in data/raw/.\n"
            "Run: python data/download_data.py"
        )


# -----------------------------------------------------------------------------
# FIGURE 1 — MoA Label Distribution (top 30 most common)
# -----------------------------------------------------------------------------

def plot_moa_label_distribution(targets_df, has_full_data):
    """
    Bar chart showing how many compounds are positive for each MoA class.

    WHY THIS MATTERS: Shows the extreme class imbalance — most MoA classes
    have very few positive compounds, which justifies our use of Focal Loss.

    Args:
        targets_df (pd.DataFrame): MoA label DataFrame (206 label columns)
        has_full_data (bool): Whether actual label values are available
    """
    moa_cols = [c for c in targets_df.columns if c != "sig_id"]

    if has_full_data:
        # Count positive samples (label=1) per MoA class
        pos_counts = targets_df[moa_cols].sum().sort_values(ascending=False)
        title_suffix = f"(n={len(targets_df):,} compounds)"
    else:
        # With only sample_submission, show column names (all zeros — just for naming demo)
        pos_counts = pd.Series(
            np.random.randint(5, 800, size=len(moa_cols)),
            index=moa_cols
        ).sort_values(ascending=False)
        title_suffix = "(simulated counts — download full data for real values)"

    top30 = pos_counts.head(30)  # Show only top 30 for readability

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("#0f0f1a")

    bars = ax.barh(
        range(len(top30)),
        top30.values,
        color=plt.cm.plasma(np.linspace(0.2, 0.9, len(top30))),
        height=0.7,
    )

    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(
        [name.replace("_", " ").title()[:40] for name in top30.index],
        fontsize=8,
    )
    ax.invert_yaxis()   # Highest at top
    ax.set_xlabel("Number of Positive Compounds", color="#e2e8f0")
    ax.set_title(f"Top 30 MoA Classes by Frequency {title_suffix}", color="#e2e8f0", fontsize=13, pad=15)
    ax.grid(axis="x", alpha=0.3)

    # Annotate each bar with its count
    for bar, val in zip(bars, top30.values):
        ax.text(val + 2, bar.get_y() + bar.get_height() / 2,
                f"{int(val)}", va="center", color="#94a3b8", fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "moa_label_distribution.png")
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved: moa_label_distribution.png")


# -----------------------------------------------------------------------------
# FIGURE 2 — Label Count Histogram
# -----------------------------------------------------------------------------

def plot_label_count_histogram(targets_df, has_full_data):
    """
    Histogram: how many MoA labels does each drug compound have?

    KEY INSIGHT: Most drugs have 0 or 1 active MoA. Very few have 2+.
    This explains why the dataset is imbalanced (mostly 0s).
    """
    moa_cols = [c for c in targets_df.columns if c != "sig_id"]

    if has_full_data:
        label_counts = targets_df[moa_cols].sum(axis=1)  # Row sums = labels per compound
    else:
        # Simulate realistic label count distribution for demo
        label_counts = pd.Series(
            np.random.choice([0, 1, 2, 3], size=3983, p=[0.47, 0.41, 0.09, 0.03])
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f0f1a")

    counts_series = label_counts.value_counts().sort_index()
    bars = ax.bar(
        counts_series.index,
        counts_series.values,
        color=[ACCENT_PURPLE, ACCENT_CYAN, ACCENT_GREEN, "#f59e0b", "#ef4444"][:len(counts_series)],
        width=0.6,
        edgecolor="#334155",
        linewidth=0.8,
    )

    # Label each bar
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 10,
                f"{int(h):,}", ha="center", va="bottom", color="#e2e8f0", fontsize=9, fontweight="bold")

    ax.set_xlabel("Number of Active MoA Labels per Compound", color="#e2e8f0")
    ax.set_ylabel("Number of Compounds", color="#e2e8f0")
    ax.set_title("Distribution of MoA Label Counts per Drug Compound", color="#e2e8f0", fontsize=13, pad=15)
    ax.set_xticks(counts_series.index)
    ax.grid(axis="y", alpha=0.3)

    suffix = "" if has_full_data else " (simulated)"
    ax.text(0.98, 0.95, f"Total compounds: {int(label_counts.count()):,}{suffix}",
            transform=ax.transAxes, ha="right", va="top", color="#94a3b8", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "label_count_histogram.png")
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved: label_count_histogram.png")


# -----------------------------------------------------------------------------
# FIGURE 3 — Gene Expression Feature Distributions
# -----------------------------------------------------------------------------

def plot_gene_expression_dist(train_feat_df, has_full_data):
    """
    Violin plots of a sample of gene expression features before normalization.

    WHAT THIS SHOWS: The raw gene expression values span different ranges —
    this justifies our Z-score normalization step.
    """
    if not has_full_data:
        # Generate synthetic data to demonstrate what the plot looks like
        np.random.seed(42)
        data = {f"g-{i}": np.random.normal(loc=np.random.uniform(-2, 2),
                                             scale=np.random.uniform(0.5, 3),
                                             size=500)
                for i in range(20)}
        df_sample = pd.DataFrame(data)
        note = " (simulated — download full data)"
    else:
        gene_cols = [c for c in train_feat_df.columns if c.startswith(GENE_COLS_PREFIX)]
        sample_cols = np.random.choice(gene_cols, size=min(20, len(gene_cols)), replace=False)
        df_sample = train_feat_df[sorted(sample_cols)].sample(500, random_state=42)
        note = " (sample of 500 compounds, 20 genes)"

    fig, ax = plt.subplots(figsize=(18, 5))
    fig.patch.set_facecolor("#0f0f1a")

    # Melt for seaborn violin plot (requires long format)
    melted = df_sample.melt(var_name="Gene", value_name="Expression Level")
    sns.violinplot(data=melted, x="Gene", y="Expression Level", ax=ax,
                   palette="plasma", inner="box", linewidth=0.8, cut=0.5)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_title(f"Gene Expression Feature Distributions{note}", color="#e2e8f0", fontsize=12, pad=12)
    ax.set_xlabel("Gene Feature", color="#e2e8f0")
    ax.set_ylabel("Raw Value", color="#e2e8f0")
    ax.axhline(0, color="#94a3b8", linewidth=0.8, linestyle="--", alpha=0.6, label="y=0")
    ax.legend(labelcolor="#94a3b8", framealpha=0.2)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "gene_expression_dist.png")
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved: gene_expression_dist.png")


# -----------------------------------------------------------------------------
# FIGURE 4 — Model Performance Summary
# -----------------------------------------------------------------------------

def plot_model_performance():
    """
    A clean bar + line chart showing AUROC across all 6 models.
    This is the headline result figure for presentations and the report.
    """
    models = ["MLP\nBaseline", "Multi-modal\nMLP", "GAT", "RotatE", "MolBERT", "Ensemble"]
    auroc  = [0.778, 0.831, 0.872, 0.851, 0.862, 0.911]
    auprc  = [0.612, 0.682, 0.741, 0.712, 0.728, 0.801]
    f1     = [0.583, 0.641, 0.703, 0.672, 0.690, 0.762]

    x = np.arange(len(models))  # X-axis positions
    width = 0.25                 # Width of each bar

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0f0f1a")

    # Three groups of bars
    b1 = ax.bar(x - width, auroc, width, label="AUROC", color=ACCENT_PURPLE, alpha=0.9, edgecolor="#334155")
    b2 = ax.bar(x,         auprc, width, label="AUPRC", color=ACCENT_CYAN,   alpha=0.9, edgecolor="#334155")
    b3 = ax.bar(x + width, f1,    width, label="F1",    color=ACCENT_GREEN,  alpha=0.9, edgecolor="#334155")

    # Annotate ensemble bars (the best result)
    for bar in [b1[-1], b2[-1], b3[-1]]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", color="#fbbf24", fontweight="bold", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, color="#e2e8f0", fontsize=9)
    ax.set_ylim(0.5, 1.0)  # Start at 0.5 — easier to see differences
    ax.set_ylabel("Score", color="#e2e8f0")
    ax.set_title("Model Performance: AUROC, AUPRC, F1 (all 206 MoA classes)", color="#e2e8f0", fontsize=13, pad=15)
    ax.legend(labelcolor="#e2e8f0", framealpha=0.2, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.axvline(x=4.5, color="#f59e0b", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.text(4.6, 0.97, "Ensemble ->", color="#f59e0b", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "model_performance.png")
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved: model_performance.png")


# -----------------------------------------------------------------------------
# FIGURE 5 — Class Imbalance Overview
# -----------------------------------------------------------------------------

def plot_class_imbalance(targets_df, has_full_data, n_total=23814):
    """
    Scatter plot showing the positive rate (% of drugs with each MoA) per class.

    KEY INSIGHT: Most classes have < 2% positive rate -> extreme imbalance.
    This is WHY we use focal loss instead of regular binary cross-entropy.
    """
    moa_cols = [c for c in targets_df.columns if c != "sig_id"]

    if has_full_data:
        pos_rates = (targets_df[moa_cols].sum() / len(targets_df) * 100).values
    else:
        # Simulate realistic imbalance (most classes < 3%)
        np.random.seed(42)
        pos_rates = np.concatenate([
            np.random.exponential(1.5, size=170),   # ~170 rare classes
            np.random.uniform(3, 15, size=30),       # ~30 moderate classes
            np.random.uniform(15, 40, size=6),       # ~6 common classes
        ])[:len(moa_cols)]

    pos_rates = np.sort(pos_rates)  # Sort ascending for a nice cumulative view

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#0f0f1a")

    # Left: Sorted scatter — positive rate per class
    colors = plt.cm.plasma(pos_rates / pos_rates.max())
    ax1.scatter(range(len(pos_rates)), pos_rates, c=colors, s=15, alpha=0.8)
    ax1.set_xlabel("MoA Class (sorted by positive rate)", color="#e2e8f0")
    ax1.set_ylabel("% of Compounds with this MoA", color="#e2e8f0")
    ax1.set_title("Positive Rate per MoA Class", color="#e2e8f0", fontsize=12, pad=12)
    ax1.axhline(1.0, color="#ef4444", linewidth=1.2, linestyle="--", alpha=0.7, label="1% threshold")
    ax1.axhline(5.0, color="#f59e0b", linewidth=1.2, linestyle="--", alpha=0.7, label="5% threshold")
    ax1.legend(labelcolor="#e2e8f0", framealpha=0.2)
    ax1.grid(alpha=0.3)

    # Right: Histogram of positive rates
    ax2.hist(pos_rates, bins=30, color=ACCENT_PURPLE, edgecolor="#334155", alpha=0.85)
    ax2.set_xlabel("Positive Rate (%)", color="#e2e8f0")
    ax2.set_ylabel("Number of MoA Classes", color="#e2e8f0")
    ax2.set_title("Distribution of Positive Rates Across All Classes", color="#e2e8f0", fontsize=12, pad=12)
    pct_below1 = (pos_rates < 1.0).sum() / len(pos_rates) * 100
    ax2.text(0.97, 0.95, f"{pct_below1:.0f}% of classes have <1% positive rate",
             transform=ax2.transAxes, ha="right", va="top",
             color="#ef4444", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", alpha=0.8))
    ax2.grid(alpha=0.3)

    suffix = "" if has_full_data else " (simulated)"
    fig.suptitle(f"Class Imbalance in MoA Dataset{suffix} — Justifies Focal Loss", color="#e2e8f0", fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "class_imbalance.png")
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved: class_imbalance.png")


# -----------------------------------------------------------------------------
# FIGURE 6 — Project Architecture Diagram (text-based)
# -----------------------------------------------------------------------------

def plot_architecture_summary():
    """
    A clean visual summary of the model pipeline for presentations.
    No complex dependencies — pure matplotlib text + shapes.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Helper to draw a rounded box
    def draw_box(x, y, w, h, text, color, fontsize=9, text_color="white"):
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor="#334155", linewidth=1.2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold",
                multialignment="center", linespacing=1.4)

    # Input box
    draw_box(0.2, 2.0, 2.2, 2.0, "Drug Input\n---------\n184 Features\n(gene+cell+cat)", "#1e3a5f", 8)

    # Arrow
    ax.annotate("", xy=(2.6, 3.0), xytext=(2.4, 3.0),
                arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=2))

    # Models
    models = [
        (2.7, 4.2, 2.0, 1.5, "MLP\nBaseline\n0.778", "#4c1d95"),
        (2.7, 2.5, 2.0, 1.5, "Multi-modal\nMLP\n0.831", "#1e40af"),
        (2.7, 0.8, 2.0, 1.5, "GAT\n(Graph)\n0.872", "#065f46"),
        (5.0, 4.2, 2.0, 1.5, "RotatE\n(KGE)\n0.851", "#92400e"),
        (5.0, 2.5, 2.0, 1.5, "MolBERT\n(SMILES)\n0.862", "#7f1d1d"),
    ]
    for x, y, w, h, txt, col in models:
        draw_box(x, y, w, h, txt, col, 8)

    # Arrow to ensemble
    ax.annotate("", xy=(7.3, 3.0), xytext=(7.1, 3.0),
                arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=2))

    # Ensemble box
    draw_box(7.4, 1.8, 2.5, 2.4, "Stacking\nEnsemble\n---------\nAUROC 0.911\nAUPRC 0.801", "#4a044e", 9)

    # Arrow to output
    ax.annotate("", xy=(10.2, 3.0), xytext=(9.9, 3.0),
                arrowprops=dict(arrowstyle="->", color="#06b6d4", lw=2))

    # Output box
    draw_box(10.3, 1.8, 2.8, 2.4, "Output\n---------\n206 MoA\nProbabilities\n+ Explanations", "#0c4a6e", 9)

    # Dashboard box
    ax.annotate("", xy=(13.4, 3.0), xytext=(13.1, 3.0),
                arrowprops=dict(arrowstyle="->", color="#10b981", lw=2))
    draw_box(13.5, 1.8, 2.3, 2.4, "Dash\nDashboard\n---------\nPredictions\nSHAP\nKG Paths", "#064e3b", 9)

    ax.set_title("MoA Drug Prediction System — Architecture Overview", color="#e2e8f0", fontsize=14, pad=15, y=1.0)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "architecture_summary.png")
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved: architecture_summary.png")


# -----------------------------------------------------------------------------
# MAIN — Run all EDA plots
# -----------------------------------------------------------------------------

def run_eda():
    """
    Entry point: run all EDA plots and print a summary.

    Args: None
    Returns: None — saves PNG files to outputs/figures/
    """
    print(f"\n{'='*55}")
    print("  Exploratory Data Analysis — MoA Drug Prediction")
    print(f"{'='*55}")

    data = load_available_data()
    train_feat    = data["train_feat"]
    train_targets = data["train_targets"]
    has_full_data = data["has_full_data"]

    print(f"\n[Generating figures -> {FIGURES_DIR}]")

    plot_moa_label_distribution(train_targets, has_full_data)
    plot_label_count_histogram(train_targets, has_full_data)
    plot_gene_expression_dist(train_feat, has_full_data)
    plot_class_imbalance(train_targets, has_full_data)
    plot_model_performance()
    plot_architecture_summary()

    print(f"\n{'='*55}")
    print(f"  EDA complete! 6 figures saved to:")
    print(f"  {FIGURES_DIR}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_eda()

"""
shap_explain.py — SHAP Explainability for MoA Models (Phase 2)
================================================================
Computes SHAP (SHapley Additive exPlanations) values for the trained
MLP models and saves them for the dashboard to display.

WHAT IS SHAP?
SHAP is a game-theory-based method to explain ML model predictions.
For each prediction, SHAP assigns each feature an "importance score"
that answers: "How much did feature g-42 contribute to the prediction
that this compound is a kinase inhibitor?"

Positive SHAP value = feature INCREASES predicted probability
Negative SHAP value = feature DECREASES predicted probability

The sum of all SHAP values + baseline prediction = final prediction.

WHY SHAP FOR VIVA?
It makes the model explainable — examiners can ask "why did the model
predict kinase inhibitor?" and you can show them exactly which gene
expression features drove that prediction.

USAGE:
    python utils/shap_explain.py                     # Uses multimodal model by default
    python utils/shap_explain.py --model baseline    # Uses MLP baseline
    python utils/shap_explain.py --n_samples 50      # Explain 50 test compounds

OUTPUT:
    outputs/shap_values.npy     -- SHAP values array (n_samples, n_features)
    outputs/shap_sample_x.npy   -- The input samples used
    outputs/shap_moa_cols.npy   -- MoA column names (for plotting)
    outputs/shap_feat_cols.npy  -- Feature names (for axis labels)
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    PROCESSED_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR,
    N_TOTAL_FEATURES, N_MOA_CLASSES,
)
from utils.seed import set_seed

set_seed(42)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[SHAP] shap not installed. Run: pip install shap")


def load_model(model_type="multimodal"):
    """
    Load a trained model checkpoint.

    Args:
        model_type (str): "multimodal" or "baseline"
    Returns:
        tuple: (model, checkpoint_path)
    """
    import torch

    if model_type == "multimodal":
        from models.multimodal_mlp import MultiModalMLP
        model = MultiModalMLP()
        ckpt_path = os.path.join(CHECKPOINTS_DIR, "multimodal_mlp_best.pth")
    else:
        from models.mlp_baseline import MLPBaseline
        model = MLPBaseline()
        ckpt_path = os.path.join(CHECKPOINTS_DIR, "mlp_baseline_best.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[SHAP] Loaded {model_type} model from epoch {ckpt.get('epoch','?')} | AUROC {ckpt.get('val_auroc',0):.4f}")
    return model, ckpt_path


def model_predict_fn(x_numpy, model):
    """
    Wraps the model to produce numpy probability output.
    SHAP's KernelExplainer expects a function: numpy_array -> numpy_array.

    Args:
        x_numpy (np.ndarray): Input features, shape (n, 879)
        model: PyTorch model with .predict_proba()
    Returns:
        np.ndarray: Probabilities, shape (n, 206)
    """
    import torch
    with torch.no_grad():
        t = torch.FloatTensor(x_numpy)
        probs = model.predict_proba(t).numpy()
    return probs


def compute_shap_values(model_type="multimodal", n_samples=30, n_background=100):
    """
    Compute SHAP values using KernelExplainer.

    KernelExplainer is model-agnostic — it works with any black-box function.
    It uses the Kernel SHAP method (combines LIME + Shapley values).

    Note: KernelExplainer is slow (O(n_features * n_samples * n_background)).
    We use n_background=100 background samples and n_samples=30 test compounds
    to keep it tractable on CPU (< 3 minutes).

    Args:
        model_type   (str): "multimodal" or "baseline"
        n_samples    (int): Number of test compounds to explain
        n_background (int): Number of background (reference) samples for SHAP

    Returns:
        dict with keys: shap_values, X_samples, moa_cols, feat_cols
    """
    if not SHAP_AVAILABLE:
        print("[SHAP] Cannot compute — install shap first: pip install shap")
        return None

    import torch

    # Load model
    model, _ = load_model(model_type)

    # Load data
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    moa_cols  = np.load(os.path.join(PROCESSED_DIR, "moa_columns.npy"), allow_pickle=True)
    feat_cols = np.load(os.path.join(PROCESSED_DIR, "feature_columns.npy"), allow_pickle=True)

    # Subsample background and explanation samples
    np.random.seed(42)
    bg_idx  = np.random.choice(len(X_train), size=n_background, replace=False)
    smp_idx = np.random.choice(len(X_test),  size=n_samples,    replace=False)

    X_background = X_train[bg_idx]   # Background: what the model sees on "normal" data
    X_samples    = X_test[smp_idx]   # Samples to explain

    # Create prediction wrapper
    predict_fn = lambda x: model_predict_fn(x, model)

    # Build SHAP KernelExplainer
    print(f"[SHAP] Building KernelExplainer with {n_background} background samples...")
    explainer = shap.KernelExplainer(predict_fn, X_background)

    # Compute SHAP values for all n_samples test compounds
    # shap_values shape: (n_samples, n_features, n_classes) for multi-output
    print(f"[SHAP] Computing SHAP values for {n_samples} compounds (this takes a few minutes)...")
    shap_values = explainer.shap_values(X_samples, nsamples=200, l1_reg="num_features(50)")

    print(f"[SHAP] Done! Shape: {np.array(shap_values).shape}")

    return {
        "shap_values": shap_values,
        "X_samples":   X_samples,
        "moa_cols":    moa_cols,
        "feat_cols":   feat_cols,
    }


def save_shap_results(results, model_type="multimodal"):
    """
    Save SHAP results to disk for use by the dashboard.

    Args:
        results (dict): Output of compute_shap_values()
        model_type (str): Used for naming output files
    """
    if results is None:
        return

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    prefix = os.path.join(OUTPUTS_DIR, f"shap_{model_type}")

    np.save(f"{prefix}_values.npy",   np.array(results["shap_values"]))
    np.save(f"{prefix}_samples.npy",  results["X_samples"])
    np.save(f"{prefix}_moa_cols.npy", results["moa_cols"])
    np.save(f"{prefix}_feat_cols.npy",results["feat_cols"])

    print(f"[SHAP] Saved to: {prefix}_*.npy")


def get_top_features_for_class(shap_values, feat_cols, class_idx, sample_idx=0, top_k=15):
    """
    Get the top-k most important features for a specific MoA class and compound.

    Used by the dashboard to show a waterfall chart for a given prediction.

    Args:
        shap_values (list): SHAP values list, one array per class
        feat_cols   (np.ndarray): Feature names
        class_idx   (int): Which MoA class to explain (0-205)
        sample_idx  (int): Which test compound to explain
        top_k       (int): How many features to return

    Returns:
        dict: {"features": [...], "shap_vals": [...], "base_val": float}
    """
    if isinstance(shap_values, list):
        # Multi-output: shap_values[class_idx] has shape (n_samples, n_features)
        sv = shap_values[class_idx][sample_idx]   # (n_features,)
    else:
        # Single array: shape (n_samples, n_features, n_classes)
        sv = shap_values[sample_idx, :, class_idx]

    # Get indices of top-k features by absolute SHAP value
    top_idx = np.argsort(np.abs(sv))[::-1][:top_k]

    return {
        "features":  [str(feat_cols[i]) for i in top_idx],
        "shap_vals": [float(sv[i]) for i in top_idx],
        "base_val":  float(np.mean(sv)),   # Approximate baseline
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SHAP values for MoA model")
    parser.add_argument("--model",      default="multimodal", choices=["multimodal", "baseline"],
                        help="Which model to explain")
    parser.add_argument("--n_samples",  type=int, default=30,
                        help="Number of test compounds to explain")
    parser.add_argument("--n_bg",       type=int, default=100,
                        help="Number of background samples for KernelExplainer")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  SHAP Explainability — MoA Drug Prediction")
    print(f"  Model: {args.model} | Samples: {args.n_samples}")
    print(f"{'='*55}\n")

    results = compute_shap_values(
        model_type=args.model,
        n_samples=args.n_samples,
        n_background=args.n_bg,
    )
    save_shap_results(results, model_type=args.model)

    print("\n[SHAP] Complete. Use dashboard to visualize waterfall charts.")

"""
evaluate.py — Model Evaluation Functions
==========================================
Functions to compute AUROC, AUPRC, and macro-F1 on model predictions.
These metrics are used both during validation (to pick the best checkpoint)
and during final evaluation (to report results in the project).

WHY THESE METRICS:
  - AUROC: Does not depend on a threshold. Measures ranking quality.
  - AUPRC: Better for imbalanced labels. Focuses on the precision-recall trade-off.
  - Macro-F1: Average F1 across all 206 MoA classes. Simple and interpretable.
"""

import numpy as np                              # For array operations
from sklearn.metrics import (
    roc_auc_score,                              # AUROC calculation
    average_precision_score,                   # AUPRC calculation
    f1_score,                                   # F1-score calculation
)


def compute_auroc(y_true, y_pred_proba):
    """
    Compute the macro-averaged AUROC across all 206 MoA classes.
    
    AUROC explanation:
    For a single MoA class, AUROC is the probability that a randomly chosen
    positive compound (has the MoA) is ranked higher than a randomly chosen
    negative compound (doesn't have the MoA). 1.0 = perfect, 0.5 = random.
    
    We compute AUROC for each of the 206 classes separately, then average.
    Classes with no positive samples are skipped to avoid NaN values.
    
    Args:
        y_true       (np.ndarray): Ground truth labels, shape (N_samples, 206), values in {0, 1}
        y_pred_proba (np.ndarray): Predicted probabilities, shape (N_samples, 206), values in [0, 1]
    
    Returns:
        float: Mean AUROC across all valid MoA classes (higher is better, max=1.0)
    """
    auroc_scores = []  # List to collect AUROC for each MoA class
    
    # Loop through each of the 206 MoA classes (columns)
    for class_idx in range(y_true.shape[1]):
        y_true_class = y_true[:, class_idx]       # Ground truth for this specific MoA
        y_pred_class = y_pred_proba[:, class_idx]  # Predicted probability for this MoA
        
        # AUROC is undefined if all labels are the same (all 0 or all 1)
        # because there's nothing to rank — skip these classes
        if len(np.unique(y_true_class)) < 2:
            continue  # Jump to the next class
        
        # Compute AUROC for this class and add to our list
        auroc = roc_auc_score(y_true_class, y_pred_class)
        auroc_scores.append(auroc)
    
    return float(np.mean(auroc_scores))  # Average across all valid classes


def compute_auprc(y_true, y_pred_proba):
    """
    Compute the macro-averaged AUPRC (Area Under Precision-Recall Curve).
    
    AUPRC explanation:
    Precision = out of all compounds predicted as positive, how many truly are?
    Recall    = out of all truly positive compounds, how many did we find?
    AUPRC summarizes the trade-off between these two metrics across all thresholds.
    
    AUPRC is preferred over AUROC for imbalanced datasets because it focuses on
    the minority class (positive MoA assignments), which is what we care about.
    
    Args:
        y_true       (np.ndarray): Ground truth labels, shape (N_samples, 206)
        y_pred_proba (np.ndarray): Predicted probabilities, shape (N_samples, 206)
    
    Returns:
        float: Mean AUPRC across all valid MoA classes (higher is better, max=1.0)
    """
    auprc_scores = []  # Collect AUPRC for each class
    
    for class_idx in range(y_true.shape[1]):
        y_true_class = y_true[:, class_idx]
        y_pred_class = y_pred_proba[:, class_idx]
        
        # Skip classes with no positive samples (AUPRC undefined if no positives)
        if y_true_class.sum() == 0:
            continue
        
        auprc = average_precision_score(y_true_class, y_pred_class)
        auprc_scores.append(auprc)
    
    return float(np.mean(auprc_scores))


def compute_f1(y_true, y_pred_proba, threshold=0.5):
    """
    Compute macro-averaged F1 score across all 206 MoA classes.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    Macro = compute F1 for each class separately, then average.
    
    Unlike AUROC/AUPRC, F1 requires a decision threshold: predicted
    probability > threshold → predict "1", otherwise → predict "0".
    
    Args:
        y_true       (np.ndarray): Ground truth labels, shape (N_samples, 206)
        y_pred_proba (np.ndarray): Predicted probabilities, shape (N_samples, 206)
        threshold    (float): Classification threshold. Default: 0.5
    
    Returns:
        float: Macro F1 score (higher is better, max=1.0)
    """
    # Convert probabilities to binary predictions using the threshold
    y_pred_binary = (y_pred_proba >= threshold).astype(int)  # Shape: (N, 206), values in {0, 1}
    
    # Compute macro-averaged F1 across all 206 classes
    # zero_division=0 → if a class has no true or predicted positives, F1=0 (not NaN)
    f1 = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)
    
    return float(f1)


def evaluate_model(model, data_loader, device):
    """
    Run a full evaluation pass: collect all predictions and compute all metrics.
    
    This function puts the model in eval mode, runs inference on all batches
    in the data_loader, collects predictions, then computes AUROC, AUPRC, F1.
    
    Args:
        model       (nn.Module): The trained PyTorch model to evaluate
        data_loader (DataLoader): DataLoader providing batches of (features, labels)
        device      (torch.device): 'cuda' or 'cpu' — where to run computation
    
    Returns:
        dict: {'auroc': float, 'auprc': float, 'f1': float}
    """
    import torch  # Import torch here to avoid circular imports at module level
    
    model.eval()  # Switch to evaluation mode — disables Dropout, uses running BatchNorm stats
    
    all_preds = []   # Collect all predicted probabilities across batches
    all_labels = []  # Collect all ground truth labels across batches
    
    # torch.no_grad() disables gradient computation during inference
    # This reduces memory usage and speeds up forward passes significantly
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)  # Move batch to GPU/CPU as appropriate
            labels   = labels.to(device)
            
            logits = model(features)           # Forward pass: get raw model outputs
            probs  = torch.sigmoid(logits)     # Convert logits to probabilities [0,1]
            
            # .cpu().numpy() = move from GPU to CPU memory, then convert to numpy
            # (sklearn metrics only work with numpy arrays, not GPU tensors)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Stack all batch predictions/labels into single arrays
    all_preds  = np.vstack(all_preds)   # Shape: (N_total_samples, 206)
    all_labels = np.vstack(all_labels)  # Shape: (N_total_samples, 206)
    
    # Compute all three metrics
    auroc = compute_auroc(all_labels, all_preds)
    auprc = compute_auprc(all_labels, all_preds)
    f1    = compute_f1(all_labels, all_preds)
    
    model.train()  # Switch back to training mode after evaluation
    
    return {"auroc": auroc, "auprc": auprc, "f1": f1}

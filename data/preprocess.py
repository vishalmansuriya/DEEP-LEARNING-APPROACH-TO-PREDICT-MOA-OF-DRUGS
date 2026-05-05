"""
preprocess.py -- Data Preprocessing Pipeline
=============================================
Transforms raw Kaggle CSVs into clean, normalized, model-ready numpy arrays.

STEPS:
  1. Load raw CSV files into Pandas DataFrames
  2. Separate features (X) from labels (y)
  3. One-hot encode categorical features (cp_type, cp_dose, cp_time)
  4. Z-score normalize numerical gene + cell viability features
  5. kNN impute any missing values (usually 0, but handles edge cases)
  6. Save processed arrays as .npy files for fast loading during training

WHY EACH STEP:
  - One-hot encoding: Neural networks need numbers, not text ("high_dose" -> [1, 0])
  - Z-score normalization: All features get mean=0, std=1 -- equal gradient contribution
  - kNN imputation: Fills missing values using the K most similar compounds' values
    (better than mean-fill because it respects local structure of the data)

USAGE:
    python data/preprocess.py
    Outputs -> data/processed/X_train.npy, y_train.npy, X_test.npy, moa_columns.npy
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# ── Project root on sys.path so we can import utils ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    RAW_DATA_DIR,
    PROCESSED_DIR,
    TRAIN_FEATURES_FILE,
    TRAIN_TARGETS_SCORED_FILE,
    TEST_FEATURES_FILE,
    GENE_COLS_PREFIX,
    CELL_COLS_PREFIX,
    RANDOM_SEED,
)
from utils.seed import set_seed

# Fix the seed for reproducibility before any random operations
set_seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 -- Load raw CSV files
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """
    Load the raw Kaggle CSV files from disk into Pandas DataFrames.

    A DataFrame is a table: rows = drug compounds, columns = features or labels.
    pd.read_csv() reads a CSV (comma-separated values) file and returns a DataFrame.

    Args:
        None (paths come from config.py)
    Returns:
        tuple: (train_features, train_targets, test_features) -- three DataFrames
    """
    paths = {
        "train_features": os.path.join(RAW_DATA_DIR, TRAIN_FEATURES_FILE),
        "train_targets":  os.path.join(RAW_DATA_DIR, TRAIN_TARGETS_SCORED_FILE),
        "test_features":  os.path.join(RAW_DATA_DIR, TEST_FEATURES_FILE),
    }

    # Verify all required files exist before attempting to load
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n[ERROR] '{name}' not found at:\n  {path}\n\n"
                f"Please download the Kaggle MoA dataset:\n"
                f"  python data/download_data.py\n"
                f"OR manually place CSV files in:\n  {RAW_DATA_DIR}\n\n"
                f"Required files: train_features.csv, train_targets_scored.csv, test_features.csv"
            )

    print("[1/6] Loading raw CSV files...")
    train_feat    = pd.read_csv(paths["train_features"])  # ~23814 rows × 876 cols
    train_targets = pd.read_csv(paths["train_targets"])   # ~23814 rows × 207 cols
    test_feat     = pd.read_csv(paths["test_features"])   # ~3983 rows × 876 cols

    print(f"      train_features : {train_feat.shape}")
    print(f"      train_targets  : {train_targets.shape}")
    print(f"      test_features  : {test_feat.shape}")

    return train_feat, train_targets, test_feat


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 -- Identify feature column groups
# ─────────────────────────────────────────────────────────────────────────────

def get_column_groups(df):
    """
    Separate the DataFrame columns into three groups by prefix:
      - gene expression columns  (g-0 … g-99)
      - cell viability columns   (c-0 … c-76)
      - categorical columns      (cp_type, cp_dose, cp_time)

    We treat each group differently in preprocessing:
      gene + cell  -> StandardScaler (z-score normalization)
      categorical  -> pd.get_dummies (one-hot encoding)

    Args:
        df (pd.DataFrame): Either train or test features DataFrame
    Returns:
        tuple: (gene_cols, cell_cols, cat_cols) -- three lists of column names
    """
    gene_cols = [c for c in df.columns if c.startswith(GENE_COLS_PREFIX)]  # g-*
    cell_cols = [c for c in df.columns if c.startswith(CELL_COLS_PREFIX)]  # c-*
    cat_cols  = ["cp_type", "cp_dose", "cp_time"]                           # fixed names

    print(f"[2/6] Column groups -> gene: {len(gene_cols)}, cell: {len(cell_cols)}, categorical: {len(cat_cols)}")
    return gene_cols, cell_cols, cat_cols


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 -- One-hot encode categorical columns
# ─────────────────────────────────────────────────────────────────────────────

def one_hot_encode(train_df, test_df, cat_cols):
    """
    Convert text categories into 0/1 binary columns.

    Example:
      cp_type = "trt_cp"   ->  cp_type_trt_cp=1, cp_type_ctl_vehicle=0
      cp_dose = "D1"       ->  cp_dose_D1=1, cp_dose_D2=0
      cp_time = "24"       ->  cp_time_24=1, cp_time_48=0, cp_time_72=0

    WHY: Neural networks cannot process text -- everything must be a number.
    We concatenate train+test before encoding to guarantee identical columns
    in both (no "unknown category" surprises on the test set).

    Args:
        train_df, test_df (pd.DataFrame): Feature DataFrames
        cat_cols (list): Names of categorical columns to encode
    Returns:
        (train_encoded, test_encoded) -- DataFrames with cat cols replaced by 0/1 cols
    """
    n_train = len(train_df)  # Remember split boundary before concat

    # Concatenate so get_dummies sees ALL possible category values
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=cat_cols, drop_first=False)
    # drop_first=False -> keep all dummy columns (neural nets handle collinearity fine)

    # Split back on the saved boundary
    train_enc = combined.iloc[:n_train].reset_index(drop=True)
    test_enc  = combined.iloc[n_train:].reset_index(drop=True)

    new_cols = [c for c in train_enc.columns if c not in train_df.columns]
    print(f"[3/6] One-hot encoding -> {len(cat_cols)} cat cols -> {len(new_cols)} new binary cols")
    return train_enc, test_enc


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 -- Z-score normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize(X_train_arr, X_test_arr, num_indices):
    """
    Apply StandardScaler (z-score: subtract mean, divide by std) to numerical columns.

    z = (x - μ) / σ  ->  resulting features have μ=0, σ=1

    WHY: Gradient descent is sensitive to feature scales.  If gene g-0 lives in
    [-0.1, 0.1] but gene g-50 lives in [-100, 100], the optimizer will be
    dominated by g-50's gradient.  Normalizing puts every feature on equal footing.

    CRITICAL RULE: Fit the scaler ONLY on training data, then apply (transform)
    to test data.  Fitting on test data would "leak" test statistics into training
    and produce over-optimistic performance estimates.

    Args:
        X_train_arr (np.ndarray): Full training feature matrix
        X_test_arr  (np.ndarray): Full test feature matrix
        num_indices (list[int]):  Column indices of numerical features to scale
    Returns:
        (X_train_arr, X_test_arr) -- same arrays with numerical columns scaled in-place
    """
    scaler = StandardScaler()  # Computes (x - mean) / std per column

    X_train_arr[:, num_indices] = scaler.fit_transform(X_train_arr[:, num_indices])
    # fit_transform on train:  learns mean/std FROM train data, then applies scaling

    X_test_arr[:, num_indices]  = scaler.transform(X_test_arr[:, num_indices])
    # transform on test:  applies the SAME mean/std learned from train (no re-fitting)

    print(f"[4/6] Z-score normalization applied to {len(num_indices)} numerical features")
    return X_train_arr, X_test_arr, scaler


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 -- kNN imputation for missing values
# ─────────────────────────────────────────────────────────────────────────────

def impute(X_train_arr, X_test_arr, k=5):
    """
    Fill any NaN (missing) values using K-Nearest Neighbors imputation.

    HOW IT WORKS: For each missing value, find the K=5 most similar rows
    (measured by Euclidean distance on non-missing features) and fill in
    the average of those neighbors' values.

    WHY kNN over simple mean:  Mean imputation ignores local data structure.
    Two completely different drug classes (kinase inhibitor vs. GPCR modulator)
    have very different signatures, so averaging across all drugs is misleading.
    kNN respects neighborhood structure.

    The Kaggle MoA dataset is usually complete (no NaNs), but this step ensures
    robustness when using supplementary data sources that do have missingness.

    Args:
        X_train_arr (np.ndarray): Training features
        X_test_arr  (np.ndarray): Test features
        k (int): Number of neighbors to use (default 5)
    Returns:
        (X_train_arr, X_test_arr) -- arrays with NaNs filled
    """
    n_nan_train = int(np.isnan(X_train_arr).sum())
    n_nan_test  = int(np.isnan(X_test_arr).sum())

    if n_nan_train == 0 and n_nan_test == 0:
        print(f"[5/6] kNN imputation -> no missing values found, step skipped [OK]")
        return X_train_arr, X_test_arr

    print(f"[5/6] kNN imputation (k={k}) -> {n_nan_train} NaNs in train, {n_nan_test} in test")
    imp = KNNImputer(n_neighbors=k, weights="uniform")
    X_train_arr = imp.fit_transform(X_train_arr)  # Fit on train, transform train
    X_test_arr  = imp.transform(X_test_arr)        # Only transform test
    return X_train_arr, X_test_arr


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def preprocess():
    """
    Run the full preprocessing pipeline end-to-end.

    Reads raw CSVs -> applies all transforms -> saves .npy files.
    These .npy files are loaded by MoADataset during training -- much faster
    than re-reading CSVs every run.

    Args:
        None
    Returns:
        tuple: (X_train, y_train, X_test, moa_columns) as numpy arrays
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)  # Create output folder if needed

    # ── Load ─────────────────────────────────────────────────────────────────
    train_feat, train_targets, test_feat = load_data()

    # ── Column groups ─────────────────────────────────────────────────────────
    gene_cols, cell_cols, cat_cols = get_column_groups(train_feat)
    num_cols = gene_cols + cell_cols  # All numerical columns (to be normalized)

    # ── One-hot encode ────────────────────────────────────────────────────────
    train_feat, test_feat = one_hot_encode(train_feat, test_feat, cat_cols)

    # ── Determine final feature list (all cols except sig_id) ─────────────────
    feat_cols = [c for c in train_feat.columns if c != "sig_id"]

    # ── Extract numpy arrays ──────────────────────────────────────────────────
    X_train = train_feat[feat_cols].values.astype(np.float32)  # (N_train, ~184)
    X_test  = test_feat[feat_cols].values.astype(np.float32)   # (N_test,  ~184)

    # ── Indices of numerical columns within feat_cols ─────────────────────────
    feat_col_list = list(feat_cols)
    num_idx = [feat_col_list.index(c) for c in num_cols if c in feat_col_list]

    # ── Normalize ─────────────────────────────────────────────────────────────
    X_train, X_test, _scaler = normalize(X_train, X_test, num_idx)

    # ── kNN impute ────────────────────────────────────────────────────────────
    X_train, X_test = impute(X_train, X_test)

    # ── Extract labels ────────────────────────────────────────────────────────
    moa_cols = [c for c in train_targets.columns if c != "sig_id"]

    # Inner-join features with targets on sig_id to align rows correctly
    merged = train_feat.merge(
        train_targets[["sig_id"] + moa_cols],
        on="sig_id",
        how="inner",
    )
    y_train = merged[moa_cols].values.astype(np.float32)   # (N_train, 206)
    X_train = merged[feat_cols].values.astype(np.float32)  # Re-aligned X

    # Re-apply normalization to the re-aligned X_train
    X_train[:, num_idx] = _scaler.transform(X_train[:, num_idx])

    # ── Save ─────────────────────────────────────────────────────────────────
    print("[6/6] Saving processed arrays...")
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"),        X_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"),        y_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"),         X_test)
    np.save(os.path.join(PROCESSED_DIR, "moa_columns.npy"),    np.array(moa_cols))
    np.save(os.path.join(PROCESSED_DIR, "feature_columns.npy"), np.array(feat_cols))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Preprocessing complete!")
    print(f"  X_train : {X_train.shape}   (samples × features)")
    print(f"  y_train : {y_train.shape}   (samples × MoA classes)")
    print(f"  X_test  : {X_test.shape}   (test compounds × features)")
    print(f"  MoA classes : {len(moa_cols)}")
    print(f"  Saved to : {PROCESSED_DIR}")
    print(f"{'='*55}\n")

    return X_train, y_train, X_test, moa_cols


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    preprocess()

# MoA Drug Prediction — Deep Learning System

**IILM University, Greater Noida | BTP2CSE280 | Session 2025-26**
**Guide: Dr. Jaswinder Singh**
**Team: Sneha Singh, Nainika, Vishal Mansuriya, Simarpreet Kaur, Rudranill Chaterjee**

> **Phase 2: COMPLETE** | Dashboard with SHAP live at `localhost:8050` | Multi-modal MLP AUROC: **0.7779**

---

## Project Structure

```
DEEP LEARNING APPROACH TO PREDICT MOA OF DRUGS/
|
+-- app/
|   +-- dashboard.py          <- Dash dashboard (run this to view the app)
|
+-- data/
|   +-- raw/                  <- Original Kaggle CSVs (DO NOT MODIFY)
|   |   +-- train_features.csv        (23,814 x 876)
|   |   +-- train_targets_scored.csv  (23,814 x 207)
|   |   +-- test_features.csv         (3,982 x 876)
|   |   +-- sample_submission.csv
|   +-- processed/            <- Preprocessed .npy arrays (auto-generated)
|   |   +-- X_train.npy  (23814, 879)
|   |   +-- y_train.npy  (23814, 206)
|   |   +-- X_test.npy   (3982, 879)
|   |   +-- moa_columns.npy
|   |   +-- feature_columns.npy
|   +-- download_data.py      <- Kaggle API downloader
|   +-- preprocess.py         <- Preprocessing pipeline (z-score, one-hot, kNN)
|   +-- dataset.py            <- PyTorch Dataset wrapper
|
+-- docs/
|   +-- project_context.md    <- Static scope document
|   +-- project_logs.md       <- Session-by-session progress log
|   +-- PROJECT_PRESENTATION.md <- Plain-English viva prep guide
|
+-- models/
|   +-- mlp_baseline.py       <- 6-layer MLP (879->4096->...->206) + Focal Loss
|   +-- multimodal_mlp.py     <- 3-branch MLP (Gene, Cell, Meta) with residual fusion
|
+-- train/
|   +-- train_mlp.py          <- Training loop for baseline
|   +-- train_multimodal.py   <- Training loop for multi-modal Phase 2
|   +-- evaluate.py           <- AUROC, AUPRC, F1 metric functions
|
+-- utils/
|   +-- config.py             <- ALL hyperparameters (edit here to tune)
|   +-- seed.py               <- Global seed=42 setter
|   +-- shap_explain.py       <- SHAP KernelExplainer utility
|
+-- checkpoints/
|   +-- mlp_baseline_best.pth   <- Baseline weights (AUROC 0.7525)
|   +-- multimodal_mlp_best.pth <- Multi-modal weights (AUROC 0.7779)
|
+-- outputs/
|   +-- figures/              <- 6 EDA plots (auto-generated)
|   +-- shap_multimodal_*.npy <- Precomputed SHAP explanation data
|
+-- notebooks/
|   +-- eda.py                <- EDA script
|
+-- requirements.txt
+-- README.md
```

---

## Dataset — Actual Verified Dimensions

| Feature Group | Count | Column Names |
|--------------|-------|-------------|
| Gene expression | 772 | g-* |
| Cell viability | 100 | c-* |
| Categorical (one-hot) | 7 | cp_type, cp_dose, cp_time expanded |
| **Total input** | **879** | — |
| MoA labels | 206 | scored targets |

---

## Model Results

| Model | AUROC (Target) | AUROC (Actual) | Status |
|-------|---------------|----------------|--------|
| **MLP Baseline** | 0.778 | **0.7525** | TRAINED ✅ |
| **Multi-modal MLP** | 0.831 | **0.7779** | TRAINED ✅ |
| GAT | 0.872 | — | Phase 3 (Planned) |
| RotatE | 0.851 | — | Phase 3 (Planned) |
| MolBERT | 0.862 | — | Phase 4 (Planned) |
| **Ensemble** | **0.911** | — | Phase 4 (Planned) |

---

## Quick Start

### All dependencies are already installed. Training data is in place.

### Launch the dashboard (real model already trained)
```bash
python app/dashboard.py
# Open: http://localhost:8050
```

### Re-run training if needed
```bash
python train/train_mlp.py --dry-run   # 1-epoch sanity check
python train/train_mlp.py             # Full training (stops around epoch 37)
```

### Re-run preprocessing if needed
```bash
python data/preprocess.py
```

### Regenerate EDA figures
```bash
python notebooks/eda.py
# Figures saved to: outputs/figures/
```

---

## Configuration
All hyperparameters in `utils/config.py`:
- `N_TOTAL_FEATURES = 879` — actual input size (verified from real data)
- `N_MOA_CLASSES = 206`
- `MLP_HIDDEN_DIMS = [4096, 2048, 1024, 512, 256]`
- `BATCH_SIZE = 256` | `MAX_EPOCHS = 200` | `EARLY_STOP_PATIENCE = 15`
- `FOCAL_GAMMA = 2.0` | `LEARNING_RATE = 1e-3` | `RANDOM_SEED = 42`

---

## For Viva Preparation
Read `docs/PROJECT_PRESENTATION.md` — plain English explanation of every component, including Q&A prep for common examiner questions.


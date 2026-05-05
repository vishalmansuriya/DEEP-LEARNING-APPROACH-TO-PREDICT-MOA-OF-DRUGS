# MoA Drug Prediction — Project Context
> This file is the static source of truth for the project scope. Do NOT modify unless the scope changes.

## What We Are Building
A deep learning system that predicts the **Mechanism of Action (MoA)** of drugs given their biological feature vector.

## Dataset — ACTUAL VERIFIED DIMENSIONS (updated 2026-05-06)
- **Primary**: Kaggle "MoA Prediction" competition (lish-moa)
  - 23,814 training compounds | 3,982 test compounds
  - **772** gene expression features (g-*) — confirmed from real data
  - **100** cell viability features (c-*) — confirmed from real data
  - 3 categorical features (cp_type, cp_dose, cp_time) → 7 after one-hot encoding
  - **Total input features: 879** (772 + 100 + 7)
  - 206 binary MoA labels (multi-label classification)
  - No missing values in the Kaggle dataset
- **Supplementary** (Phase 2+): ChEMBL, LINCS L1000, KEGG, FAERS, STRING

## Model Pipeline — Build Status
| # | Model | Target AUROC | Actual AUROC | Status |
|---|-------|-------------|--------------|--------|
| 1 | MLP Baseline | 0.778 | **0.7525** | TRAINED ✅ |
| 2 | Multi-modal MLP | 0.831 | **0.7779** | TRAINED ✅ |
| 3 | GAT (Graph Attention) | 0.872 | — | Phase 3 |
| 4 | RotatE (KG Embeddings) | 0.851 | — | Phase 3 |
| 5 | MolBERT | 0.862 | — | Phase 4 |
| 6 | Stacking Ensemble | **0.911** | — | Phase 4 |

## Tech Stack
- Deep Learning: PyTorch 2.11.0+cpu
- Data: NumPy 2.4.2, Pandas 3.0.2, Scikit-learn 1.8.0
- Dashboard: Dash 4.1.0, Plotly 6.5.2, Flask 3.1.3
- Graph ML (Phase 3): PyTorch Geometric, DGL
- Cheminformatics (Phase 2): RDKit
- KG Database (Phase 3): Neo4j v5.x
- Explainability (Phase 2): SHAP, Captum

## Working Conventions
- `random_seed = 42` everywhere
- Focal loss (gamma=2.0) for class imbalance
- AdamW optimizer + cosine annealing LR with warm restarts
- Early stopping: patience=15
- 5-fold cross-validation (to be applied in Phase 2+)

## Build Order — Phase Completion
- **Phase 1**: COMPLETE — MLP Baseline trained, dashboard live at localhost:8050
- **Phase 2**: COMPLETE — Multi-modal MLP trained (AUROC 0.7779), SHAP integrated
- **Phase 3**: IN PROGRESS — KG built, GAT + RotatE + FusedModel implemented (needs full training)
- **Phase 4**: Pending — MolBERT + Ensemble + Docker + FastAPI

- Optimizer: AdamW, cosine annealing LR

## Build Order
- Phase 1: Dataset EDA + MLP Baseline + Basic Dash Dashboard
- Phase 2: Multi-modal MLP + SHAP
- Phase 3: Knowledge Graph + GAT + RotatE
- Phase 4: MolBERT + Ensemble + Docker + FastAPI

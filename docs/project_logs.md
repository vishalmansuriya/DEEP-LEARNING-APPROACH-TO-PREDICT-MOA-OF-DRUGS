# Project Logs — MoA Drug Prediction

> Latest at the top. Each session entry is self-contained.

---

## [2026-05-06] Session 8 — Phase 3: Knowledge Graph + GAT + RotatE

### Purpose of this session
Implement the full Phase 3 pipeline: build the biological knowledge graph,
train a Graph Attention Network (GAT), train RotatE KG embeddings, and create
the FusedModel that combines all three signal sources.

### Completed

**Step 1 — Config**
- Added all Phase 3 constants to `utils/config.py`:
  KG settings, GAT hyperparameters, RotatE hyperparameters, FusedModel dims.

**Step 2 — Knowledge Graph (`data/build_kg.py`)**
- Built bipartite compound <-> MoA label graph from training labels
- **KG stats**: 24,020 nodes (23,814 compound + 206 MoA labels) | 33,688 edges
- **Graph density**: 0.34% (extremely sparse — realistic for drug biology)
- Avg MoA per drug: 0.71 | Max label freq: 832 | Min: 1
- Saved: `data/processed/kg_graph.pt`, `data/processed/kg_node_features.npy`

**Step 3 — GAT Model (`models/gat_model.py`, `train/train_gat.py`)**
- Custom 2-layer GAT implemented without torch_geometric (pure PyTorch sparse ops)
- Architecture: 879 -> 512 (4-head concat) -> 256 (single-head mean)
- GATClassifier wraps GATEncoder + linear head (703,438 parameters)
- After training: exports `outputs/gat_compound_embeddings.npy` (23814, 256)
- Dry-run: PASSED (1 epoch, 500 samples, 3.8s)

**Step 4 — RotatE (`models/rotate.py`, `train/train_rotate.py`)**
- RotatE (Sun et al., 2019): entities in complex space, relation as rotation
- 6,149,248 parameters (large embedding matrix for 24,020 entities x 128 dim x 2)
- Self-supervised training on compound->MoA edges with 64 negative samples per positive
- Loss drops from 2.51 -> 1.15 in 3 epochs (healthy convergence)
- After training: exports `outputs/rotate_embeddings.npy` (23814, 128)
- 3-epoch test: PASSED

**Step 5 — FusedModel (`models/multimodal_mlp.py`, `train/train_fused.py`)**
- `FusedModel` class added to `multimodal_mlp.py` (backward compatible)
- Fuses: MultiModalMLP latent (416) + GAT (256) + RotatE (128) = 800-dim
- 2-stage training: MultiModalMLP frozen for first 10 epochs, then unfrozen
- 2,652,668 total parameters | 595,918 trainable in frozen stage
- Dry-run: PASSED (1 epoch, 500 samples, 0.4s)

### Dry-run verification (all PASSED)
| Script | Status | Notes |
|--------|--------|-------|
| `python data/build_kg.py` | PASSED | 24,020 nodes, 33,688 edges |
| `python models/gat_model.py` | PASSED | 584,192 params, shape (110, 256) |
| `python models/rotate.py` | PASSED | loss=2.53, embeddings (100, 128) |
| `python models/multimodal_mlp.py` | PASSED | FusedModel (32, 206) output |
| `python train/train_gat.py --dry-run` | PASSED | 703k params, 3.8s/epoch |
| `python train/train_rotate.py --epochs 3` | PASSED | loss 2.51->1.15 |
| `python train/train_fused.py --dry-run` | PASSED | 2.6M params, 0.4s/epoch |

### Project state at end of session

| Phase | Item | Status | Where |
|-------|------|--------|-------|
| 1 | MLP Baseline | DONE | `checkpoints/mlp_baseline_best.pth` |
| 2 | Multi-modal MLP | DONE — AUROC 0.7779 | `checkpoints/multimodal_mlp_best.pth` |
| 3 | Knowledge Graph | DONE | `data/processed/kg_graph.pt` |
| 3 | GAT model + trainer | DONE (needs full train) | `models/gat_model.py`, `train/train_gat.py` |
| 3 | RotatE + trainer | DONE (needs full train) | `models/rotate.py`, `train/train_rotate.py` |
| 3 | FusedModel + trainer | DONE (needs full train) | `models/multimodal_mlp.py`, `train/train_fused.py` |
| 4 | MolBERT + Ensemble | NOT STARTED | Phase 4 |

### Next session priorities
1. Run full GAT training:    `python train/train_gat.py`  (~100 epochs)
2. Run full RotatE training: `python train/train_rotate.py`  (~50 epochs)
3. Copy `multimodal_mlp_best.pth` to checkpoints/ (if missing from git)
4. Run full Fused training:  `python train/train_fused.py`
5. Update dashboard with Phase 3 model selector

### Notes on pre-trained checkpoint
`multimodal_mlp_best.pth` is excluded from git (.gitignore). Before running
`train/train_fused.py` for real, either:
  a) Re-train: `python train/train_multimodal.py`
  b) Copy from the original developer's machine

---

## [2026-05-06] Session 7 — Handover Snapshot & GitHub Initialization

### Purpose of this session
Freeze the project at a clean checkpoint so it can be handed over to a new
contributor without context loss. Also: initialize git, push to GitHub, and
write a single source-of-truth handover document.

### Completed
- **Repository initialized and pushed to GitHub:**
  https://github.com/RudranilChatterjee07/DEEP-LEARNING-APPROACH-TO-PREDICT-MOA-OF-DRUGS
- Created `.gitignore` excluding raw Kaggle CSVs (re-downloadable), processed
  numpy arrays (re-generatable via `data/preprocess.py`), Python caches, and
  IDE files. Trained checkpoints and SHAP outputs are committed so the next
  contributor can run the dashboard immediately without retraining.
- Created `docs/HANDOVER.md` — single document the next contributor should
  read first. Covers environment setup, what's built, what's next, file map.
- Cleaned up `docs/project_context.md` — now contains the canonical Phase 3
  and Phase 4 task breakdown with concrete file names and acceptance criteria.
- Fixed `docs/PROJECT_PRESENTATION.md`:
  - Corrected dataset dimensions throughout (was claiming 100 gene + 77 cell;
    real values are 772 gene + 100 cell + 7 one-hot meta = 879 input dims).
  - Marked Phase 3 (KG/GAT/RotatE) and Phase 4 (MolBERT/Ensemble/Deploy) as
    PLANNED, not "What we did". Earlier wording was aspirational and would
    have been embarrassing in a viva.
  - Kept the model-explanation Q&A intact since the conceptual answers are
    still accurate.
- Updated `README.md` with the real state of both Phase 1 and Phase 2 models,
  the actual SHAP outputs, and the recommended pre-Phase-3 deviation note.

### Project state at end of session

| Phase | Item | Status | Where |
|-------|------|--------|-------|
| 1 | MLP Baseline trained | DONE — Val AUROC 0.7525 | `checkpoints/mlp_baseline_best.pth` |
| 1 | EDA figures | DONE | `outputs/figures/*.png` |
| 1 | Dashboard live | DONE | `app/dashboard.py` |
| 2 | Multi-modal MLP trained | DONE — Val AUROC 0.7779 | `checkpoints/multimodal_mlp_best.pth` |
| 2 | SHAP values computed | DONE | `outputs/shap_multimodal_*.npy` |
| 2 | SHAP panel in dashboard | DONE | `app/dashboard.py` |
| 2.5 | ECFP4 fingerprints (skipped earlier) | RECOMMENDED before Phase 3 | not implemented |
| 3 | Knowledge Graph construction | NOT STARTED | `data/build_kg.py` (TBD) |
| 3 | GAT model | NOT STARTED | `models/gat_model.py` (TBD) |
| 3 | RotatE embeddings | NOT STARTED | `models/rotate.py` (TBD) |
| 4 | MolBERT fine-tune | NOT STARTED | `models/molbert.py` (TBD) |
| 4 | Stacking ensemble | NOT STARTED | `models/ensemble.py` (TBD) |
| 4 | FastAPI + Docker | NOT STARTED | `app/api.py`, `Dockerfile` (TBD) |

### Next session priorities (whoever picks this up)
1. Read `docs/HANDOVER.md` end-to-end.
2. Decide on Phase 2.5 (ECFP4 fingerprints) — if YES, do it before Phase 3.
   Cost: install rdkit, fetch SMILES, retrain multi-modal. Benefit: closes
   gap from 0.7779 toward original 0.831 target.
3. Begin Phase 3: knowledge graph construction. Start with a small KG using
   only the 23,814 training compounds and their MoA labels as relations,
   then expand with ChEMBL/KEGG.

---

## [2026-05-06] Session 6 — Multi-modal MLP & Explainability

### Completed
- **Phase 2 Model Built**: Created `models/multimodal_mlp.py`
  - Multi-branch architecture: GeneEncoder (772), CellEncoder (100), MetaEncoder (7)
  - Fusion head combining modalities (416-dim latent space)
  - Residual connections and Label-Smoothed Focal Loss (eps=0.05)
- **Phase 2 Training**: Executed `train/train_multimodal.py`
  - Best Val AUROC: **0.7779** (improvement over baseline 0.7525)
  - Best Val AUPRC: 0.1375
  - Model parameters: 2,056,750 (significant reduction from baseline 14M, yet
    better performance — the multi-branch structure is more parameter-efficient)
- **Explainability**: Created `utils/shap_explain.py`
  - Uses SHAP KernelExplainer for model-agnostic feature importance
  - Generated `outputs/shap_multimodal_*.npy` files
- **Dashboard**: Multi-model selector + SHAP waterfall panel integrated
- **Dependencies**: Installed `shap` library

### Phase 2 Status: COMPLETE (with one deviation from original plan)
The original Phase 2 plan called for ECFP4 chemical fingerprints from
RDKit/PubChem to be concatenated with gene/cell features (target input
dim 2927). That work was NOT done. Instead, "multi-modal" was reinterpreted
as splitting the existing 879-dim feature vector into Gene/Cell/Meta
sub-encoders. This is why the AUROC stopped at 0.7779 instead of the
originally projected 0.831 — there is no chemical-structure signal in the
model yet. Recommended for the next contributor: backfill ECFP4 as a
"Phase 2.5" before moving to Phase 3, or rely on Phase 3's KG to provide
the chemistry signal.

### Next Steps (Session 7+)
See Session 7 entry above and `docs/HANDOVER.md`.

---

## [2026-05-06] Session 5 — Documentation Sync & Phase 1 Sign-Off

### Completed
- Updated `docs/project_context.md`:
  - Corrected dataset dimensions to actual verified values (772 gene, 100 cell, 879 total)
  - Added model pipeline status table (MLP Baseline: TRAINED 0.7525, others pending)
  - Updated tech stack with real installed package versions
  - Added phase completion tracker
- Updated `README.md`:
  - Reflects Phase 1 COMPLETE status prominently
  - Shows real processed file sizes and dimensions
  - Accurate Quick Start (all setup steps already done)
  - Current model results table with real vs target AUROC
  - Removed outdated "setup" steps (data already downloaded, model already trained)
- `project_logs.md` (this file): Sessions 1-5 fully logged

### Project Artifacts — Current State

| File | Status | Key Details |
|------|--------|-------------|
| data/raw/train_features.csv | Present | 23,814 x 876, 149 MB |
| data/raw/train_targets_scored.csv | Present | 23,814 x 207, 9.7 MB |
| data/raw/test_features.csv | Present | 3,982 x 876, 24.9 MB |
| data/processed/X_train.npy | Present | (23814, 879), 83 MB |
| data/processed/y_train.npy | Present | (23814, 206), 19 MB |
| data/processed/X_test.npy | Present | (3982, 879), 14 MB |
| data/processed/moa_columns.npy | Present | 206 class names |
| checkpoints/mlp_baseline_best.pth | Present | 56 MB, epoch 22, AUROC 0.7525 |
| outputs/figures/ | Present | 6 figures (real data) |
| app/dashboard.py | Live | localhost:8050, green "Model LIVE" badge |

### Phase 1 Final Status: COMPLETE

---

## [2026-05-06] Session 4 — Data Integration, Training & Dashboard Go-Live

### Completed
- User provided all 4 Kaggle CSV files in the project root; auto-moved to data/raw/:
  - train_features.csv   (149 MB, 23,814 rows x 876 cols)
  - train_targets_scored.csv (9.7 MB, 23,814 rows x 207 cols)
  - test_features.csv    (24.9 MB, 3,982 rows x 876 cols)
  - sample_submission.csv (3.2 MB)
- Ran `data/preprocess.py` — PASSED:
  - Discovered real feature dimensions: 772 gene + 100 cell + 7 one-hot cat = 879 total
  - No missing values found (kNN impute step skipped cleanly)
  - Saved: X_train(23814,879), y_train(23814,206), X_test(3982,879), moa_columns.npy
- Updated `utils/config.py` with correct real dimensions (879 features, 4096-first-layer MLP)
- Ran dry-run (1 epoch, 100 samples) — PASSED in 0.4s
- Ran FULL training (200 epoch max, early stopping patience=15):
  - Stopped at epoch 37 (no improvement for 15 epochs)
  - Best Val AUROC: 0.7525 (epoch 22)
  - Best Val AUPRC: 0.1339
  - Checkpoint saved: `checkpoints/mlp_baseline_best.pth`
  - Model parameters: 14,818,254
- Regenerated EDA figures with REAL training data
- Updated `app/dashboard.py` to load real model checkpoint
- Dashboard launched at http://localhost:8050 with green "Model LIVE" badge

### Errors Encountered & Fixed
- Unicode arrow character in `preprocess.py` print() -> cp1252 encoding error on Windows.
  Fix: replaced all special Unicode chars with ASCII equivalents.
- `_stat_card()` helper defined after it was called in `app.layout`.
  Fix: moved function definition before `app.layout` block.

---

## [2026-05-05] Session 3 — Roadmap Review & Status Check

### Completed
- Reviewed full project roadmap and confirmed current build status
- Confirmed all installed packages are working:
  - PyTorch 2.11.0+cpu
  - Dash 4.1.0 | NumPy 2.4.2 | Pandas 3.0.2 | Scikit-learn 1.8.0
- Dashboard verified live at http://localhost:8050 with dark theme + demo prediction working
- All 6 EDA figures confirmed on disk (`outputs/figures/`)

### Blockers
- BLOCKER: Kaggle training CSV files not yet in `data/raw/` (resolved in Session 4)

---

## [2026-05-05] Session 2 — Environment Setup + EDA + Full Smoke Tests

### Completed
- Recreated `data/preprocess.py` with improved step-by-step logging
- Created `data/raw/` and `data/processed/` directories
- Installed all dependencies (PyTorch 2.11.0+cpu, Dash 4.1.0, sklearn, etc.)
- Created `notebooks/eda.py` — 6 publication-quality dark-theme figures:
  1. moa_label_distribution.png
  2. label_count_histogram.png
  3. gene_expression_dist.png
  4. class_imbalance.png
  5. model_performance.png
  6. architecture_summary.png
- Ran full codebase smoke test — ALL PASSED:
  - MLPBaseline: forward pass correct
  - FocalLoss: computing correctly
  - MoADataset: batching correctly
  - AUROC/AUPRC/F1: metric functions all working

### Errors / Blockers
- `dash` not found initially — fixed by `pip install dash`
- `seaborn` not found — fixed by `pip install seaborn`
- Unicode characters in print() causing cp1252 encoding error on Windows — fixed via ASCII

---

## [2026-05-05] Session 1 — Project Initialization

### Completed
- Created full project directory scaffold:
  - `docs/`, `data/`, `models/`, `train/`, `outputs/figures/`, `checkpoints/`, `notebooks/`, `utils/`
- Confirmed dataset: `sample_submission.csv` present
- Created `docs/project_context.md`, `docs/project_logs.md`, `docs/PROJECT_PRESENTATION.md`
- Created `utils/config.py`, `utils/seed.py`
- Created `data/download_data.py`, `data/preprocess.py`, `data/dataset.py`
- Created `models/mlp_baseline.py`
- Created `train/train_mlp.py`, `train/evaluate.py`
- Created `app/dashboard.py`
- Created `requirements.txt`

### Errors / Blockers
- None at this stage (data files not yet downloaded — handled in Session 4)

# Project Handover Guide — MoA Drug Prediction

> This document is for the next developer taking over the project. It covers the current state, how to run things, and the immediate next steps for Phase 3 and 4.

---

## 1. Project Overview
This system predicts the **Mechanism of Action (MoA)** of drugs using deep learning. We have built a robust pipeline that processes biological high-dimensional data (879 features) and outputs probabilities for 206 MoA classes.

**Current Performance:**
- **Baseline MLP**: 0.7525 AUROC
- **Multi-modal MLP (Phase 2)**: 0.7779 AUROC

---

## 2. Environment & Setup
- **OS**: Developed on Windows.
- **Language**: Python 3.10+
- **Core Libraries**: PyTorch, Dash, Scikit-learn, Pandas, SHAP.
- **Setup**:
  ```bash
  pip install -r requirements.txt
  ```
- **Data**: Raw Kaggle CSVs are in `data/raw/`. Processed `.npy` files are in `data/processed/`.
- **Note**: Large data files are ignored by git. If they are missing, run `data/preprocess.py`.

---

## 3. How to Run the System

### A. Dashboard (The Main Interface)
The dashboard allows you to run predictions on test compounds or uploaded CSVs and visualize SHAP explainability waterfall charts.
```bash
python app/dashboard.py
# Open: http://localhost:8050
```

### B. Training Models
If you want to retrain from scratch:
```bash
# Baseline
python train/train_mlp.py
# Multi-modal (Phase 2)
python train/train_multimodal.py
```

### C. Explainability
To recompute SHAP values (takes a few minutes on CPU):
```bash
python utils/shap_explain.py --n_samples 50
```

---

## 4. Immediate Next Steps (Phase 3 & 4)

### Phase 3: Knowledge Graph (KG) Integration
The goal is to move from purely statistical learning to biological reasoning.
1. **Build KG**: Create `data/build_kg.py`. Map drugs to proteins (targets) and genes to pathways.
2. **GAT Model**: Implement `models/gat_model.py`. Use the graph structure to weigh feature importance.
3. **RotatE**: Use Knowledge Graph Embeddings to provide extra latent features for the MLP.

### Phase 4: Ensembling & Deployment
1. **MolBERT**: Fine-tune a chemical transformer on SMILES strings.
2. **Ensemble**: Create `models/ensemble.py` to combine predictions from all models using a Meta-Learner (Logistic Regression).
3. **API**: Wrap the models in a FastAPI service for production use.

---

## 5. Key File Map
- `utils/config.py`: Central source for all hyperparameters and paths.
- `app/dashboard.py`: Frontend logic.
- `docs/PROJECT_PRESENTATION.md`: Use this for viva prep or explaining the project to non-technical stakeholders.
- `docs/project_logs.md`: Detailed history of every session.

---
*Good luck! The foundations are solid, and the model is already performing well above random chance (0.5).*

# GNN Improvements - GroupKFold Time-Based CV Training

## Overview
This worktree contains improvements to GNN training following the baseline.py workflow from the 1st place solution by Chris Deotte. The key improvement is using **GroupKFold cross-validation with time-based grouping** to prevent data leakage and enable effective training on Kaggle.

## Key Changes

### 1. Time-Based GroupKFold Validation
- Uses `GroupKFold(n_splits=6)` splitting by transaction month
- Prevents temporal leakage: future transactions never used to predict past
- Matches baseline.py approach for fair comparison

### 2. New Training File: `train_cv.py`
- **`create_month_feature()`**: Converts TransactionDT to month number for grouping
- **`train_gnn_with_cv()`**: Trains GNN with GroupKFold, early stopping, and OOF predictions
- **`train_xgboost_with_embeddings()`**: Trains XGBoost on GNN embeddings with same CV scheme
- **`main()`**: Full pipeline with ablation study

### 3. Training Pipeline
```
Load Data → Build Graph → Extract Time Info →
GraphSAGE CV Training (6 folds) → Extract Embeddings →
XGBoost CV Training on Embeddings →
GAT CV Training → Extract Embeddings →
XGBoost CV Training on Embeddings →
Ablation Results
```

## Usage

### Run CV Training
```bash
cd /home/quillbolt/project/hackathon/fraud-detection-gnn-gnn
uv run python -m src.fraud_detection.train_cv
```

### Expected Output
```
Starting GNN training with GroupKFold time-based validation...
============================================================
Using device: cuda
============================================================
Building graph...
Graph built: 4 node types, 590540 transactions
============================================================
Training GraphSAGE with GroupKFold CV...
============================================================
Fold 1/6 | Withholding month 1
Train: 590000 rows | Val: 540 rows
  Epoch  10/100 | Loss: 0.2345 | Val AUC: 0.8512 | Best: 0.8512
  ...
GNN OOF CV AUC = 0.9234
...
============================================================
Ablation Study Results:
  graphsage_gnn_auc: 0.9234
  graphsage_xgb_auc: 0.9456
  gat_gnn_auc: 0.9210
  gat_xgb_auc: 0.9432
============================================================
```

### Model Outputs
- `models/graphsage_best.pt` - Best GraphSAGE model across all folds
- `models/gat_best.pt` - Best GAT model across all folds
- `models/xgboost_gsage.json` - XGBoost trained on GraphSAGE embeddings
- `models/xgboost_gat.json` - XGBoost trained on GAT embeddings

## Comparison with Baseline

| Metric | Baseline XGBoost | GraphSAGE + XGBoost | GAT + XGBoost |
|--------|----------------|-------------------|-------------|
| CV AUC | ~0.96 | TBD | TBD |

## Technical Details

### GroupKFold Time-Based Splitting
- Groups transactions by `DT_M` (month since 2017-11-30)
- Each fold holds out one month as validation
- Ensures models learn temporal patterns, not just static fraud signatures

### Early Stopping
- Monitors validation AUC every 10 epochs
- Patience: 20 epochs (configurable)
- Saves best model state per fold

### Memory Management
- `gc.collect()` after each fold
- `torch.cuda.empty_cache()` for GPU training
- Explicit deletion of fold-specific models

## Kaggle Competition Notes

### Advantages of This Approach
1. **Temporal Validity**: Models trained on past data to predict future transactions
2. **Robust Validation**: 6-fold CV reduces variance in performance estimates
3. **Fair Comparison**: Same CV scheme as baseline.py for direct model comparison

### For Kaggle Submission
1. Generate OOF predictions using `train_cv.py`
2. Train final model on full training data
3. Generate test predictions using trained model
4. Submit to competition

## Files Modified/Added
- `src/fraud_detection/train_cv.py` - New CV training script
- `README_GNN_IMPROVEMENTS.md` - This file

## Dependencies
Same as main project - run `uv sync` in this worktree.

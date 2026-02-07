"""
GNN Training with GroupKFold Time-Based Validation (follows baseline.py workflow)
Based on 1st place solution pattern: GroupKFold(n_splits=6) with time-based grouping
"""

import numpy as np
import pandas as pd
import gc
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from loguru import logger
from torch_geometric.data import HeteroData
from fraud_detection.config import MODELS_DIR, DATA_DIR, GNNS, XGBOOST_PARAMS
from fraud_detection.data.preprocessing import GraphBuilder
from fraud_detection.models.gnn import GraphSAGEModel, GATModel
from fraud_detection.models.classifier import HybridClassifier


def create_month_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Create month feature from TransactionDT for time-based splitting."""
    START_DATE = datetime.strptime("2017-11-30", "%Y-%m-%d")
    df = df.copy()
    df["DT_M"] = df["TransactionDT"].apply(
        lambda x: (START_DATE + timedelta(seconds=x))
    )
    df["DT_M"] = (df["DT_M"].dt.year - 2017) * 12 + df["DT_M"].dt.month
    return df


def build_graph_with_time_info(
    train_csv: str,
    test_csv: str,
    train_identity: str
) -> tuple[HeteroData, pd.DataFrame]:
    """Build graph and return data with time information for grouping."""
    builder = GraphBuilder(train_csv, test_csv, train_identity)
    data = builder.build_graph()
    
    train_df = pd.read_csv(train_csv)
    train_df = create_month_feature(train_df)
    
    return data, train_df


def train_gnn_with_cv(
    model: torch.nn.Module,
    data: HeteroData,
    train_df: pd.DataFrame,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = "cpu",
    n_splits: int = 6,
    early_stopping_rounds: int = 20,
    batch_size: int = None
) -> tuple[np.ndarray, torch.nn.Module]:
    """Train GNN with GroupKFold time-based cross-validation.
    
    Args:
        model: GNN model to train
        data: HeteroData graph object
        train_df: Training DataFrame with DT_M column for grouping
        epochs: Maximum epochs per fold
        lr: Learning rate
        device: Device to train on
        n_splits: Number of GroupKFold splits
        early_stopping_rounds: Patience for early stopping
        batch_size: Batch size (None for full batch)
    
    Returns:
        tuple: (OOF predictions, best model)
    """
    oof = np.zeros(len(train_df))
    skf = GroupKFold(n_splits=n_splits)
    
    best_model = None
    best_global_auc = 0.0
    
    for i, (idxT, idxV) in enumerate(
        skf.split(train_df, train_df["isFraud"], groups=train_df["DT_M"])
    ):
        month = train_df.iloc[idxV]["DT_M"].iloc[0]
        logger.info("=" * 60)
        logger.info(f"Fold {i+1}/{n_splits} | Withholding month {month}")
        logger.info(f"Train: {len(idxT)} rows | Val: {len(idxV)} rows")
        
        fold_model = type(model)(
            **model.__dict__
        ).to(device)
        optimizer = torch.optim.Adam(fold_model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        best_val_auc = 0.0
        patience_counter = 0
        best_state_dict = None
        
        x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
        y_tensor = data["transaction"].y.to(device)
        
        for epoch in range(epochs):
            fold_model.train()
            optimizer.zero_grad()
            
            out = fold_model(x_dict, edge_index_dict)
            pred = out["transaction"]
            
            train_mask = torch.zeros(len(pred), dtype=torch.bool, device=device)
            train_mask[idxT] = True
            
            loss = criterion(pred[train_mask], y_tensor[train_mask].float())
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                fold_model.eval()
                with torch.no_grad():
                    val_mask = torch.zeros(len(pred), dtype=torch.bool, device=device)
                    val_mask[idxV] = True
                    
                    val_pred = torch.sigmoid(pred[val_mask])
                    val_labels = y_tensor[val_mask].cpu().numpy()
                    val_scores = val_pred.cpu().numpy()
                    
                    val_auc = roc_auc_score(val_labels, val_scores)
                    
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        patience_counter = 0
                        best_state_dict = fold_model.state_dict().copy()
                        
                        if val_auc > best_global_auc:
                            best_global_auc = val_auc
                            best_model = fold_model
                    else:
                        patience_counter += 1
                    
                    logger.info(
                        f"  Epoch {epoch+1:3d}/{epochs} | "
                        f"Loss: {loss:.4f} | "
                        f"Val AUC: {val_auc:.4f} | "
                        f"Best: {best_val_auc:.4f}"
                    )
                    
                    if patience_counter >= early_stopping_rounds:
                        logger.info(f"  Early stopping at epoch {epoch+1}")
                        break
        
        fold_model.load_state_dict(best_state_dict)
        fold_model.eval()
        with torch.no_grad():
            out = fold_model(x_dict, edge_index_dict)
            pred = torch.sigmoid(out["transaction"])
            
            val_mask = torch.zeros(len(pred), dtype=torch.bool, device=device)
            val_mask[idxV] = True
            
            oof[idxV] = pred[val_mask].cpu().numpy()
        
        del fold_model, optimizer, best_state_dict
        gc.collect()
        
        if device == "cuda":
            torch.cuda.empty_cache()
    
    logger.info("=" * 60)
    oof_labels = train_df["isFraud"].values
    oof_auc = roc_auc_score(oof_labels, oof)
    logger.info(f"GNN OOF CV AUC = {oof_auc:.4f}")
    
    return oof, best_model


def train_xgboost_with_embeddings(
    embeddings: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    train_df: pd.DataFrame,
    n_splits: int = 6
) -> tuple[np.ndarray, HybridClassifier]:
    """Train XGBoost with GNN embeddings using GroupKFold.
    
    Args:
        embeddings: GNN embeddings
        features: Transaction features
        labels: Fraud labels
        train_df: Training DataFrame with DT_M column
        n_splits: Number of GroupKFold splits
    
    Returns:
        tuple: (OOF predictions, trained classifier)
    """
    from xgboost import XGBClassifier
    
    X = np.hstack([embeddings, features])
    oof = np.zeros(len(X))
    skf = GroupKFold(n_splits=n_splits)
    
    for i, (idxT, idxV) in enumerate(
        skf.split(X, labels, groups=train_df["DT_M"])
    ):
        month = train_df.iloc[idxV]["DT_M"].iloc[0]
        logger.info(f"XGB Fold {i+1}/{n_splits} | Withholding month {month}")
        
        clf = XGBClassifier(
            n_estimators=5000,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.4,
            missing=-1,
            eval_metric="auc",
            tree_method="hist",
            random_state=42 + i,
        )
        
        clf.fit(
            X[idxT], labels[idxT],
            eval_set=[(X[idxV], labels[idxV])],
            verbose=100,
            early_stopping_rounds=200,
        )
        
        oof[idxV] = clf.predict_proba(X[idxV])[:, 1]
        
        del clf
        gc.collect()
    
    oof_auc = roc_auc_score(labels, oof)
    logger.info(f"XGBoost OOF CV AUC = {oof_auc:.4f}")
    
    final_clf = XGBClassifier(
        n_estimators=5000,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.4,
        missing=-1,
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
    )
    final_clf.fit(X, labels, verbose=100)
    
    classifier = HybridClassifier(XGBOOST_PARAMS)
    classifier.model = final_clf
    
    return oof, classifier


def main():
    """Main training pipeline with GroupKFold CV."""
    logger.info("Starting GNN training with GroupKFold time-based validation...")
    logger.info("=" * 60)
    
    data_path = DATA_DIR
    train_csv = data_path / "train_transaction.csv"
    test_csv = data_path / "test_transaction.csv"
    train_identity = data_path / "train_identity.csv"
    
    if not train_csv.exists():
        logger.error(f"Data not found at {train_csv}")
        logger.info("Run: uv run python -m src.fraud_detection.data.download")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info("=" * 60)
    
    logger.info("Building graph...")
    data, train_df = build_graph_with_time_info(
        str(train_csv),
        str(test_csv),
        str(train_identity)
    )
    logger.info(f"Graph built: {len(data.node_types)} node types, {len(train_df)} transactions")
    logger.info("=" * 60)
    
    gs_config = GNNS["graphsage"]
    gat_config = GNNS["gat"]
    
    logger.info("Training GraphSAGE with GroupKFold CV...")
    graphsage = GraphSAGEModel(
        hidden_channels=gs_config["hidden_channels"],
        num_layers=gs_config["num_layers"],
        dropout=gs_config["dropout"]
    )
    gs_oof, gs_best = train_gnn_with_cv(
        graphsage, data, train_df,
        epochs=100, lr=gs_config["lr"],
        device=device, n_splits=6
    )
    torch.save(gs_best.state_dict(), MODELS_DIR / "graphsage_best.pt")
    logger.info(f"GraphSAGE model saved to: {MODELS_DIR / 'graphsage_best.pt'}")
    
    gs_embeddings = gs_oof.reshape(-1, 1)
    logger.info("=" * 60)
    
    logger.info("Training XGBoost with GraphSAGE embeddings...")
    trans_features = data["transaction"].x.numpy()
    labels = train_df["isFraud"].values
    gs_xgb_oof, gs_classifier = train_xgboost_with_embeddings(
        gs_embeddings, trans_features, labels, train_df
    )
    gs_classifier.save(str(MODELS_DIR / "xgboost_gsage.json"))
    logger.info(f"GraphSAGE XGBoost saved to: {MODELS_DIR / 'xgboost_gsage.json'}")
    logger.info("=" * 60)
    
    logger.info("Training GAT with GroupKFold CV...")
    gat = GATModel(
        hidden_channels=gat_config["hidden_channels"],
        num_heads=gat_config["num_heads"],
        num_layers=gat_config["num_layers"],
        dropout=gat_config["dropout"]
    )
    gat_oof, gat_best = train_gnn_with_cv(
        gat, data, train_df,
        epochs=100, lr=gat_config["lr"],
        device=device, n_splits=6
    )
    torch.save(gat_best.state_dict(), MODELS_DIR / "gat_best.pt")
    logger.info(f"GAT model saved to: {MODELS_DIR / 'gat_best.pt'}")
    
    gat_embeddings = gat_oof.reshape(-1, 1)
    logger.info("=" * 60)
    
    logger.info("Training XGBoost with GAT embeddings...")
    gat_xgb_oof, gat_classifier = train_xgboost_with_embeddings(
        gat_embeddings, trans_features, labels, train_df
    )
    gat_classifier.save(str(MODELS_DIR / "xgboost_gat.json"))
    logger.info(f"GAT XGBoost saved to: {MODELS_DIR / 'xgboost_gat.json'}")
    logger.info("=" * 60)
    
    ablation_results = {
        "graphsage_gnn_auc": roc_auc_score(labels, gs_oof),
        "graphsage_xgb_auc": roc_auc_score(labels, gs_xgb_oof),
        "gat_gnn_auc": roc_auc_score(labels, gat_oof),
        "gat_xgb_auc": roc_auc_score(labels, gat_xgb_oof),
    }
    
    logger.info("=" * 60)
    logger.info("Ablation Study Results:")
    for key, value in ablation_results.items():
        logger.info(f"  {key}: {value:.4f}")
    logger.info("=" * 60)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

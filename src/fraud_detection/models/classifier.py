import torch
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from loguru import logger
from fraud_detection.config import XGBOOST_TREE_METHOD


class HybridClassifier:
    """XGBoost classifier using GNN embeddings + transaction features."""

    def __init__(self, params: dict = None):
        self.params = params or {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 3.5,
            "tree_method": XGBOOST_TREE_METHOD,
            "use_label_encoder": False,
            "eval_metric": "auc",
        }
        self.model = None

    def prepare_features(
        self,
        gnn_embeddings: np.ndarray,
        trans_features: np.ndarray
    ) -> np.ndarray:
        """Combine GNN embeddings with transaction features."""
        return np.hstack([gnn_embeddings, trans_features])

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple = None
    ) -> None:
        """Train XGBoost classifier."""
        logger.info(f"Training XGBoost on {X.shape[0]} samples, {X.shape[1]} features")
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=50,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud probability."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud label."""
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        auc = roc_auc_score(y, y_proba)
        report = classification_report(y, y_pred, output_dict=True)
        return {"auc": auc, "report": report}

    def get_feature_importance(self) -> np.ndarray:
        """Get XGBoost feature importance scores."""
        return self.model.feature_importances_

    def save(self, path: str) -> None:
        """Save model to file."""
        self.model.save_model(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from file."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        logger.info(f"Model loaded from {path}")

import shap
import numpy as np
from loguru import logger


class SHAPExplainer:
    """SHAP-based explainer for fraud predictions."""

    def __init__(self, model, feature_names: list = None):
        self.model = model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(128)]
        self.explainer = None

    def fit(self, X_train: np.ndarray) -> None:
        """Fit SHAP explainer on training data."""
        logger.info("Fitting SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model.model)
        self.explainer.expected_value = self.explainer.expected_value[0] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value

    def explain(self, X: np.ndarray) -> dict:
        """Generate SHAP values for a single prediction."""
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return {
            "shap_values": shap_values,
            "base_value": self.explainer.expected_value,
            "feature_names": self.feature_names[:X.shape[1]],
        }

    def get_top_features(self, X: np.ndarray, top_k: int = 10) -> list:
        """Get top-k important features for a prediction."""
        explanation = self.explain(X)
        shap_vals = explanation["shap_values"].flatten()
        top_indices = np.argsort(np.abs(shap_vals))[-top_k:][::-1]
        feature_names = explanation["feature_names"]
        return [
            {"feature": feature_names[i], "importance": float(shap_vals[i])}
            for i in top_indices
        ]

    def generate_summary(self, X: np.ndarray, output_path: str = None) -> None:
        """Generate SHAP summary plot."""
        explanation = self.explain(X)
        shap_values = explanation["shap_values"]
        shap.summary_plot(shap_values, X, feature_names=explanation["feature_names"], show=False)
        if output_path:
            import matplotlib.pyplot as plt
            plt.savefig(output_path)
            logger.info(f"SHAP summary saved to {output_path}")

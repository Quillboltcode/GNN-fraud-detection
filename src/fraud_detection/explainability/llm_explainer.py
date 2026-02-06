import requests
import json
from loguru import logger


class LLMExplainer:
    """Local LLM for generating natural language fraud explanations."""

    def __init__(self, endpoint: str = "http://localhost:11434", model: str = "llama3.2"):
        self.endpoint = endpoint
        self.model = model
        self.prompt_template = """
You are a fraud detection analyst. Explain why a transaction was flagged as fraudulent based on these factors:

Transaction Details:
{transaction_info}

Network Risk Factors:
{network_info}

SHAP Feature Importance:
{shap_features}

Provide a concise (2-3 sentences) explanation of why this transaction is suspicious.
"""

    def is_available(self) -> bool:
        """Check if LLM endpoint is available."""
        try:
            response = requests.get(f"{endpoint}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def explain(self, transaction_info: dict, network_info: dict, shap_features: list) -> str:
        """Generate natural language explanation."""
        prompt = self.prompt_template.format(
            transaction_info=json.dumps(transaction_info, indent=2),
            network_info=json.dumps(network_info, indent=2),
            shap_features=json.dumps(shap_features, indent=2),
        )

        try:
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=30,
            )
            if response.status_code == 200:
                return response.json().get("response", "No explanation generated.")
            else:
                logger.error(f"LLM request failed: {response.status_code}")
                return self._fallback_explanation(shap_features)
        except Exception as e:
            logger.warning(f"LLM unavailable: {e}")
            return self._fallback_explanation(shap_features)

    def _fallback_explanation(self, shap_features: list) -> str:
        """Generate rule-based fallback explanation."""
        top_risks = [f["feature"] for f in sorted(shap_features, key=lambda x: abs(x["importance"]), reverse=True)[:3]]
        return f"Transaction flagged due to anomalous patterns in: {', '.join(top_risks)}."

    def batch_explain(
        self, transactions: list, network_info: list, shap_features: list
    ) -> list:
        """Generate explanations for multiple transactions."""
        return [
            self.explain(t, n, s)
            for t, n, s in zip(transactions, network_info, shap_features)
        ]

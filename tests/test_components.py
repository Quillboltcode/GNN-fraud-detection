import pytest
import numpy as np
from demo.components.fraud_alerts import FraudAlerts


def test_fraud_alerts_render_empty():
    alerts = FraudAlerts.render([])
    assert alerts is not None


def test_fraud_alerts_render_with_data():
    alerts = [
        {"transaction_id": 1, "probability": 0.9},
        {"transaction_id": 2, "probability": 0.3},
    ]
    result = FraudAlerts.render(alerts)
    assert result is not None


def test_fraud_alerts_sorting():
    alerts = [
        {"transaction_id": 1, "probability": 0.3},
        {"transaction_id": 2, "probability": 0.9},
    ]
    alerts_sorted = sorted(alerts, key=lambda x: x["probability"], reverse=True)
    assert alerts_sorted[0]["transaction_id"] == 2

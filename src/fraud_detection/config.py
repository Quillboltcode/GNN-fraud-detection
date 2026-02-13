import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path("/kaggle/input/ieee-fraud-detection")
MODELS_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

IEEE_CIS_URL = "https://www.kaggle.com/competitions/ieee-fraud-detection"

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

XGBOOST_TREE_METHOD = "gpu_hist" if DEVICE == "cuda" else "hist"

GNNS = {
    "graphsage": {
        "hidden_channels": 128,
        "num_layers": 3,
        "dropout": 0.3,
        "lr": 0.001,
    },
    "gat": {
        "hidden_channels": 128,
        "num_heads": 4,
        "num_layers": 3,
        "dropout": 0.3,
        "lr": 0.001,
    },
}

XGBOOST_PARAMS = {
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

LLM_CONFIG = {
    "endpoint": "http://localhost:11434",
    "model": "llama3.2",
    "timeout": 30,
}

SHAP_PARAMS = {
    "nsamples": 100,
    "max_display": 10,
}

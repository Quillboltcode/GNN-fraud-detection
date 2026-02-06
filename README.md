# Fraud Detection GNN + XGBoost Hybrid

Graph Neural Network + XGBoost hybrid fraud detection with interactive visualization.

## Architecture

```
Input: Transaction + Network Context
  ↓
Heterogeneous Graph Construction (Transactions ↔ Users ↔ Devices ↔ IPs)
  ↓
GNN Encoder (GraphSAGE / GAT) → Node Embeddings
  ↓
XGBoost Classifier → Fraud Probability
  ↓
SHAP Explainer + LLM → Natural Language Explanations
```

## Setup

```bash
uv sync
```

## Usage

### Download Data (Kaggle Required)
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
python -m src.fraud_detection.data.download
```

### Train Models
```bash
python -m src.fraud_detection.train
```

### Run Demo
```bash
python -m demo.app
```

## Features

- GraphSAGE & GAT implementations with ablation study
- Hybrid GNN + XGBoost classifier
- SHAP explainability
- Local LLM explanations (Ollama-ready)
- Interactive Dash visualization
- Real-time fraud ring detection

## Kaggle Notebook

For reference implementations and comparison, see Kaggle notebooks:
- IEEE-CIS Fraud Detection competition solutions
- Graph-based fraud detection tutorials

## Notes

- Pre-trained models: Train from scratch on IEEE-CIS dataset
- GPU recommended for training
- Ollama required for LLM explanations (optional)

# Agent Instructions

## Project Overview
Fraud detection using Graph Neural Networks (GNNs) + XGBoost hybrid model with interactive visualization. The system builds heterogeneous graphs from transaction data, extracts embeddings using GraphSAGE/GAT, and classifies fraud with XGBoost. Features SHAP explainability and real-time Dash dashboard.

## Technology Stack
- Python 3.10-3.12
- PyTorch, PyTorch Geometric (HeteroConv, SAGEConv, GATConv)
- XGBoost, scikit-learn
- Dash, Plotly, NetworkX
- SHAP, Loguru
- uv for dependency management

## Key Commands

### Setup & Dependencies
- Install dependencies: `uv sync`
- Add new dependency: `uv add <package>`
- Run in virtual environment: Always prefix commands with `uv run`

### Running Applications
- Run training: `uv run python -m src.fraud_detection.train`
- Run demo dashboard: `uv run python -m demo.app`
- Run baseline XGBoost: `uv run python baseline.py`

### Testing
- Run all tests: `uv run pytest`
- Run specific test file: `uv run pytest tests/test_preprocessing.py`
- Run single test function: `uv run pytest tests/test_preprocessing.py::test_graphbuilder_initialization`
- Run with verbose output: `uv run pytest -v`
- Run with coverage: `uv run pytest --cov=src`

### Linting & Type Checking
- Run linting: `uv run ruff check .`
- Auto-fix linting issues: `uv run ruff check --fix .`
- Format code: `uv run ruff format .`
- Run type checking: `uv run pyright .`

### Git Worktrees
This project uses 3 worktrees for parallel development:
- `../fraud-detection-gnn-gnn` (branch: `feature/gnn-improvements`) - GNN model enhancements
- `../fraud-detection-gnn-dashboard` (branch: `feature/dashboard-enhancements`) - Visualization & dashboard features
- `../fraud-detection-gnn-preprocessing` (branch: `feature/preprocessing-optimizations`) - Data preprocessing improvements
- List all worktrees: `git worktree list`
- Create new worktree: `git worktree add <path> -b <branch-name>`
- Remove worktree: `git worktree remove <path>`

## Code Style Guidelines

### Import Ordering
1. Standard library (os, sys, pathlib)
2. Third-party (torch, pandas, numpy)
3. PyTorch Geometric (torch_geometric)
4. Local modules (from fraud_detection...)
Separate groups with blank lines. Use isort-style grouping.

```python
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

from fraud_detection.config import MODELS_DIR
from fraud_detection.data.preprocessing import GraphBuilder
```

### Type Hints
- Always include type hints for function parameters and return values
- Use `Optional[T]` for nullable parameters
- Use `Dict[str, torch.Tensor]` or `Tuple[T, ...]` for complex types
- Use type aliases for repeated complex types

```python
def train_gnn(
    model: torch.nn.Module,
    data: HeteroData,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = "cpu"
) -> None:
```

### Naming Conventions
- Classes: PascalCase (`GraphBuilder`, `GraphSAGEModel`, `HybridClassifier`)
- Functions/Variables: snake_case (`train_gnn`, `build_graph`, `node_features`)
- Constants: UPPER_SNAKE_CASE (`MODELS_DIR`, `HIDDEN_CHANNELS`)
- Private methods: prefix with underscore (`_encode_column`, `_build_node_features`)
- Module-level: descriptive names, avoid single letters except in comprehensions

### Docstrings
- Use Google-style docstrings for all public functions and classes
- Include Args, Returns, and Raises sections where applicable
- Keep docstrings concise but descriptive

```python
def build_graph(self) -> HeteroData:
    """Construct heterogeneous graph data object.
    
    Returns:
        HeteroData: Graph with transaction, account, device, and email nodes
            connected by heterogeneous edges.
    """
```

### Error Handling
- Use loguru for logging: `logger.error()`, `logger.info()`, `logger.warning()`
- Handle FileNotFoundError for missing data files
- Raise ValueError for invalid input parameters
- Use try-except for I/O operations

```python
if not train_csv.exists():
    logger.error(f"Data not found at {train_csv}")
    logger.info("Run: python -m src.fraud_detection.data.download")
    return
```

### PyTorch & PyTorch Geometric Patterns
- Use HeteroConv for heterogeneous graphs with edge types as tuples: `("transaction", "by", "account")`
- Always move tensors to target device before operations: `x.to(device)`
- Use `torch.no_grad()` for inference: `with torch.no_grad():`
- Save/load model state: `torch.save(model.state_dict(), path)` / `model.load_state_dict(torch.load(path))`
- For GNNs, build convolution layers in `__init__`, apply in `forward` with dict iteration

### Testing Guidelines
- Use pytest for all tests
- Use `tmp_path` fixture for temporary file creation
- Mock external dependencies (network, filesystem)
- Test both success and failure paths
- Use descriptive test names: `test_graphbuilder_build_graph`
- Arrange-Act-Assert pattern in test body
- Keep tests isolated and independent

### Project Structure
- `src/fraud_detection/` - Main package
  - `data/` - Data loading, preprocessing, graph building
  - `models/` - GNN models, classifiers
  - `explainability/` - SHAP, LLM explainers
  - `api/` - API service layer
- `demo/` - Dash dashboard components
- `tests/` - pytest test files
- `data/` - Large CSV files (gitignored)
- `models/` - Saved model files (gitignored)
- `notebooks/` - Jupyter notebooks (gitignored)

### Configuration
- Centralize constants in `src/fraud_detection/config.py`
- Use `pathlib.Path` for all file paths
- Store model paths, data paths, hyperparameters in config

### Memory Management
- Use `del` and `gc.collect()` after large tensor operations
- Convert to appropriate dtypes (float32, int16) to save memory
- Use `torch.no_grad()` context manager for inference

### Graph Construction Notes
- Heterogeneous graphs with 4 node types: transaction, account, device, email
- Bidirectional edges for each relationship (forward + reverse)
- Node features are stacked tensors with encoded categorical columns
- Edge indices stored as 2xN tensors: `torch.stack([src_idx, dst_idx], dim=0)`

import torch
import numpy as np
from pathlib import Path
from loguru import logger
from fraud_detection.config import MODELS_DIR, DATA_DIR, GNNS, XGBOOST_PARAMS
from fraud_detection.data.preprocessing import GraphBuilder
from fraud_detection.models.gnn import GraphSAGEModel, GATModel
from fraud_detection.models.classifier import HybridClassifier
from fraud_detection.explainability.shap_explainer import SHAPExplainer


def train_gnn(
    model: torch.nn.Module,
    data,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = "cpu"
) -> None:
    """Train GNN model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}

        out = model(x_dict, edge_index_dict)
        pred = out["transaction"]
        y = data["transaction"].y.to(device)

        loss = criterion(pred, y.float())
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                prob = torch.sigmoid(pred)
                pred_labels = (prob > 0.5).float()
                acc = (pred_labels == y).float().mean()
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Acc: {acc:.4f}")


def train_classifier(
    classifier: HybridClassifier,
    embeddings: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray
) -> None:
    """Train XGBoost classifier on GNN embeddings."""
    X = classifier.prepare_features(embeddings, features)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    classifier.train(X_train, y_train, eval_set=[(X_val, y_val)])

    train_metrics = classifier.evaluate(X_train, y_train)
    val_metrics = classifier.evaluate(X_val, y_val)

    logger.info(f"Train AUC: {train_metrics['auc']:.4f}")
    logger.info(f"Val AUC: {val_metrics['auc']:.4f}")


def main():
    """Main training pipeline."""
    logger.info("Starting fraud detection training pipeline...")

    data_path = DATA_DIR
    train_csv = data_path / "train_transaction.csv"
    test_csv = data_path / "test_transaction.csv"

    if not train_csv.exists():
        logger.error(f"Data not found at {train_csv}")
        logger.info("Run: python -m src.fraud_detection.data.download")
        return

    logger.info("Building graph...")
    builder = GraphBuilder(str(train_csv), str(test_csv))
    data = builder.build_graph()
    logger.info(f"Graph: {data}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    gs_config = GNNS["graphsage"]
    gat_config = GNNS["gat"]

    logger.info("Training GraphSAGE...")
    graphsage = GraphSAGEModel(
        hidden_channels=gs_config["hidden_channels"],
        num_layers=gs_config["num_layers"],
        dropout=gs_config["dropout"]
    )
    train_gnn(graphsage, data, epochs=100, lr=gs_config["lr"], device=device)

    logger.info("Extracting GraphSAGE embeddings...")
    graphsage.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
        gs_embeddings = graphsage(x_dict, edge_index_dict)["transaction"].cpu().numpy()

    logger.info("Training XGBoost with GraphSAGE embeddings...")
    classifier_gs = HybridClassifier(XGBOOST_PARAMS)
    trans_features = data["transaction"].x.numpy()
    labels = data["transaction"].y.numpy()
    train_classifier(classifier_gs, gs_embeddings, trans_features, labels)
    classifier_gs.save(str(MODELS_DIR / "xgboost_gsage.json"))
    torch.save(graphsage.state_dict(), MODELS_DIR / "graphsage_model.pt")

    logger.info("Training GAT...")
    gat = GATModel(
        hidden_channels=gat_config["hidden_channels"],
        num_heads=gat_config["num_heads"],
        num_layers=gat_config["num_layers"],
        dropout=gat_config["dropout"]
    )
    train_gnn(gat, data, epochs=100, lr=gat_config["lr"], device=device)

    logger.info("Extracting GAT embeddings...")
    gat.eval()
    with torch.no_grad():
        gat_embeddings = gat(x_dict, edge_index_dict)["transaction"].cpu().numpy()

    logger.info("Training XGBoost with GAT embeddings...")
    classifier_gat = HybridClassifier(XGBOOST_PARAMS)
    train_classifier(classifier_gat, gat_embeddings, trans_features, labels)
    classifier_gat.save(str(MODELS_DIR / "xgboost_gat.json"))
    torch.save(gat.state_dict(), MODELS_DIR / "gat_model.pt")

    logger.info("Training complete!")
    logger.info(f"Models saved to: {MODELS_DIR}")

    ablation_results = {
        "graphsage_auc": classifier_gs.evaluate(gs_embeddings, trans_features, labels)["auc"],
        "gat_auc": classifier_gat.evaluate(gat_embeddings, trans_features, labels)["auc"],
    }
    logger.info(f"Ablation Study: {ablation_results}")


if __name__ == "__main__":
    main()

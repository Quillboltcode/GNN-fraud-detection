import torch
import numpy as np
import joblib
from pathlib import Path
from loguru import logger
from typing import Dict, List, Tuple, Optional
from fraud_detection.models.gnn import GraphSAGEModel, GATModel
from fraud_detection.models.classifier import HybridClassifier


class FraudDetectionService:
    """Main service for fraud detection inference."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.graphsage_model = None
        self.gat_model = None
        self.classifier = None
        self.graph_data = None

    def load_models(self) -> None:
        """Load all trained models."""
        logger.info("Loading models...")

        config = {"hidden_channels": 128, "num_layers": 3, "dropout": 0.3}
        self.graphsage_model = GraphSAGEModel(**config).to(self.device)
        self.gat_model = GATModel(hidden_channels=128, num_heads=4, num_layers=3, dropout=0.3).to(self.device)

        gs_path = self.models_dir / "graphsage_model.pt"
        gat_path = self.models_dir / "gat_model.pt"
        xgb_path = self.models_dir / "xgboost_model.json"

        if gs_path.exists():
            self.graphsage_model.load_state_dict(torch.load(gs_path, map_location=self.device))
            self.graphsage_model.eval()
            logger.info("GraphSAGE loaded")
        if gat_path.exists():
            self.gat_model.load_state_dict(torch.load(gat_path, map_location=self.device))
            self.gat_model.eval()
            logger.info("GAT loaded")

        self.classifier = HybridClassifier()
        if xgb_path.exists():
            self.classifier.load(str(xgb_path))
            logger.info("XGBoost loaded")

    def set_graph_data(self, graph_data) -> None:
        """Set the graph data for inference."""
        self.graph_data = graph_data

    def predict(
        self,
        transaction_indices: List[int],
        use_gnn: str = "graphsage"
    ) -> List[Dict]:
        """Predict fraud for given transaction indices."""
        if self.graph_data is None:
            raise ValueError("Graph data not loaded")

        gnn_model = self.graphsage_model if use_gnn == "graphsage" else self.gat_model
        gnn_model.eval()

        with torch.no_grad():
            x_dict = self.graph_data.x_dict
            edge_index_dict = self.graph_data.edge_index_dict
            embeddings = gnn_model(x_dict, edge_index_dict)
            trans_embeddings = embeddings["transaction"][transaction_indices].cpu().numpy()

            trans_features = self.graph_data["transaction"].x[transaction_indices].cpu().numpy()
            X = self.classifier.prepare_features(trans_embeddings, trans_features)

            probabilities = self.classifier.predict_proba(X)
            predictions = self.classifier.predict(X)

        results = []
        for i, idx in enumerate(transaction_indices):
            results.append({
                "transaction_id": int(idx),
                "prediction": int(predictions[i]),
                "probability": float(probabilities[i]),
                "is_fraud": predictions[i] == 1,
            })
        return results

    def get_network_context(self, transaction_idx: int) -> Dict:
        """Get network context for a transaction."""
        if self.graph_data is None:
            return {}

        edge_types = [
            ("transaction", "by", "account"),
            ("transaction", "uses", "device"),
            ("transaction", "from_ip", "ip"),
        ]

        context = {}
        for rel in edge_types:
            if rel in self.graph_data.edge_index_dict:
                edge_index = self.graph_data.edge_index_dict[rel]
                mask = edge_index[0] == transaction_idx
                neighbors = edge_index[1][mask].tolist()
                context[f"{rel[2]}_neighbors"] = neighbors

        return context

    def detect_fraud_ring(self, threshold: float = 0.7) -> List[List[int]]:
        """Detect connected fraud rings."""
        if self.graph_data is None:
            return []

        all_probs = self.predict(list(range(len(self.graph_data["transaction"].x))))
        fraud_indices = [r["transaction_id"] for r in all_probs if r["probability"] > threshold]

        import networkx as nx
        G = nx.Graph()

        for idx in fraud_indices:
            G.add_node(idx)

            for rel in [
                ("transaction", "by", "account"),
                ("transaction", "uses", "device"),
            ]:
                if rel in self.graph_data.edge_index_dict:
                    edge_index = self.graph_data.edge_index_dict[rel]
                    mask = edge_index[0] == idx
                    neighbors = edge_index[1][mask].tolist()
                    for n in neighbors:
                        G.add_edge(idx, n)

        rings = []
        for component in nx.connected_components(G):
            if len(component) >= 2:
                rings.append(list(component))

        return rings

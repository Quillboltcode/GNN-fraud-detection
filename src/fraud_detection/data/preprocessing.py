import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder
from typing import Optional
from loguru import logger


class GraphBuilder:
    """Build heterogeneous graph from IEEE-CIS transaction data."""

    def __init__(self, train_path: str, test_path: str, identity_path: Optional[str] = None):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)
        if identity_path:
            identity = pd.read_csv(identity_path)
            self.train = self.train.merge(identity, on="TransactionID", how="left")
            self.test = self.test.merge(identity, on="TransactionID", how="left")
        self.combined = pd.concat([self.train, self.test], ignore_index=True)
        self.label_encoders = {}
        self.node_types = ["transaction", "account", "device", "email"]

    def _encode_column(self, col: str) -> np.ndarray:
        """Label encode a categorical column."""
        if col not in self.label_encoders:
            self.label_encoders[col] = LabelEncoder()
            self.combined[col] = self.combined[col].fillna("unknown")
            self.label_encoders[col].fit(self.combined[col])
        encoded = self.label_encoders[col].transform(self.combined[col].fillna("unknown"))
        return encoded

    def _build_node_features(self) -> dict:
        """Extract node features for each node type."""
        n_transactions = len(self.combined)

        productcd_encoded = self._encode_column("ProductCD")
        card4_encoded = self._encode_column("card4")
        card6_encoded = self._encode_column("card6")

        numeric_features = ["TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2"]
        trans_feature_data = self.combined[numeric_features].values.astype(np.float32)
        trans_feature_data = np.column_stack([
            trans_feature_data,
            productcd_encoded.astype(np.float32),
            card4_encoded.astype(np.float32),
            card6_encoded.astype(np.float32),
        ])

        device_features = self._encode_column("DeviceInfo")
        email_features = self._encode_column("P_emaildomain")

        return {
            "transaction": torch.tensor(trans_feature_data),
            "device": torch.tensor(device_features, dtype=torch.long).unsqueeze(1),
            "email": torch.tensor(email_features, dtype=torch.long).unsqueeze(1),
        }

    def _build_edge_indices(self) -> dict:
        """Build edge indices for heterogeneous relationships."""
        n = len(self.combined)

        edges = {
            ("transaction", "by", "account"): (
                torch.arange(n),
                torch.tensor(self._encode_column("card1"))
            ),
            ("transaction", "uses", "device"): (
                torch.arange(n),
                torch.tensor(self._encode_column("DeviceInfo"))
            ),
            ("transaction", "with_email", "email"): (
                torch.arange(n),
                torch.tensor(self._encode_column("P_emaildomain"))
            ),
        }

        return edges

    def build_graph(self) -> HeteroData:
        """Construct heterogeneous graph data object."""
        logger.info("Building heterogeneous graph...")

        data = HeteroData()

        node_features = self._build_node_features()
        for node_type, features in node_features.items():
            data[node_type].x = features

        edge_indices = self._build_edge_indices()
        for (src, rel, dst), (src_idx, dst_idx) in edge_indices.items():
            data[src, rel, dst].edge_index = torch.stack([src_idx, dst_idx], dim=0)

        if "isFraud" in self.combined.columns:
            fraud_labels = self.combined["isFraud"].values
            data["transaction"].y = torch.tensor(fraud_labels, dtype=torch.long)

        logger.info(f"Graph built: {data}")
        return data

    def get_node_counts(self) -> dict:
        """Return node counts per type."""
        return {
            "transaction": len(self.combined),
            "device": len(self.label_encoders.get("DeviceInfo", [])),
            "email": len(self.label_encoders.get("P_emaildomain", [])),
        }

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv
from typing import Dict, Tuple
from loguru import logger


class GraphSAGEModel(torch.nn.Module):
    """GraphSAGE for heterogeneous graph node embeddings."""

    def __init__(self, hidden_channels: int, num_layers: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        self.convs.append(
            HeteroConv({
                ("transaction", "by", "account"): SAGEConv((-1, -1), hidden_channels),
                ("transaction", "uses", "device"): SAGEConv((-1, -1), hidden_channels),
                ("transaction", "from_ip", "ip"): SAGEConv((-1, -1), hidden_channels),
                ("transaction", "with_email", "email"): SAGEConv((-1, -1), hidden_channels),
                ("account", "rev_by", "transaction"): SAGEConv((-1, -1), hidden_channels),
                ("device", "rev_uses", "transaction"): SAGEConv((-1, -1), hidden_channels),
                ("ip", "rev_from_ip", "transaction"): SAGEConv((-1, -1), hidden_channels),
                ("email", "rev_with_email", "transaction"): SAGEConv((-1, -1), hidden_channels),
            })
        )

        for _ in range(num_layers - 1):
            self.convs.append(
                HeteroConv({
                    rel: SAGEConv(hidden_channels, hidden_channels)
                    for rel in [
                        ("transaction", "by", "account"), ("account", "rev_by", "transaction"),
                        ("transaction", "uses", "device"), ("device", "rev_uses", "transaction"),
                        ("transaction", "from_ip", "ip"), ("ip", "rev_from_ip", "transaction"),
                        ("transaction", "with_email", "email"), ("email", "rev_with_email", "transaction"),
                    ]
                })
            )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Generate node embeddings."""
        for conv in self.convs:
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        return x_dict

    def get_transaction_embeddings(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract transaction node embeddings for classification."""
        return x_dict["transaction"]


class GATModel(torch.nn.Module):
    """Graph Attention Network for heterogeneous graph node embeddings."""

    def __init__(self, hidden_channels: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.attentions = torch.nn.ModuleList()

        self.convs.append(
            HeteroConv({
                rel: GATConv((-1, -1), hidden_channels, heads=num_heads, dropout=dropout)
                for rel in [
                    ("transaction", "by", "account"), ("account", "rev_by", "transaction"),
                    ("transaction", "uses", "device"), ("device", "rev_uses", "transaction"),
                    ("transaction", "from_ip", "ip"), ("ip", "rev_from_ip", "transaction"),
                    ("transaction", "with_email", "email"), ("email", "rev_with_email", "transaction"),
                ]
            })
        )

        for _ in range(num_layers - 1):
            self.convs.append(
                HeteroConv({
                    rel: GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout)
                    for rel in [
                        ("transaction", "by", "account"), ("account", "rev_by", "transaction"),
                        ("transaction", "uses", "device"), ("device", "rev_uses", "transaction"),
                        ("transaction", "from_ip", "ip"), ("ip", "rev_from_ip", "transaction"),
                        ("transaction", "with_email", "email"), ("email", "rev_with_email", "transaction"),
                    ]
                })
            )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Generate node embeddings with attention."""
        for conv in self.convs:
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        return x_dict

    def get_transaction_embeddings(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract transaction node embeddings for classification."""
        return x_dict["transaction"]

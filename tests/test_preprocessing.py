import pytest
import torch
import pandas as pd
from pathlib import Path
from fraud_detection.data.preprocessing import GraphBuilder


@pytest.fixture
def test_data_paths(tmp_path):
    train_trans = tmp_path / "train_transaction.csv"
    train_ident = tmp_path / "train_identity.csv"
    
    train_trans.write_text(
        "TransactionID,isFraud,TransactionAmt,ProductCD,card1,card2,card3,card4,card5,card6,addr1,addr2,P_emaildomain\n"
        "2987000,0,68.5,W,13926,,150.0,discover,142.0,credit,315.0,87.0,\n"
        "2987001,0,29.0,W,2755,404.0,150.0,mastercard,102.0,credit,325.0,87.0,gmail.com\n"
        "2987002,1,150.0,C,9999,200.0,125.0,visa,100.0,debit,200.0,50.0,yahoo.com\n"
        "2987003,0,45.0,H,5555,,100.0,discover,120.0,credit,,60.0,\n"
        "2987004,1,300.0,W,1234,315.0,142.0,mastercard,142.0,credit,315.0,87.0,hotmail.com\n"
    )
    
    train_ident.write_text(
        "TransactionID,DeviceInfo,DeviceType\n"
        "2987000,SAMSUNG SM-G892A,mobile\n"
        "2987001,iOS Device,mobile\n"
        "2987002,Windows PC,desktop\n"
        "2987003,Android 8.0,mobile\n"
        "2987004,iOS Device,mobile\n"
    )
    
    return str(train_trans), str(train_ident)


def test_graphbuilder_initialization(test_data_paths):
    train_trans, train_ident = test_data_paths
    builder = GraphBuilder(train_trans, train_trans, train_ident)
    assert len(builder.combined) == 10


def test_graphbuilder_build_graph(test_data_paths):
    train_trans, train_ident = test_data_paths
    builder = GraphBuilder(train_trans, train_trans, train_ident)
    graph = builder.build_graph()
    
    assert graph is not None
    assert "transaction" in graph.node_types
    assert "device" in graph.node_types
    assert "email" in graph.node_types
    assert graph["transaction"].x.shape[0] == 10


def test_graphbuilder_node_features(test_data_paths):
    train_trans, train_ident = test_data_paths
    builder = GraphBuilder(train_trans, train_trans, train_ident)
    features = builder._build_node_features()
    
    assert "transaction" in features
    assert "device" in features
    assert "email" in features
    assert features["transaction"].shape[0] == 10


def test_graphbuilder_edge_indices(test_data_paths):
    train_trans, train_ident = test_data_paths
    builder = GraphBuilder(train_trans, train_trans, train_ident)
    edges = builder._build_edge_indices()
    
    assert ("transaction", "by", "account") in edges
    assert ("transaction", "uses", "device") in edges
    assert ("transaction", "with_email", "email") in edges


def test_graphbuilder_node_counts(test_data_paths):
    train_trans, train_ident = test_data_paths
    builder = GraphBuilder(train_trans, train_trans, train_ident)
    counts = builder.get_node_counts()
    
    assert "transaction" in counts
    assert counts["transaction"] == 10
    assert "device" in counts
    assert "email" in counts


def test_graphbuilder_has_labels(test_data_paths):
    train_trans, train_ident = test_data_paths
    builder = GraphBuilder(train_trans, train_trans, train_ident)
    graph = builder.build_graph()
    
    assert graph["transaction"].y is not None
    assert graph["transaction"].y.shape[0] == 10

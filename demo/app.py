import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import time
import networkx as nx
from loguru import logger
from pathlib import Path

from demo.components.fraud_alerts import FraudAlerts
from demo.components.explanations import ExplanationsPanel

logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")

app = dash.Dash(__name__)

fake_predictions = [
    {"transaction_id": i, "probability": np.random.beta(8, 2) if i % 7 == 0 else np.random.beta(2, 8)}
    for i in range(100)
]

stats = {
    "total_transactions": 590000,
    "fraud_detected": 1247,
    "fraud_rings": 23,
    "avg_risk_score": 0.12,
}

app.layout = html.Div([
    dcc.Store(id="graph-data-store", data={"loaded": False}),
    dcc.Store(id="predictions-store", data=fake_predictions),
    dcc.Interval(id="refresh-interval", interval=5000, n_intervals=0),

    html.Div([
        html.H1("Fraud Detection Dashboard"),
        html.P("GNN + XGBoost Hybrid Model with Real-time Network Visualization"),
    ], className="app-header"),

    html.Div([
        html.Div([
            html.Div([
                html.Div(stats["total_transactions"], className="stat-value"),
                html.Div("Total Txns", className="stat-label"),
            ], className="stat-item"),
            html.Div([
                html.Div(stats["fraud_detected"], className="stat-value", style={"color": "#ff4444"}),
                html.Div("Fraud Cases", className="stat-label"),
            ], className="stat-item"),
            html.Div([
                html.Div(stats["fraud_rings"], className="stat-value", style={"color": "#ffaa00"}),
                html.Div("Fraud Rings", className="stat-label"),
            ], className="stat-item"),
            html.Div([
                html.Div(f"{stats['avg_risk_score']:.1%}", className="stat-value"),
                html.Div("Avg Risk", className="stat-label"),
            ], className="stat-item"),
        ], className="stats-bar"),

        html.Div([
            html.Div([
                html.Label("GNN Model:", style={"color": "#a0a0b0"}),
                dcc.Dropdown(
                    id="gnn-model-select",
                    options=[
                        {"label": "GraphSAGE", "value": "graphsage"},
                        {"label": "GAT", "value": "gat"},
                    ],
                    value="graphsage",
                    clearable=False,
                    style={"width": "150px"},
                ),
            ], className="control-group"),
            html.Div([
                html.Label("Risk Threshold:", style={"color": "#a0a0b0"}),
                dcc.Slider(
                    id="threshold-slider",
                    min=0.3, max=0.95, step=0.05, value=0.7,
                    marks={0.3: "30%", 0.5: "50%", 0.7: "70%", 0.9: "90%"},
                ),
            ], className="control-group", style={"flex": 1, "maxWidth": "400px"}),
            html.Div([
                html.Button("Refresh", id="refresh-btn", n_clicks=0, className="refresh-btn"),
            ]),
        ], className="controls-row"),
    ], className="controls-section"),

    html.Div([
        html.Div([
            html.H3("Transaction Network", className="card-title"),
            dcc.Graph(
                id="network-graph",
                style={"height": "500px"},
                config={"displayModeBar": True},
            ),
            html.Div(id="selected-node-info", className="node-info"),
        ], className="card", style={"grid-column": "1 / 2"}),

        html.Div([
            FraudAlerts.render(),
            ExplanationsPanel.render(),
        ], className="sidebar", style={"display": "flex", "flexDirection": "column", "gap": "24px"}),
    ], className="main-grid"),

    html.Div([
        html.Div([
            html.H4("Model Performance", className="card-title"),
            dcc.Graph(id="roc-curve", style={"height": "250px"}),
        ], className="card"),
        html.Div([
            html.H4("Fraud Distribution Over Time", className="card-title"),
            dcc.Graph(id="fraud-timeline", style={"height": "250px"}),
        ], className="card"),
    ], style={"display": "grid", "grid-template-columns": "1fr 1fr", "gap": "24px", "marginTop": "24px"}),

], className="app-container")


@app.callback(
    Output("network-graph", "figure"),
    Input("refresh-interval", "n_intervals"),
    Input("gnn-model-select", "value"),
    Input("threshold-slider", "value"),
)
def update_network_graph(n, model, threshold):
    nodes = {}
    fraud_ring = []
    edges = []

    for i in range(min(50, len(fake_predictions))):
        prob = fake_predictions[i]["probability"]
        is_fraud = prob > threshold
        node_type = "fraud" if is_fraud else "normal"
        nodes[i] = {
            "label": f"Tx {i}",
            "type": node_type,
            "probability": prob,
            "account_neighbors": list(np.random.choice(range(1000, 1050), size=np.random.randint(1, 4), replace=False)),
            "device_neighbors": list(np.random.choice(range(2000, 2020), size=np.random.randint(1, 2), replace=False)),
        }
        if is_fraud:
            fraud_ring.append(i)
            for acc in nodes[i]["account_neighbors"][:2]:
                edges.append((i, acc))
            for dev in nodes[i]["device_neighbors"][:1]:
                edges.append((i, dev))

    G = nx.Graph()

    colors = {"normal": "#4488ff", "fraud": "#ff4444", "account": "#44ff88", "device": "#ffaa00"}
    sizes = {"normal": 10, "fraud": 25, "account": 15, "device": 12}

    for node_id, attrs in nodes.items():
        G.add_node(node_id, **attrs)

    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)

    edge_x, edge_y = [], []
    for edge in edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color="#555"),
        hoverinfo="none",
        mode="lines"
    )

    node_x, node_y, node_colors, node_sizes, node_text = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        attrs = nodes.get(node, {})
        node_type = attrs.get("type", "normal")
        prob = attrs.get("probability", 0)
        node_colors.append(colors.get(node_type, "#888"))
        node_sizes.append(sizes.get(node_type, 10))
        node_text.append(f"Tx {node}<br>Risk: {prob:.1%}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=2, color="white"),
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Real-time Fraud Ring Detection (Model: {model.upper()})",
            titlefont_size=14,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            plot_bgcolor="#1a1a2e",
            paper_bgcolor="#1a1a2e",
            font=dict(color="white"),
        )
    )

    return fig


@app.callback(
    Output("fraud-alerts-container", "children"),
    Input("refresh-interval", "n_intervals"),
    Input("threshold-slider", "value"),
)
def update_alerts(n, threshold):
    alerts = [p for p in fake_predictions if p["probability"] > threshold]
    alerts.sort(key=lambda x: x["probability"], reverse=True)
    return FraudAlerts.render(alerts)


@app.callback(
    Output("explanations-container", "children"),
    Input("refresh-interval", "n_intervals"),
)
def update_explanations(n):
    fake_shap = [
        {"feature": "TransactionAmt", "importance": 0.45},
        {"feature": "DeviceInfo_entropy", "importance": 0.32},
        {"feature": "IP_cluster_size", "importance": 0.28},
        {"feature": "card1_velocity", "importance": 0.21},
        {"feature": "email_domain_risk", "importance": 0.18},
    ]
    return ExplanationsPanel.render({
        "shap_features": fake_shap,
        "llm_explanation": "This transaction shows suspicious patterns: unusually high amount compared to user's historical behavior, device fingerprint mismatch, and IP address belonging to a known fraud cluster. The connected account has been flagged in 3 previous fraud cases.",
    })


@app.callback(
    Output("roc-curve", "figure"),
    Input("refresh-interval", "n_intervals"),
)
def update_roc(n):
    fpr = np.linspace(0, 1, 100)
    tpr = 1 - np.exp(-5 * fpr)
    auc = 0.924

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"ROC (AUC = {auc:.3f})",
        line=dict(color="#4488ff", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(color="#666", width=1, dash="dash"),
    ))

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor="#252540",
        paper_bgcolor="#252540",
        font=dict(color="white"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=250,
        showlegend=False,
    )
    return fig


@app.callback(
    Output("fraud-timeline", "figure"),
    Input("refresh-interval", "n_intervals"),
)
def update_timeline(n):
    hours = list(range(24))
    fraud_counts = np.random.poisson(5, 24)
    fraud_counts = np.abs(fraud_counts + np.random.normal(0, 2, 24)).astype(int)

    fig = go.Figure(go.Bar(
        x=hours,
        y=fraud_counts,
        marker_color=px.colors.sequential.Blues[3:],
    ))

    fig.update_layout(
        title="Fraud Cases by Hour",
        xaxis_title="Hour",
        yaxis_title="Fraud Cases",
        plot_bgcolor="#252540",
        paper_bgcolor="#252540",
        font=dict(color="white"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=250,
    )
    return fig


if __name__ == "__main__":
    logger.info("Starting Fraud Detection Dashboard...")
    app.run(debug=True, host="0.0.0.0", port=8050)

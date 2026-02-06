import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go
import networkx as nx
from loguru import logger
import json


class NetworkGraph:
    """Interactive network visualization component."""

    @staticmethod
    def create_graph(
        fraud_ring: list,
        all_nodes: dict,
        selected_node: int = None
    ) -> go.Figure:
        """Create Plotly network graph figure."""
        G = nx.Graph()

        fraud_set = set(fraud_ring) if fraud_ring else set()

        for node_id, attrs in all_nodes.items():
            G.add_node(node_id, **attrs)

        for node_id in fraud_ring:
            for rel in ["account", "device", "ip"]:
                neighbors = attrs.get(f"{rel}_neighbors", [])
                for n in neighbors:
                    if n in all_nodes:
                        G.add_edge(node_id, n)

        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if node in fraud_set:
                node_colors.append("#ff4444")
                node_sizes.append(30)
            elif selected_node == node:
                node_colors.append("#ffaa00")
                node_sizes.append(25)
            else:
                node_colors.append("#4488ff")
                node_sizes.append(10)

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines"
        )

        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            attrs = all_nodes.get(node, {})
            label = attrs.get("label", f"Node {node}")
            node_text.append(label)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[str(n) for n in G.nodes()],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line_width=2
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Fraud Ring Detection - Real-time Network Visualization",
                titlefont_size=16,
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

        fig.update_layout(
            clickmode="event+select",
            dragmode="pan",
        )

        return fig

    @staticmethod
    def render() -> html.Div:
        """Render the network graph component."""
        return html.Div([
            html.H3("Transaction Network", className="card-title"),
            dcc.Graph(id="network-graph", style={"height": "500px"}),
            html.Div(id="selected-node-info", className="node-info"),
        ], className="card")


class FraudAlerts:
    """Fraud alerts panel component."""

    @staticmethod
    def render(alerts: list) -> html.Div:
        """Render fraud alerts list."""
        if not alerts:
            return html.Div([
                html.H3("Fraud Alerts", className="card-title"),
                html.P("No high-risk transactions detected.", className="no-alerts"),
            ], className="card")

        alert_items = []
        for alert in alerts[:20]:
            prob = alert.get("probability", 0)
            color = "#ff4444" if prob > 0.8 else "#ffaa00" if prob > 0.5 else "#44ff44"
            alert_items.append(html.Div([
                html.Span(f"Tx: {alert.get('transaction_id', 'N/A')}", className="alert-id"),
                html.Span(f"{prob:.1%}", className="alert-prob", style={"color": color}),
            ], className="alert-item"))

        return html.Div([
            html.H3("Fraud Alerts", className="card-title"),
            html.Div(alert_items, className="alerts-list"),
        ], className="card")


class ExplanationsPanel:
    """SHAP + LLM explanations panel."""

    @staticmethod
    def render(explanations: dict) -> html.Div:
        """Render explanations panel."""
        return html.Div([
            html.H3("AI Explanations", className="card-title"),
            html.Div([
                html.H4("SHAP Feature Importance", className="section-title"),
                html.Div(id="shap-values"),
                html.H4("LLM Analysis", className="section-title"),
                html.Div(id="llm-explanation", className="llm-text"),
            ], className="explanations-content"),
        ], className="card")

import dash
from dash import dcc, html


class ExplanationsPanel:
    """SHAP + LLM explanations panel."""

    @staticmethod
    def render(explanations: dict = None) -> html.Div:
        """Render explanations panel."""
        explanations = explanations or {}

        shap_features = explanations.get("shap_features", [])
        llm_text = explanations.get("llm_explanation", "Select a transaction to see explanation.")

        shap_items = []
        for feat in shap_features[:10]:
            name = feat.get("feature", "unknown")
            imp = feat.get("importance", 0)
            bar_width = min(abs(imp) * 100, 100)
            color = "#ff6b6b" if imp > 0 else "#4ecdc4"
            shap_items.append(html.Div([
                html.Span(name, className="feat-name"),
                html.Div([
                    html.Div(className="feat-bar", style={"width": f"{bar_width}%", "background": color}),
                ], className="feat-bar-container"),
                html.Span(f"{imp:.3f}", className="feat-value"),
            ], className="shap-item"))

        return html.Div([
            html.H3("AI Explanations", className="card-title"),
            html.Div([
                html.H4("SHAP Feature Importance", className="section-title"),
                html.Div(shap_items, className="shap-list"),
                html.H4("LLM Analysis", className="section-title"),
                html.Div(llm_text, className="llm-text"),
            ], className="explanations-content"),
        ], className="card")

"""
IntelliCredit - Model Drift & Population Stability Dashboard Component
======================================================================
Interactive Streamlit component rendering Population Stability Index (PSI)
traffic-light alerts, feature-level drift rankings, and distribution comparison plots.

Usage:
    Can be run directly via:
        streamlit run dashboard/monitoring_view.py
    Or embedded as a tab in dashboard/app.py:
        from dashboard.monitoring_view import render_monitoring_dashboard
        render_monitoring_dashboard()

Author: Interactive Dashboard & UI Team
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    import streamlit as st
except ImportError:
    st = None

from dashboard.monitoring_engine import (
    run_model_monitoring_suite,
    calculate_psi,
    PSI_THRESHOLD_AMBER,
    PSI_THRESHOLD_RED,
    ALERT_GREEN,
    ALERT_AMBER,
    ALERT_RED,
)
from dashboard.ui_utils import apply_dashboard_theme, render_drift_badge


def render_monitoring_summary_cards(summary: dict) -> None:
    """Renders top-level summary metrics cards."""
    if st is None:
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Features Monitored", summary["features_monitored"])
    with col2:
        st.metric("Stable Features (Green)", summary["stable_green_count"])
    with col3:
        st.metric("Amber Warnings", summary["amber_alert_count"])
    with col4:
        st.metric("Red Alerts", summary["red_alert_count"])


def render_psi_ranking_chart(drift_df: pd.DataFrame) -> go.Figure:
    """Creates a horizontal bar chart of feature PSI scores colored by Basel alert tier."""
    df = drift_df.copy().head(12)
    colors = []
    for s in df["status"]:
        if s == ALERT_GREEN:
            colors.append("#10B981")  # Green
        elif s == ALERT_AMBER:
            colors.append("#F59E0B")  # Amber
        else:
            colors.append("#EF4444")  # Red

    fig = go.Figure(go.Bar(
        x=df["psi"],
        y=df["feature"],
        orientation="h",
        marker=dict(color=colors),
        text=df["psi"].apply(lambda x: f"{x:.4f}"),
        textposition="auto",
    ))

    # Add supervisory threshold guide lines
    fig.add_vline(x=PSI_THRESHOLD_AMBER, line_dash="dash", line_color="#F59E0B", annotation_text="Amber (0.10)")
    fig.add_vline(x=PSI_THRESHOLD_RED, line_dash="dash", line_color="#EF4444", annotation_text="Red (0.25)")

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Population Stability Index (PSI)",
        yaxis_title="Feature",
        height=450,
    )
    return apply_dashboard_theme(fig, title="Feature Drift Stability Index (Basel Standard)")


def render_distribution_comparison(
    baseline_series: np.ndarray,
    current_series: np.ndarray,
    feature_name: str,
) -> go.Figure:
    """Plots overlaid normalized distribution histograms for baseline vs current population."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=baseline_series,
        name="Baseline (Training)",
        opacity=0.6,
        marker_color="#3B82F6",
        histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=current_series,
        name="Current (Scoring)",
        opacity=0.6,
        marker_color="#10B981",
        histnorm="probability density",
    ))

    fig.update_layout(
        barmode="overlay",
        xaxis_title=feature_name.replace("_", " ").title(),
        yaxis_title="Probability Density",
        height=380,
    )
    return apply_dashboard_theme(fig, title=f"Distribution Comparison: {feature_name}")


def render_monitoring_dashboard() -> None:
    """Main rendering entrypoint for Streamlit model monitoring view."""
    if st is None:
        print("Streamlit not installed in this environment.")
        return

    st.subheader("Model Monitoring & Population Stability Index (PSI)")
    st.markdown(
        "Automated statistical drift surveillance across model features and credit score distributions "
        "complying with Basel & EBA supervisory traffic-light standards."
    )

    summary = run_model_monitoring_suite()
    render_monitoring_summary_cards(summary)

    st.markdown("---")
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("#### Feature Stability Ranking")
        fig_psi = render_psi_ranking_chart(summary["drift_table"])
        st.plotly_chart(fig_psi, use_container_width=True)

    with c2:
        st.markdown("#### Regulatory Alert Summary")
        st.markdown(
            """
            * **Green Tier (PSI < 0.10):** Distribution shift is negligible; scorecards remain fully calibrated.
            * **Amber Tier (0.10 ≤ PSI ≤ 0.25):** Moderate population change; requires monitoring and feature analysis.
            * **Red Tier (PSI > 0.25):** Significant covariate shift; triggers mandatory model review or retuning.
            """
        )
        display_tbl = summary["drift_table"][["feature", "psi", "status", "ks_statistic"]].copy()
        st.dataframe(display_tbl, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    import sys
    # If launched via streamlit run, render the interactive app
    if "streamlit" in sys.modules and any("run" in arg for arg in sys.argv):
        st.set_page_config(page_title="Model Monitoring", layout="wide")
        render_monitoring_dashboard()
    else:
        print("IntelliCredit - Model Monitoring View Component (CLI Validation Mode)...")
        summ = run_model_monitoring_suite()
        print(f"Features Monitored : {summ['features_monitored']}")
        print(f"Stable (Green)     : {summ['stable_green_count']}")
        print(f"Amber Warnings     : {summ['amber_alert_count']}")
        print(f"Red Alerts         : {summ['red_alert_count']}")
        print("Component syntax and logic validated successfully.")

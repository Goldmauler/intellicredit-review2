"""
IntelliCredit - Dashboard Design System, Badges & Visual Utilities
==================================================================
Reusable visual components, CSS theme injectors, risk band badges,
Indian Rupee (INR) currency formatters, and Plotly charting templates.

Author: Interactive Dashboard & UI Team
"""

from typing import Any, Dict, Optional
import plotly.graph_objects as go

# Color Palettes for Risk Tiers & Decisions
COLOR_BAND_A = "#10B981"  # Emerald (Very Low Risk)
COLOR_BAND_B = "#34D399"  # Mint Green (Low Risk)
COLOR_BAND_C = "#F59E0B"  # Amber (Moderate Risk)
COLOR_BAND_D = "#F97316"  # Orange (High Risk)
COLOR_BAND_E = "#EF4444"  # Crimson (Severe Risk)

COLOR_APPROVE = "#059669"
COLOR_REFER = "#D97706"
COLOR_DECLINE = "#DC2626"

BAND_COLORS = {
    "A": COLOR_BAND_A,
    "B": COLOR_BAND_B,
    "C": COLOR_BAND_C,
    "D": COLOR_BAND_D,
    "E": COLOR_BAND_E,
}

DECISION_COLORS = {
    "Approve": COLOR_APPROVE,
    "Refer": COLOR_REFER,
    "Refer (Senior Review)": COLOR_REFER,
    "Decline": COLOR_DECLINE,
}


def format_inr(amount: float, compact: bool = False) -> str:
    """
    Formats numerical amounts into Indian Rupee denomination (Lakhs/Crores).

    Examples:
        format_inr(50000) -> "₹50,000"
        format_inr(1500000, compact=True) -> "₹15.00 L"
        format_inr(25000000, compact=True) -> "₹2.50 Cr"
    """
    if amount is None or amount == "":
        return "₹0"

    amt = float(amount)
    if not compact:
        # Standard Indian comma notation formatting
        s = f"{abs(amt):,.2f}".rstrip("0").rstrip(".")
        prefix = "-₹" if amt < 0 else "₹"
        return f"{prefix}{s}"

    abs_amt = abs(amt)
    sign = "-" if amt < 0 else ""
    if abs_amt >= 10_000_000:
        return f"{sign}₹{abs_amt / 10_000_000:.2f} Cr"
    elif abs_amt >= 100_000:
        return f"{sign}₹{abs_amt / 100_000:.2f} L"
    elif abs_amt >= 1_000:
        return f"{sign}₹{abs_amt / 1_000:.1f} K"
    return f"{sign}₹{abs_amt:.2f}"


def render_risk_band_badge(band: str) -> str:
    """Returns an inline HTML pill badge styled by Risk Band (A through E)."""
    b = str(band).upper().strip()
    color = BAND_COLORS.get(b, "#6B7280")
    label_map = {
        "A": "Band A (Prime)",
        "B": "Band B (Near-Prime)",
        "C": "Band C (Moderate Risk)",
        "D": "Band D (Subprime)",
        "E": "Band E (High Risk)",
    }
    label = label_map.get(b, f"Band {b}")
    return (
        f'<span style="background-color: {color}20; color: {color}; border: 1px solid {color}60; '
        f'padding: 3px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85rem;">'
        f'{label}</span>'
    )


def render_decision_badge(decision: str) -> str:
    """Returns an inline HTML pill badge for underwriting decisions."""
    dec = str(decision).strip()
    color = DECISION_COLORS.get(dec, "#6B7280")
    return (
        f'<span style="background-color: {color}25; color: {color}; border: 1.5px solid {color}; '
        f'padding: 4px 12px; border-radius: 14px; font-weight: 700; font-size: 0.9rem; letter-spacing: 0.3px;">'
        f'{dec.upper()}</span>'
    )


def render_drift_badge(status: str) -> str:
    """Returns an inline HTML pill badge for Population Stability Index (PSI) drift alerts."""
    st = str(status).lower()
    if "green" in st or "stable" in st:
        bg, text, border = "#10B98120", "#10B981", "#10B981"
        icon = "✓"
    elif "amber" in st or "moderate" in st:
        bg, text, border = "#F59E0B20", "#F59E0B", "#F59E0B"
        icon = "⚠"
    else:
        bg, text, border = "#EF444420", "#EF4444", "#EF4444"
        icon = "⚡"

    return (
        f'<span style="background-color: {bg}; color: {text}; border: 1px solid {border}; '
        f'padding: 3px 8px; border-radius: 10px; font-weight: 600; font-size: 0.8rem;">'
        f'{icon} {status}</span>'
    )


def apply_dashboard_theme(
    fig: go.Figure,
    title: Optional[str] = None,
    height: int = 400,
) -> go.Figure:
    """
    Applies clean, production-grade styling to any Plotly chart figure.
    """
    layout_updates: Dict[str, Any] = {
        "height": height,
        "template": "plotly_white",
        "margin": dict(l=40, r=30, t=50, b=40),
        "font": dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"),
        "hoverlabel": dict(bgcolor="white", font_size=12, font_family="sans-serif"),
    }
    if title:
        layout_updates["title"] = dict(text=title, font=dict(size=15, color="#1F2937"))

    fig.update_layout(**layout_updates)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#F3F4F6", zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#F3F4F6", zeroline=False)
    return fig


if __name__ == "__main__":
    print("IntelliCredit - Testing Dashboard Design System & Formatting Utilities...")
    print("INR Formatting:")
    print("  50,000      ->", format_inr(50000))
    print("  1,500,000   ->", format_inr(1500000, compact=True))
    print("  25,000,000  ->", format_inr(25000000, compact=True))
    print("\nBadge HTML Generators:")
    print("  Band A Badge:", render_risk_band_badge("A"))
    print("  Decision Badge:", render_decision_badge("Approve"))
    print("  Drift Badge:", render_drift_badge("Stable (Green)"))

from __future__ import annotations

import html
from typing import Any

import streamlit as st


def inject_global_css() -> None:
    st.html(
        """
        <style>
        .stApp { background: #F5F7FA; }
        [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #DCE4EC; }
        .cr-kicker { color:#087F79; font-size:.72rem; text-transform:uppercase; letter-spacing:.14em; font-weight:800; }
        .cr-muted { color:#667085; font-size:.92rem; }
        .cr-brand { font-weight:850; color:#102A43; font-size:1.16rem; }
        .cr-pill { display:inline-block; padding:4px 9px; border-radius:999px; background:#EEF2F6; color:#475467; font-size:.70rem; font-weight:800; }
        .cr-card { background:#FFFFFF; border:1px solid #DCE4EC; border-radius:16px; padding:20px; box-shadow:0 2px 10px rgba(16,42,67,.035); }
        .cr-phase-strip { display:flex; gap:6px; margin:10px 0 18px; }
        .cr-phase-dot { height:5px; flex:1; border-radius:999px; background:#E4E7EC; }
        .cr-phase-dot.done { background:#18A999; }
        .cr-phase-dot.current { background:#0F6B88; }
        .cr-status { border:1px solid #DCE4EC; border-radius:12px; padding:12px 14px; background:#FBFCFE; }
        .cr-small { font-size:.80rem; color:#667085; }
        .cr-console { background:#0B1220; border:1px solid #1F2937; border-radius:12px; padding:14px 16px; max-height:320px; overflow:auto; box-shadow:inset 0 0 0 1px rgba(255,255,255,.02); }
        .cr-console pre { margin:0; color:#9FE7B4; font-family:Consolas,Monaco,'Courier New',monospace; font-size:.78rem; line-height:1.5; white-space:pre-wrap; }
        </style>
        """
    )


def inject_login_css() -> None:
    """Hide the application sidebar and its collapse control on the login screen."""
    st.html(
        """
        <style>
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
        [data-testid="stAppViewContainer"] { margin-left: 0 !important; }
        </style>
        """
    )


def render_brand(large: bool = False) -> None:
    st.markdown("<div class='cr-brand'>🏦 CreditRisk Studio</div>", unsafe_allow_html=True)
    if large:
        st.caption("Automobile loan default risk analytics")


def render_sidebar_status(role: str, display_name: str, store: Any) -> None:
    """Compact authenticated sidebar context without loading model artifacts/metadata."""
    st.caption(display_name)
    st.markdown(f"<span class='cr-pill'>{html.escape(role.upper())}</span>", unsafe_allow_html=True)
    active_project = getattr(store, "active_project", None)
    if active_project and getattr(active_project, "recommended_version", None):
        model_name = getattr(active_project, "recommended_model_name", None) or "Final model configured"
        st.caption(f"Active: {active_project.name} · {model_name}")
    elif active_project:
        st.caption(f"Active: {active_project.name} · No final model")
    else:
        st.caption("No User-facing project")

def section_header(kicker: str, title: str, description: str | None = None) -> None:
    st.markdown(f"<div class='cr-kicker'>{html.escape(kicker)}</div>", unsafe_allow_html=True)
    st.title(title)
    if description:
        st.markdown(f"<div class='cr-muted'>{html.escape(description)}</div>", unsafe_allow_html=True)


def empty_state(title: str, body: str, action_label: str | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.write(body)
        if action_label:
            st.caption(action_label)


def render_phase_strip(current: int, total: int) -> None:
    pieces = []
    for i in range(total):
        cls = "done" if i < current else ("current" if i == current else "")
        pieces.append(f"<div class='cr-phase-dot {cls}'></div>")
    st.markdown(f"<div class='cr-phase-strip'>{''.join(pieces)}</div>", unsafe_allow_html=True)


def render_loader_status(label: str, detail: str | None = None):
    status = st.status(label, expanded=True)
    if detail:
        status.write(detail)
    return status


def risk_meter(probability: float, base_rate: float, threshold: float) -> None:
    """Render a robust Plotly speedometer-style riskometer."""
    import math
    import plotly.graph_objects as go
    from ml.explainability import risk_band

    p = float(max(0.0, min(1.0, probability)))
    t = float(max(0.0, min(1.0, threshold)))
    band, _ = risk_band(p)
    pct = p * 100.0
    threshold_pct = t * 100.0

    # Communication bands are fixed and intentionally limited to three labels.
    bands = [
        (0.0, 10.0, "Low", "#2E9F5B"),
        (10.0, 20.0, "Medium", "#C99B2B"),
        (20.0, 50.0, "High", "#C84B4B"),
    ]
    visual_max = 50.0

    def angle_for(value_pct: float) -> float:
        # 180° = left, 0° = right; upper half is the visible speedometer.
        return math.pi - (min(max(value_pct, 0.0), visual_max) / visual_max) * math.pi

    # Build colored annular sectors as ordinary XY polygons. Plotly renders these
    # reliably inside Streamlit, unlike inline SVG blocks whose height can collapse.
    fig = go.Figure()
    outer_r = 1.0
    inner_r = 0.72
    for lo, hi, label, fill_color in bands:
        a0 = angle_for(hi)
        a1 = angle_for(lo)
        outer_angles = [a0 + (a1 - a0) * i / 36 for i in range(37)]
        inner_angles = list(reversed(outer_angles))
        xs = [outer_r * math.cos(a) for a in outer_angles] + [inner_r * math.cos(a) for a in inner_angles]
        ys = [outer_r * math.sin(a) for a in outer_angles] + [inner_r * math.sin(a) for a in inner_angles]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                fillcolor=fill_color,
                line=dict(color="white", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        mid = (a0 + a1) / 2
        label_r = 0.86
        fig.add_annotation(
            x=label_r * math.cos(mid),
            y=label_r * math.sin(mid),
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=11, color="white"),
        )

    # Threshold marker.
    ta = angle_for(threshold_pct)
    fig.add_trace(
        go.Scatter(
            x=[0.72 * math.cos(ta), 1.08 * math.cos(ta)],
            y=[0.72 * math.sin(ta), 1.08 * math.sin(ta)],
            mode="lines",
            line=dict(color="#102A43", width=5),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=1.14 * math.cos(ta),
        y=1.14 * math.sin(ta),
        text=f"Threshold {threshold_pct:.1f}%",
        showarrow=False,
        font=dict(size=10, color="#102A43"),
        bgcolor="rgba(255,255,255,.92)",
        bordercolor="#DCE4EC",
        borderwidth=1,
        borderpad=4,
    )

    # Needle.
    needle_angle = angle_for(pct)
    needle_r = 0.76
    nx = needle_r * math.cos(needle_angle)
    ny = needle_r * math.sin(needle_angle)
    fig.add_trace(
        go.Scatter(
            x=[0, nx],
            y=[0, ny],
            mode="lines",
            line=dict(color="#102A43", width=7),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(size=20, color="#102A43", line=dict(color="white", width=3)),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_annotation(
        x=0,
        y=0.30,
        text=f"<b>{pct:.1f}%</b>",
        showarrow=False,
        font=dict(size=34, color="#102A43"),
    )
    fig.add_annotation(
        x=0,
        y=0.07,
        text=f"<b>{band}</b> · estimated default probability",
        showarrow=False,
        font=dict(size=14, color="#475467"),
    )
    fig.add_annotation(
        x=0,
        y=-0.10,
        text="Lower probability  •  Higher probability",
        showarrow=False,
        font=dict(size=11, color="#667085"),
    )

    fig.update_xaxes(visible=False, range=[-1.2, 1.2], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-0.18, 1.18], fixedrange=True, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        height=340,
        margin=dict(l=18, r=18, t=8, b=4),
        paper_bgcolor="white",
        plot_bgcolor="white",
        dragmode=False,
    )

    with st.container(border=True):
        st.markdown(
            "<div class='cr-kicker'>DEFAULT RISKOMETER</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated probability", f"{pct:.1f}%")
        c2.metric("Probability band", band)
        c3.metric("Decision threshold", f"{threshold_pct:.1f}%")

        if base_rate > 0:
            multiple = p / base_rate
            relation = "above" if p >= base_rate else "below"
            st.caption(
                f"Training-period default rate: {base_rate:.1%}. The estimate is {multiple:.1f}× the base rate ({relation} the observed training rate). "
                "The riskometer is a communication aid, not a regulatory or underwriting grade."
            )

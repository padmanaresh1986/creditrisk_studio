from __future__ import annotations

import html
from typing import Any

import streamlit as st


def inject_global_css() -> None:
    st.html(
        """
        <style>
        .stApp { background: #F7F9FC; }
        [data-testid="stMainBlockContainer"] { max-width: 1600px; padding: 2rem 2rem 3rem; }
        [data-testid="stSidebar"] { position:relative !important; background:linear-gradient(180deg,#0A2B57 0%,#082247 100%); border-right:1px solid #173D6B; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"], [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#D8E6F8; }
        [data-testid="stSidebar"] .cr-brand { color:#FFFFFF; padding:8px 4px 20px; }
        [data-testid="stSidebar"] .cr-brand-subtitle { color:#AEC5E3; }
        [data-testid="stSidebar"] .cr-brand-mark { background:linear-gradient(145deg,#1678F2,#49A0FF); box-shadow:0 7px 18px rgba(0,0,0,.2); }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] a { color:#D8E6F8; border-radius:8px; margin:2px 8px; }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] { background:#14569B; color:#FFFFFF; }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover { background:#123D6D; color:#FFFFFF; }
        [data-testid="stSidebarContent"] { position:relative; display:flex; flex-direction:column; height:100vh; min-height:100vh; overflow-y:auto; }
        [data-testid="stSidebarLogo"], [data-testid="stSidebarLogo"] img { height:52px !important; min-height:52px !important; max-height:none !important; width:auto !important; max-width:220px !important; margin:4px 0 18px !important; }
        [data-testid="stSidebarUserContent"] { position:static; display:flex; flex:1 1 auto; min-height:0; flex-direction:column; margin:0 !important; padding:0 16px 16px; }
        [data-testid="stSidebarUserContent"] .st-key-sidebar-account { position:absolute; left:16px; right:16px; bottom:16px; margin:0 !important; padding-top:14px; border-top:1px solid rgba(216,230,248,.16); }
        [data-testid="stSidebar"] button { border-color:transparent; color:#FFFFFF; background:transparent; justify-content:flex-start; }
        [data-testid="stSidebar"] button:hover { border-color:transparent; background:#123D6D; color:#FFFFFF; }
        [data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"] { background:#FFFFFF; border-color:#DCE7F3; border-radius:16px; box-shadow:0 4px 18px rgba(22,70,125,.045); }
        [data-testid="stMain"] div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] { background:#104B82 !important; border-color:#104B82 !important; border-radius:10px; color:#FFFFFF !important; font-weight:700; min-height:3rem; }
        [data-testid="stMain"] div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover { background:#0B3B69 !important; border-color:#0B3B69 !important; }
        [data-testid="stMain"] div[data-testid="stFormSubmitButton"] button[kind="secondaryFormSubmit"] { background:#FFFFFF !important; border:1px solid #C7D4E3 !important; border-radius:10px; color:#15365D !important; font-weight:650; min-height:3rem; }
        .cr-app-header { display:flex; align-items:center; justify-content:space-between; gap:1rem; }
        .cr-app-title { color:#102A52; font-size:2rem; font-weight:800; line-height:1.12; margin:0; }
        .cr-app-subtitle { color:#64748B; margin-top:.25rem; font-size:.96rem; }
        .cr-ready { display:inline-flex; align-items:center; gap:8px; white-space:nowrap; border-radius:999px; padding:9px 14px; background:#DDF7EC; color:#087443; font-weight:700; font-size:.88rem; }
        .cr-user-chip { display:flex; align-items:center; justify-content:flex-end; gap:8px; color:#19345D; font-weight:650; white-space:nowrap; padding:8px 10px; }
        .cr-user-icon { display:inline-flex; align-items:center; justify-content:center; border-radius:50%; width:34px; height:34px; background:#2878E8; color:white; }
        .cr-how-card { background:#FFFFFF; border:1px solid #DCE7F3; border-radius:16px; padding:18px; box-shadow:0 4px 18px rgba(22,70,125,.045); }
        .cr-section-heading { display:flex; align-items:center; gap:13px; margin:0 0 15px; }
        .cr-section-icon { display:flex; align-items:center; justify-content:center; flex:0 0 48px; height:48px; border-radius:50%; background:#E0F0FF; color:#1765C0; font-size:1.35rem; font-weight:800; }
        .cr-section-icon svg { width:24px; height:24px; fill:none; stroke:currentColor; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }
        .cr-section-copy b { display:block; color:#102A52; font-size:1.18rem; }
        .cr-section-copy span { display:block; color:#687C98; font-size:.87rem; margin-top:2px; }
        .cr-how-title { color:#102A52; font-size:1.15rem; font-weight:750; margin:0 0 18px; }
        .cr-how-icon { display:inline-flex; width:34px; height:34px; align-items:center; justify-content:center; vertical-align:middle; margin-right:8px; border-radius:50%; background:#FFF3D5; color:#D99000; }
        .cr-how-icon svg { width:21px; height:21px; fill:none; stroke:currentColor; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }
        .cr-how-step { display:flex; gap:12px; margin:0 0 18px; color:#18345E; }
        .cr-step-num { display:flex; flex:0 0 30px; height:30px; align-items:center; justify-content:center; border-radius:50%; background:#D9EAFF; color:#1765C0; font-weight:800; }
        .cr-step-copy b { font-size:.9rem; }
        .cr-step-copy p { color:#687C98; font-size:.82rem; line-height:1.4; margin:4px 0 0; }
        .cr-secure { background:#EAF7EE; color:#146340; padding:13px; border-radius:11px; font-size:.83rem; line-height:1.45; }
        .cr-secure b { display:block; margin-bottom:3px; }
        .cr-sidebar-profile { display:flex; align-items:center; gap:10px; padding:4px 4px 12px; color:#FFFFFF; font-size:.94rem; font-weight:650; }
        .cr-sidebar-avatar { display:flex; align-items:center; justify-content:center; flex:0 0 38px; height:38px; border-radius:50%; background:#DDEBFA; color:#24436B; }
        .cr-sidebar-avatar svg { width:24px; height:24px; }
        .cr-sidebar-section { color:#D8E6F8; font-size:.76rem; font-weight:700; margin:10px 0 4px; }
        @media (max-width: 800px) { [data-testid="stMainBlockContainer"] { padding:1rem; } .cr-app-title { font-size:1.55rem; } }
        .cr-kicker { color:#087F79; font-size:.72rem; text-transform:uppercase; letter-spacing:.14em; font-weight:800; }
        .cr-muted { color:#667085; font-size:.92rem; }
        .cr-brand { display:flex; align-items:center; gap:12px; font-weight:850; color:#102A43; font-size:1.16rem; }
        .cr-brand-mark { display:inline-flex; align-items:center; justify-content:center; flex:0 0 auto; width:42px; height:42px; border-radius:14px; color:#fff; background:linear-gradient(145deg,#0F6B88,#18A999); box-shadow:0 7px 18px rgba(15,107,136,.20); }
        .cr-brand-mark svg { width:25px; height:25px; }
        .cr-brand-copy { display:flex; flex-direction:column; gap:2px; }
        .cr-brand-subtitle { color:#667085; font-size:.76rem; font-weight:500; letter-spacing:.01em; }
        .cr-login-intro { color:#667085; line-height:1.55; }
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
        [data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 5vh; }
        [data-testid="stForm"] { border:0; padding:0; }
        div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:20px; box-shadow:0 18px 55px rgba(16,42,67,.09); }
        .stButton > button[kind="primaryFormSubmit"] { min-height:3rem; border-radius:10px; font-weight:700; }
        @media (max-width: 700px) { [data-testid="stMainBlockContainer"] { padding: 1.25rem 1rem; } }
        </style>
        """
    )


def render_brand(large: bool = False) -> None:
    mark_size = 54 if large else 42
    st.markdown(
        f"""<div class='cr-brand'><span class='cr-brand-mark' style='width:{mark_size}px;height:{mark_size}px'>
        <svg viewBox='0 0 32 32' fill='none' aria-hidden='true'><path d='M16 3.5 27 8v7.1c0 6.7-4.5 11.3-11 13.4C9.5 26.4 5 21.8 5 15.1V8l11-4.5Z' stroke='currentColor' stroke-width='2.2' stroke-linejoin='round'/><path d='m10.5 16.2 3.6 3.5 7.6-8' stroke='currentColor' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/></svg>
        </span><span class='cr-brand-copy'><span>CreditRisk Studio</span><span class='cr-brand-subtitle'>Automobile loan risk analytics</span></span></div>""",
        unsafe_allow_html=True,
    )
    if large:
        st.markdown("<div class='cr-login-intro'>A guided workspace for reviewing application details and understanding estimated default risk.</div>", unsafe_allow_html=True)

def render_sidebar_status(role: str, display_name: str, store: Any) -> None:
    """Compact authenticated sidebar context without loading model artifacts/metadata."""
    if role.lower() == "user":
        safe_name = html.escape(display_name)
        st.markdown(
            f"<div class='cr-sidebar-profile'><span class='cr-sidebar-avatar'><svg viewBox='0 0 24 24' aria-hidden='true'><circle cx='12' cy='8' r='4' fill='#24436B'/><path d='M4 21c.4-4.1 3.5-6.5 8-6.5s7.6 2.4 8 6.5' fill='#24436B'/></svg></span><span>{safe_name}</span></div>",
            unsafe_allow_html=True,
        )
        return
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

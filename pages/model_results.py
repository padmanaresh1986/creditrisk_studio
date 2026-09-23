from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.ui import empty_state, section_header
from core.state import get_store

store = get_store()
section_header("ADMIN WORKSPACE", "Model Results", "Review holdout evidence for the administrator-selected final prediction model.")
if not store.has_models:
    empty_state("No model results yet", "Results will appear after candidate training and validation have been completed.", "Training Studio → run the workflow")
    st.stop()

records = [r for r in store.models.values() if r.is_published]
if not records:
    empty_state("No final model results yet", "Model Results is populated only after the administrator completes final evaluation and sets the final prediction model.", "Training Studio → complete Model Release & User Access")
    st.stop()

portfolio = pd.DataFrame([
    {
        "Model": r.model_name,
        "Status": r.status,
        "CV PR-AUC": r.cv_metrics.get("PR-AUC", float("nan")),
        "CV Recall": r.cv_metrics.get("Recall", float("nan")),
        "CV F1": r.cv_metrics.get("F1", float("nan")),
        "Holdout PR-AUC": r.final_metrics.get("PR-AUC", float("nan")),
        "Holdout ROC-AUC": r.final_metrics.get("ROC-AUC", float("nan")),
        "Holdout Recall": r.final_metrics.get("Recall", float("nan")),
        "Holdout F1": r.final_metrics.get("F1", float("nan")),
        "Threshold": r.threshold,
    }
    for r in records
])

st.dataframe(portfolio.round(4), width="stretch", hide_index=True)

cv_data = portfolio.dropna(subset=["CV PR-AUC"])
if not cv_data.empty:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(cv_data, x="Model", y="CV PR-AUC", text_auto=".4f", title="Validation PR-AUC"), width="stretch")
    with c2:
        metrics = [c for c in ["CV Recall", "CV F1", "Holdout Recall", "Holdout F1"] if portfolio[c].notna().any()]
        if metrics:
            st.plotly_chart(px.bar(portfolio, x="Model", y=metrics, barmode="group", title="Secondary metrics"), width="stretch")

selected_name = st.selectbox("Published model detail", sorted({r.model_name for r in records}))
rec = next(r for r in records if r.model_name == selected_name)
with st.container(border=True):
    st.markdown(f"## {rec.model_name}")
    a, b, c, d = st.columns(4)
    a.metric("CV PR-AUC", f"{rec.cv_metrics.get('PR-AUC', float('nan')):.4f}")
    b.metric("Holdout PR-AUC", f"{rec.final_metrics.get('PR-AUC', float('nan')):.4f}" if rec.final_metrics else "—")
    c.metric("Threshold", f"{rec.threshold:.1%}")
    d.metric("Features", rec.feature_count)
    st.caption(f"{rec.family} • status: {rec.status}")
    with st.expander("Model parameters", expanded=False):
        st.json(rec.params, expanded=False)

# Detailed final evaluation is available after Phase 06.
final_bundle = st.session_state.get("final_bundle") or {}
if selected_name in final_bundle:
    out = final_bundle[selected_name]
    st.markdown("### Holdout evaluation")
    st.dataframe(out["metrics"].round(4), width="stretch", hide_index=True)
    cm = out["confusion"]
    cm_df = pd.DataFrame([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]], index=["Actual Non-default", "Actual Default"], columns=["Predicted Non-default", "Predicted Default"])
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.imshow(cm_df, text_auto=True, aspect="auto", title=f"Confusion matrix @ {out['threshold']:.1%}"), width="stretch")
    with c2:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(out["test_y"], out["probabilities"])
        fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line={"dash": "dash"}))
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig, width="stretch")
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(out["test_y"], out["probabilities"])
    st.plotly_chart(px.line(x=recall, y=precision, labels={"x": "Recall", "y": "Precision"}, title="Precision-Recall curve"), width="stretch")
else:
    st.info("Holdout metrics and curves will appear after Phase 06 final evaluation.")

with st.expander("How to read these results", expanded=False):
    st.write("PR-AUC and ROC-AUC are ranking metrics and do not depend on the displayed classification threshold. Precision, Recall and F1 change when the operating threshold changes. Holdout metrics are shown only after the candidate configuration and threshold have been finalised.")

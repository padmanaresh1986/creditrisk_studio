from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

from components.ui import empty_state, risk_meter, section_header
from core.model_catalog import available_user_models, load_model_context
from core.logging_utils import emit_exception, emit_log, render_console
from core.state import get_store
from ml.feature_engineering import raw_required_columns

store = get_store()
current_session_user = st.session_state.get("auth_user") or {}
current_session_username = str(current_session_user.get("username", ""))
active_project = store.active_project
project_text = f"Active project: {active_project.name}. " if active_project else ""
section_header("PREDICTION CENTER", "Bulk Prediction", "Score a CSV or Excel application file, inspect individual records, and download the scored workbook.")
models = available_user_models()
if not models:
    empty_state("Bulk scoring is not available yet", "No final prediction model has been released for the active project. Complete training and have an administrator set one final model.", "Training Studio → Model Release & User Access")
    st.stop()

selected = models[0]
st.caption("Predictions use the model approved for this project.")

busy = bool(st.session_state.get("busy", False))
uploaded = st.file_uploader("Upload applicant file", type=["csv", "xlsx", "xls"], disabled=busy)
if uploaded is None:
    empty_state("Waiting for an applicant file", "Upload a scoring file that follows the raw predictor schema. A target column, when present, is ignored during scoring.", "CSV, XLSX and XLS are supported")
    st.stop()

@st.cache_data(show_spinner=False)
def read_upload(name: str, payload: bytes):
    from io import BytesIO
    buf = BytesIO(payload)
    suffix = name.lower().split(".")[-1]
    if suffix == "csv":
        return pd.read_csv(buf, low_memory=False)
    if suffix == "xlsx":
        return pd.read_excel(buf, engine="openpyxl")
    return pd.read_excel(buf, engine="xlrd")


try:
    with st.spinner("Reading applicant file…"):
        raw = read_upload(uploaded.name, uploaded.getvalue())
except Exception as exc:
    st.error(f"Could not read file: {exc}")
    st.stop()

required = raw_required_columns()
missing = [c for c in required if c not in raw.columns]
if missing:
    st.error(f"The file is missing {len(missing)} required predictor columns.")
    with st.expander("Missing columns", expanded=False):
        st.dataframe(pd.DataFrame({"Missing column": missing}), width="stretch", hide_index=True)
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(raw):,}")
c2.metric("Columns", f"{len(raw.columns):,}")
c3.metric("Missing required cells", f"{int(raw[required].isna().sum().sum()):,}")

st.markdown("### Preview")
st.dataframe(raw.head(12), width="stretch", hide_index=True)

if st.button("Run bulk scoring", type="primary", width="stretch", icon=":material/play_arrow:", disabled=busy):
    st.session_state.bulk_logs = []
    st.session_state.busy = True
    emit_log(
        f"Bulk scoring started | file={uploaded.name!r} | rows={len(raw):,} | model={selected.model_name!r} | version={selected.version}",
        "INFO",
        "bulk",
    )
    try:
        from core.ml_runtime import score_raw_dataframe
        with st.status(f"Scoring {len(raw):,} rows with {selected.model_name}…", expanded=True) as status:
            status.write("Validating the applicant schema.")
            emit_log(f"Schema validated | columns={len(raw.columns)}", "DEBUG", "bulk")
            status.write("Applying feature engineering and the trained preprocessing pipeline.")
            scored, _ = score_raw_dataframe(raw, selected.version)
            emit_log(f"Scoring pipeline complete | output_shape={scored.shape}", "DEBUG", "bulk")
            status.write("Calculating probabilities and thresholded classifications.")
            st.session_state.bulk_scored = scored
            st.session_state.bulk_model_version = selected.version
            st.session_state.bulk_source_name = uploaded.name
            st.session_state.bulk_owner_username = current_session_username
            st.session_state.bulk_owner_role = str(current_session_user.get("role", ""))
            st.session_state.bulk_owner_project_id = selected.project_id
            emit_log("Bulk scoring complete", "INFO", "bulk")
            status.update(label="Bulk scoring complete", state="complete")
        st.rerun()
    except Exception as exc:
        emit_exception(exc, channel="bulk", context="Bulk scoring failed")
        st.error(f"Scoring failed: {type(exc).__name__}: {exc}")
    finally:
        st.session_state.busy = False

render_console("bulk", expanded=False)

scored = st.session_state.get("bulk_scored")
owner_mismatch = (
    st.session_state.get("bulk_owner_username") != current_session_username
    or st.session_state.get("bulk_owner_role") != str(current_session_user.get("role", ""))
    or st.session_state.get("bulk_owner_project_id") != selected.project_id
)
if owner_mismatch:
    st.session_state.bulk_scored = None
    st.session_state.bulk_model_version = None
    st.session_state.bulk_source_name = None
    scored = None

if scored is None or st.session_state.get("bulk_model_version") != selected.version:
    st.info("Run bulk scoring to populate the results table for the selected model.")
    st.stop()

p1, p2, p3, p4 = st.columns(4)
p1.metric("Rows scored", f"{len(scored):,}")
p2.metric("Predicted default", f"{int(scored['Predicted_Default'].sum()):,}")
p3.metric("Predicted default rate", f"{scored['Predicted_Default'].mean():.1%}")
p4.metric("Mean probability", f"{scored['Default_Probability'].mean():.1%}")

view_rows = st.number_input("Rows available for drill-down", 10, min(1000, len(scored)), min(100, len(scored)), 10)
preview = scored.head(int(view_rows)).copy()
preview.insert(0, "_Row", range(1, len(preview) + 1))
event = st.dataframe(preview, width="stretch", height=430, hide_index=True, on_select="rerun", selection_mode="single-row", key="bulk_results")
selected_rows = list(event.selection.rows) if hasattr(event, "selection") else []

if selected_rows:
    idx = selected_rows[0]
    row = preview.iloc[idx].drop(labels=["_Row"])
    row_num = int(preview.iloc[idx]["_Row"])
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown(f"### Record #{row_num}")
        detail = pd.DataFrame({
            "Field": [str(v) for v in row.index],
            "Value": ["" if pd.isna(v) else str(v) for v in row.to_list()],
        })
        st.dataframe(detail, width="stretch", height=420, hide_index=True)
    with right:
        st.markdown("### Risk detail")
        risk_meter(float(row["Default_Probability"]), float(row["Training_Default_Rate"]), float(row["Threshold"]))
        raw_row = row[raw.columns].to_frame().T
        with st.spinner("Preparing row-level explanation…"):
            from core.ml_runtime import load_model, score_raw_dataframe
            scored_row, engineered = score_raw_dataframe(raw_row, selected.version)
            ctx = load_model_context(selected.version)
            from ml.explainability import clean_display_feature, explain_local, local_shap_values
            applicant = engineered.reindex(columns=ctx["feature_columns"])
            model = load_model(selected.version)
            shap_df, _ = local_shap_values(model, applicant)
        exp = explain_local(shap_df, ctx.get("dictionary", {}), top_n=4)
        if exp["higher_risk"]:
            st.markdown("**Contributors moving the estimate upward**")
            for item in exp["higher_risk"]:
                st.write(f"**{item['feature']}** — {item['impact']:+.3f} · {item['definition']}")
        if exp["lower_risk"]:
            st.markdown("**Contributors moving the estimate downward**")
            for item in exp["lower_risk"]:
                st.write(f"**{item['feature']}** — {item['impact']:+.3f} · {item['definition']}")
        top = shap_df.head(8).copy()
        top["Feature"] = top["Feature"].map(clean_display_feature)
        st.plotly_chart(px.bar(top.sort_values("SHAP_Value"), x="SHAP_Value", y="Feature", orientation="h", title="Local model contributions"), width="stretch")

st.markdown("### Probability distribution")
st.plotly_chart(px.histogram(scored, x="Default_Probability", nbins=30, title=f"Default probability distribution — {selected.model_name}"), width="stretch")

from core.ml_runtime import build_excel_export
st.download_button(
    "Download scored Excel workbook",
    data=build_excel_export(scored),
    file_name=f"creditrisk_scored_{selected.model_name.lower().replace(' ', '_')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary",
    width="stretch",
    icon=":material/download:",
)

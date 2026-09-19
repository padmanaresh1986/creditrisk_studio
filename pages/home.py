from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from components.ui import empty_state, section_header
from core.state import get_store
from core.projects import get_project_manager

store = get_store()
manager = get_project_manager()
section_header("ADMIN WORKSPACE", "Admin Dashboard", "Manage saved training projects, review model evidence and choose which trained project is available to Users.")

projects = manager.list_projects()
if projects:
    active_project = manager.active_project()
    published = store.user_models()
    default_name = store.active_model_name if published else "—"
    with st.container(border=True):
        ctx1, ctx2, ctx3 = st.columns(3)
        ctx1.metric("Administrator", "ADMIN")
        ctx2.metric("User-facing project", active_project.name if active_project else "—")
        ctx3.metric("Default model", default_name)

if not projects:
    empty_state("No projects yet", "Create a named project in Training Studio. The project will persist its source data, trained model artifacts, analysis outputs and release metadata on disk.", "Training Studio → 01 Data Setup")
    st.markdown("### Workspace")
    cols = st.columns(4)
    cols[0].metric("Projects", "—")
    cols[1].metric("Trained models", "—")
    cols[2].metric("Active user project", "—")
    cols[3].metric("Last run", "—")
    st.stop()

# Project library
st.markdown("### Project library")
proj_rows = manager.project_summary()
st.dataframe(pd.DataFrame(proj_rows).drop(columns=["Project ID"]), width="stretch", hide_index=True)

options = {f"{p.name} · {p.status} · {p.project_id}": p.project_id for p in projects}
current_id = store.current_project_id or store.active_project_id
keys = list(options)
current_label = next((label for label, pid in options.items() if pid == current_id), keys[0])
selected_label = st.selectbox("Project to inspect", keys, index=keys.index(current_label))
selected_project_id = options[selected_label]
selected_project = manager.get_project(selected_project_id)
assert selected_project is not None
store.select_project(selected_project_id)

project_models = [r for r in store.models.values() if r.project_id == selected_project_id]
if project_models:
    final_models = [r for r in project_models if r.final_metrics]
    st.markdown("### Selected project")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{selected_project.rows:,}")
    c2.metric("Default rate", f"{selected_project.default_rate:.2%}")
    c3.metric("Model versions", len(project_models))
    c4.metric("Evaluated models", len(final_models))

    rows=[]
    for rec in sorted(project_models, key=lambda r: r.trained_at_utc or ""):
        rows.append({
            "Model": rec.model_name,
            "Version": rec.version,
            "Status": rec.status,
            "CV PR-AUC": rec.cv_metrics.get("PR-AUC", float("nan")),
            "Holdout PR-AUC": rec.final_metrics.get("PR-AUC", float("nan")),
            "Recall": rec.final_metrics.get("Recall", float("nan")),
            "F1": rec.final_metrics.get("F1", float("nan")),
            "Threshold": rec.threshold,
        })
    st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)

    eligible = [r for r in store.latest_final_models(selected_project_id) if Path(r.artifact_path).exists()]
    if eligible:
        eligible_sorted = sorted(eligible, key=lambda r: float(r.cv_metrics.get("PR-AUC", -1)), reverse=True)
        labels = {f"{r.model_name} · CV PR-AUC {r.cv_metrics.get('PR-AUC', float('nan')):.4f} · {r.version}": r.version for r in eligible_sorted}
        choice_label = st.selectbox("Default model for the selected project", list(labels))
        recommended_version = labels[choice_label]
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Set this project as User Project", type="primary", width="stretch", icon=":material/publish:", disabled=store.current_project_id is None):
                store.set_user_project(selected_project_id, recommended_version)
                st.success(f"{selected_project.name} is now the User-facing project. Default model: {next(r.model_name for r in eligible if r.version == recommended_version)}")
                st.rerun()
        with b2:
            active = store.active_project
            if active and active.project_id == selected_project_id:
                st.success("This is the active User project.", icon=":material/check_circle:")
            else:
                st.info("This project is not currently visible to Users.")

    with st.expander("Project storage", expanded=False):
        st.code(selected_project.root_path)
        st.write("Source data, model versions, analysis CSVs, run logs and project metadata are stored below this folder. Paths are recorded relative to the project for portability.")
else:
    empty_state("Project created, training not yet run", "This project has no saved model versions yet. Open Training Studio to run the lifecycle.", "Training Studio → run the remaining phases")


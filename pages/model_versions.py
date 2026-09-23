from __future__ import annotations

import pandas as pd
import streamlit as st

from components.ui import empty_state, section_header
from core.projects import get_project_manager
from core.state import get_store

store = get_store()
manager = get_project_manager()
section_header("ADMIN WORKSPACE", "Model & Project Registry", "Durable local registry of project datasets, trained model artifacts, versions and the single User-facing final model.")

projects = manager.list_projects()
if not projects:
    empty_state("No projects saved", "Projects appear here after an administrator creates and processes a dataset in Training Studio.", "Nothing is preloaded")
    st.stop()

st.markdown("### Saved projects")
proj_df = pd.DataFrame(manager.project_summary())
st.dataframe(proj_df, width="stretch", hide_index=True)

st.markdown("### Project details")
labels = [f"{p.name} · {p.status} · {p.project_id}" for p in projects]
selected_label = st.selectbox("Project", labels)
selected_id = next(p.project_id for p in projects if f"{p.name} · {p.status} · {p.project_id}" == selected_label)
selected = manager.get_project(selected_id)
assert selected is not None

rows=[]
for rec in [r for r in store.models.values() if r.project_id == selected_id and r.is_trained]:
    rows.append({
        "Model": rec.model_name, "Version": rec.version, "Status": rec.status,
        "CV PR-AUC": rec.cv_metrics.get("PR-AUC", float("nan")),
        "Holdout PR-AUC": rec.final_metrics.get("PR-AUC", float("nan")),
        "Threshold": rec.threshold, "Features": rec.feature_count,
        "Artifact": rec.artifact_path,
    })
if rows:
    st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)
else:
    empty_state("No model versions for this project", "Run the modelling lifecycle before publishing this project.", "Training Studio")

eligible=[r for r in store.latest_final_models(selected_id) if r.artifact_path]
if eligible:
    default_sorted=sorted(eligible,key=lambda r: float(r.cv_metrics.get("PR-AUC",-1)),reverse=True)
    choice_map={f"{r.model_name} · {r.version}": r.version for r in default_sorted}
    recommendation=st.selectbox("Final prediction model for this project", list(choice_map))
    if st.button("Set project + model for Users", type="primary", icon=":material/publish:", width="stretch"):
        store.set_user_project(selected_id, choice_map[recommendation])
        st.success("Project is now the active User-facing project.")
        st.rerun()

with st.expander("What is persisted?", expanded=False):
    st.write("Each project owns its source training files, candidate and released model artifacts, model metadata, analysis CSVs and run logs. The persistent registry is local to this application installation; it is not a replacement for a multi-user database or enterprise model registry.")
    st.code("projects/<project-slug>__<project-id>/\n├── project.json\n├── data/\n│   ├── training dataset\n│   └── data dictionary\n├── models/\n│   ├── candidates/\n│   ├── releases/\n│   └── model_index.json\n├── analysis/\n├── runs/\n└── exports/", language="text")

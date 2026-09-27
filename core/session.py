from __future__ import annotations

import streamlit as st


def init_app_state() -> None:
    defaults = {
        "studio_phase": 0,
        "completed_phases": set(),
        "dataset_locked": False,
        "dataset_name": None,
        "project_id": None,
        "project_name": None,
        "project_root": None,
        "training_df": None,
        "dictionary_df": None,
        "training_bundle": None,
        "cv_bundle": None,
        "imbalance_df": None,
        "threshold_bundle": None,
        "final_bundle": None,
        "tuning_bundle": None,
        "xai_bundle": {},
        "analysis_cache": None,
        "training_logs": [],
        "prediction_logs": [],
        "bulk_logs": [],
        "app_logs": [],
        "busy": False,
        "busy_label": "",
        "navigation_notice": None,
        "bulk_scored": None,
        "bulk_model_version": None,
        "bulk_source_name": None,
        "bulk_owner_username": None,
        "bulk_owner_role": None,
        "bulk_owner_project_id": None,
        "quick_result": None,
        "last_run_id": None,
        "phase4_complete": False,
        "phase5_complete": False,
        "phase6_complete": False,
        "phase7_complete": False,
        "eda_cache_ready": False,
        "project_selection": None,
        "selected_model_names": ["Random Forest", "XGBoost", "L1 Logistic Regression"],
        "final_model_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            if isinstance(value, dict): st.session_state[key] = value.copy()
            elif isinstance(value, list): st.session_state[key] = list(value)
            elif isinstance(value, set): st.session_state[key] = set(value)
            else: st.session_state[key] = value


def clear_training_state(keep_dataset: bool = True) -> None:
    for key in [
        "training_bundle", "cv_bundle", "imbalance_df", "threshold_bundle", "final_bundle", "tuning_bundle",
        "xai_bundle", "analysis_cache", "quick_result", "bulk_scored", "bulk_model_version", "bulk_source_name",
        "bulk_owner_username", "bulk_owner_role", "bulk_owner_project_id",
    ]:
        st.session_state[key] = {} if key == "xai_bundle" else None
    st.session_state.completed_phases = set()
    for key in ["phase4_complete", "phase5_complete", "phase6_complete", "phase7_complete", "eda_cache_ready"]:
        st.session_state[key] = False
    st.session_state.training_logs = []
    st.session_state.prediction_logs = []
    st.session_state.bulk_logs = []
    st.session_state.selected_model_names = ["Random Forest", "XGBoost", "L1 Logistic Regression"]
    st.session_state.final_model_name = None
    if not keep_dataset:
        for key in ["training_df", "dictionary_df"]:
            st.session_state[key] = None
        for key in ["dataset_name", "project_id", "project_name", "project_root"]:
            st.session_state[key] = None
        st.session_state.dataset_locked = False
    st.session_state.studio_phase = 0


def add_log(message: str) -> None:
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.training_logs = (st.session_state.training_logs + [f"[{ts}] {message}"])[-1200:]


def set_busy(label: str) -> None:
    st.session_state.busy = True
    st.session_state.busy_label = label


def clear_busy() -> None:
    st.session_state.busy = False
    st.session_state.busy_label = ""


def reset_session_for_logout() -> None:
    """Clear all Streamlit session state at the authentication boundary.

    Persistent projects, datasets, model artifacts and release metadata live on
    disk and are intentionally preserved. All transient UI, prediction, training,
    and role-specific state is removed so a later login starts with a clean session.
    """
    # Snapshot keys because the mapping is modified during iteration.
    for key in list(st.session_state.keys()):
        try:
            del st.session_state[key]
        except KeyError:
            pass

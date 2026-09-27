from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from core.state import get_store
from ml.schema import CATEGORICAL_FEATURES_FINAL, RANDOM_STATE


@st.cache_data(show_spinner=False)
def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_record(version: str):
    rec = get_store().get_model(version)
    if rec is None:
        raise KeyError(f"Model version '{version}' is not available.")
    return rec


def available_user_models():
    return get_store().user_models()


def load_model_context(version: str) -> dict[str, Any]:
    rec = model_record(version)
    if not rec.context_path or not Path(rec.context_path).exists():
        raise FileNotFoundError("The selected model context is not available.")
    return _load_json(rec.context_path)


@st.cache_data(show_spinner=False)
def prediction_input_profile(version: str) -> dict[str, Any]:
    """Load precomputed form metadata without rereading the project CSV when possible."""
    ctx = load_model_context(version)
    embedded = ctx.get("input_profile")
    if isinstance(embedded, dict) and embedded.get("columns") is not None:
        return embedded

    # Backward-compatible fallback for old model contexts produced before the
    # embedded input profile was introduced.
    rec = model_record(version)
    from ml.feature_engineering import raw_required_columns
    project = get_store().manager.get_project(rec.project_id)
    if project is None:
        return {"columns": {}, "source": "model-context-defaults"}
    path = Path(project.root_path) / "data" / project.dataset_filename
    if not path.exists():
        return {"columns": {}, "source": "model-context-defaults"}
    import pandas as pd
    df = pd.read_csv(path, usecols=lambda c: c in raw_required_columns(), low_memory=False)
    profile: dict[str, Any] = {"columns": {}, "source": str(path)}
    for col in raw_required_columns():
        if col not in df.columns:
            continue
        series = df[col]
        info: dict[str, Any] = {}
        if col in CATEGORICAL_FEATURES_FINAL:
            info["kind"] = "categorical"
            info["options"] = [v for v in pd.unique(series.dropna())][:100]
        else:
            num = pd.to_numeric(series, errors="coerce")
            info["kind"] = "numeric"
            if num.notna().any():
                info.update({"min": float(num.min()), "max": float(num.max()), "median": float(num.median())})
        profile["columns"][col] = info
    return profile

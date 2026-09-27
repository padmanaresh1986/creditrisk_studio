from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    class _Stub:
        @staticmethod
        def cache_resource(**kwargs): return lambda fn: fn
        @staticmethod
        def cache_data(**kwargs): return lambda fn: fn
    st = _Stub()

from ml.feature_engineering import clean_raw_dataframe, engineer_features, raw_required_columns
from ml.explainability import local_shap_values
from ml.schema import CATEGORICAL_FEATURES_FINAL


@st.cache_resource(show_spinner=False)
def _load_joblib(path: str):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


from core.model_catalog import available_user_models, load_model_context, model_record, prediction_input_profile


@st.cache_data(show_spinner=False)
def cached_global_feature_importance(version: str):
    from ml.explainability import global_feature_importance, clean_display_feature
    model = load_model(version)
    frame = global_feature_importance(model).head(25).copy()
    frame["Display Feature"] = frame["Feature"].map(clean_display_feature)
    return frame


def load_model(version: str):
    rec = model_record(version)
    if not rec.artifact_path or not Path(rec.artifact_path).exists():
        raise FileNotFoundError("The selected model artifact is not available in this Streamlit process.")
    return _load_joblib(rec.artifact_path)


def _validate_optional_score(label: str, value: Any) -> None:
    if value is None:
        return
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")


def prepare_quick_applicant(version: str, inputs: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a dtype-safe raw applicant row, then reuse the authoritative feature pipeline."""
    ctx = load_model_context(version)
    rec = model_record(version)
    raw_defaults = dict(ctx["raw_feature_defaults"])
    raw = pd.DataFrame([{c: raw_defaults.get(c) for c in ctx["raw_input_columns"]}])

    direct = inputs.get("raw_overrides") if isinstance(inputs, dict) else None
    overrides: dict[str, Any] = dict(direct or {})

    # Backward-compatible aliases for the original compact form.
    alias_map = {
        "client_income": "Client_Income",
        "credit_amount": "Credit_Amount",
        "loan_annuity": "Loan_Annuity",
        "child_count": "Child_Count",
        "family_members": "Client_Family_Members",
        "score_source_1": "Score_Source_1",
        "score_source_2": "Score_Source_2",
        "score_source_3": "Score_Source_3",
        "population_region_relative": "Population_Region_Relative",
        "phone_change": "Phone_Change",
        "credit_bureau": "Credit_Bureau",
        "social_circle_default": "Social_Circle_Default",
        "own_house_age": "Own_House_Age",
    }
    for old_key, raw_key in alias_map.items():
        if old_key in inputs and inputs.get(old_key) is not None:
            overrides[raw_key] = inputs.get(old_key)

    # Derived duration aliases supplied by the UI.
    if inputs.get("age_years") is not None and "Age_Days" not in overrides:
        overrides["Age_Days"] = -(float(inputs["age_years"]) * 365.25)
    if inputs.get("employment_years") is not None and "Employed_Days" not in overrides:
        overrides["Employed_Days"] = -(float(inputs["employment_years"]) * 365.25)

    for col, value in list(overrides.items()):
        if col in raw.columns and value is not None:
            raw.loc[0, col] = value

    # Validate the core financial/profile inputs whenever supplied.
    for col, label in [("Client_Income", "Annual income"), ("Credit_Amount", "Loan amount"), ("Loan_Annuity", "Loan annuity")]:
        if col in raw.columns:
            value = pd.to_numeric(raw.loc[0, col], errors="coerce")
            if pd.notna(value) and float(value) <= 0:
                raise ValueError(f"{label} must be greater than 0.")
    if "Age_Days" in raw.columns:
        age_value = pd.to_numeric(raw.loc[0, "Age_Days"], errors="coerce")
        if pd.notna(age_value):
            age = abs(float(age_value)) / 365.25
            if not 18 <= age <= 100:
                raise ValueError("Age must be between 18 and 100 years.")
    if "Employed_Days" in raw.columns:
        emp_value = pd.to_numeric(raw.loc[0, "Employed_Days"], errors="coerce")
        if pd.notna(emp_value) and float(emp_value) != 365243:
            employment = abs(float(emp_value)) / 365.25
            if "Age_Days" in raw.columns:
                age_value = pd.to_numeric(raw.loc[0, "Age_Days"], errors="coerce")
                if pd.notna(age_value) and employment > abs(float(age_value)) / 365.25:
                    raise ValueError("Employment years must not exceed age.")
    if "Child_Count" in raw.columns and "Client_Family_Members" in raw.columns:
        children = pd.to_numeric(raw.loc[0, "Child_Count"], errors="coerce")
        family = pd.to_numeric(raw.loc[0, "Client_Family_Members"], errors="coerce")
        if pd.notna(children) and children < 0:
            raise ValueError("Children cannot be negative.")
        if pd.notna(family) and family < 1:
            raise ValueError("Family members must be at least 1.")
        if pd.notna(children) and pd.notna(family) and children > family:
            raise ValueError("Children cannot exceed family members.")

    for label, key in [("Score Source 1", "Score_Source_1"), ("Score Source 2", "Score_Source_2"), ("Score Source 3", "Score_Source_3")]:
        _validate_optional_score(label, overrides.get(key))

    # Normalize numeric inputs before feature engineering. This also protects
    # selected bulk rows whose mixed dtype can become object-backed Series.
    for col in [
        "Client_Income", "Credit_Amount", "Loan_Annuity", "Population_Region_Relative",
        "Age_Days", "Employed_Days", "Registration_Days", "ID_Days", "Own_House_Age",
        "Client_Family_Members", "Child_Count", "Application_Process_Day", "Application_Process_Hour",
        "Score_Source_1", "Score_Source_2", "Score_Source_3", "Social_Circle_Default",
        "Phone_Change", "Credit_Bureau", "Car_Owned", "Bike_Owned", "Active_Loan", "House_Own",
        "Homephone_Tag", "Workphone_Working", "Cleint_City_Rating",
    ]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    cleaned = clean_raw_dataframe(raw, drop_constant=False).drop(columns=["Mobile_Tag"], errors="ignore")
    engineered = engineer_features(cleaned)
    applicant = engineered.reindex(columns=ctx["feature_columns"])

    supplied_raw = {c for c in overrides if c in raw.columns and overrides.get(c) is not None}
    assumptions = [c for c in raw.columns if c not in supplied_raw]
    meta = {
        "raw": raw,
        "engineered": engineered,
        "assumed_fields": assumptions,
        "supplied_fields": len(supplied_raw),
        "supplied_raw_fields": sorted(supplied_raw),
        "base_rate": float(ctx.get("base_default_rate", 0.0)),
        "project_name": rec.project_name,
        "model_name": rec.model_name,
    }
    return applicant, meta


def predict_quick(version: str, inputs: dict[str,Any]):
    from core.state import get_store
    rec=model_record(version)
    threshold=float(rec.threshold)
    applicant,meta=prepare_quick_applicant(version,inputs)
    model=load_model(version)
    probability=float(model.predict_proba(applicant)[0,1])
    prediction=int(probability>=threshold)
    from ml.explainability import risk_band
    band, _ = risk_band(probability)
    base=max(meta["base_rate"],1e-9)
    relative=float(probability/base)
    return applicant,{"model_name":rec.model_name,"version":rec.version,"probability":probability,"threshold":threshold,"prediction":prediction,"risk_decision":"Higher Default Risk" if prediction else "Lower Default Risk","risk_band":band,"relative_to_training_rate":relative,"assumption_count":len(meta["assumed_fields"]),"base_rate":meta["base_rate"],"assumed_fields":meta["assumed_fields"],"supplied_fields":meta["supplied_fields"]}


def score_raw_dataframe(raw_df: pd.DataFrame, version: str):
    rec=model_record(version); model=load_model(version); ctx=load_model_context(version)
    clean=clean_raw_dataframe(raw_df,drop_constant=False).drop(columns=["Mobile_Tag"],errors="ignore")
    engineered=engineer_features(clean)
    X=engineered.reindex(columns=ctx["feature_columns"])
    probs=model.predict_proba(X)[:,1]
    pred=(probs>=float(rec.threshold)).astype(int)
    required=raw_required_columns(); present=[c for c in required if c in raw_df.columns]
    missing=raw_df[present].isna().sum(axis=1)
    result=raw_df.copy()
    result["Model"]=rec.model_name; result["Model_Version"]=rec.version; result["Default_Probability"]=probs; result["Predicted_Default"]=pred
    result["Risk_Decision"]=np.where(pred==1,"Higher Default Risk","Lower Default Risk"); result["Threshold"]=float(rec.threshold)
    result["Missing_Input_Fields"]=missing.astype(int); result["Input_Completeness_%"]=(100*(1-missing/max(len(required),1))).round(1)
    result["Training_Default_Rate"]=float(ctx.get("base_default_rate",0.0))
    return result,engineered


def build_excel_export(scored_df: pd.DataFrame):
    buf=BytesIO()
    summary=pd.DataFrame([
        {"Metric":"Rows scored","Value":int(len(scored_df))},
        {"Metric":"Model","Value":str(scored_df["Model"].iloc[0]) if len(scored_df) else ""},
        {"Metric":"Threshold","Value":float(scored_df["Threshold"].iloc[0]) if len(scored_df) else ""},
        {"Metric":"Average default probability","Value":float(scored_df["Default_Probability"].mean()) if len(scored_df) else ""},
        {"Metric":"Predicted higher-risk rows","Value":int(scored_df["Predicted_Default"].sum()) if len(scored_df) else 0},
    ])
    with pd.ExcelWriter(buf,engine="openpyxl") as writer:
        summary.to_excel(writer,sheet_name="Summary",index=False)
        scored_df.to_excel(writer,sheet_name="Scored_Applications",index=False)
    buf.seek(0); return buf.getvalue()


def local_explanation(applicant: pd.DataFrame, version: str):
    return local_shap_values(load_model(version),applicant)

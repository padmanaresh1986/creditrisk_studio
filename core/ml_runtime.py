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


@st.cache_resource(show_spinner=False)
def _load_joblib(path: str):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_record(version: str):
    from core.state import get_store
    rec = get_store().models.get(version)
    if rec is None:
        raise KeyError(f"Model version '{version}' is not available.")
    return rec


def available_user_models():
    from core.state import get_store
    return get_store().user_models()


def load_model(version: str):
    rec = model_record(version)
    if not rec.artifact_path or not Path(rec.artifact_path).exists():
        raise FileNotFoundError("The selected model artifact is not available in this Streamlit process.")
    return _load_joblib(rec.artifact_path)


def load_model_context(version: str):
    rec = model_record(version)
    if not rec.context_path or not Path(rec.context_path).exists():
        raise FileNotFoundError("The selected model context is not available.")
    return _load_json(rec.context_path)


def dictionary_map(version: str) -> dict[str, str]:
    ctx = load_model_context(version)
    return dict(ctx.get("dictionary", {}))


def prepare_quick_applicant(version: str, inputs: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    for key in ["client_income","credit_amount","loan_annuity"]:
        if float(inputs[key]) <= 0:
            raise ValueError(f"{key.replace('_',' ').title()} must be greater than 0.")
    age=float(inputs["age_years"]); employment=float(inputs["employment_years"])
    children=int(inputs["child_count"]); family=int(inputs["family_members"])
    if not 18 <= age <= 100: raise ValueError("Age must be between 18 and 100 years.")
    if not 0 <= employment <= age: raise ValueError("Employment years must be between 0 and age.")
    if family < 1 or children < 0 or children > family: raise ValueError("Children must be between 0 and family members.")
    for label,key in [("Score Source 1","score_source_1"),("Score Source 2","score_source_2"),("Score Source 3","score_source_3")]:
        val=inputs.get(key)
        if val is not None and not 0 <= float(val) <= 1: raise ValueError(f"{label} must be between 0 and 1.")
    ctx=load_model_context(version)
    raw_defaults=dict(ctx["raw_feature_defaults"])
    raw=pd.DataFrame([{c:raw_defaults.get(c) for c in ctx["raw_input_columns"]}])
    overrides={
        "Client_Income":float(inputs["client_income"]),"Credit_Amount":float(inputs["credit_amount"]),"Loan_Annuity":float(inputs["loan_annuity"]),
        "Child_Count":children,"Client_Family_Members":family,"Age_Days":-(age*365.25),"Employed_Days":-(employment*365.25),
        "Score_Source_1":inputs.get("score_source_1"),"Score_Source_2":inputs.get("score_source_2"),"Score_Source_3":inputs.get("score_source_3"),
    }
    for col,val in overrides.items():
        if col in raw.columns: raw.loc[0,col]=val
    cleaned=clean_raw_dataframe(raw,drop_constant=False).drop(columns=["Mobile_Tag"],errors="ignore")
    engineered=engineer_features(cleaned)
    applicant=engineered.reindex(columns=ctx["feature_columns"])
    supplied_raw={c for c,v in overrides.items() if c in raw.columns and v is not None}
    assumptions=[c for c in raw.columns if c not in supplied_raw]
    supplied=len(supplied_raw)
    return applicant,{"raw":raw,"engineered":engineered,"assumed_fields":assumptions,"supplied_fields":supplied,"base_rate":float(ctx.get("base_default_rate",0.0))}


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

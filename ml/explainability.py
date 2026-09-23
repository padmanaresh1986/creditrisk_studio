from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ENGINEERED_DEFINITIONS = {
    "Loan_to_Income_Ratio": "Loan amount relative to annual client income.",
    "Annuity_to_Income_Ratio": "Loan annuity relative to annual client income.",
    "Credit_per_Family_Member": "Loan credit amount normalized by the number of family members.",
    "Children_to_Family_Ratio": "Number of children relative to household family size.",
    "Age_Years": "Client age expressed in years from the source day-based field.",
    "Employment_Years": "Employment duration expressed in years from the source day-based field.",
    "Registration_Years": "Registration duration expressed in years from the source day-based field.",
    "ID_Change_Years": "Elapsed time represented by the identity-change day field, expressed in years.",
    "Average_Credit_Score": "Mean of the available external score-source values.",
    "Min_Credit_Score": "Lowest available external score-source value.",
    "Max_Credit_Score": "Highest available external score-source value.",
    "Credit_Score_Range": "Difference between the highest and lowest available score-source values.",
    "Available_Credit_Scores": "Count of external score-source values that are present.",
    "Application_Hour_Sin": "Cyclical sine encoding of the application hour.",
    "Application_Hour_Cos": "Cyclical cosine encoding of the application hour.",
    "Client_Income_Log": "Log-transformed version of client income used to reduce the influence of a long right tail.",
    "Credit_Amount_Log": "Log-transformed version of loan credit amount.",
    "Loan_Annuity_Log": "Log-transformed version of the loan annuity.",
}

# For user-facing explanations, aggregate derived model features back to the
# raw application fields. When a derived feature depends on multiple inputs,
# its SHAP contribution is split evenly across those source inputs. This is a
# presentation aid, not a causal attribution.
DERIVED_TO_RAW = {
    "Loan_to_Income_Ratio": ["Credit_Amount", "Client_Income"],
    "Annuity_to_Income_Ratio": ["Loan_Annuity", "Client_Income"],
    "Credit_per_Family_Member": ["Credit_Amount", "Client_Family_Members"],
    "Children_to_Family_Ratio": ["Child_Count", "Client_Family_Members"],
    "Age_Years": ["Age_Days"],
    "Employment_Years": ["Employed_Days"],
    "Registration_Years": ["Registration_Days"],
    "ID_Change_Years": ["ID_Days"],
    "Average_Credit_Score": ["Score_Source_1", "Score_Source_2", "Score_Source_3"],
    "Min_Credit_Score": ["Score_Source_1", "Score_Source_2", "Score_Source_3"],
    "Max_Credit_Score": ["Score_Source_1", "Score_Source_2", "Score_Source_3"],
    "Credit_Score_Range": ["Score_Source_1", "Score_Source_2", "Score_Source_3"],
    "Available_Credit_Scores": ["Score_Source_1", "Score_Source_2", "Score_Source_3"],
    "Score_Source_1_Missing": ["Score_Source_1"],
    "Score_Source_3_Missing": ["Score_Source_3"],
    "Client_Occupation_Missing": ["Client_Occupation"],
    "Credit_Bureau_Missing": ["Credit_Bureau"],
    "Social_Circle_Default_Missing": ["Social_Circle_Default"],
    "Application_Hour_Sin": ["Application_Process_Hour"],
    "Application_Hour_Cos": ["Application_Process_Hour"],
    "Client_Income_Log": ["Client_Income"],
    "Credit_Amount_Log": ["Credit_Amount"],
    "Loan_Annuity_Log": ["Loan_Annuity"],
}


def _one_class_shap_values(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 3:
        return arr[:, :, 1] if arr.shape[-1] > 1 else arr[:, :, 0]
    return arr


def local_shap_values(pipeline, X_raw: pd.DataFrame):
    import shap

    preprocessor = pipeline.named_steps["preprocessing"]
    estimator = pipeline.named_steps["model"]
    transformed = preprocessor.transform(X_raw)
    names = list(preprocessor.get_feature_names_out())
    if hasattr(estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(estimator)
    elif hasattr(estimator, "coef_"):
        explainer = shap.LinearExplainer(estimator, transformed)
    else:
        explainer = shap.Explainer(estimator, transformed)
    explanation = explainer(transformed)
    values = _one_class_shap_values(explanation.values)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    base = np.asarray(getattr(explanation, "base_values", getattr(explainer, "expected_value", 0.0)))
    base_value = float(base.reshape(-1)[-1]) if base.size else 0.0
    return pd.DataFrame({
        "Feature": names,
        "SHAP_Value": values[0],
        "Abs_SHAP": np.abs(values[0]),
        "Transformed_Value": np.asarray(transformed)[0],
    }).sort_values("Abs_SHAP", ascending=False).reset_index(drop=True), base_value


def global_feature_importance(pipeline) -> pd.DataFrame:
    prep = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]
    names = list(prep.get_feature_names_out())
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        return pd.DataFrame({"Feature": names, "Importance": values}).sort_values("Importance", ascending=False).reset_index(drop=True)
    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_).reshape(-1)
        return pd.DataFrame({"Feature": names, "Coefficient": values, "Importance": np.abs(values)}).sort_values("Importance", ascending=False).reset_index(drop=True)
    raise ValueError("Model does not expose native importance or coefficients.")


def permutation_importance_frame(pipeline, X_eval, y_eval, n_repeats: int = 5, random_state: int = 42):
    from sklearn.inspection import permutation_importance
    prep = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]
    transformed = prep.transform(X_eval)
    result = permutation_importance(
        model,
        transformed,
        y_eval,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return pd.DataFrame({
        "Feature": prep.get_feature_names_out(),
        "Importance_Mean": result.importances_mean,
        "Importance_STD": result.importances_std,
    }).sort_values("Importance_Mean", ascending=False).reset_index(drop=True)


def clean_display_feature(feature: str) -> str:
    return feature.replace("numeric__", "").replace("categorical__", "")


def _raw_feature_for_encoded(name: str, raw_columns: list[str]) -> list[str]:
    cleaned = clean_display_feature(str(name))
    if cleaned in raw_columns:
        return [cleaned]
    # Match the longest raw column name as a prefix, which handles one-hot
    # encoded names such as Client_Marital_Status_M.
    matches = [c for c in raw_columns if cleaned.startswith(c + "_")]
    if matches:
        return [max(matches, key=len)]
    return []


def raw_dependencies(feature: str, raw_columns: list[str]) -> list[str]:
    cleaned = clean_display_feature(str(feature))
    if cleaned in DERIVED_TO_RAW:
        return [c for c in DERIVED_TO_RAW[cleaned] if c in raw_columns]
    if cleaned.endswith("_Missing"):
        raw = cleaned[:-8]
        if raw in raw_columns:
            return [raw]
    return _raw_feature_for_encoded(cleaned, raw_columns)


def aggregate_input_contributions(
    shap_df: pd.DataFrame,
    raw_values: pd.DataFrame,
    dictionary: dict[str, str] | None = None,
    supplied_fields: list[str] | set[str] | None = None,
) -> pd.DataFrame:
    """Aggregate encoded/engineered SHAP evidence back to raw user inputs.

    A derived feature that uses multiple raw inputs splits its signed SHAP
    contribution equally between those inputs. This keeps the chart legible
    while explicitly avoiding causal language.
    """
    # Backward compatibility for older positional callers that accidentally
    # supplied (dictionary, supplied_fields) in the opposite order.
    if not isinstance(dictionary, dict) and isinstance(supplied_fields, dict):
        dictionary, supplied_fields = supplied_fields, dictionary

    if raw_values is None or raw_values.empty:
        return pd.DataFrame(columns=["Raw_Feature", "SHAP_Value", "Abs_SHAP", "Definition", "Value", "Direction"])

    raw_columns = list(raw_values.columns)
    supplied = set(supplied_fields or raw_columns)
    contributions: dict[str, float] = {c: 0.0 for c in supplied if c in raw_columns}

    for _, row in shap_df.iterrows():
        deps = raw_dependencies(str(row["Feature"]), raw_columns)
        deps = [d for d in deps if d in contributions]
        if not deps:
            continue
        share = float(row["SHAP_Value"]) / len(deps)
        for dep in deps:
            contributions[dep] += share

    dictionary = dictionary or {}
    rows: list[dict[str, Any]] = []
    for raw_name, impact in contributions.items():
        if abs(impact) < 1e-12:
            continue
        value = raw_values.iloc[0][raw_name]
        rows.append({
            "Raw_Feature": raw_name,
            "SHAP_Value": float(impact),
            "Abs_SHAP": abs(float(impact)),
            "Definition": feature_definition(raw_name, dictionary),
            "Value": value,
            "Direction": "Moves toward default" if impact > 0 else "Moves away from default",
        })

    return pd.DataFrame(rows).sort_values("Abs_SHAP", ascending=False).reset_index(drop=True)


def feature_definition(feature: str, dictionary: dict[str, str] | None = None) -> str:
    name = clean_display_feature(feature)
    if name in ENGINEERED_DEFINITIONS:
        return ENGINEERED_DEFINITIONS[name]
    dictionary = dictionary or {}
    if name in dictionary:
        return dictionary[name]
    if name.endswith("_Missing"):
        raw = name[:-8]
        base = dictionary.get(raw, raw.replace("_", " "))
        return f"Availability indicator for {base}."
    for raw in sorted(dictionary, key=len, reverse=True):
        if name.startswith(raw + "_"):
            category = name[len(raw) + 1 :]
            return f"{dictionary[raw]} Encoded category: {category}."
    return name.replace("_", " ")


def explain_local(shap_df: pd.DataFrame, dictionary: dict[str, str], top_n: int = 4) -> dict[str, list[dict[str, Any]]]:
    positive = shap_df[shap_df["SHAP_Value"] > 0].head(top_n)
    negative = shap_df[shap_df["SHAP_Value"] < 0].sort_values("SHAP_Value").head(top_n)

    def pack(frame):
        rows = []
        for _, row in frame.iterrows():
            rows.append({
                "feature": clean_display_feature(str(row["Feature"])),
                "impact": float(row["SHAP_Value"]),
                "definition": feature_definition(str(row["Feature"]), dictionary),
            })
        return rows

    return {"higher_risk": pack(positive), "lower_risk": pack(negative)}


def risk_band(probability: float) -> tuple[str, float]:
    """Return a stable three-band probability label and percent position."""
    p = float(np.clip(probability, 0.0, 1.0))
    if p < 0.10:
        return "Low", p * 100.0
    if p < 0.20:
        return "Medium", p * 100.0
    return "High", p * 100.0

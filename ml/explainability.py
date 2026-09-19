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
    """Return a stable six-band probability label and percent position."""
    p = float(np.clip(probability, 0.0, 1.0))
    labels = ["Very Low", "Low", "Moderate", "Moderately High", "High", "Very High"]
    bounds = [0.02, 0.05, 0.10, 0.20, 0.35]
    for idx, boundary in enumerate(bounds):
        if p < boundary:
            return labels[idx], p * 100.0
    return labels[-1], p * 100.0

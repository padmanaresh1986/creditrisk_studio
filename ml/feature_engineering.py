
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import (
    CATEGORICAL_FEATURES_FINAL,
    ID_COLUMN,
    NUMERIC_AS_TEXT_FEATURES,
    TARGET,
)

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce")
    return numerator / denominator.replace(0, np.nan)

def detect_constant_features(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col != TARGET and df[col].nunique(dropna=False) <= 1
    ]

def clean_raw_dataframe(df: pd.DataFrame, drop_constant: bool = True) -> pd.DataFrame:
    """Apply the application's data-quality cleaning before feature engineering."""
    out = df.copy()

    # Numeric fields stored as strings in the source.
    numeric_to_convert = list(NUMERIC_AS_TEXT_FEATURES)
    numeric_to_convert += ["Credit_Amount", "Score_Source_3", "Application_Process_Hour"]

    for col in numeric_to_convert:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Application-specific sentinel.
    if "Employed_Days" in out.columns:
        out.loc[out["Employed_Days"] == 365243, "Employed_Days"] = np.nan

    # Constant features are detected on the raw frame.
    if drop_constant:
        constants = detect_constant_features(out)
        if constants:
            out = out.drop(columns=constants, errors="ignore")

    return out

def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Exact feature-engineering formulas used by the application."""
    df = df_clean.copy()

    # Duration features. Build each only when its source field is available.
    if "Age_Days" in df.columns:
        df["Age_Years"] = df["Age_Days"].abs() / 365.25
    if "Employed_Days" in df.columns:
        df["Employment_Years"] = df["Employed_Days"].abs() / 365.25
    if "Registration_Days" in df.columns:
        df["Registration_Years"] = df["Registration_Days"].abs() / 365.25
    if "ID_Days" in df.columns:
        df["ID_Change_Years"] = df["ID_Days"].abs() / 365.25

    # Financial ratios. Build only when all source fields are present.
    if "Credit_Amount" in df.columns:
        df["Credit_Amount"] = pd.to_numeric(df["Credit_Amount"], errors="coerce")
    if "Credit_Amount" in df.columns and "Client_Income" in df.columns:
        df["Loan_to_Income_Ratio"] = safe_divide(df["Credit_Amount"], df["Client_Income"])
    if "Loan_Annuity" in df.columns and "Client_Income" in df.columns:
        df["Annuity_to_Income_Ratio"] = safe_divide(df["Loan_Annuity"], df["Client_Income"])
    if "Credit_Amount" in df.columns and "Client_Family_Members" in df.columns:
        df["Credit_per_Family_Member"] = safe_divide(df["Credit_Amount"], df["Client_Family_Members"])
    if "Child_Count" in df.columns and "Client_Family_Members" in df.columns:
        df["Children_to_Family_Ratio"] = safe_divide(df["Child_Count"], df["Client_Family_Members"])

    # Score aggregations.
    score_columns = [c for c in ["Score_Source_1", "Score_Source_2", "Score_Source_3"] if c in df.columns]
    for c in score_columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if score_columns:
        df["Average_Credit_Score"] = df[score_columns].mean(axis=1)
        df["Min_Credit_Score"] = df[score_columns].min(axis=1)
        df["Max_Credit_Score"] = df[score_columns].max(axis=1)
        df["Credit_Score_Range"] = df[score_columns].max(axis=1) - df[score_columns].min(axis=1)
        df["Available_Credit_Scores"] = df[score_columns].notna().sum(axis=1)

    # Missingness indicators.
    for col in [
        "Score_Source_1",
        "Score_Source_3",
        "Client_Occupation",
        "Credit_Bureau",
        "Social_Circle_Default",
    ]:
        if col in df.columns:
            df[f"{col}_Missing"] = df[col].isna().astype(int)

    # Cyclical application hour.
    if "Application_Process_Hour" in df.columns:
        df["Application_Hour_Sin"] = np.sin(
            2 * np.pi * df["Application_Process_Hour"] / 24
        )
        df["Application_Hour_Cos"] = np.cos(
            2 * np.pi * df["Application_Process_Hour"] / 24
        )

    # Log transforms.
    for col in ["Client_Income", "Credit_Amount", "Loan_Annuity"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_Log"] = np.log1p(values.clip(lower=0))

    return df

def prepare_training_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Clean + engineer, then separate target/ID exactly as the application does."""
    cleaned = clean_raw_dataframe(df, drop_constant=True)
    engineered = engineer_features(cleaned)

    if TARGET not in engineered.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    X = engineered.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    y = engineered[TARGET].astype(int)
    audit = {
        "raw_rows": int(len(df)),
        "raw_columns": int(len(df.columns)),
        "clean_columns": int(len(cleaned.columns)),
        "engineered_columns": int(len(engineered.columns)),
        "predictor_columns": int(X.shape[1]),
        "constant_features_removed": [
            c for c in detect_constant_features(df) if c != TARGET
        ],
    }
    return X, y, audit

def raw_required_columns() -> list[str]:
    """Raw predictor schema excluding the ID and target."""
    # Imported here to avoid a duplicated hard-coded list.
    return [
        "Client_Income",
        "Car_Owned",
        "Bike_Owned",
        "Active_Loan",
        "House_Own",
        "Child_Count",
        "Credit_Amount",
        "Loan_Annuity",
        "Accompany_Client",
        "Client_Income_Type",
        "Client_Education",
        "Client_Marital_Status",
        "Client_Gender",
        "Loan_Contract_Type",
        "Client_Housing_Type",
        "Population_Region_Relative",
        "Age_Days",
        "Employed_Days",
        "Registration_Days",
        "ID_Days",
        "Own_House_Age",
        "Homephone_Tag",
        "Workphone_Working",
        "Client_Occupation",
        "Client_Family_Members",
        "Cleint_City_Rating",
        "Application_Process_Day",
        "Application_Process_Hour",
        "Client_Permanent_Match_Tag",
        "Client_Contact_Work_Tag",
        "Type_Organization",
        "Score_Source_1",
        "Score_Source_2",
        "Score_Source_3",
        "Social_Circle_Default",
        "Phone_Change",
        "Credit_Bureau",
    ]

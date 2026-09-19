import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.feature_engineering import clean_raw_dataframe, engineer_features, prepare_training_matrix
from ml.explainability import risk_band


def sample_raw_df():
    return pd.DataFrame({
        "ID": [1, 2],
        "Default": [0, 1],
        "Client_Income": [750000, 500000],
        "Credit_Amount": [1000000, 750000],
        "Loan_Annuity": [50000, 45000],
        "Age_Days": [-12783, -14610],
        "Employed_Days": [-2922, 365243],
        "Registration_Days": [-1800, -2200],
        "ID_Days": [-1000, -1200],
        "Child_Count": [0, 2],
        "Client_Family_Members": [2, 4],
        "Score_Source_1": [np.nan, 0.2],
        "Score_Source_2": [0.7, 0.3],
        "Score_Source_3": [0.6, 0.4],
        "Client_Occupation": ["manager", "sales"],
        "Credit_Bureau": [2, 5],
        "Social_Circle_Default": [0, 1],
        "Application_Process_Hour": [10, 23],
    })


def test_sentinel_is_missing_before_duration_engineering():
    df = pd.DataFrame({"Employed_Days": [365243, -365]})
    clean = clean_raw_dataframe(df, drop_constant=False)
    assert pd.isna(clean.loc[0, "Employed_Days"])
    engineered = engineer_features(clean)
    assert pd.isna(engineered.loc[0, "Employment_Years"])
    assert abs(engineered.loc[1, "Employment_Years"] - (365 / 365.25)) < 1e-9


def test_engineering_creates_expected_features():
    engineered = engineer_features(clean_raw_dataframe(sample_raw_df(), drop_constant=False))
    for col in [
        "Loan_to_Income_Ratio",
        "Annuity_to_Income_Ratio",
        "Average_Credit_Score",
        "Score_Source_1_Missing",
        "Application_Hour_Sin",
        "Application_Hour_Cos",
        "Client_Income_Log",
    ]:
        assert col in engineered.columns


def test_training_matrix_separates_target_and_identifier():
    X, y, audit = prepare_training_matrix(sample_raw_df())
    assert "Default" not in X.columns
    assert "ID" not in X.columns
    assert len(y) == 2
    assert audit["predictor_columns"] == X.shape[1]


def test_risk_band_has_six_stable_probability_bands():
    band, score = risk_band(0.188)
    assert band == "Moderately High"
    assert 0 <= score <= 100

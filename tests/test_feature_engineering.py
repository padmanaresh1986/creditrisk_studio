import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.feature_engineering import clean_raw_dataframe, engineer_features, prepare_training_matrix
from ml.explainability import risk_band, aggregate_input_contributions


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


def test_risk_band_has_three_stable_probability_bands():
    cases = [
        (0.0, "Low"),
        (0.249, "Low"),
        (0.25, "Medium"),
        (0.499, "Medium"),
        (0.50, "High"),
        (1.0, "High"),
    ]
    for probability, expected_band in cases:
        band, score = risk_band(probability)
        assert band == expected_band
        assert score == probability * 100


def test_application_hour_mixed_excel_types_are_numeric_before_cyclical_encoding():
    raw = pd.DataFrame({
        "Application_Process_Hour": ["10", np.float64(23.0)],
        "Age_Days": [-(35 * 365.25), -(40 * 365.25)],
    })
    clean = clean_raw_dataframe(raw, drop_constant=False)
    assert pd.api.types.is_numeric_dtype(clean["Application_Process_Hour"])
    engineered = engineer_features(clean)
    assert np.isfinite(engineered["Application_Hour_Sin"]).all()
    assert np.isfinite(engineered["Application_Hour_Cos"]).all()


def test_raw_input_contribution_aggregation_maps_engineered_features_to_entered_fields():
    shap = pd.DataFrame({
        "Feature": ["numeric__Loan_to_Income_Ratio", "numeric__Client_Income_Log", "categorical__Bike_Owned_1.0"],
        "SHAP_Value": [0.030, -0.010, 0.005],
        "Abs_SHAP": [0.030, 0.010, 0.005],
        "Transformed_Value": [0.1, 13.0, 1.0],
    })
    raw = pd.DataFrame({
        "Client_Income": [750000],
        "Credit_Amount": [1000000],
        "Bike_Owned": [1.0],
    })
    out = aggregate_input_contributions(shap, raw, {}, {"Client_Income", "Credit_Amount", "Bike_Owned"})
    contrib = dict(zip(out["Raw_Feature"], out["SHAP_Value"]))
    assert abs(contrib["Client_Income"] - 0.005) < 1e-9
    assert abs(contrib["Credit_Amount"] - 0.015) < 1e-9
    assert abs(contrib["Bike_Owned"] - 0.005) < 1e-9


def test_raw_input_contribution_aggregation_accepts_legacy_positional_order():
    shap = pd.DataFrame({
        "Feature": ["numeric__Client_Income_Log"],
        "SHAP_Value": [0.010],
        "Abs_SHAP": [0.010],
        "Transformed_Value": [13.0],
    })
    raw = pd.DataFrame({"Client_Income": [750000]})
    dictionary = {"Client_Income": "Client income in $."}
    # Historical call order was (shap_df, raw_values, supplied_fields, dictionary).
    out = aggregate_input_contributions(shap, raw, {"Client_Income"}, dictionary)
    assert len(out) == 1
    assert out.iloc[0]["Raw_Feature"] == "Client_Income"

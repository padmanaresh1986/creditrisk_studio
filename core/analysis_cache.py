from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    class _Stub:
        @staticmethod
        def cache_data(**kwargs):
            return lambda fn: fn
    st = _Stub()

from ml.eda import target_distribution, missingness_table, numeric_histogram, boxplot_figure, categorical_default_rate, missingness_vs_default, correlation_figure, feature_selection_tables
from ml.feature_engineering import clean_raw_dataframe, engineer_features, detect_constant_features


@st.cache_data(show_spinner=False)
def cached_understanding(df: pd.DataFrame):
    summary = pd.DataFrame({
        "Feature": df.columns,
        "Raw dtype": df.dtypes.astype(str).values,
        "Unique values": [df[c].nunique(dropna=True) for c in df.columns],
        "Missing %": (df.isna().mean() * 100).round(2).values,
    })
    miss = missingness_table(df).reset_index().rename(columns={"index": "Feature"})
    constants = detect_constant_features(df)
    td = target_distribution(df)
    return {"summary": summary, "missing": miss, "constants": constants, "target": td}


@st.cache_data(show_spinner=False)
def cached_eda(df: pd.DataFrame):
    clean = clean_raw_dataframe(df, drop_constant=True)
    eng = engineer_features(clean)
    hist = numeric_histogram(df)
    box = boxplot_figure(df)
    missing_default = missingness_vs_default(df)
    corr_fig, corr = correlation_figure(df)
    chi, mi = feature_selection_tables(df)
    categorical_choices = [
        c for c in [
            "Client_Education", "Client_Income_Type", "Client_Marital_Status",
            "Client_Gender", "Loan_Contract_Type", "Client_Housing_Type", "Cleint_City_Rating"
        ] if c in df.columns
    ]
    cat_figs = {c: categorical_default_rate(df, c) for c in categorical_choices}
    new_features = [c for c in eng.columns if c not in clean.columns]
    return {
        "hist": hist,
        "box": box,
        "missing_default": missing_default,
        "corr_fig": corr_fig,
        "corr": corr,
        "chi": chi,
        "mi": mi,
        "categorical_choices": categorical_choices,
        "cat_figs": cat_figs,
        "clean": clean,
        "engineered": eng,
        "new_features": new_features,
    }

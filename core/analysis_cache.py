from __future__ import annotations

import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    class _Stub:
        @staticmethod
        def cache_data(**kwargs):
            return lambda fn: fn
    st = _Stub()


def _df_cache_key(df: pd.DataFrame) -> str:
    """Return a stable lightweight cache key stored on project dataframes.

    Streamlit normally hashes every cell of a DataFrame when it is passed to
    ``st.cache_data``. For the 6,000x40 training extract that repeated hashing
    is unnecessary because project creation already computes a content fingerprint.
    """
    key = df.attrs.get("_creditrisk_cache_key")
    if key:
        return str(key)
    return f"fallback:{len(df)}:{len(df.columns)}:{','.join(map(str, df.columns))}"


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _df_cache_key})
def cached_understanding(df: pd.DataFrame):
    from ml.eda import target_distribution, missingness_table
    from ml.feature_engineering import detect_constant_features
    summary = pd.DataFrame({
        "Feature": df.columns,
        "Raw dtype": df.dtypes.astype(str).values,
        "Unique values": [df[c].nunique(dropna=True) for c in df.columns],
        "Missing %": (df.isna().mean() * 100).round(2).values,
    })
    miss = missingness_table(df).reset_index().rename(columns={"index": "Feature"})
    return {
        "summary": summary,
        "missing": miss,
        "constants": detect_constant_features(df),
        "target": target_distribution(df),
    }


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _df_cache_key})
def cached_eda_base(df: pd.DataFrame):
    """Cache only the cleaned/engineered data needed by EDA. Plotly figures are built on demand."""
    from ml.feature_engineering import clean_raw_dataframe, engineer_features
    clean = clean_raw_dataframe(df, drop_constant=True)
    eng = engineer_features(clean)
    categorical_choices = [
        c for c in [
            "Client_Education", "Client_Income_Type", "Client_Marital_Status",
            "Client_Gender", "Loan_Contract_Type", "Client_Housing_Type", "Cleint_City_Rating"
        ] if c in clean.columns
    ]
    return {
        "clean": clean,
        "engineered": eng,
        "categorical_choices": categorical_choices,
        "new_features": [c for c in eng.columns if c not in clean.columns],
    }


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _df_cache_key})
def cached_feature_selection(df: pd.DataFrame):
    from ml.eda import feature_selection_tables
    from ml.feature_engineering import clean_raw_dataframe
    clean = clean_raw_dataframe(df, drop_constant=True)
    return feature_selection_tables(clean=clean)

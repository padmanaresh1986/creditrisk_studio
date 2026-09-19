from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

from .feature_engineering import clean_raw_dataframe, engineer_features
from .schema import (
    BINARY_CATEGORICAL_FEATURES,
    NOMINAL_CATEGORICAL_FEATURES,
    ORDINAL_CATEGORICAL_FEATURES,
    RANDOM_STATE,
    TARGET,
)


def cleaned_engineered(df):
    return engineer_features(clean_raw_dataframe(df, drop_constant=True))


def target_distribution(df):
    counts = df[TARGET].value_counts().sort_index()
    return pd.DataFrame({
        "Class": ["Non-default", "Default"],
        "Count": [int(counts.get(0, 0)), int(counts.get(1, 0))],
    })


def missingness_table(df):
    out = df.isna().sum().sort_values(ascending=False).to_frame("Missing_Count")
    out["Missing_Percentage"] = (out["Missing_Count"] / len(df) * 100).round(2)
    return out


def missingness_figure(df):
    miss = missingness_table(df).head(15).reset_index().rename(columns={"index": "Feature"})
    return px.bar(miss.sort_values("Missing_Percentage"), x="Missing_Percentage", y="Feature", orientation="h",
                  title="Top Features by Missing Value Percentage", labels={"Missing_Percentage": "Missing Values (%)"})


def numeric_histogram(df):
    clean = cleaned_engineered(df)
    selected = [c for c in [
        "Client_Income", "Credit_Amount", "Loan_Annuity", "Age_Years", "Employment_Years",
        "Score_Source_1", "Score_Source_2", "Score_Source_3", "Credit_Bureau", "Social_Circle_Default"
    ] if c in clean.columns]
    fig = make_subplots(rows=5, cols=2, subplot_titles=selected, vertical_spacing=0.09)
    for idx, col in enumerate(selected):
        r, c = divmod(idx, 2)
        series = pd.to_numeric(clean[col], errors="coerce").dropna()
        fig.add_trace(go.Histogram(x=series, nbinsx=30, name=col, showlegend=False), row=r+1, col=c+1)
    fig.update_layout(height=1150, title_text="Distribution of Important Numerical Features")
    return fig


def boxplot_figure(df):
    clean = cleaned_engineered(df)
    selected = [c for c in ["Client_Income", "Credit_Amount", "Loan_Annuity", "Age_Years", "Employment_Years"] if c in clean.columns]
    fig = make_subplots(rows=len(selected), cols=1, shared_xaxes=False, subplot_titles=[f"Boxplot: {c}" for c in selected])
    for i, col in enumerate(selected, start=1):
        fig.add_trace(go.Box(x=pd.to_numeric(clean[col], errors="coerce"), name=col, showlegend=False), row=i, col=1)
    fig.update_layout(height=max(360, 250 * len(selected)), title_text="Boxplots of Selected Numerical Features")
    return fig


def categorical_default_rate(df, col):
    clean = cleaned_engineered(df)
    rate = clean.groupby(col, dropna=False)[TARGET].mean().mul(100).sort_values(ascending=False).reset_index(name="Default Rate (%)")
    rate[col] = rate[col].fillna("Missing").astype(str)
    return px.bar(rate.sort_values("Default Rate (%)"), x="Default Rate (%)", y=col, orientation="h",
                  title=f"Default Rate by {col}", text_auto=".1f")


def missingness_vs_default(df):
    clean = cleaned_engineered(df)
    rows=[]
    for c in ["Score_Source_1", "Score_Source_3", "Client_Occupation", "Credit_Bureau", "Social_Circle_Default"]:
        if c not in clean.columns: continue
        group = clean.groupby(clean[c].isna())[TARGET].mean().mul(100)
        rows += [{"Feature": c, "Missing": "Present" if not flag else "Missing", "Default Rate (%)": float(val)} for flag,val in group.items()]
    return px.bar(pd.DataFrame(rows), x="Feature", y="Default Rate (%)", color="Missing", barmode="group",
                  title="Default Rate by Missingness Status")


def correlation_figure(df):
    clean = cleaned_engineered(df)
    numeric = clean.select_dtypes(include=np.number).drop(columns=[TARGET, "ID"], errors="ignore")
    corr = numeric.corr(method="spearman")
    fig = px.imshow(corr, text_auto=False, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Spearman Correlation Matrix")
    fig.update_layout(height=900)
    return fig, corr


def feature_selection_tables(df):
    clean = cleaned_engineered(df)
    categorical = [c for c in (BINARY_CATEGORICAL_FEATURES + NOMINAL_CATEGORICAL_FEATURES + ORDINAL_CATEGORICAL_FEATURES) if c in clean.columns]
    chi_rows=[]
    for col in categorical:
        table=pd.crosstab(clean[col].fillna("Missing"), clean[TARGET])
        chi2,p,dof,_=chi2_contingency(table)
        chi_rows.append({"Feature":col,"Chi2":chi2,"P_Value":p,"Degrees_of_Freedom":dof})
    chi_df=pd.DataFrame(chi_rows).sort_values("P_Value")

    mi_features=[c for c in [
        "Client_Income", "Credit_Amount", "Loan_Annuity", "Population_Region_Relative", "Age_Days", "Employed_Days",
        "Registration_Days", "ID_Days", "Own_House_Age", "Phone_Change", "Credit_Bureau", "Social_Circle_Default",
        "Age_Years", "Employment_Years", "Registration_Years", "ID_Change_Years", "Loan_to_Income_Ratio",
        "Annuity_to_Income_Ratio", "Credit_per_Family_Member", "Children_to_Family_Ratio", "Average_Credit_Score",
        "Credit_Score_Range", "Available_Credit_Scores"
    ] if c in clean.columns]
    mi_data=clean[mi_features].apply(pd.to_numeric,errors="coerce")
    mi_data=mi_data.fillna(mi_data.median(numeric_only=True))
    mi_scores=mutual_info_classif(mi_data, clean[TARGET], random_state=RANDOM_STATE)
    mi_df=pd.DataFrame({"Feature":mi_features,"Mutual_Information":mi_scores}).sort_values("Mutual_Information",ascending=False)
    return chi_df, mi_df

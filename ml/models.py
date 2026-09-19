from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor
from .schema import RANDOM_STATE

@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    description: str
    strengths: str
    best_params: dict[str, Any]
    tuning_grid: dict[str, list[Any]]


MODEL_SPECS: dict[str, ModelSpec] = {
    "Random Forest": ModelSpec(
        name="Random Forest",
        family="Bagging / tree ensemble",
        description="An ensemble of randomized decision trees that can capture nonlinear relationships and interactions.",
        strengths="Useful for mixed tabular data and offers direct feature-importance diagnostics.",
        best_params={
            "model__n_estimators": 400,
            "model__max_depth": None,
            "model__max_features": "log2",
            "model__min_samples_leaf": 5,
        },
        tuning_grid={
            "model__n_estimators": [300, 400],
            "model__max_features": ["log2"],
            "model__min_samples_leaf": [5],
            "model__max_depth": [None],
        },
    ),
    "XGBoost": ModelSpec(
        name="XGBoost",
        family="Gradient boosting",
        description="Sequentially builds decision trees so later trees focus on errors made earlier.",
        strengths="Captures nonlinear interactions with a flexible boosting objective for tabular data.",
        best_params={
            "model__n_estimators": 200,
            "model__learning_rate": 0.03,
            "model__max_depth": 4,
            "model__subsample": 0.8,
            "model__colsample_bytree": 0.8,
        },
        tuning_grid={
            "model__n_estimators": [150, 200],
            "model__learning_rate": [0.03],
            "model__max_depth": [3, 4],
            "model__subsample": [0.8],
            "model__colsample_bytree": [0.8],
        },
    ),
    "L1 Logistic Regression": ModelSpec(
        name="L1 Logistic Regression",
        family="Linear / sparse",
        description="A linear classifier with L1 regularization that can shrink weak coefficients toward zero.",
        strengths="Compact and comparatively easy to inspect through coefficient magnitude.",
        best_params={"model__C": 0.01},
        tuning_grid={"model__C": [0.005, 0.01, 0.05, 0.1]},
    ),
}


def build_model_pipeline(model_name: str, X_train, y_train):
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {model_name}")
    preprocessor, numeric, categorical = build_preprocessor(X_train)
    if model_name == "Random Forest":
        estimator = RandomForestClassifier(
            n_estimators=400, max_depth=None, max_features="log2", min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
        )
    elif model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is required for the XGBoost candidate.") from exc
        ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        estimator = XGBClassifier(
            n_estimators=200, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE, scale_pos_weight=ratio,
        )
    else:
        estimator = LogisticRegression(
            penalty="l1", solver="liblinear", C=0.01, class_weight="balanced",
            max_iter=2000, random_state=RANDOM_STATE,
        )
    return Pipeline([("preprocessing", preprocessor), ("model", estimator)]), numeric, categorical

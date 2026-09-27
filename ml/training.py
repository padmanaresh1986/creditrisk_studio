from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline

from .evaluation import select_f1_threshold
from .feature_engineering import clean_raw_dataframe, engineer_features, prepare_training_matrix
from .models import MODEL_SPECS, build_model_pipeline
from .schema import RANDOM_STATE, TARGET


ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "model"


def _resolve_artifact_root(artifact_root=None) -> Path:
    return Path(artifact_root) if artifact_root else ARTIFACT_ROOT


def _save_dataframe(project_root: Path | None, name: str, df: pd.DataFrame) -> None:
    if project_root is None:
        return
    analysis_dir = Path(project_root) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(analysis_dir / f"{name}.csv", index=False)


def dataset_fingerprint(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _series_default(series: pd.Series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.80:
        return float(numeric.median())
    mode = series.dropna().mode()
    return mode.iloc[0] if len(mode) else None


def _build_input_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Precompute lightweight prediction-form metadata so User pages do not re-read CSVs."""
    from .feature_engineering import raw_required_columns
    from .schema import CATEGORICAL_FEATURES_FINAL
    profile = {"columns": {}}
    for col in raw_required_columns():
        if col not in df.columns:
            continue
        series = df[col]
        if col in CATEGORICAL_FEATURES_FINAL:
            vals = [v for v in pd.unique(series.dropna())][:100]
            profile["columns"][col] = {"kind": "categorical", "options": vals}
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            info = {"kind": "numeric"}
            if numeric.notna().any():
                info.update({"min": float(numeric.min()), "max": float(numeric.max()), "median": float(numeric.median())})
            profile["columns"][col] = info
    return profile


def build_context(df: pd.DataFrame, feature_columns: list[str], base_rate: float, dictionary: dict[str, str]) -> dict[str, Any]:
    cleaned = clean_raw_dataframe(df, drop_constant=True)
    raw = cleaned.drop(columns=[TARGET, "ID"], errors="ignore")
    raw_defaults = {c: _series_default(raw[c]) for c in raw.columns}
    return {
        "feature_columns": feature_columns,
        "raw_feature_defaults": raw_defaults,
        "base_default_rate": float(base_rate),
        "dictionary": dictionary,
        "raw_input_columns": raw.columns.tolist(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_profile": _build_input_profile(df),
    }


def split_training_data(df: pd.DataFrame):
    X, y, audit = prepare_training_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    audit.update({
        "train_rows": int(len(X_train)), "holdout_rows": int(len(X_test)),
        "train_default_rate": float(y_train.mean()), "holdout_default_rate": float(y_test.mean()),
    })
    return X_train, X_test, y_train, y_test, audit


def _safe_version(name: str) -> str:
    return name.lower().replace(" ", "-").replace("/", "-")


def _save_context(path: Path, context: dict[str, Any]) -> str:
    out = path.with_suffix(".json")
    out.write_text(json.dumps(context, indent=2, default=str), encoding="utf-8")
    return str(out)


def train_candidates(df: pd.DataFrame, dictionary: dict[str, str], log=None, progress=None, artifact_root=None, project_id: str = "", project_name: str = "", project_root=None, model_names: list[str] | None = None):
    log = log or (lambda _msg: None)
    progress = progress or (lambda _value, _text: None)
    X_train, X_test, y_train, y_test, audit = split_training_data(df)
    fp = dataset_fingerprint(df)
    artifact_root = _resolve_artifact_root(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    candidates_dir = artifact_root / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    names = list(model_names) if model_names is not None else list(MODEL_SPECS)
    names = [name for name in names if name in MODEL_SPECS]
    if not names:
        raise ValueError("Select at least one supported model to train.")
    total_models = len(names)
    # Compute project-level prediction metadata once. Previously this cleaned the full
    # training dataframe and rebuilt the input profile once per candidate model.
    base_context = build_context(df, X_train.columns.tolist(), float(y_train.mean()), dictionary)
    log(f"Prepared {X_train.shape[1]} model features from {len(df):,} labelled rows.")
    log(f"Training partition: {len(X_train):,} rows; holdout partition: {len(X_test):,} rows.")
    log(f"Holdout target rate is {y_test.mean():.2%}. The holdout will not be used for tuning or threshold selection.")
    for i, name in enumerate(names, 1):
        progress((i-1)/total_models, f"Training {name}")
        log(f"Training {name} ({i}/{len(names)})…")
        pipeline, numeric, categorical = build_model_pipeline(name, X_train, y_train)
        started = datetime.now(timezone.utc)
        pipeline.fit(X_train, y_train)
        transformed_count = len(pipeline.named_steps["preprocessing"].get_feature_names_out())
        version = f"candidate-{_safe_version(name)}-{started.strftime('%Y%m%d%H%M%S%f')}-{uuid4().hex[:6]}"
        model_path = candidates_dir / f"{version}.joblib"
        joblib.dump(pipeline, model_path)
        context = dict(base_context)
        context["model_name"] = name
        context["version"] = version
        context["project_id"] = project_id
        context["project_name"] = project_name
        context["numeric_features"] = numeric
        context["categorical_features"] = categorical
        context["transformed_feature_count"] = transformed_count
        context["params"] = pipeline.named_steps["model"].get_params()
        context_path = _save_context(model_path, context)
        results[name] = {
            "model_name": name, "family": MODEL_SPECS[name].family,
            "description": MODEL_SPECS[name].description, "strengths": MODEL_SPECS[name].strengths,
            "pipeline": pipeline, "artifact_path": str(model_path), "context_path": context_path,
            "version": version, "dataset_fingerprint": fp,
            "params": {k.replace("model__", ""): v for k, v in pipeline.named_steps["model"].get_params().items() if k in MODEL_SPECS[name].best_params},
            "feature_count": int(X_train.shape[1]), "transformed_feature_count": int(transformed_count),
            "trained_at_utc": started.isoformat(), "base_default_rate": float(y_train.mean()),
        }
        progress(i/total_models, f"Completed {name}")
        log(f"Completed {name}; artifact saved.")
    return {"models": results, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "audit": audit, "dataset_fingerprint": fp, "dictionary": dictionary, "project_id": project_id, "project_name": project_name, "project_root": str(project_root) if project_root else "", "artifact_root": str(artifact_root)}


def tune_candidates(training_bundle: dict, n_splits: int = 3, log=None, progress=None):
    log = log or (lambda _msg: None)
    progress = progress or (lambda _value, _text: None)
    X_train, y_train = training_bundle["X_train"], training_bundle["y_train"]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    tuned = {}
    model_items = list(training_bundle["models"].items())
    total_models = max(len(model_items), 1)
    for i, (name, info) in enumerate(model_items, 1):
        spec = MODEL_SPECS[name]
        log(f"Tuning {name} with {n_splits}-fold CV and a compact search space…")
        progress((i-1)/total_models, f"Tuning {name}")
        search = GridSearchCV(
            estimator=clone(info["pipeline"]), param_grid=spec.tuning_grid,
            scoring="average_precision", cv=cv, n_jobs=1, refit=True, return_train_score=False,
        )
        started = datetime.now(timezone.utc)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        info["pipeline"] = best
        info["params"] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        info["best_search_pr_auc"] = float(search.best_score_)
        info["tuning_duration_seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
        info["tuning_results"] = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(15)[["params","mean_test_score","std_test_score","rank_test_score"]]
        # Overwrite candidate artifact/context with tuned estimator.
        joblib.dump(best, info["artifact_path"])
        context = json.loads(Path(info["context_path"]).read_text(encoding="utf-8"))
        context["params"] = best.named_steps["model"].get_params()
        context["tuned"] = True
        Path(info["context_path"]).write_text(json.dumps(context, indent=2, default=str), encoding="utf-8")
        tuned[name] = info
        progress(i/total_models, f"Tuning complete: {name}")
        log(f"Best {name} search PR-AUC: {search.best_score_:.4f}; params={search.best_params_}.")
    training_bundle["models"] = tuned
    training_bundle["tuned"] = True
    return training_bundle


def cross_validate_candidates(training_bundle: dict, n_splits: int = 5, log=None, progress=None):
    log = log or (lambda _msg: None)
    progress = progress or (lambda _value, _text: None)
    X_train, y_train = training_bundle["X_train"], training_bundle["y_train"]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows, folds, oof = [], [], {}
    total = len(training_bundle["models"]) * n_splits
    done = 0
    for name, info in training_bundle["models"].items():
        probs = np.zeros(len(X_train), dtype=float)
        fold_rows=[]
        log(f"Starting {n_splits}-fold cross-validation for {name}…")
        for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), 1):
            model = clone(info["pipeline"])
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            p = model.predict_proba(X_train.iloc[va_idx])[:, 1]
            probs[va_idx] = p
            yv = y_train.iloc[va_idx]
            pred = (p >= 0.50).astype(int)
            row = {
                "Model": name, "Fold": fold,
                "Accuracy": accuracy_score(yv,pred), "Precision": precision_score(yv,pred,zero_division=0),
                "Recall": recall_score(yv,pred,zero_division=0), "F1": f1_score(yv,pred,zero_division=0),
                "ROC-AUC": roc_auc_score(yv,p), "PR-AUC": average_precision_score(yv,p),
            }
            fold_rows.append(row); folds.append(row); done += 1
            progress(done/total, f"Cross-validation • {name} • fold {fold}/{n_splits}")
            log(f"  Fold {fold}/{n_splits}: PR-AUC={row['PR-AUC']:.4f}; Recall={row['Recall']:.1%}; F1={row['F1']:.4f}.")
        fd = pd.DataFrame(fold_rows)
        oof[name]=probs
        rows.append({
            "Model":name, "Accuracy":fd["Accuracy"].mean(), "Precision":fd["Precision"].mean(),
            "Recall":fd["Recall"].mean(), "F1":fd["F1"].mean(), "ROC-AUC":fd["ROC-AUC"].mean(),
            "PR-AUC":fd["PR-AUC"].mean(), "PR-AUC STD":fd["PR-AUC"].std(ddof=0),
        })
    cv_df=pd.DataFrame(rows)
    project_root = Path(training_bundle.get("project_root")) if training_bundle.get("project_root") else None
    _save_dataframe(project_root, "cross_validation_results", cv_df)
    _save_dataframe(project_root, "cross_validation_folds", pd.DataFrame(folds))
    return {"cv_results":cv_df, "fold_results":pd.DataFrame(folds), "oof_predictions":oof, "n_splits":n_splits, "project_root": training_bundle.get("project_root", ""), "project_id": training_bundle.get("project_id", "")}


def class_imbalance_comparison(training_bundle: dict, n_splits: int = 5, log=None, progress=None):
    log = log or (lambda _msg: None)
    progress = progress or (lambda _value, _text: None)
    X_train, y_train = training_bundle["X_train"], training_bundle["y_train"]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    out=[]
    model_items = list(training_bundle["models"].items())
    total_models = max(len(model_items), 1)
    for i,(name,info) in enumerate(model_items,1):
        balanced=clone(info["pipeline"]); standard=clone(info["pipeline"])
        est=standard.named_steps["model"]
        if hasattr(est,"class_weight"): est.set_params(class_weight=None)
        if hasattr(est,"scale_pos_weight"): est.set_params(scale_pos_weight=1.0)
        for strategy,pipe in [("Balanced",balanced),("Standard",standard)]:
            from sklearn.model_selection import cross_validate
            scores=cross_validate(pipe,X_train,y_train,cv=cv,scoring={"pr":"average_precision","rec":"recall"},n_jobs=1)
            out.append({"Model":name,"Strategy":strategy,"CV PR-AUC":float(scores["test_pr"].mean()),"CV Recall @ 0.50":float(scores["test_rec"].mean())})
        progress(i/total_models,f"Imbalance comparison • {name}")
        log(f"Imbalance comparison complete for {name}.")
    out_df = pd.DataFrame(out)
    project_root = Path(training_bundle.get("project_root")) if training_bundle.get("project_root") else None
    _save_dataframe(project_root, "imbalance_comparison", out_df)
    return out_df


def optimize_thresholds(cv_bundle: dict, y_train, log=None):
    log = log or (lambda _msg: None)
    best_rows=[]; curves={}
    for name, probs in cv_bundle["oof_predictions"].items():
        threshold, curve, best = select_f1_threshold(y_train, probs)
        curves[name]=curve
        best_rows.append({"Model":name,"Threshold":float(threshold),"Precision":float(best["Precision"]),"Recall":float(best["Recall"]),"F1":float(best["F1"])})
        log(f"Selected threshold for {name} from training OOF probabilities: {threshold:.2f}.")
    best_df = pd.DataFrame(best_rows)
    project_root = Path(cv_bundle.get("project_root")) if cv_bundle.get("project_root") else None
    _save_dataframe(project_root, "threshold_summary", best_df)
    return {"best_thresholds":best_df,"curves":curves, "project_root": str(project_root) if project_root else ""}


def final_evaluate_all(training_bundle: dict, threshold_bundle: dict, log=None, progress=None, artifact_root=None):
    log = log or (lambda _msg: None)
    progress = progress or (lambda _value, _text: None)
    X_train,X_test=training_bundle["X_train"],training_bundle["X_test"]
    y_train,y_test=training_bundle["y_train"],training_bundle["y_test"]
    artifact_root = _resolve_artifact_root(artifact_root or training_bundle.get("artifact_root"))
    final_dir=artifact_root/"releases"
    final_dir.mkdir(parents=True,exist_ok=True)
    outputs={}
    thresholds=threshold_bundle["best_thresholds"].set_index("Model")
    model_items = list(training_bundle["models"].items())
    total_models = max(len(model_items), 1)
    for i,(name,info) in enumerate(model_items,1):
        threshold=float(thresholds.loc[name,"Threshold"])
        log(f"Fitting {name} on the complete training partition…")
        model=clone(info["pipeline"])
        started=datetime.now(timezone.utc)
        model.fit(X_train,y_train)
        probs=model.predict_proba(X_test)[:,1]
        metric_rows=[]
        for t in [0.50,threshold]:
            pred=(probs>=t).astype(int)
            metric_rows.append({"Threshold":float(t),"Accuracy":accuracy_score(y_test,pred),"Precision":precision_score(y_test,pred,zero_division=0),"Recall":recall_score(y_test,pred,zero_division=0),"F1":f1_score(y_test,pred,zero_division=0),"ROC-AUC":roc_auc_score(y_test,probs),"PR-AUC":average_precision_score(y_test,probs)})
        eval_df=pd.DataFrame(metric_rows)
        pred=(probs>=threshold).astype(int)
        tn,fp,fn,tp=confusion_matrix(y_test,pred,labels=[0,1]).ravel()
        version=f"release-{_safe_version(name)}-{started.strftime('%Y%m%d%H%M%S%f')}-{uuid4().hex[:6]}"
        model_path=final_dir/f"{version}.joblib"; joblib.dump(model,model_path)
        context=json.loads(Path(info["context_path"]).read_text(encoding="utf-8"))
        context.update({"version":version,"threshold":threshold,"final_metrics":eval_df.iloc[-1].to_dict(),"confusion":{"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp)}})
        context_path=_save_context(model_path,context)
        outputs[name]={
            "model_name":name,"family":info["family"],"pipeline":model,"artifact_path":str(model_path),"context_path":context_path,
            "version":version,"threshold":threshold,"metrics":eval_df,"probabilities":probs,"test_y":y_test,"confusion":{"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp)},
            "final_pr_auc":float(eval_df.iloc[-1]["PR-AUC"]),"final_roc_auc":float(eval_df.iloc[-1]["ROC-AUC"]),"final_recall":float(eval_df.iloc[-1]["Recall"]),"final_precision":float(eval_df.iloc[-1]["Precision"]),"final_f1":float(eval_df.iloc[-1]["F1"]),
            "dataset_fingerprint":training_bundle["dataset_fingerprint"],"base_default_rate":float(training_bundle["y_train"].mean()),"params":info["params"],"trained_at_utc":started.isoformat(),
            "project_id":training_bundle.get("project_id", ""),"project_name":training_bundle.get("project_name", ""),
        }
        info.update({"pipeline":model,"artifact_path":str(model_path),"context_path":context_path,"version":version,"selected_threshold":threshold,"final_metrics":outputs[name]})
        progress(i/total_models,f"Holdout evaluation • {name}")
        log(f"{name}: holdout PR-AUC={outputs[name]['final_pr_auc']:.4f}; ROC-AUC={outputs[name]['final_roc_auc']:.4f}; Recall={outputs[name]['final_recall']:.1%}; F1={outputs[name]['final_f1']:.4f}.")
    final_rows = []
    for name, out in outputs.items():
        row = out["metrics"].iloc[-1].to_dict()
        row.update({"Model": name, "Threshold": out["threshold"], "TN": out["confusion"]["TN"], "FP": out["confusion"]["FP"], "FN": out["confusion"]["FN"], "TP": out["confusion"]["TP"]})
        final_rows.append(row)
    _save_dataframe(Path(training_bundle.get("project_root")) if training_bundle.get("project_root") else None, "final_metrics", pd.DataFrame(final_rows))
    return outputs


def run_full_lifecycle(df: pd.DataFrame, dictionary: dict[str,str], config: dict[str,Any], log=None, progress=None):
    log=log or (lambda _msg: None); progress=progress or (lambda _v,_t:None)
    include_tuning=bool(config.get("include_tuning", False))
    # Progress budget: training 15%, tuning 20%, CV 25%, imbalance 10%, threshold 5%, holdout 25%.
    selected_models = config.get("model_names") or list(MODEL_SPECS)
    bundle=train_candidates(df,dictionary,log,lambda p,t:progress(0.15*p,t), model_names=selected_models)
    progress(0.15,f"{len(selected_models)} selected model(s) trained")
    if include_tuning:
        bundle=tune_candidates(bundle,n_splits=int(config.get("tuning_folds",3)),log=log,progress=lambda p,t:progress(0.15+0.20*p,t))
    else:
        log("Hyperparameter tuning was skipped for this lifecycle run.")
    cv=cross_validate_candidates(bundle,n_splits=int(config.get("cv_folds",5)),log=log,progress=lambda p,t:progress(0.35+0.25*p,t))
    imb=class_imbalance_comparison(bundle,n_splits=int(config.get("cv_folds",5)),log=log,progress=lambda p,t:progress(0.60+0.10*p,t))
    threshold=optimize_thresholds(cv,bundle["y_train"],log=log)
    progress(0.75,"Threshold optimisation complete")
    final=final_evaluate_all(bundle,threshold,log=log,progress=lambda p,t:progress(0.75+0.25*p,t))
    return {"training_bundle":bundle,"cv_bundle":cv,"imbalance_df":imb,"threshold_bundle":threshold,"final_bundle":final}

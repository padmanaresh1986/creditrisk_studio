from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.ui import empty_state, render_phase_strip, section_header
from core.analysis_cache import cached_eda, cached_understanding
from core.session import add_log, clear_busy, clear_training_state, set_busy
from core.state import get_store
from core.projects import get_project_manager
from ml.feature_engineering import clean_raw_dataframe, engineer_features, detect_constant_features
from ml.models import MODEL_SPECS


store = get_store()
project_manager = get_project_manager()
PHASES = [
    ("01", "Data Setup"),
    ("02", "Understand & Quality"),
    ("03", "EDA & Feature Engineering"),
    ("04", "Preprocessing & Model Lab"),
    ("05", "Validation, Imbalance & Tuning"),
    ("06", "Threshold, Evaluation & Explainability"),
    ("07", "Model Release & User Access"),
]


def current_df() -> pd.DataFrame | None:
    return st.session_state.training_df


def dictionary_map() -> dict[str, str]:
    dd = st.session_state.dictionary_df
    if dd is None or dd.empty or not {"Variable", "Description"}.issubset(dd.columns):
        return {}
    return dict(zip(dd["Variable"].astype(str), dd["Description"].astype(str)))


def _sync_candidate_records() -> list[str]:
    """Synchronise final/candidate artifacts into the durable project registry.

    Returns the versions that are currently represented in the session bundles.
    This makes release robust even after a previous phase was rerun or the
    in-process model cache was cleared.
    """
    bundle = st.session_state.get("training_bundle") or {}
    cv_bundle = st.session_state.get("cv_bundle") or {}
    cv_df = cv_bundle.get("cv_results", pd.DataFrame())
    final = st.session_state.get("final_bundle") or {}
    if not bundle:
        return []
    synced_versions: list[str] = []
    for name, info in bundle.get("models", {}).items():
        cv_row = cv_df[cv_df["Model"] == name].iloc[0].to_dict() if not cv_df.empty and (cv_df["Model"] == name).any() else {}
        out = final.get(name)
        spec = MODEL_SPECS.get(name)
        payload = {
            "model_name": name,
            "project_id": st.session_state.get("project_id", ""),
            "project_name": st.session_state.get("project_name", ""),
            "family": info.get("family") or (spec.family if spec else ""),
            "version": out.get("version") if out else info.get("version"),
            "cv_metrics": {k: float(cv_row[k]) for k in ["PR-AUC", "PR-AUC STD", "Recall", "Precision", "F1", "ROC-AUC", "Accuracy"] if k in cv_row},
            "final_metrics": ({k: float(out["metrics"].iloc[-1][k]) for k in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]} if out else {}),
            "threshold": float(out.get("threshold", 0.50)) if out else 0.50,
            "artifact_path": out.get("artifact_path", info.get("artifact_path", "")) if out else info.get("artifact_path", ""),
            "context_path": out.get("context_path", info.get("context_path", "")) if out else info.get("context_path", ""),
            "dataset_fingerprint": info.get("dataset_fingerprint", bundle.get("dataset_fingerprint", "")),
            "trained_at_utc": out.get("trained_at_utc", info.get("trained_at_utc", "")) if out else info.get("trained_at_utc", ""),
            "params": info.get("params", {}),
            "feature_count": int(bundle.get("X_train", pd.DataFrame()).shape[1]),
            "transformed_feature_count": int(info.get("transformed_feature_count", 0)),
            "explanation_ready": bool(out),
            "default_probability_base_rate": float(out.get("base_default_rate", bundle.get("y_train", pd.Series(dtype=float)).mean() or 0.0)) if out else float(bundle.get("y_train", pd.Series(dtype=float)).mean() or 0.0),
        }
        version = payload["version"]
        existing = store.models.get(version) if version else None
        if existing:
            store.replace_candidate(version, **{k: v for k, v in payload.items() if k != "version"})
        elif version:
            store.register_model(payload)
        if version:
            synced_versions.append(version)
    return synced_versions


def _navigate_to_phase(target: int) -> None:
    """Use a Streamlit callback for reliable sequential navigation."""
    phase = int(st.session_state.studio_phase)
    target = int(target)
    if st.session_state.busy:
        return
    if target > phase and not phase_complete(phase):
        return
    if target > phase:
        st.session_state.completed_phases.add(phase)
        if phase == 0:
            st.session_state.dataset_locked = True
            add_log(f"Dataset locked for this run: {st.session_state.dataset_name}.")
    label = PHASES[target][1]
    st.session_state.navigation_notice = f"Opening {PHASES[target][0]} · {label}"
    st.session_state.studio_phase = target


def _next_phase() -> None:
    phase = int(st.session_state.studio_phase)
    if phase >= len(PHASES) - 1 or not phase_complete(phase) or st.session_state.busy:
        return
    _navigate_to_phase(phase + 1)


def _previous_phase() -> None:
    phase = int(st.session_state.studio_phase)
    if phase <= 0 or st.session_state.busy:
        return
    _navigate_to_phase(phase - 1)


def render_navigation_feedback() -> None:
    notice = st.session_state.get("navigation_notice")
    if notice:
        st.session_state.navigation_notice = None
        st.toast(notice, icon=":material/swap_horiz:")


def phase_nav() -> None:
    with st.sidebar:
        st.markdown("### Training workflow")
        for i, (num, label) in enumerate(PHASES):
            marker = "▶" if i == st.session_state.studio_phase else ("✓" if i in st.session_state.completed_phases else "○")
            future = i > st.session_state.studio_phase
            st.button(
                f"{marker} {num}  {label}",
                key=f"phase_{i}",
                use_container_width=True,
                disabled=st.session_state.busy or future or i == st.session_state.studio_phase,
                on_click=_navigate_to_phase,
                args=(i,),
            )
        if st.session_state.dataset_locked:
            st.success(f"Project: {st.session_state.project_name or 'Current project'}", icon=":material/folder:")
            st.caption(f"Dataset: {st.session_state.dataset_name}")
            st.button(
                "Start a new project", icon=":material/add:", use_container_width=True,
                disabled=st.session_state.busy, on_click=_start_new_dataset
            )
        st.caption("Each project is saved on disk with its source data, model artifacts, analysis outputs and release metadata.")


def _start_new_dataset() -> None:
    if st.session_state.busy:
        return
    clear_training_state(keep_dataset=False)
    store.select_project(None)
    add_log("New project started. Previous projects remain saved on disk.")
    st.session_state.navigation_notice = "Data Setup · ready for a new project"


def phase_complete(phase_index: int) -> bool:
    if phase_index == 0:
        return current_df() is not None
    if phase_index in (1, 2):
        return current_df() is not None
    if phase_index == 3:
        return bool(st.session_state.get("phase4_complete"))
    if phase_index == 4:
        return bool(st.session_state.get("phase5_complete"))
    if phase_index == 5:
        return bool(st.session_state.get("phase6_complete"))
    if phase_index == 6:
        return bool(st.session_state.get("phase7_complete"))
    return False


def bottom_navigation() -> None:
    st.divider()
    left, mid, right = st.columns([1, 1, 1])
    phase = int(st.session_state.studio_phase)
    with left:
        st.button(
            "Back", icon=":material/chevron_left:", use_container_width=True,
            disabled=st.session_state.busy or phase == 0, on_click=_previous_phase,
        )
    with mid:
        st.caption(f"Phase {phase + 1} of {len(PHASES)}", text_alignment="center")
    with right:
        can_next = phase < len(PHASES) - 1 and phase_complete(phase)
        st.button(
            "Next", type="primary", icon=":material/chevron_right:", use_container_width=True,
            disabled=st.session_state.busy or not can_next, on_click=_next_phase,
        )


def render_console() -> None:
    """Render a terminal-style processing console that is consistently dark."""
    with st.expander("Processing console", expanded=False):
        if st.session_state.training_logs:
            import html
            log_text = "\n".join(st.session_state.training_logs)
            safe = html.escape(log_text)
            st.html(
                f"""
                <div style="background:#171C26;border:1px solid #2B3442;border-radius:12px;
                            padding:14px 16px;max-height:320px;overflow:auto;
                            box-shadow:inset 0 0 0 1px rgba(255,255,255,.025);">
                    <pre style="margin:0;background:transparent !important;color:#E6EDF3 !important;
                                font-family:Consolas,Monaco,'Courier New',monospace !important;
                                font-size:.78rem !important;line-height:1.55 !important;
                                white-space:pre-wrap !important;">{safe}</pre>
                </div>
                """
            )
        else:
            st.caption("No processing activity yet.")

def render_dataset_banner(df: pd.DataFrame) -> None:
    with st.container(border=True):
        a, b, c, d = st.columns([2.1, 1, 1, 1])
        a.write(f"**Project:** {st.session_state.project_name or "Current project"}  \n\n**Dataset:** {st.session_state.dataset_name} 🔒")
        b.metric("Rows", f"{len(df):,}")
        c.metric("Columns", len(df.columns))
        d.metric("Default rate", f"{df['Default'].mean():.2%}")
        st.caption("This project is locked for the current run. Start a new project to use a different dataset; saved projects are not overwritten.")


def run_phase4_training() -> None:
    from ml.training import train_candidates
    selected_names = list(st.session_state.get("selected_model_names") or MODEL_SPECS.keys())
    set_busy(f"Training {len(selected_names)} selected model(s)")
    try:
        with st.status("Training candidate models…", expanded=True) as status:
            bar = st.progress(0.0, text="Preparing modelling matrix")
            live = st.empty()

            def log(message: str) -> None:
                add_log(message)
                live.code("\n".join(st.session_state.training_logs[-20:]), language="text")

            def progress(value: float, text: str) -> None:
                bar.progress(max(0.0, min(1.0, value)), text=text)

            add_log(f"Phase 04 started: training {len(selected_names)} selected model(s): {", ".join(selected_names)}.")
            project_models_root = Path(st.session_state.project_root) / "models"
            bundle = train_candidates(
                current_df(), dictionary_map(), log=log, progress=progress,
                artifact_root=project_models_root, project_id=st.session_state.project_id,
                project_name=st.session_state.project_name, project_root=st.session_state.project_root,
                model_names=selected_names,
            )
            st.session_state.training_bundle = bundle
            st.session_state.phase4_complete = True
            _sync_candidate_records()
            status.update(label=f"{len(selected_names)} selected model(s) trained", state="complete")
        st.toast(f"{len(selected_names)} selected model(s) trained")
    except Exception as exc:
        add_log(f"ERROR during candidate training: {exc}")
        st.error(f"Training failed: {exc}")
    finally:
        clear_busy()


def run_phase5_validation() -> None:
    from ml.training import class_imbalance_comparison, tune_candidates, cross_validate_candidates
    set_busy("Validation, imbalance and tuning")
    try:
        with st.status("Running validation, imbalance analysis and tuning…", expanded=True) as status:
            bar = st.progress(0.0, text="Starting")
            live = st.empty()

            def log(message: str) -> None:
                add_log(message)
                live.code("\n".join(st.session_state.training_logs[-20:]), language="text")

            def progress(value: float, text: str) -> None:
                bar.progress(max(0.0, min(1.0, value)), text=text)

            bundle = st.session_state.training_bundle
            add_log(f"Phase 05 started: tuning and validating {len(bundle["models"])} selected model(s) before final comparison.")
            bundle = tune_candidates(bundle, n_splits=2, log=log, progress=lambda p, t: progress(0.45 * p, t))
            st.session_state.training_bundle = bundle
            cv = cross_validate_candidates(bundle, n_splits=5, log=log, progress=lambda p, t: progress(0.45 + 0.35 * p, t))
            imb = class_imbalance_comparison(bundle, n_splits=5, log=log, progress=lambda p, t: progress(0.80 + 0.20 * p, t))
            st.session_state.cv_bundle = cv
            st.session_state.imbalance_df = imb
            st.session_state.tuning_bundle = bundle
            st.session_state.phase5_complete = True
            _sync_candidate_records()
            status.update(label="Validation, imbalance analysis and tuning complete", state="complete")
        st.toast("Validation, tuning and imbalance analysis complete")
    except Exception as exc:
        add_log(f"ERROR during validation/tuning: {exc}")
        st.error(f"Validation/tuning failed: {exc}")
    finally:
        clear_busy()


def run_phase6_final() -> None:
    from ml.training import final_evaluate_all, optimize_thresholds
    set_busy("Threshold and final evaluation")
    try:
        with st.status("Optimising thresholds and evaluating all models…", expanded=True) as status:
            bar = st.progress(0.0, text="Generating training OOF probabilities")
            live = st.empty()

            def log(message: str) -> None:
                add_log(message)
                live.code("\n".join(st.session_state.training_logs[-20:]), language="text")

            def progress(value: float, text: str) -> None:
                bar.progress(max(0.0, min(1.0, value)), text=text)

            bundle = st.session_state.training_bundle
            cv = st.session_state.cv_bundle
            add_log("Phase 06 started: selecting operating thresholds from training OOF probabilities.")
            threshold = optimize_thresholds(cv, bundle["y_train"], log=log)
            progress(0.25, "Threshold optimisation complete")
            final = final_evaluate_all(bundle, threshold, log=log, progress=lambda p, t: progress(0.25 + 0.75 * p, t), artifact_root=bundle.get("artifact_root"))
            st.session_state.threshold_bundle = threshold
            st.session_state.final_bundle = final
            st.session_state.phase6_complete = True
            _sync_candidate_records()
            run_id = store.record_run(
                st.session_state.dataset_name or "",
                {"Project": st.session_state.get("project_name", ""), "Models trained": len(st.session_state.get("training_bundle", {}).get("models", {})), "CV folds": 5, "Lifecycle": "Train → CV → Imbalance → Tuning → OOF Threshold → Holdout → XAI"},
                project_id=st.session_state.get("project_id"), logs=st.session_state.get("training_logs", []),
            )
            st.session_state.last_run_id = run_id
            status.update(label="Threshold selection and final evaluation complete", state="complete")
        st.toast("Final holdout evaluation complete")
    except Exception as exc:
        add_log(f"ERROR during final evaluation: {exc}")
        st.error(f"Final evaluation failed: {exc}")
    finally:
        clear_busy()


def run_pca_diagnostic(df: pd.DataFrame) -> None:
    from sklearn.decomposition import PCA
    set_busy("Calculating PCA diagnostic")
    try:
        with st.status("Calculating PCA explained variance…", expanded=True) as status:
            clean = clean_raw_dataframe(df, drop_constant=True)
            eng = engineer_features(clean).drop(columns=["Default", "ID"], errors="ignore")
            from ml.preprocessing import build_preprocessor
            preprocessor, _, _ = build_preprocessor(eng)
            X = preprocessor.fit_transform(eng)
            pca = PCA(n_components=0.95, random_state=42)
            pca.fit(X)
            st.session_state.pca_diag = {"components": int(pca.n_components_), "cum": np.cumsum(pca.explained_variance_ratio_)}
            add_log(f"PCA diagnostic complete: {pca.n_components_} components retain at least 95% variance.")
            status.update(label="PCA diagnostic complete", state="complete")
        st.rerun()
    except Exception as exc:
        add_log(f"ERROR during PCA diagnostic: {exc}")
        st.error(f"PCA diagnostic failed: {exc}")
    finally:
        clear_busy()


phase_nav()
section_header("ADMIN WORKSPACE", "Training Studio", "A guided, on-demand modelling workspace. The product starts empty and only shows analytical results created during the current training run.")
phase = st.session_state.studio_phase
render_navigation_feedback()
df = current_df()
render_phase_strip(phase, len(PHASES))

if phase > 0 and df is not None:
    render_dataset_banner(df)

# -----------------------------------------------------------------------------
# Phase 01
# -----------------------------------------------------------------------------
if phase == 0:
    st.markdown("## 01 · Data Setup")
    st.write("Create a named project, select the labelled training dataset and data dictionary, then explicitly create the project. Selecting files alone does not process or save anything.")

    if not st.session_state.dataset_locked:
        project_name = st.text_input(
            "Project name",
            value=st.session_state.get("project_name") or "",
            max_chars=80,
            placeholder="Example: Auto Loan Default — Baseline Run",
            help="This name becomes the top-level project identity used in the saved project folder and project library.",
            disabled=st.session_state.busy,
        )
        csv = st.file_uploader(
            "Training dataset", type=["csv"], disabled=st.session_state.busy,
            help="Labelled CSV containing the Default target. It is saved into the new project only after you press Create & Process Project.",
        )
        ddfile = st.file_uploader(
            "Data dictionary", type=["csv", "xlsx"], disabled=st.session_state.busy,
            help="Definitions used in the admin explanations and user prediction context.",
        )
        training_ready = bool(project_name.strip()) and csv is not None and ddfile is not None
        st.caption("Select a project name and both files. The Create & Process Project button activates only when all three are present.")
        create_clicked = st.button(
            "Create & Process Project", type="primary", icon=":material/create_new_folder:",
            use_container_width=True, disabled=st.session_state.busy or not training_ready,
        )
        if csv is not None:
            st.info(f"Training dataset selected: **{csv.name}** ({getattr(csv, 'size', 0) / 1024:.1f} KB).")
        if ddfile is not None:
            st.caption(f"Data dictionary selected: **{ddfile.name}**")

        if create_clicked and training_ready:
            set_busy("Creating project and processing dataset")
            try:
                with st.status("Creating project…", expanded=True) as status:
                    progress_bar = st.progress(0.0, text="Reading training dataset")
                    add_log(f"New project requested: {project_name.strip()!r}.")
                    new_df = pd.read_csv(csv, low_memory=False)
                    progress_bar.progress(0.40, text="Validating training schema")
                    if "Default" not in new_df.columns:
                        raise ValueError("The training file must contain a 'Default' target column.")
                    if new_df.empty:
                        raise ValueError("The training file is empty.")
                    if new_df["Default"].dropna().nunique() != 2:
                        raise ValueError("The 'Default' target must contain both 0 and 1 classes.")
                    progress_bar.progress(0.65, text="Reading data dictionary")
                    if ddfile.name.lower().endswith(".csv"):
                        dictionary = pd.read_csv(ddfile)
                    else:
                        dictionary = pd.read_excel(ddfile)
                    progress_bar.progress(0.75, text="Creating project folder")
                    project = project_manager.create_project(project_name, csv.name, ddfile.name, new_df)
                    train_bytes = csv.getvalue()
                    dict_bytes = ddfile.getvalue()
                    project_manager.copy_uploads(project.project_id, train_bytes, csv.name, dict_bytes, ddfile.name)
                    clear_training_state(keep_dataset=False)
                    st.session_state.training_df = new_df
                    st.session_state.dataset_name = csv.name
                    st.session_state.dictionary_df = dictionary
                    st.session_state.project_id = project.project_id
                    st.session_state.project_name = project.name
                    st.session_state.project_root = project.root_path
                    st.session_state.dataset_locked = False
                    store.select_project(project.project_id)
                    progress_bar.progress(1.0, text="Project created and dataset processed")
                    add_log(f"Created project {project.name!r} ({project.project_id}).")
                    add_log(f"Saved training dataset and data dictionary under {project.root_path}.")
                    add_log(f"Dataset fingerprint: {project.dataset_fingerprint}.")
                    status.update(label="Project created successfully", state="complete")
                st.success(f"Project **{project.name}** is ready. Click **Next** to lock the project inputs for this training run.")
            except Exception as exc:
                add_log(f"ERROR during project creation: {exc}")
                st.error(f"Project setup failed: {exc}")
            finally:
                clear_busy()

        # Re-read session state after an upload transaction so the same rerun
        # immediately renders the newly processed project and preview.
        df = current_df()
    else:
        st.success(f"Project **{st.session_state.project_name}** is locked for this run.", icon=":material/lock:")
        st.caption("Saved source files are immutable for this run. Start a new project to train with a different dataset.")
        df = current_df()

    if df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Default rate", f"{df['Default'].mean():.2%}")
        st.dataframe(df.head(10), width="stretch", hide_index=True)
        with st.expander("What happens here?", expanded=False):
            st.write("A named project folder is created and the original training dataset plus data dictionary are copied into its data folder. The dataset is then held in session memory for the current workflow. No models are trained in this phase.")
    else:
        empty_state("No project created yet", "Enter a project name, select the labelled training CSV and its data dictionary, then click Create & Process Project.", "Nothing is preloaded at startup")

# -----------------------------------------------------------------------------
# Phase 02
# -----------------------------------------------------------------------------
elif phase == 1:
    from ml.schema import BINARY_CATEGORICAL_FEATURES, NOMINAL_CATEGORICAL_FEATURES, ORDINAL_CATEGORICAL_FEATURES, CALENDAR_CATEGORICAL_FEATURES
    st.markdown("## 02 · Understand & Quality")
    st.caption("This phase is descriptive: understand the target, raw schema, missingness, constants, and semantic feature types before modelling.")
    with st.spinner("Preparing data-quality views…"):
        u = cached_understanding(df)
    tabs = st.tabs(["Target & schema", "Data quality", "Semantics", "Dictionary"])
    with tabs[0]:
        st.plotly_chart(px.bar(u["target"], x="Class", y="Count", text="Count", title="Target distribution"), width="stretch")
        st.dataframe(u["summary"], width="stretch", hide_index=True)
    with tabs[1]:
        miss = u["missing"]
        chart = px.bar(miss.head(15).sort_values("Missing_Percentage"), x="Missing_Percentage", y="Feature", orientation="h", title="Highest missingness")
        st.plotly_chart(chart, width="stretch")
        st.dataframe(miss.head(20), width="stretch", hide_index=True)
        st.write(f"Duplicate rows: **{df.duplicated().sum():,}** • Constant features: **{len(u['constants'])}**")
        if u["constants"]:
            st.info("Constant fields will not contribute variation to the modelling matrix and are removed before modelling.")
    with tabs[2]:
        rows = []
        groups = [
            ("Binary categorical", BINARY_CATEGORICAL_FEATURES),
            ("Nominal categorical", NOMINAL_CATEGORICAL_FEATURES),
            ("Ordinal", ORDINAL_CATEGORICAL_FEATURES),
            ("Calendar categorical", CALENDAR_CATEGORICAL_FEATURES),
        ]
        for group, cols in groups:
            for col in cols:
                if col in df.columns:
                    rows.append({"Semantic type": group, "Feature": col})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    with tabs[3]:
        dd = st.session_state.dictionary_df
        if dd is not None:
            st.dataframe(dd, width="stretch", hide_index=True)
        else:
            st.info("No data dictionary was uploaded. The workflow can continue, but detailed field definitions will be limited.")
    with st.expander("Why this phase matters", expanded=False):
        st.write("Target imbalance influences which metrics are useful. Missingness can itself carry information, and constant fields add no variation. Semantic feature types determine how values are imputed, encoded and scaled later.")
    with st.expander("How to read these charts", expanded=False):
        st.write("Use the target chart to understand class balance and the missingness chart to locate incomplete fields. These are descriptive observations, not evidence that a field causes default.")

# -----------------------------------------------------------------------------
# Phase 03
# -----------------------------------------------------------------------------
elif phase == 2:
    st.markdown("## 03 · EDA & Feature Engineering")
    with st.spinner("Preparing exploratory analysis…"):
        e = cached_eda(df)
    tabs = st.tabs(["Distributions", "Default rates", "Correlation", "Feature engineering", "Filter selection"])
    with tabs[0]:
        st.plotly_chart(e["hist"], width="stretch")
        with st.expander("How to read the distributions", expanded=False):
            st.write("Look for skew, long tails, concentration and plausible extremes. Financial variables often have long right tails; log-derived features are added to provide an alternative representation.")
        st.plotly_chart(e["box"], width="stretch")
        with st.expander("How to read the boxplots", expanded=False):
            st.write("The box contains the central 50% of observations. Isolated points may be outliers, but an extreme value is not automatically a data error.")
    with tabs[1]:
        if e["categorical_choices"]:
            choice = st.selectbox("Categorical field", e["categorical_choices"], key="eda_cat_choice")
            st.plotly_chart(e["cat_figs"][choice], width="stretch")
        st.plotly_chart(e["missing_default"], width="stretch")
        with st.expander("How to interpret default-rate comparisons", expanded=False):
            st.write("These bars show observed default rates within groups in the labelled data. Group differences are descriptive and do not establish causal effects.")
    with tabs[2]:
        st.plotly_chart(e["corr_fig"], width="stretch")
        upper = e["corr"].abs().where(np.triu(np.ones(e["corr"].shape), k=1).astype(bool)).stack().sort_values(ascending=False)
        st.dataframe(upper.head(20).to_frame("Absolute correlation"), width="stretch")
        with st.expander("How to read the correlation heatmap", expanded=False):
            st.write("Spearman values near +1 or -1 indicate strong monotonic association. Correlation is a redundancy diagnostic; it is not proof of causation.")
    with tabs[3]:
        st.dataframe(pd.DataFrame({"Engineered feature": e["new_features"]}), width="stretch", hide_index=True)
        examples = [c for c in [
            "Loan_to_Income_Ratio", "Annuity_to_Income_Ratio", "Credit_per_Family_Member", "Children_to_Family_Ratio",
            "Average_Credit_Score", "Min_Credit_Score", "Max_Credit_Score", "Credit_Score_Range", "Available_Credit_Scores",
            "Age_Years", "Employment_Years", "Registration_Years", "ID_Change_Years", "Application_Hour_Sin", "Application_Hour_Cos",
            "Client_Income_Log", "Credit_Amount_Log", "Loan_Annuity_Log"
        ] if c in e["engineered"].columns]
        st.dataframe(e["engineered"][examples].describe().T.round(4), width="stretch")
        with st.expander("Why these features are created", expanded=False):
            st.write("Financial ratios represent burden relative to income or household size; score aggregates summarise available external scores; missingness indicators encode availability; hour encodings preserve circular time structure; log features provide a compressed view of skewed financial amounts.")
    with tabs[4]:
        chi, mi = e["chi"], e["mi"]
        st.markdown("**Chi-square — categorical features**")
        st.dataframe(chi.round(6), width="stretch", hide_index=True)
        if not chi.empty:
            st.plotly_chart(px.bar(chi.head(15).sort_values("P_Value", ascending=False), x="P_Value", y="Feature", orientation="h", title="Chi-square p-values"), width="stretch")
        st.markdown("**Mutual information — numeric / engineered features**")
        st.dataframe(mi.round(6), width="stretch", hide_index=True)
        st.plotly_chart(px.bar(mi.head(15).sort_values("Mutual_Information"), x="Mutual_Information", y="Feature", orientation="h", title="Mutual information ranking"), width="stretch")
        with st.expander("How to interpret filter selection", expanded=False):
            st.write("Chi-square and mutual information are model-independent diagnostics. A weak univariate relationship does not automatically mean a feature is useless because models can learn interactions and nonlinear patterns.")

# -----------------------------------------------------------------------------
# Phase 04
# -----------------------------------------------------------------------------
elif phase == 3:
    st.markdown("## 04 · Preprocessing & Model Lab")
    st.caption("Choose which model families to train. You can train one, two, or all three. Only the models you select will enter the remaining evaluation lifecycle.")
    available_names = list(MODEL_SPECS)
    if not st.session_state.training_bundle:
        current_selection = st.session_state.get("selected_model_names") or available_names
        selected_names = st.multiselect(
            "Models to train",
            available_names,
            default=[n for n in available_names if n in current_selection],
            format_func=lambda n: f"{n} · {MODEL_SPECS[n].family}",
            disabled=st.session_state.busy,
            help="Select at least one model. The same engineered feature matrix and preprocessing architecture are used for all selected candidates.",
        )
        st.session_state.selected_model_names = selected_names
        cards = st.columns(3)
        for col, name in zip(cards, available_names):
            spec = MODEL_SPECS[name]
            with col:
                with st.container(border=True):
                    st.markdown(f"### {name}")
                    st.caption(spec.family)
                    st.write(spec.description)
                    st.caption(spec.strengths)
                    selected_tag = "SELECTED" if name in selected_names else "NOT SELECTED"
                    st.caption(selected_tag)
                    with st.expander("Model settings", expanded=False):
                        st.json(spec.best_params, expanded=False)
        st.info(f"{len(selected_names)} model(s) selected. Training will run only when you start it.")
        if st.button("Train selected models", type="primary", width="stretch", icon=":material/play_arrow:", disabled=st.session_state.busy or not selected_names):
            run_phase4_training()
    else:
        bundle = st.session_state.training_bundle
        selected_names = list(bundle["models"].keys())
        st.success(f"{len(selected_names)} selected model(s) are trained. Use Next to continue to validation.")
        rows = []
        for name, info in bundle["models"].items():
            rows.append({
                "Model": name,
                "Family": info.get("family", ""),
                "Training status": "Trained",
                "Features": int(bundle["X_train"].shape[1]),
                "Transformed features": info.get("transformed_feature_count", 0),
                "Tuned": "Yes" if bundle.get("tuned") else "Not yet",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        cols = st.columns(len(selected_names))
        for col, (name, info) in zip(cols, bundle["models"].items()):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {name}")
                    st.caption(info.get("family", ""))
                    st.metric("Feature count", int(info.get("feature_count", bundle["X_train"].shape[1])))
                    st.metric("Transformed features", int(info.get("transformed_feature_count", 0)))
                    with st.expander("Trained parameters", expanded=False):
                        st.json(info.get("params", {}), expanded=False)
        with st.expander("How preprocessing works", expanded=False):
            st.write("Numeric fields use median imputation and standardisation. Categorical fields use most-frequent imputation and one-hot encoding with unknown categories tolerated. PCA remains a diagnostic branch; the operational candidates use the non-PCA representation so business fields remain interpretable.")
        if st.button("Run PCA explained-variance diagnostic", icon=":material/analytics:", disabled=st.session_state.busy):
            run_pca_diagnostic(df)
        diag = st.session_state.get("pca_diag")
        if diag:
            c1, c2 = st.columns([1, 2.4])
            c1.metric("Components for ≥95% variance", diag["components"])
            fig = px.line(x=np.arange(1, len(diag["cum"]) + 1), y=diag["cum"] * 100, labels={"x": "Principal component", "y": "Cumulative explained variance (%)"}, title="PCA cumulative explained variance")
            c2.plotly_chart(fig, width="stretch")

# -----------------------------------------------------------------------------
# Phase 05
# -----------------------------------------------------------------------------
elif phase == 4:
    st.markdown("## 05 · Validation, Imbalance & Tuning")
    bundle = st.session_state.training_bundle
    if not bundle:
        empty_state("Validation is waiting", "Train at least one selected model in Phase 04 before running validation.", "Use Back to return to Phase 04")
    elif not st.session_state.get("phase5_complete"):
        st.info("This phase performs all validation work for all three candidates: compact hyperparameter tuning, 5-fold stratified cross-validation and class-imbalance comparison.")
        if st.button("Run validation + imbalance + tuning", type="primary", width="stretch", icon=":material/model_training:", disabled=st.session_state.busy):
            run_phase5_validation()
    else:
        cvdf = st.session_state.cv_bundle["cv_results"].sort_values("PR-AUC", ascending=False)
        st.success("All three candidates completed validation, imbalance analysis and tuning.")
        st.dataframe(cvdf.round(4), width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(cvdf, x="Model", y="PR-AUC", error_y="PR-AUC STD", text_auto=".4f", title="Cross-validated PR-AUC with fold variability"), width="stretch")
        with c2:
            st.plotly_chart(px.scatter(cvdf, x="Recall", y="Precision", size="F1", text="Model", title="Precision vs Recall — validation"), width="stretch")
        st.plotly_chart(px.bar(cvdf, x="Model", y=["Accuracy", "Recall", "F1", "ROC-AUC"], barmode="group", title="Secondary validation metrics"), width="stretch")
        imb = st.session_state.imbalance_df
        st.markdown("### Class imbalance experiment")
        st.dataframe(imb.round(4), width="stretch", hide_index=True)
        st.plotly_chart(px.bar(imb, x="Model", y="CV PR-AUC", color="Strategy", barmode="group", title="Balanced vs standard training"), width="stretch")
        with st.expander("How to read validation", expanded=False):
            st.write("PR-AUC is the main ranking signal for this rare-event task. Its mean summarises average fold performance, while the standard deviation shows fold-to-fold variability. Precision, Recall and F1 at the 0.50 operating point are secondary diagnostics.")
        with st.expander("Why tuning happens before threshold selection", expanded=False):
            st.write("Changing model parameters can change probability distributions. The threshold is therefore selected only after the candidate model configuration has been finalised, and the threshold search uses training out-of-fold predictions rather than the untouched holdout partition.")
        st.markdown("### Tuning details")
        tune_rows = []
        for name, info in bundle["models"].items():
            tune_rows.append({
                "Model": name,
                "Best search PR-AUC": info.get("best_search_pr_auc", np.nan),
                "Search duration (sec)": info.get("tuning_duration_seconds", np.nan),
                "Best parameters": json.dumps(info.get("params", {}), default=str),
            })
        st.dataframe(pd.DataFrame(tune_rows), width="stretch", hide_index=True)
        with st.expander("What class imbalance means", expanded=False):
            st.write("A rare positive class can make accuracy look strong while default detection is weak. Balanced training gives the minority class more influence during fitting; the comparison shows how that choice changes validation behaviour.")

# -----------------------------------------------------------------------------
# Phase 06
# -----------------------------------------------------------------------------
elif phase == 5:
    st.markdown("## 06 · Threshold, Evaluation & Explainability")
    bundle = st.session_state.training_bundle
    final = st.session_state.final_bundle
    if not bundle or not st.session_state.cv_bundle:
        empty_state("Final evaluation is waiting", "Complete Phase 05 first so every candidate has a tuned configuration and validation evidence.", "Use Back to return to Phase 05")
    elif not st.session_state.get("phase6_complete"):
        st.info("This phase selects thresholds from training OOF probabilities, evaluates all three candidates once on the untouched holdout partition, and prepares model-specific explainability outputs.")
        if st.button("Run thresholds + final evaluation", type="primary", width="stretch", icon=":material/flag:", disabled=st.session_state.busy):
            run_phase6_final()
    else:
        thresholds = st.session_state.threshold_bundle["best_thresholds"]
        st.markdown("### Operating thresholds")
        st.dataframe(thresholds.round(4), width="stretch", hide_index=True)
        selected_names = list(final.keys())
        chosen = st.selectbox("Model to inspect", selected_names, key="phase6_model")
        out = final[chosen]
        st.markdown(f"### Final holdout evaluation · {chosen}")
        st.dataframe(out["metrics"].round(4), width="stretch", hide_index=True)
        cm = out["confusion"]
        cm_df = pd.DataFrame(
            [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]],
            index=["Actual Non-default", "Actual Default"],
            columns=["Predicted Non-default", "Predicted Default"],
        )
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.imshow(cm_df, text_auto=True, aspect="auto", title=f"Confusion matrix @ threshold {out['threshold']:.2f}"), width="stretch")
        with c2:
            from sklearn.metrics import roc_curve, precision_recall_curve
            fpr, tpr, _ = roc_curve(out["test_y"], out["probabilities"])
            rf = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model"))
            rf.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line={"dash": "dash"}))
            rf.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
            st.plotly_chart(rf, width="stretch")
        precision, recall, _ = precision_recall_curve(out["test_y"], out["probabilities"])
        st.plotly_chart(px.line(x=recall, y=precision, labels={"x": "Recall", "y": "Precision"}, title="Precision-Recall curve"), width="stretch")

        from ml.explainability import clean_display_feature, explain_local, feature_definition, global_feature_importance, local_shap_values, permutation_importance_frame
        st.markdown("### Global feature importance")
        fi = global_feature_importance(out["pipeline"]).head(25).copy()
        fi["Display Feature"] = fi["Feature"].map(clean_display_feature)
        st.plotly_chart(px.bar(fi.sort_values("Importance"), x="Importance", y="Display Feature", orientation="h", title="Top model feature importance"), width="stretch")
        with st.expander("Feature definitions", expanded=False):
            dd = dictionary_map()
            st.dataframe(pd.DataFrame([{"Feature": clean_display_feature(f), "Definition": feature_definition(f, dd)} for f in fi["Feature"]]), width="stretch", hide_index=True)

        with st.expander("Permutation importance", expanded=False):
            key = f"perm::{chosen}"
            if key not in st.session_state.xai_bundle:
                st.caption("This calculation is intentionally on-demand because it repeatedly shuffles evaluation features.")
                if st.button("Calculate permutation importance", icon=":material/shuffle:", disabled=st.session_state.busy):
                    set_busy("Calculating permutation importance")
                    try:
                        with st.status("Calculating permutation importance…", expanded=True) as status:
                            st.write("Measuring the change in Average Precision when transformed features are disrupted.")
                            st.session_state.xai_bundle[key] = permutation_importance_frame(out["pipeline"], bundle["X_test"], bundle["y_test"], n_repeats=5)
                            status.update(label="Permutation importance ready", state="complete")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Permutation importance failed: {exc}")
                    finally:
                        clear_busy()
            perm = st.session_state.xai_bundle.get(key)
            if perm is not None:
                top = perm.head(20).copy()
                top["Display Feature"] = top["Feature"].map(clean_display_feature)
                st.plotly_chart(px.bar(top.sort_values("Importance_Mean"), x="Importance_Mean", y="Display Feature", orientation="h", error_x="Importance_STD", title="Permutation importance — mean Δ PR-AUC"), width="stretch")
                st.dataframe(top[["Display Feature", "Importance_Mean", "Importance_STD"]], width="stretch", hide_index=True)

        with st.expander("Local SHAP explanation", expanded=False):
            st.caption("The sample applicant is only an illustration of how a local explanation is generated. It is not a benchmark record.")
            if st.button("Generate SHAP for a sample applicant", icon=":material/psychology:", disabled=st.session_state.busy):
                set_busy("Generating SHAP explanation")
                try:
                    with st.status("Generating SHAP explanation…", expanded=True) as status:
                        from core.ml_runtime import prepare_quick_applicant
                        applicant, _ = prepare_quick_applicant(out["version"], {
                            "client_income": 750000, "credit_amount": 1000000, "loan_annuity": 50000,
                            "age_years": 35, "employment_years": 8, "child_count": 0, "family_members": 2,
                            "score_source_1": None, "score_source_2": None, "score_source_3": None,
                        })
                        sdf, _ = local_shap_values(out["pipeline"], applicant)
                        st.session_state.xai_bundle[f"shap::{chosen}"] = sdf
                        status.update(label="SHAP explanation ready", state="complete")
                    st.rerun()
                except Exception as exc:
                    st.error(f"SHAP generation failed: {exc}")
                finally:
                    clear_busy()
            payload = st.session_state.xai_bundle.get(f"shap::{chosen}")
            if payload is not None:
                top = payload.head(10).copy()
                top["Display Feature"] = top["Feature"].map(clean_display_feature)
                st.plotly_chart(px.bar(top.sort_values("SHAP_Value"), x="SHAP_Value", y="Display Feature", orientation="h", title="Local SHAP contributions"), width="stretch")
                dd = dictionary_map()
                explanation = explain_local(payload, dd, top_n=4)
                if explanation["higher_risk"]:
                    st.markdown("**Contributors moving the estimate upward**")
                    for item in explanation["higher_risk"]:
                        st.write(f"**{item['feature']}** — {item['impact']:+.3f} · {item['definition']}")
                if explanation["lower_risk"]:
                    st.markdown("**Contributors moving the estimate downward**")
                    for item in explanation["lower_risk"]:
                        st.write(f"**{item['feature']}** — {item['impact']:+.3f} · {item['definition']}")
        with st.expander("How to read final evaluation", expanded=False):
            st.write("The holdout metrics are calculated after the model and threshold have been finalised. ROC-AUC and PR-AUC describe ranking over thresholds; Precision, Recall and F1 describe the displayed operating threshold. Feature importance and SHAP show model evidence, not causality.")

# -----------------------------------------------------------------------------
# Phase 07
# -----------------------------------------------------------------------------
elif phase == 6:
    st.markdown("## 07 · Model Release & User Access")
    final = st.session_state.final_bundle
    cv = st.session_state.cv_bundle
    if not final or not cv:
        empty_state("Release is waiting", "Complete final evaluation before selecting the final prediction model.", "Use Back to return to Phase 06")
    elif not st.session_state.get("phase7_complete"):
        rows = []
        for name, out in final.items():
            row = cv["cv_results"].query("Model == @name").iloc[0]
            rows.append({
                "Model": name, "Family": out["family"],
                "CV PR-AUC": row["PR-AUC"], "CV PR-AUC STD": row["PR-AUC STD"],
                "Threshold": out["threshold"], "Test PR-AUC": out["final_pr_auc"],
                "Test ROC-AUC": out["final_roc_auc"], "Test Recall": out["final_recall"], "Test F1": out["final_f1"],
            })
        release = pd.DataFrame(rows).sort_values("CV PR-AUC", ascending=False).reset_index(drop=True)
        st.markdown("### Final prediction model")
        st.write("Select exactly one evaluated model for the User-facing prediction experience. Other trained models remain available to Admins for review and history.")
        names = release["Model"].tolist()
        default_model = st.session_state.get("final_model_name") or names[0]
        selected_final = st.radio(
            "Final prediction model", names, index=names.index(default_model) if default_model in names else 0,
            horizontal=True, disabled=st.session_state.busy,
        )
        st.session_state.final_model_name = selected_final
        chosen_row = release[release["Model"] == selected_final].iloc[0]
        cards = st.columns(4)
        cards[0].metric("CV PR-AUC", f"{chosen_row['CV PR-AUC']:.4f}")
        cards[1].metric("Holdout PR-AUC", f"{chosen_row['Test PR-AUC']:.4f}")
        cards[2].metric("Recall", f"{chosen_row['Test Recall']:.1%}")
        cards[3].metric("F1", f"{chosen_row['Test F1']:.3f}")
        with st.expander("View evaluated models", expanded=False):
            st.dataframe(release.round(4), width="stretch", hide_index=True)
        st.info(f"The selected model will be the only model available to Users for this active project. Current selection: **{selected_final}**.")
        if st.button("Set as final prediction model", type="primary", width="stretch", icon=":material/publish:", disabled=st.session_state.busy):
            set_busy("Publishing final prediction model")
            try:
                out = final[selected_final]
                synced = set(_sync_candidate_records())
                if out["version"] not in synced or out["version"] not in store.models:
                    raise RuntimeError(f"Final model artifact was not registered: {out['version']}")
                store.set_user_project(st.session_state.project_id, out["version"])
                st.session_state.phase7_complete = True
                st.session_state.completed_phases.add(6)
                add_log(f"Set final prediction model for Users: {selected_final} ({out['version']}).")
                st.success(f"{selected_final} is now the final prediction model for Users.")
            except Exception as exc:
                add_log(f"ERROR during final model release: {exc}")
                st.error(f"Model release failed: {exc}")
            finally:
                clear_busy()
    else:
        published = store.user_models()
        st.success("Final prediction model is configured for Users.")
        if published:
            rec = published[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Project", rec.project_name)
            c2.metric("Final model", rec.model_name)
            c3.metric("Threshold", f"{rec.threshold:.1%}")
            st.caption(f"Users do not select a model; this administrator-selected model is used automatically. Version: {rec.version}")
        st.markdown("### Finish")
        if st.button("Finish and open Admin Dashboard", type="primary", width="stretch", icon=":material/dashboard:", disabled=st.session_state.busy):
            st.switch_page("pages/home.py")

bottom_navigation()
render_console()

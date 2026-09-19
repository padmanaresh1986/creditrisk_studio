# CreditRisk Studio — End-to-end final build status

## Product principles

- Starts empty: no dataset, metrics, model artifacts or predictions are preloaded.
- Training is explicitly launched by the administrator.
- User predictions become available only after model release.
- All three candidate model families are processed through the training lifecycle.
- The workflow is sequential and uses explicit Next/Back navigation without confirmation gates.
- Expensive controls are disabled while processing; status/progress feedback is shown.
- EDA and file parsing are cached for faster reruns.
- Model artifacts are cached after loading.
- CPU-bound training remains in a single controlled Streamlit operation with progress/logging; no unsafe background mutation of Streamlit session state is used.
- Named projects, source datasets, model artifacts, analysis outputs and run logs are persisted under the local `projects/` library; the in-process registry is hydrated from that library at startup.

## Training lifecycle

1. Data Setup
2. Understand & Quality
3. EDA & Feature Engineering
4. Preprocessing & Model Lab
5. Validation, Imbalance & Tuning
6. Threshold, Evaluation & Explainability
7. Model Release & User Access

The administrator completes every phase manually. Completion of a long-running action marks the current phase ready, but the app never jumps to the next phase automatically.

## User experience

- Three released models are selectable at prediction time.
- The designated default model is highlighted and preselected.
- Quick Prediction includes a six-band speedometer-style probability riskometer, threshold marker, concise interpretation, local SHAP evidence, assumptions, and data-dictionary definitions.
- Bulk Prediction supports CSV/XLS/XLSX scoring, row selection, row-level explanation and Excel export.

## Startup and data cleanliness

- Training artifacts from development validation are removed from the package.
- The `data/` directory does not contain the training dataset.
- `.streamlit/secrets.toml` is excluded; only the example file is shipped.
- No static metric values are rendered as live application results.

## Validation completed

- Python compilation: passed.
- Automated tests: 9 passed.
- Full three-model training/2-fold tuning/5-fold validation/imbalance/OOF threshold/holdout evaluation exercised on the supplied 6,000-row dataset during development.
- The selected Random Forest pipeline reproduced the supplied dataset's established holdout metrics at threshold 0.32 during development.


## Project persistence and release model

- Each training run begins by creating a named project folder under `projects/`.
- The project folder stores the original training dataset and data dictionary, candidate model artifacts, released model artifacts, analysis CSVs, run logs and metadata.
- Model version paths are persisted relative to the project folder so the registry can be rehydrated after an application restart.
- The project library supports multiple trained projects. Only one project is marked **Active for Users** at a time.
- When a project is activated, the latest evaluated version of each candidate family is exposed to Users, with one selected default model preselected.

## Latest maintenance fixes
- Data Setup now uses an explicit Upload & Process Dataset transaction; file selection alone does not commit or process the dataset.
- Processing console uses a dark terminal-style panel and remains collapsed by default.
- Fixed Model Lab/registry synchronisation by importing MODEL_SPECS where candidate records are synchronised.
- Model release re-synchronises final artifacts before publishing and reports a clear validation error if a version is missing.
- No asynchronous worker/threading was added; expensive work remains explicit and serialized with busy-state protection.


## Latest UX maintenance fix
- Data Setup file uploaders are outside a Streamlit form so file selection reruns the page and enables the Upload & Process Dataset button immediately when both required files are selected.
- The transaction button remains disabled during processing and the dataset is committed only after the explicit action.

## v6.1 navigation update

- Unauthenticated users now enter through a hidden single-page navigation route.
- The authenticated sidebar is created only after login.
- The login screen hides the sidebar and collapse control.
- Authenticated role-based navigation is unchanged.

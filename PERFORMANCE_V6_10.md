# CreditRisk Studio — Performance Pass v6.10

This performance pass preserves the complete training and prediction functionality while reducing repeated I/O, eager work and model-registry hydration.

## Key changes

- Project index remains mtime-cached.
- Per-project model indexes are mtime-cached.
- AppStore startup loads lightweight project metadata only; model metadata is hydrated lazily per project.
- Run history remains lazy.
- Prediction-form metadata is embedded in model context at training time, so User pages do not re-read the training CSV on every page load. Older model contexts retain a compatible CSV fallback.
- Global feature importance on the Explainability page is cached by model version.
- Model Results reads the single active User-facing final model instead of scanning every published record.
- Explainability and model registry pages load model metadata only for the project being inspected.
- No background threads were introduced for ordinary page navigation. Expensive training/tuning operations remain explicit, serialized and progress-logged.

## Execution model

```text
Start application
      │
      ├── Authentication shell only
      │
      └── After login
              │
              ├── Project metadata
              ├── Requested page
              └── Requested project/model resources only
                           │
                           ├── st.cache_data
                           └── st.cache_resource
```

## Performance principle

Cache deterministic data transformations and lightweight metadata; load model artifacts only when needed; never train, tune or calculate expensive explainability just because a page was opened.

## Follow-up implementation completed 2026-09-24

The performance implementation was further hardened after observing page-navigation latency:

- Project metadata objects are now cached using the project-index file mtime, avoiding repeated JSON-to-dataclass reconstruction during normal reruns.
- The authenticated sidebar no longer calls the model registry on every rerun. It uses project-level release metadata instead, avoiding model-index hydration merely to display the current project/default label.
- The Admin Dashboard avoids hydrating the active model merely to render the compact context cards.
- The `app.py` workspace initialization no longer wraps lightweight registry creation in a spinner, removing a misleading full-page loading state from ordinary navigation.
- EDA cache functions use a lightweight project dataset fingerprint rather than hashing the complete DataFrame on every Streamlit rerun. This is the most important change for repeated EDA navigation.
- Phase 04/05 and other training views use more lazy Plotly imports so a page that only displays controls does not import charting packages unnecessarily.
- Model Results, Explainability, Run History and Bulk Prediction defer heavier imports until after their empty-state/availability checks.

No background threads were introduced. Expensive training/tuning/explainability operations remain explicit and serialized, with progress and console logs.

### Lazy-import guardrail

All Streamlit pages must declare plotting aliases (`px`, `go`) explicitly at page scope when those aliases are referenced. Lazy-loading a library is acceptable; conditional creation of aliases is not. A regression test checks for undeclared `px`/`np`/`go` aliases.

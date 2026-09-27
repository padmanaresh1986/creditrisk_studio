## v6.8 Performance Pass

Page navigation and startup were optimized with lazy imports, cached project metadata, lazy run-history hydration, on-demand EDA views, and deferred statistical feature-selection calculations. No model-training or prediction behavior was changed.

# CreditRisk Studio — Project Persistence v6.4

## Bulk prediction fix

This release fixes two related bulk-prediction issues observed with CSV/Excel row drill-down:

1. **Arrow display compatibility** — row-detail display values are converted to homogeneous strings before rendering in `st.dataframe`. This is presentation-only; scoring values remain untouched.
2. **Mixed-type application hour** — the shared cleaning layer explicitly coerces `Application_Process_Hour` to numeric before `sin`/`cos` cyclical feature generation. This handles Excel/CSV rows that become object-typed after row selection, including rows containing `numpy.float64` values in an object series.

Regression coverage now includes mixed string/numeric application-hour inputs.

Validation:
- 13 tests passed
- Python `compileall` passed
- An Excel-like mixed-type row-selection reproduction was executed successfully


Prediction UX update: Quick Prediction now exposes broader risk-relevant raw inputs, score sources use their true normalized 0–1 scale with explicit missing-value semantics, and local SHAP evidence is aggregated to the raw fields actually entered by the user for a red/green directional contribution chart.

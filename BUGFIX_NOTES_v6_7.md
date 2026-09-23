# CreditRisk Studio v6.7 — Prediction Diagnostics & SHAP Bug Fix

## Fixed

### Quick Prediction: `list indices must be integers or slices, not str`
The raw-input SHAP aggregation call in `pages/user_prediction.py` had the final two positional arguments reversed. The function signature expects `dictionary` before `supplied_fields`. The call now uses keyword arguments, and the aggregation function includes a defensive compatibility guard for the historical positional order.

### Diagnostic logging
Quick Prediction and Bulk Prediction now emit DEBUG/INFO/ERROR messages to both the Streamlit server terminal and a collapsed in-app console. Exceptions include the exception type/message in the UI console and the full traceback in the server terminal.

## Regression coverage

A regression test reproduces the historical positional-argument order and verifies it no longer fails.

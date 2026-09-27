# CreditRisk Studio — Performance Pass v6.8

This pass preserves model/training functionality while reducing page-load work.

## Changes

- Project index is cached in-process and refreshed when its file timestamp changes.
- Run history is loaded lazily; opening the application no longer scans every project run folder.
- Heavy Plotly, scikit-learn and EDA imports are deferred until the selected page/phase needs them.
- EDA phase uses a lightweight cached cleaned/engineered dataset and renders only the selected EDA view.
- Chi-square and mutual-information feature-selection calculations are on-demand.
- Dashboard/model pages defer pandas/Plotly imports until actual project/model data exists.
- Model artifacts remain lazily loaded and `st.cache_resource` backed.
- No background threads were added for ordinary navigation; expensive training actions remain explicit and progress/logged.

## Intended UX

Application startup → lightweight shell → role navigation → page-specific lazy imports → cached data/resource use → explicit long-running actions.

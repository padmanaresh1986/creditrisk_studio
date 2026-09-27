# CreditRisk Studio v6.11

## Fixes

- Removed the Training Studio tuning-table dependency on a page-level `np.nan` fallback by using `float("nan")`.
- Added regression coverage for the tuning fallback.
- Completing a released training run and choosing **Finish and open Admin Dashboard** now clears only transient Training Studio state (`keep_dataset=False`) and deselects the session project; persisted project artifacts remain on disk.
- The next visit to Training Studio therefore starts at **01 · Data Setup** ready for a new project.

## Validation

- 26 automated tests passed.
- Python compilation passed.


### v6.12 runtime alias fix
Chart aliases are page-local module dependencies rather than conditional imports. This preserves lazy page loading while preventing NameError failures in later Training Studio/Bulk Prediction chart sections. Quick Prediction explicitly imports NumPy for its local contribution formatting.

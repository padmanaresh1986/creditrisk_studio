from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_analysis_cache_uses_lightweight_dataframe_key():
    source = (ROOT / "core" / "analysis_cache.py").read_text(encoding="utf-8")
    assert "hash_funcs={pd.DataFrame: _df_cache_key}" in source
    assert "_creditrisk_cache_key" in source


def test_app_does_not_wrap_lightweight_registry_in_spinner():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'store = get_store()' in source
    assert 'with st.spinner("Preparing your workspace' not in source


def test_sidebar_uses_project_metadata_for_context():
    source = (ROOT / "components" / "ui.py").read_text(encoding="utf-8")
    assert "recommended_model_name" in source
    assert "store.user_models()" not in source


def test_dataframe_cache_key_changes_for_distinct_project_keys():
    import importlib.util
    path = ROOT / "core" / "analysis_cache.py"
    spec = importlib.util.spec_from_file_location("analysis_cache_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a = pd.DataFrame({"x": [1, 2, 3]})
    b = pd.DataFrame({"x": [1, 2, 3]})
    a.attrs["_creditrisk_cache_key"] = "A"
    b.attrs["_creditrisk_cache_key"] = "B"
    assert module._df_cache_key(a) == "A"
    assert module._df_cache_key(b) == "B"

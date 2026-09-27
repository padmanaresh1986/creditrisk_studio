from pathlib import Path


PAGES = Path(__file__).resolve().parents[1] / "pages"


def test_plotly_express_alias_is_declared_in_pages_that_use_it():
    offenders = []
    for path in PAGES.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "px." in source and "import plotly.express as px" not in source:
            offenders.append(path.name)
    assert not offenders, f"Pages use px without declaring the page-local import: {offenders}"


def test_numpy_alias_is_declared_or_locally_imported_in_pages_that_use_it():
    offenders = []
    for path in PAGES.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "np." in source and "import numpy as np" not in source:
            offenders.append(path.name)
    assert not offenders, f"Pages use np without declaring/importing it: {offenders}"


def test_plotly_graph_objects_alias_is_declared_in_pages_that_use_it():
    offenders = []
    for path in PAGES.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "go." in source and "import plotly.graph_objects as go" not in source:
            offenders.append(path.name)
    assert not offenders, f"Pages use go without declaring the page-local import: {offenders}"


def test_bulk_prediction_has_page_scope_pandas_import():
    source = (Path(__file__).parents[1] / "pages" / "bulk_prediction.py").read_text(encoding="utf-8")
    assert "import pandas as pd" in source.split("from components.ui", 1)[0]

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_unauthenticated_navigation_is_hidden():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'pg = st.navigation([login_page], position="hidden")' in source
    assert 'default=True' in source
    assert 'url_path=""' not in source
    assert 'pg.run()\n    st.stop()' in source


def test_authenticated_navigation_is_role_aware():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'pages = build_authenticated_navigation(user["role"])' in source
    assert 'pg = st.navigation(pages, position="sidebar", expanded=True)' in source

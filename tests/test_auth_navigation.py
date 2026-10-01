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


def test_logout_clears_transient_session_state_at_role_boundary():
    source = (ROOT / "core" / "auth.py").read_text(encoding="utf-8")
    session_source = (ROOT / "core" / "session.py").read_text(encoding="utf-8")
    assert "reset_session_for_logout()" in source
    assert "for key in list(st.session_state.keys())" in session_source
    assert '"quick_result"' in session_source
    assert '"bulk_scored"' in session_source


def test_store_is_not_initialized_before_authentication():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    auth_gate = source.index('if not is_authenticated():')
    store_init = source.index('store = get_store()')
    assert store_init > auth_gate


def test_bulk_prediction_defaults_missing_busy_state_to_idle():
    source = (ROOT / "pages" / "bulk_prediction.py").read_text(encoding="utf-8")
    assert 'busy = bool(st.session_state.get("busy", False))' in source
    assert "disabled=st.session_state.busy" not in source

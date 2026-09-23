from __future__ import annotations

import streamlit as st

from core.logging_utils import emit_log

from components.ui import (
    inject_global_css,
    inject_login_css,
    render_brand,
    render_sidebar_status,
)
from core.auth import authenticate, current_user, is_authenticated, logout
from core.session import init_app_state
from core.state import get_store

st.set_page_config(
    page_title="CreditRisk Studio",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()
init_app_state()
store = get_store()
if not st.session_state.get("app_log_initialized"):
    emit_log("Application session initialized", "DEBUG", "app")
    st.session_state.app_log_initialized = True


def login_screen() -> None:
    """Render the unauthenticated entry screen only.

    Navigation is hidden while this page is active; the authenticated
    navigation is created only after a valid login and rerun.
    """
    inject_login_css()
    st.markdown("<div style='height:10vh'></div>", unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1.1, 1])
    with center:
        with st.container(border=True):
            render_brand(large=True)
            st.subheader("Sign in")
            st.caption("Local demo authentication for the academic application")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button(
                    "Sign in", type="primary", width="stretch"
                )
            if submitted:
                user = authenticate(username.strip(), password)
                if user:
                    st.session_state.auth_user = user
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            with st.expander("Demo credentials"):
                st.code("admin / Admin@123\nuser / User@123")


def build_authenticated_navigation(role: str):
    admin_pages = [
        st.Page("pages/home.py", title="Admin Dashboard", icon=":material/dashboard:"),
        st.Page("pages/training_studio.py", title="Training Studio", icon=":material/science:"),
        st.Page("pages/model_results.py", title="Model Results", icon=":material/analytics:"),
        st.Page("pages/explainability.py", title="Explainability Lab", icon=":material/psychology:"),
        st.Page("pages/model_versions.py", title="Model Versions", icon=":material/inventory_2:"),
        st.Page("pages/run_history.py", title="Run History", icon=":material/history:"),
    ]
    user_pages = [
        st.Page("pages/user_prediction.py", title="Quick Prediction", icon=":material/online_prediction:"),
        st.Page("pages/bulk_prediction.py", title="Bulk Prediction", icon=":material/upload_file:"),
    ]
    common = [st.Page("pages/about.py", title="About", icon=":material/info:")]
    if role == "admin":
        return {"Workspace": admin_pages, "Prediction": user_pages, "Information": common}
    return {"Prediction": user_pages, "Information": common}


# Use dynamic navigation so unauthenticated sessions have a hidden, single-page
# login route, rather than exposing authenticated routes in the sidebar.
if not is_authenticated():
    login_page = st.Page(
        login_screen,
        title="Sign in",
        icon=":material/login:",
        default=True,
    )
    pg = st.navigation([login_page], position="hidden")
    pg.run()
    st.stop()

user = current_user()
render_sidebar_status(user["role"], user["display_name"], store)
if st.sidebar.button("Sign out", icon=":material/logout:"):
    logout()
    st.rerun()

pages = build_authenticated_navigation(user["role"])
pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()

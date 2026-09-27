from __future__ import annotations

import streamlit as st

from components.ui import empty_state, section_header
from core.state import get_store

store = get_store()
store.ensure_runs_loaded()
section_header("ADMIN WORKSPACE", "Run History", "A concise audit view of saved processing runs across the local project library.")
if not store.training_runs:
    empty_state("No runs yet", "Run History will populate as you complete the training lifecycle.", "Nothing runs automatically")
    st.stop()

import pandas as pd
st.dataframe(pd.DataFrame(list(store.training_runs.values())), width="stretch", hide_index=True)
with st.expander("Processing console", expanded=False):
    if st.session_state.get("training_logs"):
        st.code("\n".join(st.session_state.training_logs), language="text")
    else:
        st.caption("No processing logs in this session.")
with st.expander("About run history", expanded=False):
    st.write("The history is persisted under each project’s runs folder. It is suitable for this local academic application, but it is not an enterprise audit store.")

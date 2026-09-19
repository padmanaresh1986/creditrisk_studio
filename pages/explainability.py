from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.ui import empty_state, section_header
from core.ml_runtime import load_model, load_model_context
from core.state import get_store

store = get_store()
section_header("ADMIN WORKSPACE", "Explainability Lab", "Inspect model-specific feature evidence and definitions for models produced by the application.")
if not store.has_models:
    empty_state("Explainability is empty", "Feature importance is available only after models have been trained.", "Training Studio → Preprocessing & Model Lab")
    st.stop()

records = list(store.models.values())
names = sorted({r.model_name for r in records})
selected = st.selectbox("Model", names)
rec = next(r for r in records if r.model_name == selected)

with st.spinner("Loading model explanation data…"):
    model = load_model(rec.version)
    ctx = load_model_context(rec.version)
    from ml.explainability import clean_display_feature, feature_definition, global_feature_importance, permutation_importance_frame
    fi = global_feature_importance(model).head(25).copy()
    fi["Display Feature"] = fi["Feature"].map(clean_display_feature)

st.plotly_chart(px.bar(fi.sort_values("Importance"), x="Importance", y="Display Feature", orientation="h", title="Top model feature importance"), width="stretch")
st.dataframe(fi[["Display Feature", "Importance"]], width="stretch", hide_index=True)

with st.expander("Feature definitions", expanded=False):
    dictionary = ctx.get("dictionary", {})
    st.dataframe(pd.DataFrame([{"Feature": clean_display_feature(f), "Definition": feature_definition(f, dictionary)} for f in fi["Feature"]]), width="stretch", hide_index=True)

with st.expander("Permutation importance", expanded=False):
    final = st.session_state.get("final_bundle") or {}
    bundle = st.session_state.get("training_bundle")
    if selected not in final or not bundle:
        st.info("Permutation importance becomes available after final evaluation for this model in the current session.")
    else:
        key = f"perm::{rec.version}"
        if key not in st.session_state.xai_bundle:
            st.caption("This is an on-demand calculation because it repeatedly shuffles transformed evaluation features.")
            if st.button("Calculate permutation importance", icon=":material/shuffle:", disabled=st.session_state.busy):
                st.session_state.busy = True
                try:
                    with st.status("Calculating permutation importance…", expanded=True) as status:
                        perm = permutation_importance_frame(final[selected]["pipeline"], bundle["X_test"], bundle["y_test"], n_repeats=5)
                        st.session_state.xai_bundle[key] = perm
                        status.update(label="Permutation importance ready", state="complete")
                    st.rerun()
                finally:
                    st.session_state.busy = False
        perm = st.session_state.xai_bundle.get(key)
        if perm is not None:
            top = perm.head(20).copy()
            top["Display Feature"] = top["Feature"].map(clean_display_feature)
            st.plotly_chart(px.bar(top.sort_values("Importance_Mean"), x="Importance_Mean", y="Display Feature", orientation="h", error_x="Importance_STD", title="Permutation importance — mean Δ PR-AUC"), width="stretch")
            st.dataframe(top[["Display Feature", "Importance_Mean", "Importance_STD"]], width="stretch", hide_index=True)

with st.expander("How to interpret explainability", expanded=False):
    st.write("Native importance is specific to the fitted algorithm. Permutation importance measures how evaluation performance changes when a transformed feature is disrupted. SHAP provides local contribution evidence. These outputs describe model behaviour and should not be read as causal effects.")

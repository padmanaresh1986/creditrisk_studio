import streamlit as st
from components.ui import section_header

section_header("INFORMATION","About CreditRisk Studio","A Streamlit-only automobile loan default analytics application built for an academic machine-learning capstone.")
with st.container(border=True):
    st.markdown("### Product workflow")
    st.write("Upload labelled training data → understand and validate the data → engineer features → select model families → train selected models → validate and tune → choose operating thresholds → evaluate on a holdout partition → inspect explainability → set one final prediction model.")
with st.container(border=True):
    st.markdown("### Prediction experience")
    st.write("Users can select any model released by the administrator. The recommended model is highlighted and preselected, while the other released candidates remain available for comparison.")
with st.container(border=True):
    st.markdown("### Explainability")
    st.write("Local explanations use SHAP contributions and data-dictionary definitions where available. These explanations describe how the model used the supplied values; they do not establish causal effects.")
with st.container(border=True):
    st.markdown("### Operational boundary")
    st.write("The application is intentionally Streamlit-only. Interactive workflow state remains in the current session, while named training projects, source datasets, trained model artifacts, analysis outputs and run logs are persisted locally under the projects/ directory. The application does not train or load a model at startup.")

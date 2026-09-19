from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.ui import empty_state, risk_meter, section_header
from core.ml_runtime import available_user_models, local_explanation, predict_quick
from core.state import get_store

store = get_store()
active_project = store.active_project
project_text = f"Active project: {active_project.name}. " if active_project else ""
section_header("PREDICTION CENTER", "Quick Prediction", project_text + "Estimate model-based default probability for an applicant using a model released by the administrator.")
models = available_user_models()
if not models:
    empty_state("Prediction is not available yet", "No model has been released. Complete training and Model Release before using the prediction center.", "Training Studio → Model Release")
    st.stop()

models = sorted(models, key=lambda r: (r.version != store.recommended_version, -float(r.cv_metrics.get("PR-AUC", -1))))
default_name = store.active_model_name if store.recommended_version else models[0].model_name

st.markdown("### Model selection")
st.caption("The highlighted model is the administrator-selected default. You may switch to any other model released for user prediction.")
selector_names = [r.model_name for r in models]
choice = st.radio("Choose a published model", selector_names, index=selector_names.index(default_name) if default_name in selector_names else 0, horizontal=True, label_visibility="collapsed")
selected = next(r for r in models if r.model_name == choice)

cards = st.columns(len(models))
for col, rec in zip(cards, models):
    with col:
        with st.container(border=True):
            st.caption("DEFAULT" if rec.status == "Default" else "PUBLISHED")
            st.markdown(f"### {rec.model_name}")
            st.caption(rec.family)
            st.metric("Validation PR-AUC", f"{rec.cv_metrics.get('PR-AUC', float('nan')):.4f}")
            st.metric("Operating threshold", f"{rec.threshold:.1%}")
            if rec.status == "Default":
                st.caption("Preselected for convenience; you can still choose another published model.")

with st.form("quick_prediction_form"):
    with st.container(border=True):
        st.markdown("### Applicant profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            income = st.number_input("Annual income", min_value=1.0, value=750000.0, step=5000.0, format="%.2f")
            loan = st.number_input("Loan amount", min_value=1.0, value=1000000.0, step=5000.0, format="%.2f")
        with c2:
            annuity = st.number_input("Loan annuity / expected payment", min_value=1.0, value=50000.0, step=1000.0, format="%.2f")
            age = st.number_input("Age", min_value=18.0, max_value=100.0, value=35.0, step=1.0)
        with c3:
            employment = st.number_input("Employment years", min_value=0.0, max_value=100.0, value=8.0, step=1.0)
            children = st.number_input("Children", min_value=0, max_value=20, value=0, step=1)
            family = st.number_input("Family members", min_value=1, max_value=20, value=2, step=1)

        st.markdown("### Optional credit scores")
        s1, s2, s3 = st.columns(3)
        with s1:
            score1 = st.number_input("Score Source 1", min_value=0.0, max_value=1.0, value=None, step=0.01, placeholder="0–1")
        with s2:
            score2 = st.number_input("Score Source 2", min_value=0.0, max_value=1.0, value=None, step=0.01, placeholder="0–1")
        with s3:
            score3 = st.number_input("Score Source 3", min_value=0.0, max_value=1.0, value=None, step=0.01, placeholder="0–1")

        submitted = st.form_submit_button("Assess default risk", type="primary", width="stretch", icon=":material/online_prediction:", disabled=st.session_state.busy)

if submitted:
    st.session_state.busy = True
    try:
        inputs = {
            "client_income": income,
            "credit_amount": loan,
            "loan_annuity": annuity,
            "age_years": age,
            "employment_years": employment,
            "child_count": children,
            "family_members": family,
            "score_source_1": score1,
            "score_source_2": score2,
            "score_source_3": score3,
        }
        with st.status(f"Assessing with {selected.model_name}…", expanded=True) as status:
            st.write("Preparing the applicant feature vector.")
            from core.ml_runtime import load_model_context
            applicant, meta = __import__("core.ml_runtime", fromlist=["prepare_quick_applicant"]).prepare_quick_applicant(selected.version, inputs)
            status.write("Running the selected model.")
            model = __import__("core.ml_runtime", fromlist=["load_model"]).load_model(selected.version)
            probability = float(model.predict_proba(applicant)[0, 1])
            prediction = int(probability >= float(selected.threshold))
            from ml.explainability import risk_band
            band, _ = risk_band(probability)
            result = {
                "model_name": selected.model_name,
                "version": selected.version,
                "probability": probability,
                "prediction": prediction,
                "threshold": float(selected.threshold),
                "risk_band": band,
                "base_rate": float(meta.get("base_rate", 0.0)),
                "assumed_fields": meta.get("assumed_fields", []),
                "supplied_fields": int(meta.get("supplied_fields", 0)),
                "assumption_count": len(meta.get("assumed_fields", [])),
                "inputs": inputs,
            }
            status.write("Preparing local explanation evidence.")
            shap_df, _ = local_explanation(applicant, selected.version)
            result["shap_df"] = shap_df
            result["dictionary"] = load_model_context(selected.version).get("dictionary", {})
            st.session_state.quick_result = result
            status.update(label="Prediction complete", state="complete")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    finally:
        st.session_state.busy = False

result = st.session_state.get("quick_result")
if result and result.get("version") == selected.version:
    st.markdown("---")
    with st.container(border=True):
        st.markdown("## Default Risk Assessment")
        risk_meter(result["probability"], result["base_rate"], result["threshold"])
        decision_text = "Higher Default Risk" if result["prediction"] else "Lower Default Risk"
        if result["prediction"]:
            st.warning(f"**{decision_text}** at the configured threshold.")
        else:
            st.success(f"**{decision_text}** at the configured threshold.")

        probability = result["probability"]
        base_rate = result["base_rate"]
        multiple = probability / base_rate if base_rate else 0.0
        threshold = result["threshold"]
        threshold_statement = "above" if probability >= threshold else "below"
        st.write(
            f"The selected model estimates a **{probability:.1%}** probability of default. "
            f"The estimate is **{multiple:.1f}×** the model's training-period default rate of **{base_rate:.1%}**. "
            f"It is **{threshold_statement}** the operating threshold of **{threshold:.1%}**, so the current classification is **{('Default' if result['prediction'] else 'Non-default')}**."
        )
        st.caption("This is a model estimate based on the supplied and assumed application fields. It is not a guaranteed outcome, a causal explanation, or a regulatory credit rating.")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("### Why the model moved the estimate")
        from ml.explainability import explain_local
        explanation = explain_local(result["shap_df"], result["dictionary"], top_n=4)
        if explanation["higher_risk"]:
            st.markdown("**Contributors moving the estimate upward**")
            for item in explanation["higher_risk"]:
                st.write(f"**{item['feature']}** — {item['impact']:+.3f} · {item['definition']}")
        if explanation["lower_risk"]:
            st.markdown("**Contributors moving the estimate downward**")
            for item in explanation["lower_risk"]:
                st.write(f"**{item['feature']}** — {item['impact']:+.3f} · {item['definition']}")

    with right:
        st.markdown("### Local explanation")
        top = result["shap_df"].head(10).copy()
        from ml.explainability import clean_display_feature
        top["Feature"] = top["Feature"].map(clean_display_feature)
        st.plotly_chart(px.bar(top.sort_values("SHAP_Value"), x="SHAP_Value", y="Feature", orientation="h", title="Top local model contributions"), width="stretch")

    with st.expander("Assumptions and data completeness", expanded=False):
        st.write(f"Directly supplied fields: **{result['supplied_fields']}**. Assumed/defaulted raw fields: **{result['assumption_count']}**.")
        if result.get("assumed_fields"):
            st.dataframe(pd.DataFrame({"Assumed model input": result["assumed_fields"]}), width="stretch", hide_index=True)

    with st.expander("Try a different operating threshold", expanded=False):
        threshold = st.slider("Classification threshold", 0.05, 0.95, float(result["threshold"]), 0.01)
        pred = int(result["probability"] >= threshold)
        st.write(f"At **{threshold:.0%}**, the selected model would classify this applicant as **{'Default' if pred else 'Non-default'}**.")
        st.caption("Changing the threshold changes the classification rule, not the underlying probability estimate.")

    with st.expander("Data dictionary context", expanded=False):
        wanted = ["Client_Income", "Credit_Amount", "Loan_Annuity", "Age_Days", "Employed_Days", "Score_Source_1", "Score_Source_2", "Score_Source_3", "Client_Family_Members"]
        d = result["dictionary"]
        st.dataframe(pd.DataFrame([{"Field": k, "Definition": d[k]} for k in wanted if k in d]), width="stretch", hide_index=True)

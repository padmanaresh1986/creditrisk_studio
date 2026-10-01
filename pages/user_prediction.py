from __future__ import annotations

from typing import Any
import html
import math

import streamlit as st

from core.logging_utils import emit_exception, emit_log, render_console

from components.ui import empty_state, risk_meter
from core.model_catalog import available_user_models, load_model_context, prediction_input_profile
from core.state import get_store
from ml.explainability import aggregate_input_contributions, feature_definition, risk_band

store = get_store()
current_session_user = st.session_state.get("auth_user") or {}
current_session_username = str(current_session_user.get("username", ""))
models = available_user_models()
if not models:
    empty_state(
        "Prediction is not available yet",
        "No final prediction model has been released for the active project. Complete training and have an administrator set one final model.",
        "Training Studio: Model Release",
    )
    st.stop()

selected = models[0]
display_name = html.escape(str(current_session_user.get("display_name") or current_session_username or "User"))
# Keep the page heading below Streamlit's fixed top toolbar.
st.html(
    """
    <style>
        [data-testid="stHeader"] { display: none !important; }
        [data-testid="stMainBlockContainer"] { padding-top: .75rem !important; }
    [data-testid="stMainBlockContainer"] h1 { margin-top: 0 !important; }
    @media (max-width: 800px) {
            [data-testid="stMainBlockContainer"] { padding-top: .75rem !important; }
    }
    </style>
    """
)
header_title, header_status, header_user = st.columns([6.2, 1.35, 1.65], vertical_alignment="center")
with header_title:
    st.title("Credit Risk Assessment")
    st.caption("Enter applicant and loan information to estimate the probability of loan default.")
with header_status:
    st.markdown("<div class='cr-ready'>&#9679;&nbsp; Model ready</div>", unsafe_allow_html=True)
with header_user:
    st.markdown(f"<div class='cr-user-chip'><span class='cr-user-icon'>U</span>{display_name}</div>", unsafe_allow_html=True)
st.markdown("<hr style='border:0;border-top:1px solid #E3EAF2;margin:.8rem 0 1.1rem'>", unsafe_allow_html=True)

ctx = load_model_context(selected.version)
profile = prediction_input_profile(selected.version)
dictionary = dict(ctx.get("dictionary", {}))
profile_columns = profile.get("columns", {})
raw_defaults = dict(ctx.get("raw_feature_defaults", {}))


def _help(field: str, extra: str = "") -> str:
    base = dictionary.get(field, feature_definition(field, dictionary))
    return f"{base}{(' ' + extra) if extra else ''}"


def _training_default(field: str, fallback: Any = None):
    value = raw_defaults.get(field, fallback)
    return value


def _numeric_bounds(field: str, fallback_min: float, fallback_max: float):
    # UI limits are intentionally broader than observed training ranges so a
    # legitimate applicant is not blocked merely for being outside the sample.
    # Training statistics are used for sensible defaults/context, not hard limits.
    return float(fallback_min), float(fallback_max)


def _numeric_widget(field: str, label: str, default: float | None, step: float, fmt: str, min_fallback: float, max_fallback: float, help_text: str, optional: bool = False):
    lo, hi = _numeric_bounds(field, min_fallback, max_fallback)
    widget_key = f"quick_{field}"
    kwargs: dict[str, Any] = {
        "min_value": float(lo),
        "max_value": float(hi),
        "step": float(step),
        "format": fmt,
        "help": help_text,
        "key": widget_key,
    }
    if optional:
        kwargs["placeholder"] = "Leave blank to use training default"
        if widget_key not in st.session_state:
            kwargs["value"] = None
    else:
        d = float(default if default is not None and math.isfinite(default) else lo)
        d = min(max(d, lo), hi)
        if widget_key not in st.session_state:
            kwargs["value"] = d
    return st.number_input(label, **kwargs)


def _resettable_number_input(label: str, *, key: str, **kwargs: Any):
    # Let session state own a field after its first render, so clearing it to
    # None from the Reset Form callback remains blank on the next rerun.
    if key in st.session_state:
        kwargs.pop("value", None)
    return st.number_input(label, key=key, **kwargs)


def _categorical_widget(field: str, label: str, optional: bool = True):
    info = profile_columns.get(field, {})
    values = list(info.get("options", []))
    if not values:
        return None
    # A blank selection leaves this field to the model's normal default handling.
    key = f"quick_{field}"
    kwargs: dict[str, Any] = {"placeholder": "Select...", "key": key, "help": _help(field)}
    if key not in st.session_state:
        kwargs["index"] = None
    return st.selectbox(label, values, **kwargs)


def _binary_widget(field: str, label: str):
    key = f"quick_{field}"
    kwargs: dict[str, Any] = {"placeholder": "Select...", "key": key, "help": _help(field)}
    if key not in st.session_state:
        kwargs["index"] = None
    selection = st.selectbox(label, ["No (0)", "Yes (1)"], **kwargs)
    if selection is None:
        return None
    return 0.0 if selection.startswith("No") else 1.0


def _reset_quick_form() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("quick_"):
            st.session_state[key] = None
    st.session_state.quick_result = None


with st.form("quick_prediction_form"):
    form_left, form_right = st.columns([3.15, 1], gap="large")
    with form_left:
        with st.container(border=True):
            st.markdown("<div class='cr-section-heading'><span class='cr-section-icon'>&#8377;</span><span class='cr-section-copy'><b>Financial Information</b><span>Enter the applicant's financial and loan details.</span></span></div>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                income = _numeric_widget(
                    "Client_Income", "Annual income", _training_default("Client_Income", 14400.0), 500.0, "%.2f", 1.0, 1_000_000.0,
                    _help("Client_Income"),
                )
                age_default = abs(float(_training_default("Age_Days", -15758.0))) / 365.25
                age = _resettable_number_input(
                    "Age (years)", min_value=18.0, max_value=100.0, value=min(max(age_default, 18.0), 100.0), step=1.0,
                    key="quick_Age_Days",
                    help=_help("Age_Days", "The model converts this to the source day-based field."),
                )
                family_default = int(round(float(_training_default("Client_Family_Members", 2))))
                family = _resettable_number_input(
                    "Family members", min_value=1, max_value=30, value=max(family_default, 1), step=1,
                    key="quick_Client_Family_Members",
                    help=_help("Client_Family_Members"),
                )
            with c2:
                loan = _numeric_widget(
                    "Credit_Amount", "Loan amount", _training_default("Credit_Amount", 51856.2), 500.0, "%.2f", 1.0, 2_000_000.0,
                    _help("Credit_Amount"),
                )
                employment_default = abs(float(_training_default("Employed_Days", -2241.0))) / 365.25
                employment = _resettable_number_input(
                    "Employment (years)", min_value=0.0, max_value=100.0, value=min(max(employment_default, 0.0), 100.0), step=1.0,
                    key="quick_Employed_Days",
                    help=_help("Employed_Days", "Known sentinel values are handled by the data-cleaning pipeline."),
                )
            with c3:
                annuity = _numeric_widget(
                    "Loan_Annuity", "Loan annuity / expected payment", _training_default("Loan_Annuity", 2520.9), 100.0, "%.2f", 1.0, 500_000.0,
                    _help("Loan_Annuity"),
                )
                children_default = int(round(float(_training_default("Child_Count", 0))))
                children = _resettable_number_input(
                    "Children", min_value=0, max_value=20, value=max(children_default, 0), step=1,
                    key="quick_Child_Count",
                    help=_help("Child_Count"),
                )
        with st.container(border=True):
            st.markdown("<div class='cr-section-heading'><span class='cr-section-icon'><svg viewBox='0 0 24 24'><path d='M4 19V11h4v8M10 19V5h4v14M16 19V8h4v11'/></svg></span><span class='cr-section-copy'><b>External Credit Scores <span style='display:inline;color:#1765C0;font-size:.85rem'>(optional)</span></b><span>Enter normalized source scores when available. Leave blank if unavailable.</span></span></div>", unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            with s1:
                score1 = _resettable_number_input(
                    "External score 1 (0 to 1)", min_value=0.0, max_value=1.0, value=None, step=0.001,
                    key="quick_Score_Source_1",
                    placeholder="e.g. 0.64", help=_help("Score_Source_1", "Enter the normalized score supplied by the source. This dataset contains missing values, so leaving it blank is supported."),
                )
            with s2:
                score2 = _resettable_number_input(
                    "External score 2 (0 to 1)", min_value=0.0, max_value=1.0, value=None, step=0.001,
                    key="quick_Score_Source_2",
                    placeholder="e.g. 0.57", help=_help("Score_Source_2", "Enter the normalized score supplied by the source. This dataset contains missing values, so leaving it blank is supported."),
                )
            with s3:
                score3 = _resettable_number_input(
                    "External score 3 (0 to 1)", min_value=0.0, max_value=1.0, value=None, step=0.001,
                    key="quick_Score_Source_3",
                    placeholder="e.g. 0.61", help=_help("Score_Source_3", "Enter the normalized score supplied by the source. This dataset contains missing values, so leaving it blank is supported."),
                )
            st.caption("Scores use a 0–1 scale, not percentages. When a score is unavailable, leave it blank and the model's normal handling will be used.")

        with st.container(border=True):
            st.markdown("<div class='cr-section-heading'><span class='cr-section-icon'><svg viewBox='0 0 24 24'><circle cx='12' cy='8' r='4'/><path d='M4 21c.4-4.1 3.5-6.5 8-6.5s7.6 2.4 8 6.5'/></svg></span><span class='cr-section-copy'><b>Applicant Profile</b><span>Additional profile information helps complete the application.</span></span></div>", unsafe_allow_html=True)
            p1, p2, p3 = st.columns(3)
            with p1:
                education = _categorical_widget("Client_Education", "Education")
                income_type = _categorical_widget("Client_Income_Type", "Income type")
            with p2:
                marital = _categorical_widget("Client_Marital_Status", "Marital status")
                gender = _categorical_widget("Client_Gender", "Gender")
            with p3:
                housing = _categorical_widget("Client_Housing_Type", "Housing situation")
                contract = _categorical_widget("Loan_Contract_Type", "Loan contract")

        with st.expander("Additional credit, employment & application signals", expanded=False, icon=":material/tune:"):
            a1, a2, a3 = st.columns(3)
            with a1:
                population = _numeric_widget("Population_Region_Relative", "Population-region relative value", None, 0.001, "%.6f", 0.0, 100.0, _help("Population_Region_Relative"), optional=True)
                phone_change = _numeric_widget("Phone_Change", "Phone change (days)", None, 1.0, "%.0f", 0.0, 10_000.0, _help("Phone_Change"), optional=True)
                credit_bureau = _numeric_widget("Credit_Bureau", "Credit enquiries in last year", None, 1.0, "%.0f", 0.0, 100.0, _help("Credit_Bureau"), optional=True)
                social_default = _numeric_widget("Social_Circle_Default", "Social-circle defaults", None, 0.01, "%.2f", 0.0, 1.0, _help("Social_Circle_Default"), optional=True)
                own_house_age = _numeric_widget("Own_House_Age", "Own-house age (years)", None, 1.0, "%.0f", 0.0, 100.0, _help("Own_House_Age"), optional=True)
            with a2:
                occupation = _categorical_widget("Client_Occupation", "Occupation")
                organization = _categorical_widget("Type_Organization", "Type of organization")
                accompanied = _categorical_widget("Accompany_Client", "Who accompanied the client")
                city_rating = _categorical_widget("Cleint_City_Rating", "City rating")
                registration_days = _numeric_widget("Registration_Days", "Registration change (days)", None, 1.0, "%.0f", 0.0, 30_000.0, _help("Registration_Days"), optional=True)
                id_days = _numeric_widget("ID_Days", "Identity document change (days)", None, 1.0, "%.0f", 0.0, 10_000.0, _help("ID_Days"), optional=True)
            with a3:
                car_owned = _binary_widget("Car_Owned", "Car already owned")
                bike_owned = _binary_widget("Bike_Owned", "Bike already owned")
                active_loan = _binary_widget("Active_Loan", "Other active loan")
                house_owned = _binary_widget("House_Own", "House owned")
                homephone = _binary_widget("Homephone_Tag", "Homephone provided")
                workphone = _binary_widget("Workphone_Working", "Workphone reachable")
                permanent_match = _categorical_widget("Client_Permanent_Match_Tag", "Permanent-address match")
                contact_work = _categorical_widget("Client_Contact_Work_Tag", "Contact/work match")
                application_day = _categorical_widget("Application_Process_Day", "Application day")
                application_hour = _numeric_widget("Application_Process_Hour", "Application hour", None, 1.0, "%.0f", 0.0, 23.0, _help("Application_Process_Hour"), optional=True)

        # Explicitly construct only the raw fields the user actually supplied.
        raw_overrides = {
            "Client_Income": income,
            "Credit_Amount": loan,
            "Loan_Annuity": annuity,
            "Child_Count": children,
            "Client_Family_Members": family,
            "Age_Days": -(float(age) * 365.25) if age is not None else None,
            "Employed_Days": -(float(employment) * 365.25) if employment is not None else None,
            "Score_Source_1": score1,
            "Score_Source_2": score2,
            "Score_Source_3": score3,
            "Client_Education": education,
            "Client_Income_Type": income_type,
            "Client_Marital_Status": marital,
            "Client_Gender": gender,
            "Client_Housing_Type": housing,
            "Loan_Contract_Type": contract,
            "Population_Region_Relative": population,
            "Phone_Change": phone_change,
            "Credit_Bureau": credit_bureau,
            "Social_Circle_Default": social_default,
            "Own_House_Age": own_house_age,
            "Client_Occupation": occupation,
            "Type_Organization": organization,
            "Accompany_Client": accompanied,
            "Cleint_City_Rating": city_rating,
            "Registration_Days": registration_days,
            "ID_Days": id_days,
            "Car_Owned": car_owned,
            "Bike_Owned": bike_owned,
            "Active_Loan": active_loan,
            "House_Own": house_owned,
            "Homephone_Tag": homephone,
            "Workphone_Working": workphone,
            "Client_Permanent_Match_Tag": permanent_match,
            "Client_Contact_Work_Tag": contact_work,
            "Application_Process_Day": application_day,
            "Application_Process_Hour": application_hour,
        }
        raw_overrides = {k: v for k, v in raw_overrides.items() if v is not None}

    with form_right:
        st.markdown(
            """<div class="cr-how-card">
            <div class="cr-how-title"><span class="cr-how-icon"><svg viewBox="0 0 24 24"><path d="M9 18h6M10 22h4M8.2 14.5A7 7 0 1 1 15.8 14.5c-1.1.9-1.5 1.6-1.6 2.5h-4.4c-.1-.9-.5-1.6-1.6-2.5Z"/></svg></span>How it works</div>
            <div class="cr-how-step"><span class="cr-step-num">1</span><div class="cr-step-copy"><b>Enter application details</b><p>Add financial, profile, and credit-score information when available.</p></div></div>
            <div class="cr-how-step"><span class="cr-step-num">2</span><div class="cr-step-copy"><b>Predict risk</b><p>The approved project model estimates the probability of default.</p></div></div>
            <div class="cr-how-step"><span class="cr-step-num">3</span><div class="cr-step-copy"><b>Review results</b><p>See the risk category, decision, and factors behind the estimate.</p></div></div>
            <div class="cr-secure"><b>Helpful note</b>Optional fields can be left blank. The model will use its standard handling for unavailable values.</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "Predict risk",
            type="primary",
            width="stretch",
            icon=":material/online_prediction:",
            disabled=st.session_state.busy,
        )
        st.form_submit_button(
            "Reset Form",
            type="secondary",
            width="stretch",
            icon=":material/restart_alt:",
            on_click=_reset_quick_form,
            disabled=st.session_state.busy,
        )

if submitted:
    required_values = {
        "Annual income": income,
        "Loan amount": loan,
        "Loan annuity": annuity,
        "Age": age,
        "Employment": employment,
        "Children": children,
        "Family members": family,
    }
    missing_required = [label for label, value in required_values.items() if value is None]
    if missing_required:
        st.error("Please complete the required fields: " + ", ".join(missing_required) + ".")
        st.stop()
    st.session_state.prediction_logs = []
    st.session_state.busy = True
    emit_log(
        f"Prediction started | project={selected.project_name!r} | model={selected.model_name!r} | version={selected.version}",
        "INFO",
        "prediction",
    )
    try:
        from core.ml_runtime import load_model, local_explanation, prepare_quick_applicant
        inputs = {"raw_overrides": raw_overrides}
        with st.status(f"Assessing with {selected.model_name}…", expanded=True) as status:
            status.write("Validating the supplied application fields.")
            applicant, meta = prepare_quick_applicant(selected.version, inputs)
            emit_log(
                f"Prepared applicant frame | shape={applicant.shape} | supplied_raw={meta['supplied_fields']} | assumed_raw={len(meta['assumed_fields'])}",
                "DEBUG",
                "prediction",
            )
            status.write("Running the selected model.")
            model = load_model(selected.version)
            probability = float(model.predict_proba(applicant)[0, 1])
            prediction = int(probability >= float(selected.threshold))
            band, _ = risk_band(probability)
            emit_log(
                f"Model prediction complete | probability={probability:.6f} | threshold={float(selected.threshold):.6f} | class={prediction} | band={band}",
                "INFO",
                "prediction",
            )
            status.write("Generating local explanation evidence.")
            shap_df, _ = local_explanation(applicant, selected.version)
            emit_log(f"Local explanation generated | rows={len(shap_df)}", "DEBUG", "prediction")
            raw_contrib = aggregate_input_contributions(
                shap_df=shap_df,
                raw_values=meta["raw"],
                supplied_fields=meta["supplied_raw_fields"],
                dictionary=dictionary,
            )
            emit_log(f"Raw-input contribution mapping complete | rows={len(raw_contrib)}", "DEBUG", "prediction")
            result = {
                "owner_username": current_session_username,
                "owner_role": str(current_session_user.get("role", "")),
                "project_id": selected.project_id,
                "model_name": selected.model_name,
                "version": selected.version,
                "probability": probability,
                "prediction": prediction,
                "threshold": float(selected.threshold),
                "risk_band": band,
                "base_rate": float(meta.get("base_rate", 0.0)),
                "assumed_fields": meta.get("assumed_fields", []),
                "supplied_fields": int(meta.get("supplied_fields", 0)),
                "supplied_raw_fields": meta.get("supplied_raw_fields", []),
                "assumption_count": len(meta.get("assumed_fields", [])),
                "inputs": raw_overrides,
                "raw": meta["raw"],
                "shap_df": shap_df,
                "raw_contrib": raw_contrib,
                "dictionary": dictionary,
            }
            st.session_state.quick_result = result
            status.update(label="Prediction complete", state="complete")
    except Exception as exc:
        emit_exception(exc, channel="prediction", context="Prediction failed")
        st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
    finally:
        st.session_state.busy = False

render_console("prediction", expanded=False)

result = st.session_state.get("quick_result")
if result and (
    result.get("owner_username") != current_session_username
    or result.get("owner_role") != str(current_session_user.get("role", ""))
    or result.get("project_id") != selected.project_id
    or result.get("version") != selected.version
):
    # Defense-in-depth: stale result from another authenticated context must never render.
    st.session_state.quick_result = None
    result = None

if result and result.get("version") == selected.version:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
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
        st.markdown("### How the entered inputs influenced the estimate")
        raw_contrib = result["raw_contrib"].head(10).copy()
        if not raw_contrib.empty:
            raw_contrib["Display Feature"] = raw_contrib["Raw_Feature"].map(lambda x: str(x).replace("_", " "))
            raw_contrib["Direction"] = np.where(raw_contrib["SHAP_Value"] > 0, "Moves toward default", "Moves away from default")
            raw_contrib["Contribution"] = raw_contrib["SHAP_Value"].map(lambda x: f"{x:+.3f}")
            raw_contrib["Value"] = raw_contrib["Value"].map(lambda x: "Missing" if pd.isna(x) else str(x))
            table = raw_contrib[["Display Feature", "Value", "Contribution", "Direction", "Definition"]].rename(columns={"Display Feature": "Entered input"})
            st.dataframe(table, width="stretch", hide_index=True)
        else:
            st.info("No non-zero SHAP contribution could be mapped to the supplied raw fields.")

    with right:
        st.markdown("### Input contribution chart")
        chart_df = result["raw_contrib"].head(10).sort_values("SHAP_Value").copy()
        if not chart_df.empty:
            labels = [str(v).replace("_", " ") for v in chart_df["Raw_Feature"]]
            colors = ["#D64545" if v > 0 else "#2E9B57" for v in chart_df["SHAP_Value"]]
            fig = go.Figure(
                go.Bar(
                    x=chart_df["SHAP_Value"],
                    y=labels,
                    orientation="h",
                    marker_color=colors,
                    customdata=np.array([[feature_definition(str(r), result["dictionary"])] for r in chart_df["Raw_Feature"]], dtype=object),
                    hovertemplate="%{y}<br>SHAP contribution: %{x:+.3f}<br>%{customdata[0]}<extra></extra>",
                )
            )
            fig.add_vline(x=0, line_width=1)
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Model contribution toward default",
                yaxis_title=None,
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption("Red = the entered input moved the model output toward default. Green = it moved the output away from default. Derived model features are aggregated back to their raw entered inputs for readability; this is not a causal attribution.")

    with st.expander("Supplied inputs and data completeness", expanded=False):
        st.write(f"Directly supplied raw fields: **{result['supplied_fields']}**. Assumed/defaulted raw fields: **{result['assumption_count']}**.")
        supplied_names = [str(x).replace("_", " ") for x in result.get("supplied_raw_fields", [])]
        if supplied_names:
            st.dataframe(pd.DataFrame({"Supplied field": supplied_names}), width="stretch", hide_index=True)
        if result.get("assumed_fields"):
            st.caption("The remaining model inputs use the training project's standard defaults/imputation context.")
            st.dataframe(pd.DataFrame({"Assumed field": [str(x).replace("_", " ") for x in result["assumed_fields"]]}), width="stretch", hide_index=True)

    with st.expander("Try a different operating threshold", expanded=False):
        threshold = st.slider("Classification threshold", 0.05, 0.95, float(result["threshold"]), 0.01)
        pred = int(result["probability"] >= threshold)
        st.write(f"At **{threshold:.0%}**, the selected model would classify this applicant as **{'Default' if pred else 'Non-default'}**.")
        st.caption("Changing the threshold changes the classification rule, not the underlying probability estimate.")

    with st.expander("Data dictionary context", expanded=False):
        wanted = result.get("supplied_raw_fields", [])
        rows = []
        for name in wanted:
            rows.append({"Field": name, "Definition": dictionary.get(name, feature_definition(name, dictionary)), "Entered value": result["raw"].iloc[0].get(name)})
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

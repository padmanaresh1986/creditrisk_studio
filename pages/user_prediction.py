from __future__ import annotations

from typing import Any
import math

import streamlit as st

from core.logging_utils import emit_exception, emit_log, render_console

from components.ui import empty_state, risk_meter, section_header
from core.model_catalog import available_user_models, load_model_context, prediction_input_profile
from core.state import get_store
from ml.explainability import aggregate_input_contributions, feature_definition, risk_band

store = get_store()
current_session_user = st.session_state.get("auth_user") or {}
current_session_username = str(current_session_user.get("username", ""))
active_project = store.active_project
project_text = f"Active project: {active_project.name}. " if active_project else ""
section_header(
    "PREDICTION CENTER",
    "Quick Prediction",
    project_text + "Estimate default probability using published models and the application details you provide.",
)

models = available_user_models()
if not models:
    empty_state(
        "Prediction is not available yet",
        "No final prediction model has been released for the active project. Complete training and have an administrator set one final model.",
        "Training Studio → Model Release",
    )
    st.stop()

selected = models[0]
with st.container(border=True):
    st.caption("ADMINISTRATOR-SELECTED FINAL MODEL")
    st.markdown(f"### {selected.model_name}")
    st.caption(f"{selected.family} · holdout PR-AUC {selected.final_metrics.get('PR-AUC', float('nan')):.4f} · threshold {selected.threshold:.1%}")
    st.info("Model selection is managed by the administrator. Users do not need to choose a model; this final model is used automatically for the active project.")

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
    kwargs: dict[str, Any] = {
        "min_value": float(lo),
        "max_value": float(hi),
        "step": float(step),
        "format": fmt,
        "help": help_text,
    }
    if optional:
        kwargs["value"] = None
        kwargs["placeholder"] = "Leave blank to use training default"
    else:
        d = float(default if default is not None and math.isfinite(default) else lo)
        d = min(max(d, lo), hi)
        kwargs["value"] = d
    return st.number_input(label, **kwargs)


def _categorical_widget(field: str, label: str, optional: bool = True):
    info = profile_columns.get(field, {})
    values = list(info.get("options", []))
    if not values:
        return None
    # Keep training-data types intact for the model.
    options: list[Any] = ["Use training default"] if optional else []
    options.extend(values)
    default_value = _training_default(field)
    if not optional and default_value in values:
        index = values.index(default_value)
    else:
        index = 0
    selection = st.selectbox(label, options, index=index, help=_help(field))
    return None if selection == "Use training default" else selection


def _binary_widget(field: str, label: str, default_label: str = "Use training default"):
    options = [default_label, "No (0)", "Yes (1)"]
    selection = st.selectbox(label, options, help=_help(field))
    if selection == default_label:
        return None
    return 0.0 if selection.startswith("No") else 1.0


with st.form("quick_prediction_form"):
    with st.container(border=True):
        st.markdown("### Core application details")
        st.caption("These are the main financial, household and external-score inputs. Values are editable; prefilled numeric values use the training project's observed defaults.")

        c1, c2, c3 = st.columns(3)
        with c1:
            income = _numeric_widget(
                "Client_Income", "Annual income ($)", _training_default("Client_Income", 14400.0), 500.0, "%.2f", 1.0, 1_000_000.0,
                _help("Client_Income"),
            )
            loan = _numeric_widget(
                "Credit_Amount", "Loan amount ($)", _training_default("Credit_Amount", 51856.2), 500.0, "%.2f", 1.0, 2_000_000.0,
                _help("Credit_Amount"),
            )
            annuity = _numeric_widget(
                "Loan_Annuity", "Loan annuity / expected payment ($)", _training_default("Loan_Annuity", 2520.9), 100.0, "%.2f", 1.0, 500_000.0,
                _help("Loan_Annuity"),
            )
        with c2:
            age_default = abs(float(_training_default("Age_Days", -15758.0))) / 365.25
            age = st.number_input(
                "Age (years)", min_value=18.0, max_value=100.0, value=min(max(age_default, 18.0), 100.0), step=1.0,
                help=_help("Age_Days", "The model converts this to the source day-based field."),
            )
            employment_default = abs(float(_training_default("Employed_Days", -2241.0))) / 365.25
            employment = st.number_input(
                "Employment (years)", min_value=0.0, max_value=100.0, value=min(max(employment_default, 0.0), 100.0), step=1.0,
                help=_help("Employed_Days", "Known sentinel values are handled by the data-cleaning pipeline."),
            )
            children_default = int(round(float(_training_default("Child_Count", 0))))
            children = st.number_input(
                "Children", min_value=0, max_value=20, value=max(children_default, 0), step=1,
                help=_help("Child_Count"),
            )
            family_default = int(round(float(_training_default("Client_Family_Members", 2))))
            family = st.number_input(
                "Family members", min_value=1, max_value=30, value=max(family_default, 1), step=1,
                help=_help("Client_Family_Members"),
            )
        with c3:
            s1, s2, s3 = st.columns(3)
            with s1:
                score1 = st.number_input(
                    "External score 1 (0–1)", min_value=0.0, max_value=1.0, value=None, step=0.001,
                    placeholder="e.g. 0.64", help=_help("Score_Source_1", "Enter the normalized score supplied by the source. This dataset contains missing values, so leaving it blank is supported."),
                )
            with s2:
                score2 = st.number_input(
                    "External score 2 (0–1)", min_value=0.0, max_value=1.0, value=None, step=0.001,
                    placeholder="e.g. 0.57", help=_help("Score_Source_2", "Enter the normalized score supplied by the source. This dataset contains missing values, so leaving it blank is supported."),
                )
            with s3:
                score3 = st.number_input(
                    "External score 3 (0–1)", min_value=0.0, max_value=1.0, value=None, step=0.001,
                    placeholder="e.g. 0.61", help=_help("Score_Source_3", "Enter the normalized score supplied by the source. This dataset contains missing values, so leaving it blank is supported."),
                )
            score_ranges = []
        for score_col in ["Score_Source_1", "Score_Source_2", "Score_Source_3"]:
            info = profile_columns.get(score_col, {})
            if info.get("min") is not None and info.get("max") is not None:
                score_ranges.append(f"{score_col.replace('_', ' ')} observed range {float(info['min']):.3f}–{float(info['max']):.3f}")
        range_note = "; ".join(score_ranges)
        st.caption("External score fields are normalized source values, not percentages. Blank means the source value is unavailable and the model's normal imputation will be used." + (f" {range_note}." if range_note else ""))

        st.markdown("### Profile & loan context")
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

        with st.expander("Additional credit, employment & application signals", expanded=False):
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
            "Age_Days": -(float(age) * 365.25),
            "Employed_Days": -(float(employment) * 365.25),
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

        submitted = st.form_submit_button(
            "Assess default risk",
            type="primary",
            width="stretch",
            icon=":material/online_prediction:",
            disabled=st.session_state.busy,
        )

if submitted:
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

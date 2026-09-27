from __future__ import annotations

ENGINEERED_DEFINITIONS = {
    "Loan_to_Income_Ratio": "Loan amount relative to annual client income.",
    "Annuity_to_Income_Ratio": "Loan annuity relative to annual client income.",
    "Credit_per_Family_Member": "Loan credit amount normalized by the number of family members.",
    "Children_to_Family_Ratio": "Number of children relative to household family size.",
    "Age_Years": "Client age expressed in years from the source day-based field.",
    "Employment_Years": "Employment duration expressed in years from the source day-based field.",
    "Registration_Years": "Registration duration expressed in years from the source day-based field.",
    "ID_Change_Years": "Elapsed time represented by the identity-change day field, expressed in years.",
    "Average_Credit_Score": "Mean of the available external score-source values.",
    "Min_Credit_Score": "Lowest available external score-source value.",
    "Max_Credit_Score": "Highest available external score-source value.",
    "Credit_Score_Range": "Difference between the highest and lowest available score-source values.",
    "Available_Credit_Scores": "Count of external score-source values that are present.",
    "Application_Hour_Sin": "Cyclical sine encoding of the application hour.",
    "Application_Hour_Cos": "Cyclical cosine encoding of the application hour.",
    "Client_Income_Log": "Log-transformed version of client income used to reduce the influence of a long right tail.",
    "Credit_Amount_Log": "Log-transformed version of loan credit amount.",
    "Loan_Annuity_Log": "Log-transformed version of the loan annuity.",
}


def clean_display_feature(feature: str) -> str:
    return str(feature).replace("numeric__", "").replace("categorical__", "")


def feature_definition(feature: str, dictionary: dict[str, str] | None = None) -> str:
    name = clean_display_feature(feature)
    if name in ENGINEERED_DEFINITIONS:
        return ENGINEERED_DEFINITIONS[name]
    dictionary = dictionary or {}
    if name in dictionary:
        return dictionary[name]
    if name.endswith("_Missing"):
        raw = name[:-8]
        base = dictionary.get(raw, raw.replace("_", " "))
        return f"Availability indicator for {base}."
    for raw in sorted(dictionary, key=len, reverse=True):
        if name.startswith(raw + "_"):
            category = name[len(raw) + 1 :]
            return f"{dictionary[raw]} Encoded category: {category}."
    return name.replace("_", " ")

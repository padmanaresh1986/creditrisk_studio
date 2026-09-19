
from __future__ import annotations

from typing import Final

RANDOM_STATE: Final[int] = 42
TARGET: Final[str] = "Default"
ID_COLUMN: Final[str] = "ID"

BINARY_CATEGORICAL_FEATURES: Final[list[str]] = [
    "Car_Owned",
    "Bike_Owned",
    "Active_Loan",
    "House_Own",
    "Homephone_Tag",
    "Workphone_Working",
    "Client_Permanent_Match_Tag",
    "Client_Contact_Work_Tag",
]

NOMINAL_CATEGORICAL_FEATURES: Final[list[str]] = [
    "Accompany_Client",
    "Client_Income_Type",
    "Client_Marital_Status",
    "Client_Gender",
    "Loan_Contract_Type",
    "Client_Housing_Type",
    "Client_Occupation",
    "Type_Organization",
]

ORDINAL_CATEGORICAL_FEATURES: Final[list[str]] = [
    "Client_Education",
    "Cleint_City_Rating",
]

CALENDAR_CATEGORICAL_FEATURES: Final[list[str]] = [
    "Application_Process_Day",
]

CATEGORICAL_FEATURES_FINAL: Final[list[str]] = (
    BINARY_CATEGORICAL_FEATURES
    + NOMINAL_CATEGORICAL_FEATURES
    + ORDINAL_CATEGORICAL_FEATURES
    + CALENDAR_CATEGORICAL_FEATURES
)

NUMERIC_AS_TEXT_FEATURES: Final[list[str]] = [
    "Client_Income",
    "Loan_Annuity",
    "Population_Region_Relative",
    "Age_Days",
    "Employed_Days",
    "Registration_Days",
    "ID_Days",
]

TIME_FEATURES: Final[list[str]] = ["Application_Process_Hour"]

FEATURE_ENGINEERED_COLUMNS: Final[list[str]] = [
    "Age_Years",
    "Employment_Years",
    "Registration_Years",
    "ID_Change_Years",
    "Loan_to_Income_Ratio",
    "Annuity_to_Income_Ratio",
    "Credit_per_Family_Member",
    "Children_to_Family_Ratio",
    "Average_Credit_Score",
    "Min_Credit_Score",
    "Max_Credit_Score",
    "Credit_Score_Range",
    "Available_Credit_Scores",
    "Score_Source_1_Missing",
    "Score_Source_3_Missing",
    "Client_Occupation_Missing",
    "Credit_Bureau_Missing",
    "Social_Circle_Default_Missing",
    "Application_Hour_Sin",
    "Application_Hour_Cos",
    "Client_Income_Log",
    "Credit_Amount_Log",
    "Loan_Annuity_Log",
]

DEFAULT_THRESHOLD: Final[float] = 0.50

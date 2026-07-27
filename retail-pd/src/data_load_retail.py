
from pathlib import Path
 
import pandas as pd
 
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "credit"
 
# Column names per the UCI Statlog (German Credit) documentation.
# The file itself has no header row, so we assign these on load.
COLUMN_NAMES = [
    "checking_account_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_since",
    "installment_rate_pct",
    "personal_status_sex",
    "other_debtors_guarantors",
    "present_residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "num_existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "target",  # 1 = good credit, 2 = bad credit (original coding)
]
 
 
def load_credit_data() -> pd.DataFrame:
    """
    Load the German Credit dataset with column names assigned,
    and recode the target to 0/1 where 1 = default/bad credit
    (standard PD modeling convention — 1 is the event you're predicting).
    """
    file_path = DATA_DIR / "german_credit.data"
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found — run scripts/download_data.py first."
        )
 
    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
 
    # Recode target: original 1=good, 2=bad -> 0=good, 1=bad(default)
    df["target"] = df["target"].map({1: 0, 2: 1})
 
    return df
 
 
CODE_LABELS = {
    "checking_account_status": {
        "A11": "< 0 DM",
        "A12": "0-200 DM",
        "A13": ">= 200 DM / salary assigned",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "past payment delays",
        "A31": "critical account / credits elsewhere",
        "A32": "existing credits paid duly till now",
        "A33": "delay in paying in the past",
        "A34": "all credits at this bank paid duly",
    },
    "purpose": {
        "A40": "new car",
        "A41": "used car",
        "A42": "furniture/equipment",
        "A43": "radio/TV",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "other",
    },
    "savings_account": {
        "A61": "< 100 DM",
        "A62": "100-500 DM",
        "A63": "500-1000 DM",
        "A64": ">= 1000 DM",
        "A65": "unknown / no savings account",
    },
    "employment_since": {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1-4 years",
        "A74": "4-7 years",
        "A75": ">= 7 years",
    },
    "personal_status_sex": {
        "A91": "male: divorced/separated",
        "A92": "female: divorced/separated/married",
        "A93": "male: single",
        "A94": "male: married/widowed",
        "A95": "female: single",
    },
    "other_debtors_guarantors": {
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor",
    },
    "property": {
        "A121": "real estate",
        "A122": "building society savings / life insurance",
        "A123": "car or other",
        "A124": "unknown / no property",
    },
    "other_installment_plans": {
        "A141": "at another bank",
        "A142": "at department store / mail order",
        "A143": "none",
    },
    "housing": {
        "A151": "rented",
        "A152": "owned",
        "A153": "free (provided)",
    },
    "num_existing_credits": {
        "A161": "one",
        "A162": "two or three",
        "A163": "four or five",
        "A164": "six or more",
    },
    "job": {
        "A171": "unemployed/unskilled non-resident",
        "A172": "unskilled resident",
        "A173": "skilled employee/official",
        "A174": "management/self-employed/highly qualified",
    },
    "telephone": {
        "A191": "none",
        "A192": "yes, registered",
    },
    "foreign_worker": {
        "A201": "yes",
        "A202": "no",
    },
    "num_dependents": {
        "A181": "zero to two",
        "A182": "three or more",
    },
}


def apply_labels(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Returns a copy of df with coded columns replaced by readable labels.
    Original raw-coded columns are kept alongside as '<col>_code' so you
    don't lose the original values (useful to check nothing mismapped).
 
    columns: optionally restrict to a subset of CODE_LABELS keys;
             defaults to all columns present in both df and CODE_LABELS.
    """
    df = df.copy()
    cols_to_map = columns or [c for c in CODE_LABELS if c in df.columns]
 
    for col in cols_to_map:
        df[f"{col}_code"] = df[col]
        df[col] = df[col].map(CODE_LABELS[col]).fillna(df[col])
 
    return df

if __name__ == "__main__":

 
    df = load_credit_data()
    df_labeled = apply_labels(df)
 
    print(df_labeled[["checking_account_status", "checking_account_status_code",
                       "purpose", "purpose_code"]].head())

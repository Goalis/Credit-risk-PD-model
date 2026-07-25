
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
 
 
if __name__ == "__main__":
    df = load_credit_data()
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Default rate: {df['target'].mean():.1%}")
    print(df.head())

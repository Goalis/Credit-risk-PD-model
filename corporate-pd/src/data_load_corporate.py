
from pathlib import Path
 
import pandas as pd
from scipy.io import arff
 
DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data" / "credit" / "corporate" / "polish_bankruptcy"
)
 
 
def _load_single_arff(file_path: Path, horizon: int) -> pd.DataFrame:
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
 
    # class column is stored as bytes (b'0'/b'1') — decode and rename
    df["class"] = df["class"].apply(lambda x: int(x.decode()) if isinstance(x, bytes) else int(x))
    df = df.rename(columns={"class": "target"})  # 1 = bankrupt, 0 = not
 
    df["forecast_horizon_years"] = horizon
    return df
 
 
def load_corporate_credit_data(horizon="all") -> pd.DataFrame:
    """
    horizon: 1-5 to load a single forecasting-horizon file,
             or "all" to load and concatenate all 5.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"{DATA_DIR} not found — run scripts/download_data.py first."
        )
 
    if horizon == "all":
        frames = []
        for h in range(1, 6):
            file_path = DATA_DIR / f"{h}year.arff"
            if file_path.exists():
                frames.append(_load_single_arff(file_path, h))
        if not frames:
            raise FileNotFoundError(f"No .arff files found in {DATA_DIR}")
        return pd.concat(frames, ignore_index=True)
    else:
        file_path = DATA_DIR / f"{horizon}year.arff"
        return _load_single_arff(file_path, horizon)
 
 
if __name__ == "__main__":
    df = load_corporate_credit_data(horizon="all")
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Overall bankruptcy rate: {df['target'].mean():.1%}")
    print(df.groupby("forecast_horizon_years")["target"].mean())

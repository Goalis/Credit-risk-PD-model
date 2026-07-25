"""
download_data.py
Pulls all datasets used across the risk-management-portfolio projects.

Install dependencies first:
    pip install yfinance pandas requests pandas_datareader kaggle

Kaggle datasets require a Kaggle API token:
    1. Create an account at kaggle.com
    2. Go to Account -> Create New API Token, download kaggle.json
    3. Place it at ~/.kaggle/kaggle.json (chmod 600 on Linux/Mac)
Without that file, the Kaggle downloads below will be skipped with an instruction printed.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent / "data"
MARKET_DIR = BASE_DIR / "market"
CREDIT_DIR = BASE_DIR / "credit"
MACRO_DIR = BASE_DIR / "macro"

for d in (MARKET_DIR, CREDIT_DIR, MACRO_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Two non-overlapping windows: the model is built/calibrated on WINDOWS["dev"],
# then tracked on WINDOWS["monitoring"] out-of-time data it has never seen.
WINDOWS = {
    "dev": {"start": "2015-01-01", "end": "2023-12-31"},
    "monitoring": {"start": "2024-01-01", "end": None},  # None = up to today
}


# ---------------------------------------------------------------------------
# 1. Market data (for VaR calculator, portfolio dashboard, Monte Carlo sim)
# ---------------------------------------------------------------------------
def download_market_data(window: str):
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed — run: pip install yfinance")
        return

    start = WINDOWS[window]["start"]
    end = WINDOWS[window]["end"]

    tickers = ["SPY", "TLT", "GLD", "AAPL", "MSFT", "JPM"]
    print(f"Downloading market data for {tickers} ({window}: {start} to {end or 'today'}) ...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    out_path = MARKET_DIR / f"prices_{window}.csv"
    data.to_csv(out_path)
    print(f"Saved market data to {out_path}")


# ---------------------------------------------------------------------------
# 2. Credit risk / PD model data
# ---------------------------------------------------------------------------
def download_uci_german_credit():
    import requests

    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "statlog/german/german.data"
    )
    out_path = CREDIT_DIR / "german_credit.data"
    print("Downloading UCI German Credit dataset ...")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        print(f"Saved to {out_path}")
    except Exception as e:
        print(f"Failed to download German Credit dataset: {e}")


def download_kaggle_datasets():
    kaggle_cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_cfg.exists():
        print(
            "Kaggle credentials not found at ~/.kaggle/kaggle.json — skipping "
            "Kaggle downloads. See docstring at top of this file for setup."
        )
        return

    datasets = {
        "give-me-some-credit": "brycecf/give-me-some-credit-dataset",
        "home-credit-default-risk": "c/home-credit-default-risk",
    }
    for name, kaggle_ref in datasets.items():
        dest = CREDIT_DIR / name
        dest.mkdir(exist_ok=True)
        print(f"Downloading Kaggle dataset {kaggle_ref} ...")
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", kaggle_ref,
                 "-p", str(dest), "--unzip"],
                check=True,
            )
            print(f"Saved to {dest}")
        except subprocess.CalledProcessError as e:
            print(f"Kaggle download failed for {kaggle_ref}: {e}")


# ---------------------------------------------------------------------------
# 3. Macro data (for stress testing simulator, liquidity risk calibration)
# ---------------------------------------------------------------------------
def download_fred_series(window: str):
    try:
        import pandas_datareader.data as web
    except ImportError:
        print("pandas_datareader not installed — run: pip install pandas_datareader")
        return

    start = WINDOWS[window]["start"]
    end = WINDOWS[window]["end"]

    # No API key required for basic FRED series via pandas_datareader
    series = {
        "GDP": "GDP",
        "unemployment_rate": "UNRATE",
        "fed_funds_rate": "FEDFUNDS",
        "10y_treasury": "GS10",
    }
    print(f"Downloading FRED macro series ({window}: {start} to {end or 'today'}) ...")
    for name, code in series.items():
        try:
            df = web.DataReader(code, "fred", start=start, end=end)
            out_path = MACRO_DIR / f"{name}_{window}.csv"
            df.to_csv(out_path)
            print(f"Saved {name} to {out_path}")
        except Exception as e:
            print(f"Failed to download {name} ({code}): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download datasets for the risk-management-portfolio."
    )
    parser.add_argument(
        "--window",
        choices=["dev", "monitoring", "both"],
        default="both",
        help=(
            "Which period to download for time-series data (market, FRED). "
            "'dev' = model build/calibration window, 'monitoring' = out-of-time "
            "window for the monitoring report, 'both' = download each once. "
            "Credit datasets (UCI/Kaggle) are static and downloaded once regardless."
        ),
    )
    args = parser.parse_args()

    windows_to_run = ["dev", "monitoring"] if args.window == "both" else [args.window]

    for w in windows_to_run:
        download_market_data(w)
        download_fred_series(w)

    download_uci_german_credit()
    download_kaggle_datasets()

    print("\nDone. Check the data/ directory for downloaded files.")

"""Point-in-time company risk ratings, with an optional licensed import."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_ESG = ROOT / "data" / "raw" / "sp500_esg_ceo_info-filtered.csv"


def load_ratings(current_path=None):
    ratings = pd.read_csv(HISTORICAL_ESG, usecols=["Ticker", "ESG Score", "ESG Score Date"])
    ratings = ratings.rename(columns={"ESG Score": "risk", "ESG Score Date": "available_date"})
    ratings["available_date"] = pd.to_datetime(ratings["available_date"], format="%d/%m/%Y")
    ratings["source"] = "Original Sustainalytics dataset"
    if current_path:
        current = pd.read_csv(current_path, usecols=["Ticker", "risk", "available_date", "source"])
        current["available_date"] = pd.to_datetime(current["available_date"], format="%Y-%m-%d")
        if current["source"].isna().any():
            raise ValueError("Every imported ESG rating requires a source.")
        ratings = pd.concat([ratings, current], ignore_index=True)
    ratings["Ticker"] = ratings["Ticker"].str.upper().str.strip()
    ratings["risk"] = pd.to_numeric(ratings["risk"], errors="coerce")
    ratings = ratings.dropna(subset=["Ticker", "available_date", "risk"])
    return ratings.loc[ratings["risk"].between(0, 100)].sort_values("available_date", kind="stable")


def align_risk(ratings, ticker, sessions):
    series = ratings.loc[ratings["Ticker"] == ticker].drop_duplicates("available_date", keep="last")
    index = pd.DatetimeIndex(sessions)
    dates = index.tz_localize(None).normalize()
    if series.empty:
        return pd.Series(np.nan, index=index)
    scores = series.set_index("available_date")["risk"]
    return pd.Series(scores.reindex(dates, method="ffill").to_numpy(), index=index)

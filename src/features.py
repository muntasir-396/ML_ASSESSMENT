"""
features.py
-----------
Turn raw columns into useful numbers that a model can learn from.

Every feature here is designed to be intuitive and explainable:

  DISTANCE features  → how far is the truck going?
  TIME features      → what day/month/season is it?
  LOAD features      → how heavy and what type of cargo?
  MARKET features    → what is the market doing right now?
  INTERACTION        → combined signals (e.g. distance × market pressure)
"""

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Equipment encoding                                                          #
# --------------------------------------------------------------------------- #

# Reefer trucks cost more to run (refrigeration), Flatbed requires special loading.
# We map them to numbers so the model can pick this up.
EQUIPMENT_MAP = {
    "Dry Van":  0,
    "Reefer":   1,
    "Flatbed":  2,
}


# --------------------------------------------------------------------------- #
#  Build features                                                              #
# --------------------------------------------------------------------------- #

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input : a clean DataFrame (from data_prep.py)
    Output: the same DataFrame with new feature columns added
    """
    df = df.copy()

    # ---- DISTANCE ----
    # Haversine distance as a cross-check; the dataset already has 'distance',
    # but we also compute it from lat/lon ourselves for robustness.
    df["haversine_km"]   = _haversine(df)
    df["rate_per_mile"]  = df["distance"].clip(lower=1)   # safety clip at 1 mile

    # Distance buckets — short haul (< 300 mi), medium, long, very long
    df["distance_bin"] = pd.cut(
        df["distance"],
        bins=[0, 300, 700, 1500, 9999],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # ---- TIME ----
    df["month"]       = df["date"].dt.month          # 1–12: captures seasonality
    df["day_of_week"] = df["date"].dt.dayofweek      # 0=Mon … 6=Sun
    df["day_of_year"] = df["date"].dt.dayofyear      # 1–365: fine-grained trend
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)  # freight slows on weekends

    # Cyclical encoding so December (12) wraps around to January (1) smoothly
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ---- LOAD ----
    df["equipment_code"] = df["equipment"].map(EQUIPMENT_MAP).fillna(0).astype(int)
    df["load_density"]   = df["weight"] / df["distance"].clip(lower=1)  # lb per mile

    # ---- MARKET ----
    # quote_signal and market_index are the two most powerful predictors
    # (they reflect current supply/demand for trucks on that day)
    df["market_pressure"] = df["market_index"] * df["quote_signal"]   # combined signal

    # ---- INTERACTIONS ----
    df["distance_x_market"]  = df["distance"]   * df["market_index"]
    df["distance_x_quote"]   = df["distance"]   * df["quote_signal"]
    df["density_x_market"]   = df["load_density"] * df["market_index"]

    return df


# --------------------------------------------------------------------------- #
#  Feature list used for model training                                        #
# --------------------------------------------------------------------------- #

FEATURE_COLUMNS = [
    # distance
    "distance",
    "haversine_km",
    "distance_bin",
    "rate_per_mile",
    # time
    "month",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    # load
    "equipment_code",
    "weight",
    "load_density",
    # market
    "market_index",
    "quote_signal",
    "market_pressure",
    # interactions
    "distance_x_market",
    "distance_x_quote",
    "density_x_market",
]

TARGET_COLUMN = "posted_rate"


# --------------------------------------------------------------------------- #
#  Haversine helper                                                            #
# --------------------------------------------------------------------------- #

def _haversine(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the great-circle distance (km) between pickup and delivery
    using the Haversine formula.
    """
    R = 6371.0  # Earth's radius in km

    lat1 = np.radians(df["pickup_lat"])
    lon1 = np.radians(df["pickup_lon"])
    lat2 = np.radians(df["delivery_lat"])
    lon2 = np.radians(df["delivery_lon"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

"""
data_prep.py
------------
Load the raw CSVs, fix missing values, and return clean DataFrames.

Why we need this:
  - 'weight' has ~300 missing rows in train and ~165 in validation.
    We fill each one with the median weight for that equipment type
    (Dry Van, Reefer, Flatbed all have different typical weights).
  - 'market_index' has ~374 missing rows in train and ~249 in validation.
    It moves with the date, so we fill it with that day's average.
"""

import pandas as pd


# --------------------------------------------------------------------------- #
#  Load data                                                                   #
# --------------------------------------------------------------------------- #

def load_train(path: str = "data/train_test.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = _fix_missing(df, label="train")
    return df


def load_validation(path: str = "data/validation.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = _fix_missing(df, label="validation")
    return df


def load_december(path: str = "data/december_chart_inputs.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


# --------------------------------------------------------------------------- #
#  Fix missing values                                                          #
# --------------------------------------------------------------------------- #

def _fix_missing(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    Two columns can have missing values:
      1. weight        → fill with median weight for that equipment type
      2. market_index  → fill with average market_index for that date
    """
    df = df.copy()

    # --- weight ---
    missing_weight = df["weight"].isna().sum()
    if missing_weight > 0:
        # Each equipment type (Dry Van / Reefer / Flatbed) has different load weights
        median_weight_by_equipment = df.groupby("equipment")["weight"].transform("median")
        df["weight"] = df["weight"].fillna(median_weight_by_equipment)
        print(f"[{label}] Filled {missing_weight} missing weight values with equipment median.")

    # --- market_index ---
    missing_market = df["market_index"].isna().sum()
    if missing_market > 0:
        # market_index represents market conditions on a given day
        daily_avg_index = df.groupby("date")["market_index"].transform("mean")
        df["market_index"] = df["market_index"].fillna(daily_avg_index)
        # If a whole day is missing, fall back to global median
        global_median = df["market_index"].median()
        df["market_index"] = df["market_index"].fillna(global_median)
        print(f"[{label}] Filled {missing_market} missing market_index values with daily mean.")

    return df

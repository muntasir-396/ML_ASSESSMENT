"""
predict.py
----------
Load the trained model and generate the two required output files:

  1. validation_predictions.csv   — 12,000 rows (TE-000001 to TE-012000)
  2. data/december_chart_inputs.csv — 31 rows, with predicted_rate filled in

Then run score.py to validate both files and generate candidate_december.png.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
import numpy as np
import pandas as pd

from src.data_prep import load_validation, load_december, load_train
from src.features import build_features, FEATURE_COLUMNS
from src.train import train_and_evaluate, MODEL_PATH


# --------------------------------------------------------------------------- #
#  Helper: add lat/lon for the December inputs                                 #
# --------------------------------------------------------------------------- #

def _attach_latlon_for_december(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    The december_chart_inputs.csv only has city names, not lat/lon.
    We look them up from the training data so the features can be computed.
    """
    df = df.copy()

    # Build a lookup: city name → lat/lon from training data
    pickup_lookup   = train_df.drop_duplicates("pickup").set_index("pickup")[["pickup_lat", "pickup_lon"]]
    delivery_lookup = train_df.drop_duplicates("delivery").set_index("delivery")[["delivery_lat", "delivery_lon"]]

    df = df.join(pickup_lookup,   on="pickup",   how="left")
    df = df.join(delivery_lookup, on="delivery", how="left")

    # Fill in market_index and quote_signal for December using late-October averages
    # (the best estimate we have for the future from our training data)
    late_train  = train_df[train_df["date"] >= "2025-10-01"]
    avg_market  = late_train["market_index"].mean()
    avg_quote   = late_train["quote_signal"].mean()

    df["market_index"] = avg_market
    df["quote_signal"] = avg_quote
    df["weight"]       = df["weight"].fillna(32000.0)  # weight is fixed per the scorer

    return df


# --------------------------------------------------------------------------- #
#  Main prediction pipeline                                                    #
# --------------------------------------------------------------------------- #

def generate_predictions():
    # --- Step 1: Train model (or load if it already exists) ---
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH} ...")
        model = joblib.load(MODEL_PATH)
    else:
        print("No saved model found. Running training first...")
        model, _ = train_and_evaluate()

    # We still need the raw training data for lat/lon lookups
    raw_train = load_train()

    # --- Step 2: Validation predictions (12,000 rows) ---
    print("\n[1/3] Generating validation predictions...")
    val_raw  = load_validation()
    val_feat = build_features(val_raw)

    val_preds = model.predict(val_feat[FEATURE_COLUMNS])

    # Ensure all predictions are positive (score.py requires this)
    val_preds = np.maximum(val_preds, 1.0)

    validation_output = pd.DataFrame({
        "load_id":        val_raw["load_id"],
        "predicted_rate": np.round(val_preds, 2),
    })
    validation_output.to_csv("validation_predictions.csv", index=False)
    print(f"  Saved {len(validation_output):,} rows to validation_predictions.csv")

    # --- Step 3: December chart predictions (31 days) ---
    print("\n[2/3] Generating December forecast...")
    dec_raw  = load_december()
    dec_full = _attach_latlon_for_december(dec_raw, raw_train)
    dec_feat = build_features(dec_full)

    dec_preds = model.predict(dec_feat[FEATURE_COLUMNS])
    dec_preds = np.maximum(dec_preds, 1.0)

    # Write predictions back into the original file format (7 columns, same order)
    dec_output = dec_raw.copy()
    dec_output["predicted_rate"] = np.round(dec_preds, 2)
    dec_output.to_csv("data/december_chart_inputs.csv", index=False)
    print(f"  Saved 31 December predictions to data/december_chart_inputs.csv")

    # --- Step 4: Run the official scorer ---
    print("\n[3/3] Running score.py to validate and generate the chart...")
    exit_code = os.system(
        "python score.py "
        "--predictions validation_predictions.csv "
        "--december-predictions data/december_chart_inputs.csv"
    )
    if exit_code == 0:
        print("\n✅  All done! Check scorer_results/candidate_december.png for the chart.")
    else:
        print("\n❌  score.py reported an error. Check the output above.")


if __name__ == "__main__":
    generate_predictions()

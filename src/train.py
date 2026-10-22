

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_prep import load_train
from src.features import build_features, FEATURE_COLUMNS, TARGET_COLUMN


# --------------------------------------------------------------------------- #
#  Configuration                                                               #
# --------------------------------------------------------------------------- #

# Split date: train on Jan–Aug 2025, validate on Sep–Oct 2025
SPLIT_DATE = "2025-09-01"

# Random seed so results are reproducible every run
SEED = 42

# LightGBM settings
LGBM_PARAMS = {
    "objective":        "regression_l1",    # minimize MAE (robust to outliers)
    "n_estimators":     1000,
    "learning_rate":    0.05,
    "num_leaves":       127,
    "max_depth":        -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "random_state":     SEED,
    "n_jobs":           -1,
    "verbose":          -1,
}

MODEL_PATH = "model.pkl"


# --------------------------------------------------------------------------- #
#  Baseline model (for comparison in the report)                              #
# --------------------------------------------------------------------------- #

def baseline_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    """
    Simple baseline: predict the median rate per (equipment, distance_bin) group.
    If the group doesn't exist in training, fall back to global median.
    """
    global_median = train_df[TARGET_COLUMN].median()
    group_median  = train_df.groupby(["equipment", "distance_bin"])[TARGET_COLUMN].median()

    def lookup(row):
        key = (row["equipment"], row["distance_bin"])
        return group_median.get(key, global_median)

    return test_df.apply(lookup, axis=1)


# --------------------------------------------------------------------------- #
#  Evaluation helper                                                           #
# --------------------------------------------------------------------------- #

def print_metrics(label: str, y_true, y_pred) -> None:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n{'─' * 40}")
    print(f"  {label}")
    print(f"{'─' * 40}")
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  MAPE : {mape:.2f}%")


# --------------------------------------------------------------------------- #
#  Main training pipeline                                                      #
# --------------------------------------------------------------------------- #

def train_and_evaluate():
    # 1. Load & prepare data
    print("\n[1/5] Loading training data...")
    raw_train = load_train()

    # 2. Build features
    print("[2/5] Building features...")
    df = build_features(raw_train)

    # 3. Time-based train / validation split
    print(f"[3/5] Splitting data at {SPLIT_DATE} (train=Jan–Aug, val=Sep–Oct)...")
    train_mask = df["date"] < SPLIT_DATE
    train_df   = df[train_mask].copy()
    val_df     = df[~train_mask].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_val   = val_df[FEATURE_COLUMNS]
    y_val   = val_df[TARGET_COLUMN]

    print(f"  Train rows : {len(X_train):,}")
    print(f"  Val rows   : {len(X_val):,}")

    # 4. Baseline
    print("[4/5] Running baseline model...")
    baseline_preds = baseline_predict(train_df, val_df)
    print_metrics("Baseline (median per equipment+distance bin)", y_val, baseline_preds)

    # 5. LightGBM
    print("[5/5] Training LightGBM...")
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    lgbm_preds = model.predict(X_val)
    print_metrics("LightGBM", y_val, lgbm_preds)

    # Feature importance — top 10
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS
    ).sort_values(ascending=False)
    print("\n  Top 10 features by importance:")
    print(importance.head(10).to_string())

    # 6. Retrain on ALL training data (Jan–Oct) before predicting
    print("\n[Bonus] Retraining on full Jan–Oct data for final predictions...")
    X_full = df[FEATURE_COLUMNS]
    y_full = df[TARGET_COLUMN]
    final_model = lgb.LGBMRegressor(**LGBM_PARAMS)
    final_model.fit(X_full, y_full)

    # Save model for use in predict.py
    joblib.dump(final_model, MODEL_PATH)
    print(f"  Saved model to {MODEL_PATH}")

    return final_model, importance


if __name__ == "__main__":
    train_and_evaluate()

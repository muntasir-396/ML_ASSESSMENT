"""
run.py
------
One-command entry point.

  Usage:
      python run.py           → train + predict + validate + generate chart

  What it does:
      1. Cleans the data (fixes missing values)
      2. Builds features
      3. Trains the model with time-based validation (prints MAE/RMSE/MAPE)
      4. Generates validation_predictions.csv (12,000 rows)
      5. Generates December forecast and chart via score.py
"""

from src.train import train_and_evaluate
from src.predict import generate_predictions

if __name__ == "__main__":
    print("=" * 50)
    print("  Freight Rate Prediction — Full Pipeline")
    print("=" * 50)

    # Step 1 & 2: Train, validate, save model
    train_and_evaluate()

    # Step 3: Generate all output files
    generate_predictions()

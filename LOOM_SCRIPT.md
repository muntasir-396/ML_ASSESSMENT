# 2.5-Minute Loom Video Recording Script
**Spotter Machine Learning Engineer Assessment**
*Candidate: Muntasir Mamun*

> **Recording Tip:** Turn your camera ON (as requested by Spotter), share your screen showing VS Code / the repository and the generated `Freight_Rate_ML_Assessment_Report.pdf`. Speak naturally and confidently.

---

### [0:00 – 0:30] Introduction & Key Data Findings
"Hi everyone! My name is Muntasir Mamun, and today I'm walking you through my solution for the Freight Rate Prediction assessment. 

When first exploring the historical data of 48,000 loads across January through October 2025, a few key patterns emerged:
1. First, shipping rates follow an expected power-law relationship with haul distance, but with distinct pricing regimes based on equipment type—Reefers and Flatbeds command clear premiums over standard Dry Vans.
2. Second, the dynamic market index and live quote signal exhibit strong short-term elasticity, meaning pricing pressure directly amplifies rate-per-mile across tight shipping lanes."

---

### [0:30 – 1:00] Data Quality Issues & How They Were Handled
"In terms of data quality, I identified two main challenges:
1. **Missing Cargo Weights:** Around 300 loads in training and 165 loads in validation had missing weights. Since trailer types carry drastically different load weights, I avoided naive global mean imputation and instead used the median weight partitioned by equipment type.
2. **Missing Market Indices:** Around 374 rows had missing market indices. Because market index represents day-to-day macro supply-demand conditions, I imputed these using the daily date-level mean across observable loads, with a fallback to the overall median.

This ensured clean, non-leaking data preparation across both train and inference pipelines."

---

### [1:00 – 1:35] Training & Validation Approach (Temporal Splitting)
"The most crucial architectural decision here was how to split the data.

Because our target validation set spans November and December 2025—which is strictly in the future—using standard random K-Fold cross-validation would cause severe temporal data leakage and lookahead bias.

To simulate the true production task, I implemented a strict **time-based validation holdout**:
- **Training partition:** January 1st through August 31st (over 38,000 loads).
- **Validation holdout:** September 1st through October 31st (over 9,500 loads).

This out-of-time evaluation confirmed that our model and feature interactions generalize reliably to unseen future market conditions."

---

### [1:35 – 2:15] Chosen Model & Feature Engineering
"For the model, I chose a **LightGBM Regressor** with an L1 objective to minimize Mean Absolute Error, making it robust against rate outliers.

I engineered 21 domain-specific features grouped into four categories:
- **Route geometry:** Exact road distance, Haversine spherical distance, and haul-length tiers.
- **Market signals:** Combining `quote_signal` and `market_index` into an interactive market pressure metric.
- **Cross-interactions:** Most notably `distance × quote_signal`, which acts as a dynamic rate-per-mile proxy and proved to be our single most important feature.
- **Calendar harmonics:** Day of week, day of year, weekend flags, and sinusoidal encodings to capture freight seasonality.

Compared to a baseline predicting median rates by equipment and haul tier, our LightGBM model reduced the MAE from **$420.62 down to $126.22** and achieved an impressive **5.42% MAPE**—over a 3.3× performance improvement."

---

### [2:15 – 2:45] Code Walkthrough & Deliverables
"Taking a quick look at the codebase:
- It’s designed to be clean, modular, and human-readable.
- `src/data_prep.py` handles clean data ingestion and missing value resolution.
- `src/features.py` builds our 21 explainable features.
- `src/train.py` executes the time-based evaluation and retrains on the full 10-month historical data.
- `src/predict.py` generates all 12,000 validation predictions in the exact template format, as well as the December forecast.

The entire pipeline can be reproduced with a single command: `python run.py`. It runs the official `score.py` script, validating all 12,000 rows and generating our December forecast chart, which accurately reflects intra-week rate elasticity and late-December holiday spot premiums.

Thank you for your time, and I look forward to discussing the results with the team!"

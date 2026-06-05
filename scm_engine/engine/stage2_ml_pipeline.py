"""
engine/stage2_ml_pipeline.py
=============================
Supervised ML pipeline for the predictive sourcing engine.

Three models are trained:
  1. cost_pipeline    — LinearRegression  → predicts actual_delivery_cost  (₹)
  2. time_pipeline    — LinearRegression  → predicts actual_delivery_time  (days)
  3. ontime_pipeline  — LogisticRegression → predicts is_on_time            (0/1)

A shared sklearn.compose.ColumnTransformer handles:
  • One-Hot Encoding  : plant_id, warehouse_id, logistics_agency, mode, topology_type, demand_zone
  • Standard Scaling  : total_distance_km, item_qty

NOTE on topology overrides:
  Bypass markers ('BYPASS', 'WH_STOCK') are legitimate categorical values and
  are handled naturally by OneHotEncoder — no special imputation is required.
  The encoder will create indicator columns for these strings just like any other
  plant or warehouse ID.

Persists:
  models/cost_pipeline.joblib
  models/time_pipeline.joblib
  models/ontime_pipeline.joblib
"""

import os
import warnings
import joblib
import pandas as pd
import numpy as np

from sklearn.compose      import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.linear_model  import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics        import (mean_absolute_error, r2_score,
                                    accuracy_score, roc_auc_score)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH  = r"./scm_engine/data/historical_orders_clustered.csv"
MODEL_DIR  = r"./scm_engine/models"
TEST_SIZE  = 0.20
RANDOM_STATE = 42

# Categorical features that need One-Hot Encoding.
# 'warehouse_id'  includes the bypass marker 'BYPASS'  (PLANT_CUST topology)
# 'plant_id'      includes the bypass marker 'WH_STOCK' (WH_CUST  topology)
# OneHotEncoder handles these as ordinary category levels — no KeyError raised.
CAT_FEATURES = [
    "plant_id",
    "warehouse_id",
    "logistics_agency",
    "mode",
    "topology_type",   # ← critical: encodes the routing class
    "demand_zone",
]

NUM_FEATURES = [
    "total_distance_km",
    "item_qty",
]

TARGET_COST   = "actual_delivery_cost"
TARGET_TIME   = "actual_delivery_time"
TARGET_ONTIME = "is_on_time"


# ---------------------------------------------------------------------------
# Build shared pre-processor
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Return a ColumnTransformer that:
      - OHE encodes all categorical features (handle_unknown='ignore' ensures
        unseen categories at inference don't crash the pipeline)
      - Standard-scales numerical features
    """
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()

    return ColumnTransformer(
        transformers=[
            ("cat", ohe,    CAT_FEATURES),
            ("num", scaler, NUM_FEATURES),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_and_save_models(data_path: str = DATA_PATH) -> None:
    """
    Load clustered historical orders, fit three ML pipelines, evaluate on
    held-out test set, and persist model artefacts to MODEL_DIR.
    """
    print(f"[stage2] Loading data from {data_path} …")
    df = pd.read_csv(data_path)
    print(f"[stage2] Dataset shape: {df.shape}")

    # --- Feature matrix (shared across all three models) ---
    X = df[CAT_FEATURES + NUM_FEATURES].copy()

    # Safety: fill any nulls in cat columns introduced by topology overrides
    # (should already be 'BYPASS' / 'WH_STOCK' strings; NaN coords are numeric only)
    for col in CAT_FEATURES:
        X[col] = X[col].fillna("UNKNOWN").astype(str)

    y_cost   = df[TARGET_COST].values
    y_time   = df[TARGET_TIME].values
    y_ontime = df[TARGET_ONTIME].values

    X_train, X_test, yc_train, yc_test, yt_train, yt_test, yo_train, yo_test = (
        train_test_split(
            X, y_cost, y_time, y_ontime,
            test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ----------------------------------------------------------------
    # Model 1 — Cost Regression (LinearRegression)
    # ----------------------------------------------------------------
    cost_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor",    LinearRegression()),
    ])
    cost_pipeline.fit(X_train, yc_train)
    yc_pred = cost_pipeline.predict(X_test)
    print(f"\n[stage2] Cost model  — MAE: ₹{mean_absolute_error(yc_test, yc_pred):,.0f}"
          f"   R²: {r2_score(yc_test, yc_pred):.4f}")

    joblib.dump(cost_pipeline, os.path.join(MODEL_DIR, "cost_pipeline.joblib"))
    print(f"[stage2] Saved → {MODEL_DIR}/cost_pipeline.joblib")

    # ----------------------------------------------------------------
    # Model 2 — Delivery Time Regression (LinearRegression)
    # ----------------------------------------------------------------
    time_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor",    LinearRegression()),
    ])
    time_pipeline.fit(X_train, yt_train)
    yt_pred = time_pipeline.predict(X_test)
    print(f"\n[stage2] Time model  — MAE: {mean_absolute_error(yt_test, yt_pred):.3f} days"
          f"   R²: {r2_score(yt_test, yt_pred):.4f}")

    joblib.dump(time_pipeline, os.path.join(MODEL_DIR, "time_pipeline.joblib"))
    print(f"[stage2] Saved → {MODEL_DIR}/time_pipeline.joblib")

    # ----------------------------------------------------------------
    # Model 3 — On-Time Classification (LogisticRegression)
    # ----------------------------------------------------------------
    ontime_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier",   LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
    ])
    ontime_pipeline.fit(X_train, yo_train)
    yo_pred      = ontime_pipeline.predict(X_test)
    yo_pred_prob = ontime_pipeline.predict_proba(X_test)[:, 1]
    print(f"\n[stage2] On-Time model — Accuracy: {accuracy_score(yo_test, yo_pred):.4f}"
          f"   AUC-ROC: {roc_auc_score(yo_test, yo_pred_prob):.4f}")

    joblib.dump(ontime_pipeline, os.path.join(MODEL_DIR, "ontime_pipeline.joblib"))
    print(f"[stage2] Saved → {MODEL_DIR}/ontime_pipeline.joblib")

    print("\n[stage2] ✓ All three pipelines trained and saved successfully.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_and_save_models()

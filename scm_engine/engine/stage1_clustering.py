"""
engine/stage1_clustering.py
============================
K-Means geographic demand-zone clustering with automatic K selection.

The optimal number of clusters (K) is determined iteratively using both:
  • Inertia (elbow method)  — within-cluster sum of squared distances
  • Silhouette score        — measures cohesion vs. separation per cluster

For each K in CLUSTER_RANGE, both metrics are computed. The K with the
highest silhouette score is selected as optimal — a single, unambiguous
criterion that needs no second-derivative math.

Mirrors the "Sales Aggregation" technique in the Network Planning Process:
  "sales that thousands of customers generate can be geographically grouped
   into a limited number of geographic clusters without any significant loss
   in cost-estimating accuracy."

Persists:
  models/kmeans_zones.pkl        — Fitted KMeans at optimal K
  models/zone_scaler.pkl         — StandardScaler fitted on [lat, lon]
  models/cluster_metrics.csv     — Inertia + silhouette for every K tried
  data/historical_orders_clustered.csv — Orders enriched with demand zone cols
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLUSTER_RANGE = range(2, 21)   # evaluate K = 2 … 20; no single value pre-defined
RANDOM_STATE  = 42
DATA_PATH     = r"./scm_engine/data/historical_orders.csv"
OUT_CSV       = r"./scm_engine/data/historical_orders_clustered.csv"
MODEL_DIR     = r"./scm_engine/models"


# ---------------------------------------------------------------------------
# Iterative K selection: inertia + silhouette
# ---------------------------------------------------------------------------

def elbow_and_silhouette(coords_scaled: np.ndarray) -> tuple[int, pd.DataFrame]:
    """
    Evaluate KMeans quality across CLUSTER_RANGE using inertia and silhouette.

    Silhouette score ranges from -1 (poor) to +1 (perfect separation).
    The K with the highest silhouette score is returned as optimal — no
    curvature maths required.

    Parameters
    ----------
    coords_scaled : StandardScaler-transformed (lat, lon) array.

    Returns
    -------
    best_k     : Integer K with highest silhouette score.
    metrics_df : DataFrame with columns [k, inertia, silhouette, is_optimal].
    """
    inertias, silhouettes = {}, {}

    print(f"[stage1] Evaluating K = {CLUSTER_RANGE.start} … {CLUSTER_RANGE.stop - 1}")
    for k in CLUSTER_RANGE:
        km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(coords_scaled)
        inertias[k]    = km.inertia_
        silhouettes[k] = silhouette_score(coords_scaled, labels)
        print(f"         K={k:3d}  inertia={km.inertia_:8.1f}  silhouette={silhouettes[k]:.4f}")

    best_k = max(silhouettes, key=silhouettes.get)
    print(f"\n[stage1] ✓ Optimal K = {best_k}  (silhouette = {silhouettes[best_k]:.4f})")

    metrics_df = pd.DataFrame({
        "k":          list(CLUSTER_RANGE),
        "inertia":    [inertias[k]    for k in CLUSTER_RANGE],
        "silhouette": [silhouettes[k] for k in CLUSTER_RANGE],
        "is_optimal": [k == best_k    for k in CLUSTER_RANGE],
    })
    return best_k, metrics_df


# ---------------------------------------------------------------------------
# Main clustering function
# ---------------------------------------------------------------------------

def fit_demand_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-select K via silhouette score, fit KMeans, and append demand zone
    columns to the orders DataFrame.

    New columns added:
      demand_zone        — string label, e.g. 'DZ-03'
      zone_centroid_lat  — centroid latitude  of the assigned zone
      zone_centroid_lon  — centroid longitude of the assigned zone

    Parameters
    ----------
    df : Historical orders DataFrame (must have customer_lat, customer_lon).

    Returns
    -------
    df : Enriched DataFrame with demand zone columns.
    """
    coords = df[["customer_lat", "customer_lon"]].values

    # Scale so lat and lon contribute equally to Euclidean distance
    scaler        = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    # --- Iterative search — K is data-driven, not developer-defined ---
    best_k, metrics_df = elbow_and_silhouette(coords_scaled)

    # Fit final model at the elected K
    km     = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(coords_scaled)

    df = df.copy()
    df["demand_zone"] = "DZ-" + pd.Series(labels).astype(str).str.zfill(2).values

    # Map each zone to its centroid in original lat/lon space
    centroids_orig = scaler.inverse_transform(km.cluster_centers_)
    centroid_map   = {
        f"DZ-{str(i).zfill(2)}": (centroids_orig[i, 0], centroids_orig[i, 1])
        for i in range(best_k)
    }
    df["zone_centroid_lat"] = df["demand_zone"].map(lambda z: centroid_map[z][0])
    df["zone_centroid_lon"] = df["demand_zone"].map(lambda z: centroid_map[z][1])

    # --- Persist artefacts ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(km,     os.path.join(MODEL_DIR, "kmeans_zones.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "zone_scaler.pkl"))
    metrics_df.to_csv(os.path.join(MODEL_DIR, "cluster_metrics.csv"), index=False)

    print(f"[stage1] KMeans model      → {MODEL_DIR}/kmeans_zones.pkl  (K={best_k})")
    print(f"[stage1] Zone scaler       → {MODEL_DIR}/zone_scaler.pkl")
    print(f"[stage1] Cluster metrics   → {MODEL_DIR}/cluster_metrics.csv")

    # --- Zone summary ---
    summary = (
        df.groupby("demand_zone")
          .agg(
              n_orders  =("order_id",        "count"),
              total_qty =("item_qty",         "sum"),
              centroid_lat=("zone_centroid_lat", "first"),
              centroid_lon=("zone_centroid_lon", "first"),
          )
          .reset_index()
    )
    print(f"\n[stage1] Demand zone summary ({best_k} zones auto-selected):")
    print(summary.to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def assign_demand_zone(cust_lat: float, cust_lon: float) -> str:
    """
    Assign a new customer coordinate to an existing demand zone using the
    persisted KMeans model (inference path — no retraining needed).

    Parameters
    ----------
    cust_lat, cust_lon : Customer coordinates in decimal degrees.

    Returns
    -------
    demand_zone : String label, e.g. 'DZ-04'.
    """
    km     = joblib.load(os.path.join(MODEL_DIR, "kmeans_zones.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "zone_scaler.pkl"))
    label  = km.predict(scaler.transform([[cust_lat, cust_lon]]))[0]
    return f"DZ-{str(label).zfill(2)}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[stage1] Loading orders from {DATA_PATH} …")
    df = pd.read_csv(DATA_PATH)

    df_clustered = fit_demand_zones(df)   # K entirely data-driven

    df_clustered.to_csv(OUT_CSV, index=False)
    print(f"\n[stage1] Saved → {OUT_CSV}  ({len(df_clustered):,} rows)")

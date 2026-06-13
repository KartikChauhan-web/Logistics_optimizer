"""
engine/stage3_sourcing_engine.py
=================================
Predictive Sourcing & Routing Engine — Inference Module.

Core function: evaluate_purchase_order()

Given a new customer PO (lat, lon, qty, warehouse_inventory, score weights),
this module:
  1. Programmatically generates ALL *feasible* candidate routing paths across
     the three topology variants.
  2. For the WH_CUST topology, a warehouse is only included as a candidate
     if its stock_on_hand >= item_qty. Warehouses with insufficient stock are
     silently excluded from the candidate set before scoring begins.
  3. Loads the three trained ML pipelines to predict cost, delivery time,
     and on-time probability for every feasible candidate.
  4. Computes the composite sourcing score:

        sourcing_score = w_cost × norm_cost + w_rel × (1 − pred_ontime_prob)

     Both terms are *penalties* (higher = worse). norm_cost is min-max
     normalised across all candidates in the batch. The lowest score wins.

     Previous formula was:
        sourcing_score = w_cost × norm_cost − w_rel × pred_ontime_prob
     This was incorrect: subtracting reliability rewarded low-reliability
     routes. A route with cost=0 and ontime=0.9 scored −0.36, beating a
     route with cost=0 and ontime=0.99 (scored −0.396) only by accident,
     and making the sign of the reliability term dependent on the cost
     normalisation range rather than business logic.

  5. Returns a DataFrame ranked from best (lowest score) to worst, with a
     stock_on_hand column showing the inventory balance used in the check.

Why logistics_agency is excluded from inference features
---------------------------------------------------------
logistics_agency was removed from the ML feature set in stage2 because it
carries no causal signal (randomly assigned in data generation). It is still
generated in candidates for operational visibility, but is NOT passed to the
models — doing so would re-introduce the "always pick XpressBees" artefact
caused by noise-fitted OHE coefficients.
"""

import os
import joblib
import numpy as np
import pandas as pd
from itertools import product

from scm_engine.data.generate_data import PLANTS, WAREHOUSES, haversine_km

LOGISTICS_AGENCIES = ["BlueDart", "DTDC", "Delhivery", "Ecom Express", "XpressBees"]
TRANSPORT_MODES    = ["Road", "Rail", "Air", "Road+Rail"]

MODEL_DIR = r"./scm_engine/models"

# Feature columns that the stage2 models were trained on.
# logistics_agency is intentionally absent — see module docstring.
MODEL_FEATURE_COLS = [
    "plant_id",
    "warehouse_id",
    "mode",
    "topology_type",
    "demand_zone",
    "total_distance_km",
    "item_qty",
]

# ---------------------------------------------------------------------------
# Warehouse Inventory Balance Sheet
# ---------------------------------------------------------------------------

DEFAULT_WAREHOUSE_INVENTORY: dict[str, int] = {
    "WH-JDP":  1_200,
    "WH-LKO":  3_500,
    "WH-NAG":  800,
    "WH-VIZ":  2_100,
    "WH-COI":  950,
    "WH-BHO":  4_200,
    "WH-PAT":  600,
    "WH-SUR":  5_800,
    "WH-IND":  3_100,
    "WH-CHD":  1_750,
}


# ---------------------------------------------------------------------------
# Model loader (cached at module level after first call)
# ---------------------------------------------------------------------------

_MODELS: dict = {}


def _load_models() -> dict:
    global _MODELS
    if not _MODELS:
        _MODELS["cost"]   = joblib.load(os.path.join(MODEL_DIR, "cost_pipeline.joblib"))
        _MODELS["time"]   = joblib.load(os.path.join(MODEL_DIR, "time_pipeline.joblib"))
        _MODELS["ontime"] = joblib.load(os.path.join(MODEL_DIR, "ontime_pipeline.joblib"))
        print("[sourcing_engine] Models loaded from disk.")
    return _MODELS


# ---------------------------------------------------------------------------
# Candidate vector builder
# ---------------------------------------------------------------------------

def _build_candidate_vectors(
    cust_lat:            float,
    cust_lon:            float,
    item_qty:            int,
    warehouse_inventory: dict[str, int],
) -> pd.DataFrame:
    """
    Generate all *feasible* routing candidate vectors for a new PO.

    Feasibility rules per topology:
      PLANT_WH_CUST — always feasible; plant manufactures to order.
      PLANT_CUST    — always feasible; direct shipment from plant.
      WH_CUST       — feasible ONLY when warehouse_inventory[wh_id] >= item_qty.

    logistics_agency is enumerated for operational completeness (the caller
    may want to display it), but is NOT included in MODEL_FEATURE_COLS and
    will not be passed to the ML models.

    Returns
    -------
    pd.DataFrame of feasible candidate rows.
    """
    rows      = []
    plant_ids = list(PLANTS.keys())
    wh_ids    = list(WAREHOUSES.keys())

    # ----------------------------------------------------------------
    # Topology 1: PLANT_WH_CUST  — Plant → Warehouse → Customer
    # ----------------------------------------------------------------
    for plant_id, wh_id, agency, mode in product(
        plant_ids, wh_ids, LOGISTICS_AGENCIES, TRANSPORT_MODES
    ):
        p    = PLANTS[plant_id]
        w    = WAREHOUSES[wh_id]
        d_pw = haversine_km(p["lat"], p["lon"], w["lat"], w["lon"])
        d_wc = haversine_km(w["lat"], w["lon"], cust_lat, cust_lon)
        rows.append({
            "topology_type":     "PLANT_WH_CUST",
            "plant_id":          plant_id,
            "warehouse_id":      wh_id,
            "logistics_agency":  agency,
            "mode":              mode,
            "item_qty":          item_qty,
            "total_distance_km": round(d_pw + d_wc, 2),
            "stock_on_hand":     np.nan,
            "_plant_lat":        p["lat"], "_plant_lon": p["lon"],
            "_wh_lat":           w["lat"], "_wh_lon":   w["lon"],
        })

    # ----------------------------------------------------------------
    # Topology 2: PLANT_CUST  — Plant → Customer (warehouse bypassed)
    # ----------------------------------------------------------------
    for plant_id, agency, mode in product(
        plant_ids, LOGISTICS_AGENCIES, TRANSPORT_MODES
    ):
        p    = PLANTS[plant_id]
        dist = haversine_km(p["lat"], p["lon"], cust_lat, cust_lon)
        rows.append({
            "topology_type":     "PLANT_CUST",
            "plant_id":          plant_id,
            "warehouse_id":      "BYPASS",
            "logistics_agency":  agency,
            "mode":              mode,
            "item_qty":          item_qty,
            "total_distance_km": round(dist, 2),
            "stock_on_hand":     np.nan,
            "_plant_lat":        p["lat"], "_plant_lon": p["lon"],
            "_wh_lat":           np.nan,   "_wh_lon":   np.nan,
        })

    # ----------------------------------------------------------------
    # Topology 3: WH_CUST  — Warehouse → Customer
    # Hard feasibility gate on stock.
    # ----------------------------------------------------------------
    for wh_id, agency, mode in product(
        wh_ids, LOGISTICS_AGENCIES, TRANSPORT_MODES
    ):
        stock = warehouse_inventory.get(wh_id, 0)
        if stock < item_qty:
            continue

        w    = WAREHOUSES[wh_id]
        dist = haversine_km(w["lat"], w["lon"], cust_lat, cust_lon)
        rows.append({
            "topology_type":     "WH_CUST",
            "plant_id":          "WH_STOCK",
            "warehouse_id":      wh_id,
            "logistics_agency":  agency,
            "mode":              mode,
            "item_qty":          item_qty,
            "total_distance_km": round(dist, 2),
            "stock_on_hand":     stock,
            "_plant_lat":        np.nan,  "_plant_lon": np.nan,
            "_wh_lat":           w["lat"], "_wh_lon":  w["lon"],
        })

    if not rows:
        raise ValueError(
            f"No feasible candidates found for item_qty={item_qty:,}. "
            "All warehouses have insufficient stock and no plant routes were built. "
            "Check PLANTS/WAREHOUSES config."
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Demand zone assignment
# ---------------------------------------------------------------------------

def _get_demand_zone(cust_lat: float, cust_lon: float) -> str:
    try:
        from engine.stage1_clustering import assign_demand_zone
        return assign_demand_zone(cust_lat, cust_lon)
    except Exception:
        return "DZ-00"


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def evaluate_purchase_order(
    cust_lat:            float,
    cust_lon:            float,
    item_qty:            int,
    warehouse_inventory: dict[str, int] | None = None,
    w_cost:              float = 0.6,
    w_rel:               float = 0.4,
) -> pd.DataFrame:
    """
    Evaluate all feasible routing paths for a new purchase order and return
    a DataFrame ranked by composite sourcing score (ascending = better).

    Scoring formula
    ---------------
        sourcing_score = w_cost × norm_cost + w_rel × (1 − pred_ontime_prob)

    Both terms are penalties in [0, 1]:
      • norm_cost            : 0 = cheapest candidate, 1 = most expensive
      • (1 − pred_ontime_prob): 0 = perfectly reliable, 1 = always late

    The lowest combined score is the best route. w_cost + w_rel should
    sum to 1.0 for scores to remain in [0, 1], though the engine does not
    enforce this — unequal sums simply shift the absolute score range.

    Parameters
    ----------
    cust_lat            : Customer latitude (decimal degrees).
    cust_lon            : Customer longitude (decimal degrees).
    item_qty            : Order quantity (units).
    warehouse_inventory : Dict mapping warehouse_id → current stock_on_hand.
                          Defaults to DEFAULT_WAREHOUSE_INVENTORY if None.
    w_cost              : Penalty weight for normalised cost  (0–1).
    w_rel               : Penalty weight for unreliability    (0–1).

    Returns
    -------
    pd.DataFrame : Ranked feasible candidate paths.
    """
    if warehouse_inventory is None:
        warehouse_inventory = DEFAULT_WAREHOUSE_INVENTORY

    models = _load_models()

    eligible_wh = [wh for wh, stock in warehouse_inventory.items() if stock >= item_qty]
    excluded_wh = [wh for wh, stock in warehouse_inventory.items() if stock < item_qty]
    print(f"[sourcing_engine] item_qty={item_qty:,}")
    print(f"[sourcing_engine] WH_CUST eligible  : {eligible_wh or 'none'}")
    print(f"[sourcing_engine] WH_CUST excluded  : {excluded_wh or 'none'} (insufficient stock)")

    demand_zone = _get_demand_zone(cust_lat, cust_lon)
    candidates  = _build_candidate_vectors(
        cust_lat, cust_lon, item_qty, warehouse_inventory
    )
    candidates["demand_zone"] = demand_zone

    # Pass only the features the models were trained on — logistics_agency excluded.
    X_infer = candidates[MODEL_FEATURE_COLS].copy()
    for col in ["plant_id", "warehouse_id", "mode", "topology_type", "demand_zone"]:
        X_infer[col] = X_infer[col].fillna("UNKNOWN").astype(str)

    candidates["pred_delivery_cost"] = models["cost"].predict(X_infer).clip(min=0)
    candidates["pred_delivery_time"] = models["time"].predict(X_infer).clip(min=0)
    candidates["pred_ontime_prob"]   = models["ontime"].predict_proba(X_infer)[:, 1]

    # ----------------------------------------------------------------
    # Composite Sourcing Score
    #
    #   sourcing_score = w_cost × norm_cost + w_rel × (1 − pred_ontime_prob)
    #
    # Both terms are penalties — lower is better.
    # Corrected from the previous formula which *subtracted* reliability,
    # creating a sign error that favoured unreliable routes.
    # ----------------------------------------------------------------
    cost_min   = candidates["pred_delivery_cost"].min()
    cost_max   = candidates["pred_delivery_cost"].max()
    cost_range = cost_max - cost_min if cost_max > cost_min else 1.0

    candidates["norm_cost"]      = (candidates["pred_delivery_cost"] - cost_min) / cost_range
    candidates["unreliability"]  = 1.0 - candidates["pred_ontime_prob"]
    candidates["sourcing_score"] = (
        w_cost * candidates["norm_cost"]
        + w_rel * candidates["unreliability"]
    )

    result = (
        candidates
        .sort_values("sourcing_score")
        .reset_index(drop=True)
    )
    result.index      = result.index + 1
    result.index.name = "rank"

    display_cols = [
        "topology_type", "plant_id", "warehouse_id",
        "logistics_agency", "mode", "demand_zone",
        "total_distance_km", "stock_on_hand",
        "pred_delivery_cost", "pred_delivery_time",
        "pred_ontime_prob", "unreliability", "norm_cost", "sourcing_score",
        "_plant_lat", "_plant_lon", "_wh_lat", "_wh_lon",
    ]
    return result[display_cols]


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("=" * 70)
    print("Case 1: item_qty=150 — most warehouses should qualify for WH_CUST")
    df1 = evaluate_purchase_order(cust_lat=18.52, cust_lon=73.86, item_qty=150)
    wh_cust_rows = df1[df1["topology_type"] == "WH_CUST"]
    print(f"WH_CUST candidates generated: {len(wh_cust_rows)}")
    print(df1.head(5)[["topology_type", "warehouse_id", "mode", "stock_on_hand",
                        "pred_delivery_cost", "sourcing_score"]].to_string())

    print("\n" + "=" * 70)
    print("Case 2: item_qty=4000 — only WH-BHO (4200) and WH-SUR (5800) qualify")
    df2 = evaluate_purchase_order(cust_lat=18.52, cust_lon=73.86, item_qty=4000)
    wh_cust_rows2 = df2[df2["topology_type"] == "WH_CUST"]
    print(f"WH_CUST candidates generated: {len(wh_cust_rows2)}")
    print(f"Unique WH_CUST warehouses: {wh_cust_rows2['warehouse_id'].unique()}")

    print("\n" + "=" * 70)
    print("Case 3: item_qty=10000 — no warehouse qualifies, only plant routes")
    df3 = evaluate_purchase_order(cust_lat=18.52, cust_lon=73.86, item_qty=10_000)
    wh_cust_rows3 = df3[df3["topology_type"] == "WH_CUST"]
    print(f"WH_CUST candidates generated: {len(wh_cust_rows3)}")
    print(df3.head(5)[["topology_type", "warehouse_id", "mode", "stock_on_hand",
                        "pred_delivery_cost", "sourcing_score"]].to_string())

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

        sourcing_score = (w_cost × norm_cost) − (w_rel × pred_ontime_prob)

     where norm_cost is min-max normalised across all candidates in the batch.
  5. Returns a DataFrame ranked from best (lowest score) to worst, with a
     stock_on_hand column showing the inventory balance used in the check.

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

# ---------------------------------------------------------------------------
# Warehouse Inventory Balance Sheet
# ---------------------------------------------------------------------------
# Maps every warehouse_id to its current stock_on_hand (units).
# In production this would be hydrated from a live WMS / ERP query.
# Developers can override this dict or pass a custom snapshot directly into
# evaluate_purchase_order() via the `warehouse_inventory` parameter.

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
      PLANT_WH_CUST — always feasible; plant manufactures to order, warehouse
                      acts as a cross-dock / transit hub (no stock check needed).
      PLANT_CUST    — always feasible; direct shipment from plant bypasses
                      warehouse entirely.
      WH_CUST       — feasible ONLY when warehouse_inventory[wh_id] >= item_qty.
                      Warehouses with insufficient stock are excluded entirely
                      from the candidate set, not just scored poorly.

    Parameters
    ----------
    cust_lat, cust_lon   : Customer coordinates.
    item_qty             : Order quantity (units).
    warehouse_inventory  : Dict mapping warehouse_id → stock_on_hand (units).
                           Only warehouses present in this dict and with
                           stock_on_hand >= item_qty qualify for WH_CUST.

    Returns
    -------
    pd.DataFrame of feasible candidate rows, including a stock_on_hand column
    (NaN for topologies that do not consume warehouse inventory).
    """
    rows      = []
    plant_ids = list(PLANTS.keys())
    wh_ids    = list(WAREHOUSES.keys())

    # ----------------------------------------------------------------
    # Topology 1: PLANT_WH_CUST  — Plant → Warehouse → Customer
    # No warehouse stock check: product is manufactured and cross-docked.
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
            "stock_on_hand":     np.nan,   # not applicable — make-to-order
            "_plant_lat":        p["lat"], "_plant_lon": p["lon"],
            "_wh_lat":           w["lat"], "_wh_lon":   w["lon"],
        })

    # ----------------------------------------------------------------
    # Topology 2: PLANT_CUST  — Plant → Customer (warehouse bypassed)
    # No warehouse touched — no stock check needed.
    # ----------------------------------------------------------------
    for plant_id, agency, mode in product(
        plant_ids, LOGISTICS_AGENCIES, TRANSPORT_MODES
    ):
        p    = PLANTS[plant_id]
        dist = haversine_km(p["lat"], p["lon"], cust_lat, cust_lon)
        rows.append({
            "topology_type":     "PLANT_CUST",
            "plant_id":          plant_id,
            "warehouse_id":      "BYPASS",   # topology override: no warehouse leg
            "logistics_agency":  agency,
            "mode":              mode,
            "item_qty":          item_qty,
            "total_distance_km": round(dist, 2),
            "stock_on_hand":     np.nan,   # not applicable — warehouse bypassed
            "_plant_lat":        p["lat"], "_plant_lon": p["lon"],
            "_wh_lat":           np.nan,   "_wh_lon":   np.nan,
        })

    # ----------------------------------------------------------------
    # Topology 3: WH_CUST  — Warehouse → Customer (regional fulfilment)
    # Per-warehouse stock check: only include wh_id if its stock_on_hand
    # covers the full order quantity. A warehouse with 5 units on hand
    # is simply not a valid candidate for a 10,000-unit order.
    # ----------------------------------------------------------------
    for wh_id, agency, mode in product(
        wh_ids, LOGISTICS_AGENCIES, TRANSPORT_MODES
    ):
        stock = warehouse_inventory.get(wh_id, 0)

        # Hard feasibility gate — skip this warehouse if stock is insufficient
        if stock < item_qty:
            continue

        w    = WAREHOUSES[wh_id]
        dist = haversine_km(w["lat"], w["lon"], cust_lat, cust_lon)
        rows.append({
            "topology_type":     "WH_CUST",
            "plant_id":          "WH_STOCK",  # topology override: no plant leg
            "warehouse_id":      wh_id,
            "logistics_agency":  agency,
            "mode":              mode,
            "item_qty":          item_qty,
            "total_distance_km": round(dist, 2),
            "stock_on_hand":     stock,   # surfaced in output for transparency
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

    Parameters
    ----------
    cust_lat            : Customer latitude (decimal degrees).
    cust_lon            : Customer longitude (decimal degrees).
    item_qty            : Order quantity (units).
    warehouse_inventory : Dict mapping warehouse_id → current stock_on_hand.
                          Defaults to DEFAULT_WAREHOUSE_INVENTORY if None.
                          Pass a custom dict to reflect live WMS balances.
                          A warehouse is only eligible for WH_CUST routing
                          if its stock_on_hand >= item_qty.
    w_cost              : Weight applied to normalised cost  (0–1).
    w_rel               : Weight applied to on-time probability (0–1).

    Returns
    -------
    pd.DataFrame : Ranked feasible candidate paths. WH_CUST rows include a
                   stock_on_hand column; PLANT_WH_CUST and PLANT_CUST rows
                   show NaN for that column (no warehouse stock consumed).
    """
    if warehouse_inventory is None:
        warehouse_inventory = DEFAULT_WAREHOUSE_INVENTORY

    models = _load_models()

    # Log which warehouses are eligible before building candidates
    eligible_wh = [
        wh for wh, stock in warehouse_inventory.items() if stock >= item_qty
    ]
    excluded_wh = [
        wh for wh, stock in warehouse_inventory.items() if stock < item_qty
    ]
    print(f"[sourcing_engine] item_qty={item_qty:,}")
    print(f"[sourcing_engine] WH_CUST eligible  : {eligible_wh or 'none'}")
    print(f"[sourcing_engine] WH_CUST excluded  : {excluded_wh or 'none'} (insufficient stock)")

    demand_zone = _get_demand_zone(cust_lat, cust_lon)
    candidates  = _build_candidate_vectors(
        cust_lat, cust_lon, item_qty, warehouse_inventory
    )
    candidates["demand_zone"] = demand_zone

    feature_cols = [
        "plant_id", "warehouse_id", "logistics_agency",
        "mode", "topology_type", "demand_zone",
        "total_distance_km", "item_qty",
    ]
    X_infer = candidates[feature_cols].copy()

    candidates["pred_delivery_cost"] = models["cost"].predict(X_infer).clip(min=0)
    candidates["pred_delivery_time"] = models["time"].predict(X_infer).clip(min=0)
    candidates["pred_ontime_prob"]   = models["ontime"].predict_proba(X_infer)[:, 1]

    # ----------------------------------------------------------------
    # Composite Sourcing Score
    # $$\text{sourcing\_score} = (w_{\text{cost}} \times \text{norm\_cost})
    #   - (w_{\text{rel}} \times \hat{p}_{\text{on-time}})$$
    # ----------------------------------------------------------------
    cost_min   = candidates["pred_delivery_cost"].min()
    cost_max   = candidates["pred_delivery_cost"].max()
    cost_range = cost_max - cost_min if cost_max > cost_min else 1.0

    candidates["norm_cost"]      = (candidates["pred_delivery_cost"] - cost_min) / cost_range
    candidates["sourcing_score"] = (
        w_cost * candidates["norm_cost"]
        - w_rel * candidates["pred_ontime_prob"]
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
        "pred_ontime_prob", "norm_cost", "sourcing_score",
        "_plant_lat", "_plant_lon", "_wh_lat", "_wh_lon",
    ]
    return result[display_cols]


# ---------------------------------------------------------------------------
# Quick CLI test — demonstrates stock-gating behaviour
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    # Case 1: Small order — most warehouses qualify
    print("=" * 70)
    print("Case 1: item_qty=150 — most warehouses should qualify for WH_CUST")
    df1 = evaluate_purchase_order(cust_lat=18.52, cust_lon=73.86, item_qty=150)
    wh_cust_rows = df1[df1["topology_type"] == "WH_CUST"]
    print(f"WH_CUST candidates generated: {len(wh_cust_rows)}")
    print(df1.head(5)[["topology_type","warehouse_id","stock_on_hand","pred_delivery_cost","sourcing_score"]].to_string())

    # Case 2: Large order — only warehouses with sufficient stock qualify
    print("\n" + "=" * 70)
    print("Case 2: item_qty=4000 — only WH-BHO (4200) and WH-SUR (5800) qualify")
    df2 = evaluate_purchase_order(cust_lat=18.52, cust_lon=73.86, item_qty=4000)
    wh_cust_rows2 = df2[df2["topology_type"] == "WH_CUST"]
    print(f"WH_CUST candidates generated: {len(wh_cust_rows2)}")
    unique_wh = wh_cust_rows2["warehouse_id"].unique()
    print(f"Unique WH_CUST warehouses: {unique_wh}")

    # Case 3: Very large order — no warehouse qualifies, only plant routes
    print("\n" + "=" * 70)
    print("Case 3: item_qty=10000 — no warehouse qualifies, zero WH_CUST candidates")
    df3 = evaluate_purchase_order(cust_lat=18.52, cust_lon=73.86, item_qty=10_000)
    wh_cust_rows3 = df3[df3["topology_type"] == "WH_CUST"]
    print(f"WH_CUST candidates generated: {len(wh_cust_rows3)}")
    print(df3.head(5)[["topology_type","warehouse_id","stock_on_hand","pred_delivery_cost","sourcing_score"]].to_string())

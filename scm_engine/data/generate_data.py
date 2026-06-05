"""
data/generate_data.py
=====================
Synthetic historical order generator for an Indian multi-echelon logistics network.
Simulates 3,000 orders distributed across three routing topologies:
  - PLANT_WH_CUST  : Plant → Warehouse → Customer  (Traditional Echelon)
  - PLANT_CUST     : Plant → Customer               (Direct Plant Bypass)
  - WH_CUST        : Warehouse → Customer           (Regional Fulfilment)

Distances are computed using the Haversine formula (great-circle distance in km).
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# ---------------------------------------------------------------------------
# Indian Network Topology: Plants & Warehouses
# ---------------------------------------------------------------------------

PLANTS = {
    "PLT-MUM": {"name": "Mumbai Plant",     "lat": 19.0760, "lon": 72.8777, "region": "West"},
    "PLT-DEL": {"name": "Delhi Plant",      "lat": 28.7041, "lon": 77.1025, "region": "North"},
    "PLT-CHE": {"name": "Chennai Plant",    "lat": 13.0827, "lon": 80.2707, "region": "South"},
    "PLT-KOL": {"name": "Kolkata Plant",    "lat": 22.5726, "lon": 88.3639, "region": "East"},
    "PLT-HYD": {"name": "Hyderabad Plant",  "lat": 17.3850, "lon": 78.4867, "region": "South"},
    "PLT-AHM": {"name": "Ahmedabad Plant",  "lat": 23.0225, "lon": 72.5714, "region": "West"},
    "PLT-PUN": {"name": "Pune Plant",       "lat": 18.5204, "lon": 73.8567, "region": "West"},
    "PLT-BLR": {"name": "Bengaluru Plant",  "lat": 12.9716, "lon": 77.5946, "region": "South"},
}

WAREHOUSES = {
    "WH-JDP":  {"name": "Jodhpur WH",       "lat": 26.2389, "lon": 73.0243, "region": "North"},
    "WH-LKO":  {"name": "Lucknow WH",       "lat": 26.8467, "lon": 80.9462, "region": "North"},
    "WH-NAG":  {"name": "Nagpur WH",        "lat": 21.1458, "lon": 79.0882, "region": "Central"},
    "WH-VIZ":  {"name": "Visakhapatnam WH", "lat": 17.6868, "lon": 83.2185, "region": "East"},
    "WH-COI":  {"name": "Coimbatore WH",    "lat": 11.0168, "lon": 76.9558, "region": "South"},
    "WH-BHO":  {"name": "Bhopal WH",        "lat": 23.2599, "lon": 77.4126, "region": "Central"},
    "WH-PAT":  {"name": "Patna WH",         "lat": 25.5941, "lon": 85.1376, "region": "East"},
    "WH-SUR":  {"name": "Surat WH",         "lat": 21.1702, "lon": 72.8311, "region": "West"},
    "WH-IND":  {"name": "Indore WH",        "lat": 22.7196, "lon": 75.8577, "region": "Central"},
    "WH-CHD":  {"name": "Chandigarh WH",    "lat": 30.7333, "lon": 76.7794, "region": "North"},
}

# Customer demand zones spread across India (lat, lon bounding boxes per region)
DEMAND_REGIONS = {
    "North":   {"lat": (26.0, 32.0), "lon": (74.0, 80.0)},
    "South":   {"lat": (8.5,  16.0), "lon": (76.0, 80.5)},
    "East":    {"lat": (20.0, 26.0), "lon": (82.0, 88.0)},
    "West":    {"lat": (20.0, 24.0), "lon": (70.0, 75.0)},
    "Central": {"lat": (21.0, 25.0), "lon": (77.0, 82.0)},
    "NE":      {"lat": (24.0, 28.0), "lon": (88.0, 94.0)},
}

LOGISTICS_AGENCIES = ["BlueDart",  "DTDC", "Delhivery", "Ecom Express", "XpressBees"]
TRANSPORT_MODES    = ["Road", "Rail", "Air", "Road+Rail"]

# Topology mix — roughly 50 % traditional, 30 % direct plant, 20 % regional WH
TOPOLOGY_WEIGHTS = {"PLANT_WH_CUST": 0.50, "PLANT_CUST": 0.30, "WH_CUST": 0.20}

# ---------------------------------------------------------------------------
# Utility: Haversine distance (km)
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres between two (lat, lon) points."""
    R = 6371.0
    φ1, φ2 = radians(lat1), radians(lat2)
    dφ = radians(lat2 - lat1)
    dλ = radians(lon2 - lon1)
    a = sin(dφ / 2) ** 2 + cos(φ1) * cos(φ2) * sin(dλ / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# ---------------------------------------------------------------------------
# Cost & time simulation helpers
# ---------------------------------------------------------------------------

_COST_PER_KM = {"Road": 12.5, "Rail": 6.0, "Air": 45.0, "Road+Rail": 9.0}   # ₹ per km per unit
_TIME_PER_KM = {"Road": 0.020, "Rail": 0.015, "Air": 0.005, "Road+Rail": 0.018}  # days per km


def simulate_leg(origin_lat, origin_lon, dest_lat, dest_lon, qty, mode, noise_std=0.12):
    """
    Simulate cost (₹) and time (days) for a single shipping leg.
    Adds multiplicative noise to mimic real-world variability.
    """
    dist_km = haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
    cost_raw = _COST_PER_KM[mode] * dist_km * qty
    time_raw = _TIME_PER_KM[mode] * dist_km

    noise_c = np.random.normal(1.0, noise_std)
    noise_t = np.random.normal(1.0, noise_std * 0.8)

    cost = max(cost_raw * noise_c, 500.0)       # floor ₹500
    time = max(time_raw * noise_t, 0.5)         # floor 0.5 days
    return round(dist_km, 2), round(cost, 2), round(time, 3)

# ---------------------------------------------------------------------------
# Main data generation function
# ---------------------------------------------------------------------------

def generate_orders(n_orders: int = 3000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate *n_orders* synthetic historical orders.

    Returns a DataFrame with one row per order and columns covering
    node identities, coordinates, distances, cost, time, and the
    binary on-time flag.
    """
    np.random.seed(random_seed)

    plant_ids = list(PLANTS.keys())
    wh_ids    = list(WAREHOUSES.keys())
    topologies = list(TOPOLOGY_WEIGHTS.keys())
    topo_probs = list(TOPOLOGY_WEIGHTS.values())

    records = []

    for order_idx in range(n_orders):
        order_id = f"ORD-{order_idx + 1:05d}"
        topology = np.random.choice(topologies, p=topo_probs)

        # --- Customer location: sample from a random demand region ---
        region_key = np.random.choice(list(DEMAND_REGIONS.keys()))
        reg = DEMAND_REGIONS[region_key]
        cust_lat = np.random.uniform(*reg["lat"])
        cust_lon = np.random.uniform(*reg["lon"])

        qty = int(np.random.lognormal(mean=3.5, sigma=0.8))  # units, heavy-tailed
        qty = max(1, min(qty, 5000))

        agency = np.random.choice(LOGISTICS_AGENCIES)
        mode   = np.random.choice(TRANSPORT_MODES)

        # ----------------------------------------------------------------
        # Multi-topology routing overrides
        # ----------------------------------------------------------------
        if topology == "PLANT_WH_CUST":
            # Leg 1: Plant → Warehouse
            plant_id = np.random.choice(plant_ids)
            wh_id    = np.random.choice(wh_ids)
            p = PLANTS[plant_id];  w = WAREHOUSES[wh_id]

            d1, c1, t1 = simulate_leg(p["lat"], p["lon"], w["lat"], w["lon"], qty, mode)
            # Leg 2: Warehouse → Customer
            d2, c2, t2 = simulate_leg(w["lat"], w["lon"], cust_lat, cust_lon, qty, mode)

            total_dist = d1 + d2
            total_cost = c1 + c2
            total_time = t1 + t2
            wh_lat, wh_lon = w["lat"], w["lon"]

        elif topology == "PLANT_CUST":
            # Direct bypass: Plant → Customer  (warehouse is BYPASS)
            plant_id = np.random.choice(plant_ids)
            wh_id    = "BYPASS"              # << topology override marker
            p = PLANTS[plant_id]

            total_dist, total_cost, total_time = simulate_leg(
                p["lat"], p["lon"], cust_lat, cust_lon, qty, mode
            )
            wh_lat, wh_lon = np.nan, np.nan   # no warehouse node

        else:  # WH_CUST
            # Regional fulfilment: Warehouse → Customer  (plant is WH_STOCK)
            plant_id = "WH_STOCK"            # << topology override marker
            wh_id    = np.random.choice(wh_ids)
            w = WAREHOUSES[wh_id]

            total_dist, total_cost, total_time = simulate_leg(
                w["lat"], w["lon"], cust_lat, cust_lon, qty, mode
            )
            wh_lat, wh_lon = w["lat"], w["lon"]

        # On-time: probability decreases with distance and cost overruns
        on_time_prob = np.clip(0.92 - total_dist * 0.00012 - np.random.uniform(0, 0.1), 0.05, 0.98)
        is_on_time   = int(np.random.random() < on_time_prob)

        # Plant coordinates (NaN for WH_STOCK orders)
        if plant_id not in ("WH_STOCK",):
            p_info = PLANTS[plant_id]
            plant_lat, plant_lon = p_info["lat"], p_info["lon"]
        else:
            plant_lat, plant_lon = np.nan, np.nan

        records.append({
            "order_id":              order_id,
            "topology_type":         topology,
            "plant_id":              plant_id,
            "warehouse_id":          wh_id,
            "plant_lat":             plant_lat,
            "plant_lon":             plant_lon,
            "warehouse_lat":         wh_lat,
            "warehouse_lon":         wh_lon,
            "customer_lat":          round(cust_lat, 6),
            "customer_lon":          round(cust_lon, 6),
            "item_qty":              qty,
            "logistics_agency":      agency,
            "mode":                  mode,
            "total_distance_km":     total_dist,
            "actual_delivery_cost":  total_cost,
            "actual_delivery_time":  total_time,
            "is_on_time":            is_on_time,
        })

    df = pd.DataFrame(records)
    print(f"[generate_data] Generated {len(df):,} orders.")
    print(df["topology_type"].value_counts())
    return df


if __name__ == "__main__":
    import os
    os.makedirs(r"./scm_engine/data", exist_ok=True)
    df = generate_orders(n_orders=3000)
    out_path = r"./scm_engine/data/historical_orders.csv"
    df.to_csv(out_path, index=False)
    print(f"[generate_data] Saved → {out_path}")

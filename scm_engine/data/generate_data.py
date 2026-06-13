"""
data/generate_data.py
=====================
Synthetic historical order generator for an Indian multi-echelon logistics network.
Simulates 3,000 orders distributed across three routing topologies:
  - PLANT_WH_CUST  : Plant → Warehouse → Customer  (Traditional Echelon)
  - PLANT_CUST     : Plant → Customer               (Direct Plant Bypass)
  - WH_CUST        : Warehouse → Customer           (Regional Fulfilment)

Distances are computed using the Haversine formula (great-circle distance in km).

Transport cost model
---------------------
Pricing follows two real-world commercial buckets:

  FTL (Full Truckload) / FCL
    - Each truck has a fixed capacity of FTL_THRESHOLD_KG (500 kg).
    - The shipment is packed greedily: as many full trucks as needed are
      dispatched at the FTL flat rate; any remainder travels LTL.
    - FTL truck cost = flat lane charge + per-km truck rate.
      Weight inside a truck does not affect its price — you rented the vehicle.
    - Example: 1100 kg → 2 full trucks (FTL) + 100 kg remainder (LTL).

  LTL (Less-Than-Truckload) / Parcel Express
    - Applied to the entire shipment when weight < FTL_THRESHOLD_KG,
      or to the remainder after full trucks are filled.
    - Cost = base handling charge + distance component + tiered weight cost.
    - Weight tiers reflect real courier slab / rate-card pricing:
        0–50 kg   → parcel / courier rate
        50–200 kg → small pallet rate
        200–500 kg→ multi-pallet LTL rate
    - Each tier has a lower marginal ₹/kg/100km rate (bulk discounts).
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

LOGISTICS_AGENCIES = ["BlueDart", "DTDC", "Delhivery", "Ecom Express", "XpressBees"]
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
# Cost & time simulation — FTL vs LTL/Parcel model
# ---------------------------------------------------------------------------

AVG_ITEM_WEIGHT_KG = 0.5    # assumed average weight per unit
FTL_THRESHOLD_KG   = 500.0  # shipments at or above this weight are treated as FTL

# Transit time: days per km, by mode (unchanged from original)
_TIME_PER_KM = {"Road": 0.020, "Rail": 0.015, "Air": 0.005, "Road+Rail": 0.018}

# --- FTL pricing ---
# You rent the entire truck/container. Cost = flat lane charge + per-km truck rate.
# Quantity has ZERO influence; you pay for the vehicle and the driver's time.
_FTL_BASE_COST   = {"Road": 8_000,  "Rail": 5_000,  "Air": 40_000, "Road+Rail": 6_500}   # ₹ per shipment
_FTL_COST_PER_KM = {"Road": 18.0,   "Rail": 10.0,   "Air": 80.0,   "Road+Rail": 14.0}    # ₹ per km

# --- LTL / Parcel pricing ---
# You share space. Cost = base handling + distance component + tiered weight cost.
_LTL_BASE_COST   = {"Road": 400,    "Rail": 300,    "Air": 1_200,  "Road+Rail": 350}      # ₹ flat handling
_LTL_COST_PER_KM = {"Road": 3.0,    "Rail": 1.8,    "Air": 12.0,   "Road+Rail": 2.5}     # ₹ per km (distance)

# Weight tiers for LTL: (upper_kg_limit, ₹_per_kg_per_100km)
# Marginal rate decreases as weight increases — mirrors real courier slab cards.
_LTL_WEIGHT_TIERS = [
    (50,  4.50),   # Parcel / courier slab      (0 – 50 kg)
    (200, 3.20),   # Small pallet               (50 – 200 kg)
    (500, 2.10),   # Multi-pallet LTL           (200 – 500 kg)
]
_LTL_OVERFLOW_RATE = 1.60   # ₹/kg/100km for weight beyond the last tier (shouldn't trigger
                              # if FTL_THRESHOLD_KG == last tier upper bound, but kept as a
                              # safety fallback in case thresholds are tuned independently)


def _ltl_weight_cost(weight_kg: float, dist_km: float) -> float:
    """
    Tiered weight-based cost component for LTL shipments.

    Rate is expressed as ₹ per kg per 100 km, applied band-by-band so that
    each additional kg falls into the correct discount tier — identical in
    structure to how BlueDart / Delhivery rate cards work.
    """
    dist_units = dist_km / 100.0  # normalise to 100-km units
    cost       = 0.0
    prev_limit = 0.0

    for (upper, rate) in _LTL_WEIGHT_TIERS:
        band = min(weight_kg, upper) - prev_limit
        if band <= 0:
            break
        cost      += band * rate * dist_units
        prev_limit = upper
        if weight_kg <= upper:
            break
    else:
        # Overflow: weight exceeds every defined tier
        overflow = weight_kg - prev_limit
        if overflow > 0:
            cost += overflow * _LTL_OVERFLOW_RATE * dist_units

    return cost


def simulate_leg(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    qty: int,
    mode: str,
    noise_std: float = 0.12,
) -> tuple[float, float, float]:
    """
    Simulate cost (₹) and transit time (days) for a single shipping leg.

    Pricing logic
    -------------
    weight = qty × AVG_ITEM_WEIGHT_KG

    A truck has a finite capacity of FTL_THRESHOLD_KG. The shipment is packed
    greedily into as many full trucks as required; any remainder travels LTL.

    Example — weight = 1100 kg, FTL_THRESHOLD = 500 kg:
        full_trucks  = floor(1100 / 500) = 2   → 2 × FTL charge
        remainder_kg = 1100 % 500        = 100 → LTL charge on 100 kg

    FTL truck cost  = FTL_BASE_COST[mode] + FTL_COST_PER_KM[mode] × dist_km
                      (one flat charge per truck — weight inside the truck
                       is irrelevant; you rented the whole vehicle)

    LTL remainder   = LTL_BASE_COST[mode]
                    + LTL_COST_PER_KM[mode] × dist_km
                    + tiered_weight_cost(remainder_kg, dist_km)
                      (only charged when remainder_kg > 0)

    Total cost      = n_full_trucks × FTL_truck_cost + LTL_remainder_cost

    Multiplicative noise is applied once to the total cost and separately to
    transit time to simulate real-world variability (fuel surcharges, delays).

    Transit time is driven by distance and mode only — adding more trucks on
    the same lane does not extend the delivery window (they travel in parallel).

    Returns
    -------
    (distance_km, cost_inr, time_days)
    """
    dist_km   = haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
    weight_kg = qty * AVG_ITEM_WEIGHT_KG

    n_full_trucks = int(weight_kg // FTL_THRESHOLD_KG)   # number of full trucks needed
    remainder_kg  = weight_kg % FTL_THRESHOLD_KG          # leftover weight for LTL

    # Cost of each full FTL truck: flat lane charge + per-km rate.
    # The weight loaded inside the truck does not affect this charge —
    # you are paying for the vehicle, not the cargo weight.
    ftl_truck_cost = _FTL_BASE_COST[mode] + _FTL_COST_PER_KM[mode] * dist_km

    # LTL cost for the remainder (only charged when there is a remainder).
    if remainder_kg > 0:
        ltl_remainder_cost = (
            _LTL_BASE_COST[mode]
            + _LTL_COST_PER_KM[mode] * dist_km
            + _ltl_weight_cost(remainder_kg, dist_km)
        )
    else:
        ltl_remainder_cost = 0.0

    cost_raw = n_full_trucks * ftl_truck_cost + ltl_remainder_cost

    # Pure LTL shipment (weight never reached a full truck)
    if n_full_trucks == 0:
        cost_raw = (
            _LTL_BASE_COST[mode]
            + _LTL_COST_PER_KM[mode] * dist_km
            + _ltl_weight_cost(weight_kg, dist_km)
        )

    # Transit time: distance × mode rate.
    # Multiple trucks travel the same lane in parallel — delivery time is
    # not multiplied by the number of trucks.
    time_raw = _TIME_PER_KM[mode] * dist_km

    noise_c = np.random.normal(1.0, noise_std)
    noise_t = np.random.normal(1.0, noise_std * 0.8)

    cost = max(cost_raw * noise_c, 500.0)   # floor ₹ 500
    time = max(time_raw * noise_t, 0.5)     # floor 0.5 days
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

    plant_ids  = list(PLANTS.keys())
    wh_ids     = list(WAREHOUSES.keys())
    topologies = list(TOPOLOGY_WEIGHTS.keys())
    topo_probs = list(TOPOLOGY_WEIGHTS.values())

    records = []

    for order_idx in range(n_orders):
        order_id = f"ORD-{order_idx + 1:05d}"
        topology = np.random.choice(topologies, p=topo_probs)

        # --- Customer location: sample from a random demand region ---
        region_key = np.random.choice(list(DEMAND_REGIONS.keys()))
        reg        = DEMAND_REGIONS[region_key]
        cust_lat   = np.random.uniform(*reg["lat"])
        cust_lon   = np.random.uniform(*reg["lon"])

        qty    = int(np.random.lognormal(mean=3.5, sigma=0.8))  # units, heavy-tailed
        qty    = max(1, min(qty, 5000))
        agency = np.random.choice(LOGISTICS_AGENCIES)
        mode   = np.random.choice(TRANSPORT_MODES)

        # ----------------------------------------------------------------
        # Multi-topology routing
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
            # Direct bypass: Plant → Customer  (no warehouse node)
            plant_id = np.random.choice(plant_ids)
            wh_id    = "BYPASS"
            p = PLANTS[plant_id]

            total_dist, total_cost, total_time = simulate_leg(
                p["lat"], p["lon"], cust_lat, cust_lon, qty, mode
            )
            wh_lat, wh_lon = np.nan, np.nan

        else:  # WH_CUST
            # Regional fulfilment: Warehouse → Customer  (no plant involvement)
            plant_id = "WH_STOCK"
            wh_id    = np.random.choice(wh_ids)
            w = WAREHOUSES[wh_id]

            total_dist, total_cost, total_time = simulate_leg(
                w["lat"], w["lon"], cust_lat, cust_lon, qty, mode
            )
            wh_lat, wh_lon = w["lat"], w["lon"]

        # On-time probability decreases with distance and cost overruns
        on_time_prob = np.clip(
            0.92 - total_dist * 0.00012 - np.random.uniform(0, 0.1), 0.05, 0.98
        )
        is_on_time = int(np.random.random() < on_time_prob)

        # Plant coordinates (NaN for WH_STOCK orders)
        if plant_id != "WH_STOCK":
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

"""
app.py
=======
Streamlit interactive dashboard for the Predictive Sourcing & Routing Engine.

Features:
  • Sidebar  : Customer lat/lon, order qty, inventory toggle, score weights
  • KPI cards: Best cost, delivery time, on-time probability, sourcing score
  • Geo map  : Plotly Scattergeo — dynamically draws the optimal routing line
               (topology-aware: 1-hop or 2-hop path on India map)
  • Ranked shortlist table: top-N candidate paths with colour-coded scores

Run:
    streamlit run app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Map the parent directory so 'scm_engine.data' can resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="India Sourcing Engine",
    page_icon="🚚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — industrial dark theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d1117; color: #e6edf3; }
  [data-testid="stSidebar"]           { background: #161b22; }
  [data-testid="stHeader"]            { background: transparent; }
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
  }
  .metric-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase;
                  letter-spacing: 1px; margin-bottom: 6px; }
  .metric-value { font-size: 1.9rem; font-weight: 700; color: #58a6ff; }
  .metric-delta { font-size: 0.78rem; color: #3fb950; margin-top: 4px; }
  .topo-badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.5px;
  }
  .badge-pwc { background: #1f4068; color: #58a6ff; }
  .badge-pc  { background: #1a3a2a; color: #3fb950; }
  .badge-wc  { background: #3d1f1f; color: #f78166; }
  h1, h2, h3 { color: #e6edf3 !important; }
  .stDataFrame { border: 1px solid #30363d !important; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — PO inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/delivery.png", width=40)
    st.markdown("## 📦 New Purchase Order")
    st.markdown("---")

    st.markdown("### 📍 Customer Location")
    col1, col2 = st.columns(2)
    with col1:
        cust_lat = st.number_input("Latitude",  value=18.52, min_value=6.0,  max_value=36.0, step=0.01, format="%.4f")
    with col2:
        cust_lon = st.number_input("Longitude", value=73.86, min_value=67.0, max_value=97.0, step=0.01, format="%.4f")

    st.markdown("### 🏭 Order Details")
    item_qty = st.slider("Order Quantity (units)", min_value=1, max_value=10_000, value=150, step=50)

    st.markdown("### 🏪 Warehouse Inventory (units on hand)")
    st.caption("Warehouses with stock < order qty are automatically excluded from WH→CUST routing.")
    from engine.stage3_sourcing_engine import DEFAULT_WAREHOUSE_INVENTORY
    warehouse_inventory = {}
    for wh_id, default_stock in DEFAULT_WAREHOUSE_INVENTORY.items():
        warehouse_inventory[wh_id] = st.number_input(
            wh_id, min_value=0, max_value=50_000,
            value=default_stock, step=100, key=f"stock_{wh_id}"
        )

    st.markdown("### ⚖️ Score Weights")
    w_cost = st.slider("Cost Weight (w_cost)", 0.0, 1.0, 0.6, 0.05)
    w_rel  = st.slider("Reliability Weight (w_rel)", 0.0, 1.0, 0.4, 0.05)
    st.caption(f"Weights sum: **{w_cost + w_rel:.2f}** (do not need to sum to 1)")

    top_n = st.slider("Shortlist size", 5, 30, 10)
    st.markdown("---")
    run_btn = st.button("🔍 Evaluate PO", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.markdown("# 🚚 India Predictive Sourcing & Routing Engine")
st.markdown("*Multi-echelon network · ML-driven path evaluation · Three topology variants*")
st.markdown("---")

# ---------------------------------------------------------------------------
# Load models on first run
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading ML models …")
def load_engine():
    from engine.stage3_sourcing_engine import evaluate_purchase_order
    return evaluate_purchase_order


evaluate_purchase_order = load_engine()

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

if run_btn or "results" not in st.session_state:
    with st.spinner("Evaluating all candidate paths …"):
        try:
            df_results = evaluate_purchase_order(
                cust_lat=cust_lat,
                cust_lon=cust_lon,
                item_qty=item_qty,
                warehouse_inventory=warehouse_inventory,
                w_cost=w_cost,
                w_rel=w_rel,
            )
            st.session_state["results"] = df_results
            st.session_state["inputs"]  = dict(
                cust_lat=cust_lat, cust_lon=cust_lon,
                item_qty=item_qty, inv_avail=warehouse_inventory,
                w_cost=w_cost, w_rel=w_rel,
            )
        except Exception as exc:
            st.error(f"⚠️  Evaluation failed: {exc}")
            st.info("Have you run `python bootstrap.py` to train the models first?")
            st.stop()

df   = st.session_state["results"]
inp  = st.session_state["inputs"]
best = df.iloc[0]

# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------

st.markdown("### 📊 Optimal Path KPIs")
c1, c2, c3, c4 = st.columns(4)

topo_labels = {
    "PLANT_WH_CUST": "Plant → WH → Customer",
    "PLANT_CUST":    "Plant → Customer (Direct)",
    "WH_CUST":       "WH → Customer (Regional)",
}

with c1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Best Predicted Cost</div>
      <div class="metric-value">₹{best['pred_delivery_cost']:,.0f}</div>
      <div class="metric-delta">Dist: {best['total_distance_km']:,.0f} km</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Delivery Time</div>
      <div class="metric-value">{best['pred_delivery_time']:.2f} d</div>
      <div class="metric-delta">Via {best['mode']}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    prob_pct = best['pred_ontime_prob'] * 100
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">On-Time Probability</div>
      <div class="metric-value">{prob_pct:.1f}%</div>
      <div class="metric-delta">{best['logistics_agency']}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Sourcing Score</div>
      <div class="metric-value">{best['sourcing_score']:.4f}</div>
      <div class="metric-delta">{topo_labels.get(best['topology_type'], best['topology_type'])}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly Scattergeo Map — Optimal path
# ---------------------------------------------------------------------------

st.markdown("### 🗺️ Optimal Routing Path")

TOPO_COLOURS = {
    "PLANT_WH_CUST": "#58a6ff",   # blue
    "PLANT_CUST":    "#3fb950",   # green
    "WH_CUST":       "#f78166",   # red-orange
}

fig = go.Figure()

# India outline focus
fig.update_geos(
    visible=True,
    resolution=50,
    showland=True,     landcolor="#1c2128",
    showocean=True,    oceancolor="#0d1117",
    showlakes=True,    lakecolor="#161b22",
    showcountries=True, countrycolor="#30363d",
    showcoastlines=True, coastlinecolor="#444c56",
    projection_type="mercator",
    lonaxis_range=[65, 98],
    lataxis_range=[5,  38],
)
fig.update_layout(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    margin=dict(l=0, r=0, t=0, b=0),
    height=500,
    showlegend=True,
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", font=dict(color="#e6edf3")),
)

colour = TOPO_COLOURS.get(best["topology_type"], "#ffffff")

# --- Draw routing line(s) based on topology ---
if best["topology_type"] == "PLANT_WH_CUST":
    # Two-hop:  Plant → Warehouse → Customer
    path_lats = [best["_plant_lat"], best["_wh_lat"], inp["cust_lat"]]
    path_lons = [best["_plant_lon"], best["_wh_lon"], inp["cust_lon"]]
elif best["topology_type"] == "PLANT_CUST":
    # One-hop: Plant → Customer
    path_lats = [best["_plant_lat"], inp["cust_lat"]]
    path_lons = [best["_plant_lon"], inp["cust_lon"]]
else:   # WH_CUST
    # One-hop: Warehouse → Customer
    path_lats = [best["_wh_lat"], inp["cust_lat"]]
    path_lons = [best["_wh_lon"], inp["cust_lon"]]

fig.add_trace(go.Scattergeo(
    lat=path_lats, lon=path_lons,
    mode="lines",
    line=dict(width=3, color=colour),
    name=f"Optimal Route ({best['topology_type']})",
))

# --- All plants as markers ---
from data.generate_data import PLANTS, WAREHOUSES

fig.add_trace(go.Scattergeo(
    lat=[v["lat"] for v in PLANTS.values()],
    lon=[v["lon"] for v in PLANTS.values()],
    mode="markers+text",
    marker=dict(size=10, color="#f0883e", symbol="square"),
    text=[k.replace("PLT-", "") for k in PLANTS],
    textposition="top right",
    textfont=dict(size=9, color="#f0883e"),
    name="Plants",
))

# --- All warehouses as markers ---
fig.add_trace(go.Scattergeo(
    lat=[v["lat"] for v in WAREHOUSES.values()],
    lon=[v["lon"] for v in WAREHOUSES.values()],
    mode="markers+text",
    marker=dict(size=8, color="#bc8cff", symbol="diamond"),
    text=[k.replace("WH-", "") for k in WAREHOUSES],
    textposition="top right",
    textfont=dict(size=9, color="#bc8cff"),
    name="Warehouses",
))

# --- Customer marker ---
fig.add_trace(go.Scattergeo(
    lat=[inp["cust_lat"]], lon=[inp["cust_lon"]],
    mode="markers+text",
    marker=dict(size=14, color="#3fb950", symbol="star"),
    text=["Customer"],
    textposition="top right",
    textfont=dict(size=10, color="#3fb950"),
    name="Customer",
))

# Highlight active nodes with larger markers
if best["topology_type"] in ("PLANT_WH_CUST", "PLANT_CUST") and not pd.isna(best["_plant_lat"]):
    fig.add_trace(go.Scattergeo(
        lat=[best["_plant_lat"]], lon=[best["_plant_lon"]],
        mode="markers",
        marker=dict(size=16, color="#f0883e", symbol="square", line=dict(width=2, color="white")),
        name=f"Selected: {best['plant_id']}",
    ))

if best["topology_type"] in ("PLANT_WH_CUST", "WH_CUST") and not pd.isna(best["_wh_lat"]):
    fig.add_trace(go.Scattergeo(
        lat=[best["_wh_lat"]], lon=[best["_wh_lon"]],
        mode="markers",
        marker=dict(size=14, color="#bc8cff", symbol="diamond", line=dict(width=2, color="white")),
        name=f"Selected: {best['warehouse_id']}",
    ))

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Ranked shortlist table
# ---------------------------------------------------------------------------

st.markdown(f"### 🏆 Top-{top_n} Ranked Candidate Paths")

badge_map = {
    "PLANT_WH_CUST": '<span class="topo-badge badge-pwc">PLANT→WH→CUST</span>',
    "PLANT_CUST":    '<span class="topo-badge badge-pc">PLANT→CUST</span>',
    "WH_CUST":       '<span class="topo-badge badge-wc">WH→CUST</span>',
}

display_df = df.head(top_n).copy()
display_df = display_df[[
    "topology_type", "plant_id", "warehouse_id",
    "logistics_agency", "mode",
    "total_distance_km",
    "pred_delivery_cost", "pred_delivery_time",
    "pred_ontime_prob", "sourcing_score",
]].rename(columns={
    "topology_type":     "Topology",
    "plant_id":          "Plant",
    "warehouse_id":      "Warehouse",
    "logistics_agency":  "Agency",
    "mode":              "Mode",
    "total_distance_km": "Dist (km)",
    "pred_delivery_cost":"Cost (₹)",
    "pred_delivery_time":"Time (d)",
    "pred_ontime_prob":  "On-Time %",
    "sourcing_score":    "Score",
})
display_df["Cost (₹)"]   = display_df["Cost (₹)"].map("₹{:,.0f}".format)
display_df["Time (d)"]   = display_df["Time (d)"].map("{:.2f}".format)
display_df["On-Time %"]  = display_df["On-Time %"].map("{:.1%}".format)
display_df["Score"]      = display_df["Score"].map("{:.4f}".format)
display_df["Dist (km)"]  = display_df["Dist (km)"].map("{:,.0f}".format)

st.dataframe(
    display_df,
    use_container_width=True,
    height=min(40 * (top_n + 2), 520),
)

# ---------------------------------------------------------------------------
# Topology breakdown donut
# ---------------------------------------------------------------------------

st.markdown("### 📈 Candidate Topology Distribution")
topo_counts = df.head(50)["Topology" if "Topology" in df.columns else "topology_type"].value_counts()

# Re-fetch from original df
topo_counts = df["topology_type"].value_counts()
col_a, col_b = st.columns([1, 2])
with col_a:
    st.metric("Total Candidates Evaluated", f"{len(df):,}")
    st.metric("Topologies Searched", "3 variants")
    st.metric("Best Topology", best["topology_type"])
    st.metric("Best Agency",   best["logistics_agency"])
    st.metric("Best Mode",     best["mode"])

with col_b:
    fig2 = go.Figure(go.Pie(
        labels=topo_counts.index.tolist(),
        values=topo_counts.values.tolist(),
        hole=0.55,
        marker=dict(colors=["#58a6ff", "#3fb950", "#f78166"]),
        textfont=dict(color="#e6edf3"),
    ))
    fig2.update_layout(
        paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        margin=dict(l=0, r=0, t=20, b=0),
        height=260,
        legend=dict(bgcolor="#161b22", font=dict(color="#e6edf3")),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption(
    "Sourcing score formula:  **score = (w_cost × norm_cost) − (w_rel × pred_ontime_prob)**  "
    "· Lower score = more optimal path  ·  "
    "Models: LinearRegression (cost & time) + LogisticRegression (on-time)  ·  "
    "Indian network: 8 plants · 10 warehouses · 3 routing topologies"
)

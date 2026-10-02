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
# Design tokens — one place to tune the whole palette
# ---------------------------------------------------------------------------

C_BG        = "#0b0f17"   # page background
C_PANEL     = "#141a24"   # cards / sidebar
C_PANEL_2   = "#1b2331"   # inputs / raised surfaces
C_BORDER    = "#2d3748"
C_TEXT      = "#f0f6fc"   # primary text (high contrast)
C_TEXT_SOFT = "#c3cfdd"   # labels / secondary text (still clearly readable)
C_MUTED     = "#9aa9bb"   # captions

C_BLUE      = "#6cb6ff"   # PLANT → WH → CUST / cost
C_GREEN     = "#4ade80"   # PLANT → CUST / on-time
C_ORANGE    = "#ff9f5a"   # WH → CUST / score
C_PURPLE    = "#c4a2ff"   # warehouses / time
C_AMBER     = "#fbbf24"   # plants

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="India Sourcing Engine",
    page_icon="🚚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — high-contrast industrial dark theme
# ---------------------------------------------------------------------------

st.markdown(f"""
<style>
  /* ---------- Base surfaces ---------- */
  [data-testid="stAppViewContainer"] {{ background: {C_BG}; color: {C_TEXT}; }}
  [data-testid="stSidebar"]           {{ background: {C_PANEL}; border-right: 1px solid {C_BORDER}; }}
  [data-testid="stHeader"]            {{ background: transparent; }}

  /* ---------- Typography ---------- */
  h1, h2, h3, h4 {{ color: {C_TEXT} !important; letter-spacing: 0.2px; }}
  h1 {{ font-weight: 800 !important; }}
  h3 {{ border-left: 4px solid {C_BLUE}; padding-left: 10px; margin-top: 0.4rem; }}
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li {{ color: {C_TEXT_SOFT}; }}
  hr {{ border-color: {C_BORDER} !important; }}

  /* ---------- Sidebar: labels, captions, widget text ---------- */
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {{ color: {C_TEXT} !important; border-left: none; padding-left: 0; }}
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] label p {{
      color: {C_TEXT} !important;
      font-weight: 600;
      font-size: 0.88rem;
      letter-spacing: 0.4px;
  }}
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
  [data-testid="stSidebar"] .stCaption {{ color: {C_MUTED} !important; font-size: 0.8rem; }}
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] strong {{ color: {C_AMBER}; }}

  /* ---------- Number inputs (dark, readable) ---------- */
  [data-testid="stNumberInput"] div[data-baseweb="input"],
  [data-testid="stNumberInput"] div[data-baseweb="base-input"] {{
      background: {C_PANEL_2} !important;
      border-radius: 8px;
  }}
  [data-testid="stNumberInput"] div[data-baseweb="input"] {{ border: 1px solid #3b475b !important; }}
  [data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {{
      border-color: {C_BLUE} !important;
      box-shadow: 0 0 0 1px {C_BLUE};
  }}
  [data-testid="stNumberInput"] input {{
      background: {C_PANEL_2} !important;
      color: {C_TEXT} !important;
      -webkit-text-fill-color: {C_TEXT} !important;
      font-weight: 600;
  }}
  [data-testid="stNumberInput"] button {{
      background: #263043 !important;
      color: {C_TEXT} !important;
      border: none !important;
  }}
  [data-testid="stNumberInput"] button:hover {{ background: #33415a !important; color: {C_BLUE} !important; }}
  [data-testid="stNumberInput"] button svg {{ fill: {C_TEXT} !important; }}

  /* ---------- Sliders ---------- */
  [data-testid="stSlider"] [data-testid="stSliderThumbValue"] {{ color: {C_BLUE} !important; font-weight: 700; }}
  [data-testid="stSlider"] [data-testid="stTickBarMin"],
  [data-testid="stSlider"] [data-testid="stTickBarMax"] {{ color: {C_MUTED} !important; }}
  [data-testid="stSlider"] div[role="slider"] {{ background-color: {C_BLUE} !important; }}

  /* ---------- Primary button ---------- */
  [data-testid="stSidebar"] button[kind="primary"] {{
      background: linear-gradient(135deg, #2f81f7, #6cb6ff);
      color: #0b0f17; font-weight: 700; border: none; border-radius: 8px;
  }}
  [data-testid="stSidebar"] button[kind="primary"]:hover {{ filter: brightness(1.1); }}

  /* ---------- KPI cards ---------- */
  .metric-card {{
    background: linear-gradient(180deg, {C_PANEL_2} 0%, {C_PANEL} 100%);
    border: 1px solid {C_BORDER};
    border-top: 3px solid var(--accent, {C_BLUE});
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.35);
  }}
  .metric-label {{ font-size: 0.76rem; color: {C_TEXT_SOFT}; text-transform: uppercase;
                   letter-spacing: 1.2px; margin-bottom: 8px; font-weight: 600; }}
  .metric-value {{ font-size: 1.95rem; font-weight: 800; color: var(--accent, {C_BLUE}); line-height: 1.15; }}
  .metric-delta {{ font-size: 0.8rem; color: {C_TEXT_SOFT}; margin-top: 6px; }}

  /* ---------- Topology badges ---------- */
  .topo-badge {{
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.5px;
  }}
  .badge-pwc {{ background: rgba(108,182,255,0.18); color: {C_BLUE};   border: 1px solid {C_BLUE}; }}
  .badge-pc  {{ background: rgba(74,222,128,0.18);  color: {C_GREEN};  border: 1px solid {C_GREEN}; }}
  .badge-wc  {{ background: rgba(255,159,90,0.18);  color: {C_ORANGE}; border: 1px solid {C_ORANGE}; }}

  /* ---------- st.metric (Candidate Topology Distribution) ---------- */
  [data-testid="stMetric"] {{
      background: {C_PANEL};
      border: 1px solid {C_BORDER};
      border-left: 4px solid {C_BLUE};
      border-radius: 10px;
      padding: 10px 16px;
      margin-bottom: 8px;
  }}
  [data-testid="stMetricLabel"],
  [data-testid="stMetricLabel"] p {{
      color: {C_TEXT_SOFT} !important;
      font-size: 0.82rem !important;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.8px;
  }}
  [data-testid="stMetricValue"],
  [data-testid="stMetricValue"] div {{
      color: {C_TEXT} !important;
      font-weight: 800 !important;
      font-size: 1.55rem !important;
  }}

  /* ---------- Tables / captions ---------- */
  .stDataFrame {{ border: 1px solid {C_BORDER} !important; border-radius: 8px; }}
  [data-testid="stCaptionContainer"] {{ color: {C_MUTED}; }}
  [data-testid="stCaptionContainer"] strong {{ color: {C_TEXT}; }}
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
    <div class="metric-card" style="--accent:{C_BLUE};">
      <div class="metric-label">Best Predicted Cost</div>
      <div class="metric-value">₹{best['pred_delivery_cost']:,.0f}</div>
      <div class="metric-delta">Dist: {best['total_distance_km']:,.0f} km</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card" style="--accent:{C_PURPLE};">
      <div class="metric-label">Delivery Time</div>
      <div class="metric-value">{best['pred_delivery_time']:.2f} d</div>
      <div class="metric-delta">Via {best['mode']}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    prob_pct = best['pred_ontime_prob'] * 100
    st.markdown(f"""
    <div class="metric-card" style="--accent:{C_GREEN};">
      <div class="metric-label">On-Time Probability</div>
      <div class="metric-value">{prob_pct:.1f}%</div>
      <div class="metric-delta">{best['logistics_agency']}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card" style="--accent:{C_ORANGE};">
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
    "PLANT_WH_CUST": C_BLUE,     # blue
    "PLANT_CUST":    C_GREEN,    # green
    "WH_CUST":       C_ORANGE,   # orange
}

fig = go.Figure()

# India outline focus
fig.update_geos(
    visible=True,
    resolution=50,
    showland=True,     landcolor="#1e2736",
    showocean=True,    oceancolor="#0b1220",
    showlakes=True,    lakecolor="#0f1a2b",
    showcountries=True, countrycolor="#5b6b82",
    showcoastlines=True, coastlinecolor="#6b7c93",
    projection_type="mercator",
    lonaxis_range=[65, 98],
    lataxis_range=[5,  38],
)
fig.update_layout(
    paper_bgcolor=C_BG,
    plot_bgcolor=C_BG,
    margin=dict(l=0, r=0, t=0, b=0),
    height=500,
    showlegend=True,
    legend=dict(
        bgcolor=C_PANEL, bordercolor=C_BORDER, borderwidth=1,
        font=dict(color=C_TEXT, size=12),
    ),
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
    line=dict(width=4, color=colour),
    name=f"Optimal Route ({best['topology_type']})",
))

# --- All plants as markers ---
from data.generate_data import PLANTS, WAREHOUSES

fig.add_trace(go.Scattergeo(
    lat=[v["lat"] for v in PLANTS.values()],
    lon=[v["lon"] for v in PLANTS.values()],
    mode="markers+text",
    marker=dict(size=10, color=C_AMBER, symbol="square"),
    text=[k.replace("PLT-", "") for k in PLANTS],
    textposition="top right",
    textfont=dict(size=10, color=C_AMBER),
    name="Plants",
))

# --- All warehouses as markers ---
fig.add_trace(go.Scattergeo(
    lat=[v["lat"] for v in WAREHOUSES.values()],
    lon=[v["lon"] for v in WAREHOUSES.values()],
    mode="markers+text",
    marker=dict(size=8, color=C_PURPLE, symbol="diamond"),
    text=[k.replace("WH-", "") for k in WAREHOUSES],
    textposition="top right",
    textfont=dict(size=10, color=C_PURPLE),
    name="Warehouses",
))

# --- Customer marker ---
fig.add_trace(go.Scattergeo(
    lat=[inp["cust_lat"]], lon=[inp["cust_lon"]],
    mode="markers+text",
    marker=dict(size=15, color=C_GREEN, symbol="star", line=dict(width=1, color="white")),
    text=["Customer"],
    textposition="top right",
    textfont=dict(size=11, color=C_GREEN),
    name="Customer",
))

# Highlight active nodes with larger markers
if best["topology_type"] in ("PLANT_WH_CUST", "PLANT_CUST") and not pd.isna(best["_plant_lat"]):
    fig.add_trace(go.Scattergeo(
        lat=[best["_plant_lat"]], lon=[best["_plant_lon"]],
        mode="markers",
        marker=dict(size=17, color=C_AMBER, symbol="square", line=dict(width=2, color="white")),
        name=f"Selected: {best['plant_id']}",
    ))

if best["topology_type"] in ("PLANT_WH_CUST", "WH_CUST") and not pd.isna(best["_wh_lat"]):
    fig.add_trace(go.Scattergeo(
        lat=[best["_wh_lat"]], lon=[best["_wh_lon"]],
        mode="markers",
        marker=dict(size=15, color=C_PURPLE, symbol="diamond", line=dict(width=2, color="white")),
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
    donut_colours = {
        "PLANT_WH_CUST": C_BLUE,
        "PLANT_CUST":    C_GREEN,
        "WH_CUST":       C_ORANGE,
    }
    fig2 = go.Figure(go.Pie(
        labels=topo_counts.index.tolist(),
        values=topo_counts.values.tolist(),
        hole=0.55,
        marker=dict(
            colors=[donut_colours.get(t, "#8b949e") for t in topo_counts.index],
            line=dict(color=C_BG, width=3),
        ),
        textfont=dict(color="#0b0f17", size=13),
        textinfo="percent",
    ))
    fig2.update_layout(
        paper_bgcolor=C_BG,
        font=dict(color=C_TEXT),
        margin=dict(l=0, r=0, t=20, b=0),
        height=300,
        legend=dict(
            bgcolor=C_PANEL, bordercolor=C_BORDER, borderwidth=1,
            font=dict(color=C_TEXT, size=12),
        ),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption(
    "Sourcing score formula:  **score = (w_cost × norm_cost) + (w_rel × (1 − pred_ontime_prob))**  "
    "· Lower score = more optimal path  ·  "
    "Models: GradientBoosting (cost, time & on-time)  ·  "
    "Indian network: 8 plants · 10 warehouses · 3 routing topologies"
)

# 🚚 Predictive Sourcing & Routing Engine
### A Multi-Echelon Supply Chain Network — Indian Logistics Context

---

## What This Project Does

Most companies route orders the same way every time —plant to warehouse to customer — regardless of whether that path is actually optimal for that specific order.

This engine challenges that. Given a new purchase order, it evaluates every possible routing path across three distinct supply chain topologies, scores each one using machine learning predictions, 
and recommends the single best path based on cost and delivery reliability.

---

## The Three Routing Topologies

| Topology            | Path                         | When It Wins                       |
| Traditional Echelon | Plant → Warehouse → Customer | Large orders, distant customers    |
| Direct Plant Bypass | Plant → Customer             | High-value, time-sensitive orders  |
| Regional Fulfilment | Warehouse → Customer         | When regional stock is available   |

The engine does not assume one topology is always right.
It generates candidates across all three and lets the data decide.

---

## Architecture
# data/generate_data.py            # Stage 1a — Synthetic order generation
# engine/stage1_clustering.py      # Stage 1b — Geographic demand zone clustering
engine/stage2_ml_pipeline.py     # Stage 2  — ML model training
engine/stage3_sourcing_engine.py # Stage 3 — Inference & scoring
bootstrap.py                     # Runs all stages sequentially
app.py                           # Streamlit dashboard

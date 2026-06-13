"""
bootstrap.py
=============
One-shot pipeline runner.

Executes all three stages sequentially:
  Stage 0 : Ensure directory structure exists
  Stage 1a: Generate 3,000 synthetic Indian logistics orders
  Stage 1b: Cluster customer locations into demand zones (K-Means)
  Stage 2 : Train cost, time, and on-time ML pipelines

Run from the project root:
    python bootstrap.py
"""

import os
import sys
import time

# Ensure project root is on the Python path for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def banner(msg: str) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {msg}")
    print(f"{sep}")


# ---------------------------------------------------------------------------
# Stage 0 — Directory scaffold
# ---------------------------------------------------------------------------

def stage0_scaffold() -> None:
    banner("Stage 0 — Creating directory structure")
    for d in ["data", "engine", "models"]:
        os.makedirs(d, exist_ok=True)
        print(f"  ✓  {d}/")


# ---------------------------------------------------------------------------
# Stage 1a — Data generation
# ---------------------------------------------------------------------------

def stage1a_generate() -> None:
    banner("Stage 1a — Generating synthetic Indian logistics dataset (3,000 orders)")
    from data.generate_data import generate_orders

    t0 = time.time()
    df = generate_orders(n_orders=3000, random_seed=42)
    df.to_csv("data/historical_orders.csv", index=False)
    print(f"\n  ✓  data/historical_orders.csv  ({len(df):,} rows, {time.time()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# Stage 1b — Demand zone clustering
# ---------------------------------------------------------------------------

def stage1b_cluster() -> None:
    banner("Stage 1b — K-Means geographic demand-zone clustering")
    import pandas as pd
    from engine.stage1_clustering import fit_demand_zones

    t0  = time.time()
    df  = pd.read_csv("data/historical_orders.csv")
    dfc = fit_demand_zones(df)   # K auto-selected via silhouette score
    dfc.to_csv("data/historical_orders_clustered.csv", index=False)
    print(f"\n  ✓  data/historical_orders_clustered.csv  ({len(dfc):,} rows, {time.time()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# Stage 2 — ML pipeline training
# ---------------------------------------------------------------------------

def stage2_train() -> None:
    banner("Stage 2 — Training ML pipelines (cost, time, on-time)")
    from engine.stage2_ml_pipeline import train_and_save_models

    t0 = time.time()
    train_and_save_models(data_path="data/historical_orders_clustered.csv")
    print(f"\n  ✓  All models saved to models/  ({time.time()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    total_start = time.time()
    print("\n🚀  Predictive Sourcing & Routing Engine — Bootstrap")
    print("     Indian Multi-Echelon Logistics Network\n")

    stage0_scaffold()
    stage1a_generate()
    stage1b_cluster()
    stage2_train()

    total_elapsed = time.time() - total_start
    banner(f"✅  Bootstrap complete  ({total_elapsed:.1f}s total)")
    print("  Run the dashboard with:  streamlit run app.py\n")

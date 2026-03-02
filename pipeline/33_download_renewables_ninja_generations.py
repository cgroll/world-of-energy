# ---
# jupytext:
#   text_representation:
#     format_name: percent
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---

# %% [markdown]
# # Renewable Capacity Factors via Renewables.ninja
#
# Fetches hourly solar PV, onshore wind, and offshore wind capacity factors for
# Germany for the full year 2019 using the
# [renewables.ninja](https://www.renewables.ninja) API.
#
# **Inputs** (from `pipeline/31_dev_renewable_generation.py`)
# - `pv_state_aggregates.parquet`
# - `wind_onshore_state_aggregates.parquet`
# - `wind_offshore_aggregates.parquet`
#
# **Method**
# - PV and onshore wind: one API call per Bundesland (16 states each).
# - Offshore wind: two API calls — Nordsee and Ostsee — using the capacity-weighted
#   centroids from the aggregate table.
# - All calls use `capacity=1.0`; returned `electricity` values are hourly
#   capacity factors (0–1). Multiply by installed capacity (MW) in downstream
#   scripts to obtain generation in MWh.
#
# **Outputs**
# - `ninja_pv_cf.parquet`: 8760 × 16 DataFrame, columns = Bundesland names
# - `ninja_wind_onshore_cf.parquet`: same shape for onshore wind
# - `ninja_wind_offshore_cf.parquet`: 8760 × 2, columns = ["Nordsee", "Ostsee"]
#
# **API token**
# Set `RENEWABLES_NINJA_TOKEN` in your environment before running.
# Free accounts are limited to 50 requests/hour; the script sleeps between calls.

# %%
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from woe.paths import ProjPaths

load_dotenv()
paths = ProjPaths()

NINJA_TOKEN = os.environ["RENEWABLES_NINJA_TOKEN"]
NINJA_BASE = "https://www.renewables.ninja/api/data"

YEAR = 2019
DATE_FROM = f"{YEAR}-01-01"
DATE_TO = f"{YEAR}-12-31"

# Free tier: 50 requests / hour → minimum 72 s between calls.
# 16 (PV) + 16 (onshore) + 2 (offshore) = 34 requests → ~43 min wait time.
REQUEST_DELAY_S = 75

# %% [markdown]
# ## Load aggregate tables

# %%
pv_agg = pd.read_parquet(paths.pv_state_aggregates_file)
wind_onshore_agg = pd.read_parquet(paths.wind_onshore_state_aggregates_file)
wind_offshore_agg = pd.read_parquet(paths.wind_offshore_aggregates_file)

print(f"PV:            {len(pv_agg)} states")
print(f"Wind onshore:  {len(wind_onshore_agg)} states")
print(f"Wind offshore: {len(wind_offshore_agg)} regions  {wind_offshore_agg[['region','capacity_MW']].to_string(index=False)}")

# %% [markdown]
# ## Renewables.ninja API helpers

# %%
def _ninja_get(endpoint: str, params: dict) -> pd.Series:
    """Call one renewables.ninja endpoint, return hourly capacity factor Series."""
    headers = {"Authorization": f"Token {NINJA_TOKEN}"}
    r = requests.get(f"{NINJA_BASE}/{endpoint}", params=params, headers=headers)
    if not r.ok:
        raise RuntimeError(f"API {r.status_code}: {r.text}")
    payload = r.json()
    df = pd.DataFrame.from_dict(payload["data"], orient="index")
    # Keys are millisecond Unix timestamps
    df.index = pd.to_datetime(df.index.astype(int), unit="ms", utc=True)
    df.index.name = "time"
    return df["electricity"]


def fetch_pv_cf(lat: float, lon: float) -> pd.Series:
    """Hourly PV capacity factor (capacity=1 kW, south-facing 35° tilt, MERRA-2)."""
    return _ninja_get("pv", {
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
        "dataset": "merra2",
        "capacity": 1.0,
        "system_loss": 0.1,
        "tracking": 0,
        "tilt": 35,
        "azim": 180,
        "format": "json",
    })


def fetch_wind_cf(lat: float, lon: float) -> pd.Series:
    """Hourly wind capacity factor (capacity=1 kW, 100 m hub, Vestas V90 2000, MERRA-2)."""
    return _ninja_get("wind", {
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
        "dataset": "merra2",
        "capacity": 1.0,
        "height": 100,
        "turbine": "Vestas V90 2000",
        "format": "json",
    })

# %% [markdown]
# ## Fetch hourly capacity factors
#
# Each API call covers one full year (8760 h). With `capacity=1.0` the returned
# `electricity` column is the hourly capacity factor (0–1).
#
# The free renewables.ninja tier allows 50 requests/hour; a `{REQUEST_DELAY_S}`-second
# pause is inserted between calls.

# %%
pv_cols: dict[str, pd.Series] = {}
for _, row in pv_agg.iterrows():
    state = row["state"]
    print(f"  PV  {state} ...", end=" ", flush=True)
    cf = fetch_pv_cf(row["lat"], row["lon"])
    pv_cols[state] = cf
    print(f"done  (annual mean CF {cf.mean():.3f})")
    time.sleep(REQUEST_DELAY_S)

pv_cf = pd.DataFrame(pv_cols)
pv_cf.index.name = "time"
print(f"\nPV CF shape: {pv_cf.shape}")

# %%
wind_onshore_cols: dict[str, pd.Series] = {}
for _, row in wind_onshore_agg.dropna(subset=["lat", "lon"]).iterrows():
    state = row["state"]
    print(f"  Wind onshore  {state} ...", end=" ", flush=True)
    cf = fetch_wind_cf(row["lat"], row["lon"])
    wind_onshore_cols[state] = cf
    print(f"done  (annual mean CF {cf.mean():.3f})")
    time.sleep(REQUEST_DELAY_S)

wind_onshore_cf = pd.DataFrame(wind_onshore_cols)
wind_onshore_cf.index.name = "time"
print(f"\nWind onshore CF shape: {wind_onshore_cf.shape}")

# %%
wind_offshore_cols: dict[str, pd.Series] = {}
for _, row in wind_offshore_agg.iterrows():
    region = row["region"]
    print(f"  Wind offshore  {region} ...", end=" ", flush=True)
    cf = fetch_wind_cf(row["lat"], row["lon"])
    wind_offshore_cols[region] = cf
    print(f"done  (annual mean CF {cf.mean():.3f})")
    time.sleep(REQUEST_DELAY_S)

wind_offshore_cf = pd.DataFrame(wind_offshore_cols)
wind_offshore_cf.index.name = "time"
print(f"\nWind offshore CF shape: {wind_offshore_cf.shape}")

# %% [markdown]
# ## Save results

# %%
pv_cf.to_parquet(paths.ninja_pv_cf_file)
print(f"Saved PV CF            → {paths.ninja_pv_cf_file}")

wind_onshore_cf.to_parquet(paths.ninja_wind_onshore_cf_file)
print(f"Saved wind onshore CF  → {paths.ninja_wind_onshore_cf_file}")

wind_offshore_cf.to_parquet(paths.ninja_wind_offshore_cf_file)
print(f"Saved wind offshore CF → {paths.ninja_wind_offshore_cf_file}")

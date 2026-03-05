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
# # Residual Load Reconstruction
#
# Combines the BDEW+temporal XGBoost demand model (script 39, baseline — no weather
# features) with PECD-ERA5 capacity factors scaled by SMARD installed capacities to
# reconstruct an hourly **residual load** time series for Germany 2022–2025.
#
# **Residual load** (same definition for reconstruction and SMARD reference):
#
# ```
# RL = demand − solar_generation − wind_onshore_generation − wind_offshore_generation
# ```
#
# **Data sources**
# - Demand: baseline XGBoost model (`data/models/de_load_estimation/baseline.json`)
# - Capacity factors: PECD ERA5 reanalysis (`data/processed/pecd/pecd_regions.parquet`)
# - Installed capacities: SMARD monthly (`data/downloads/smard/capacities.parquet`)
# - SMARD reference: measured total load, solar, wind onshore/offshore (hourly)

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from woe.paths import ProjPaths


def show():
    """plt.show() wrapper: no-op when matplotlib uses a non-interactive backend."""
    try:
        plt.show()
    except Exception:
        pass


paths = ProjPaths()

YEARS = [2022, 2023, 2024, 2025]

# %% [markdown]
# ## Load demand predictions
#
# Pre-computed by script 39: baseline (BDEW + temporal) and weather-enhanced
# XGBoost demand models evaluated on the full PECD data range.

# %%
demand_preds = pd.read_parquet(paths.de_demand_predictions_file)
demand_preds = demand_preds[demand_preds.index.year.isin(YEARS)]

# Baseline model (BDEW + temporal features only — no weather dependency)
demand_pred = demand_preds["demand_baseline_mw"].rename("demand_pred_mw")

print(f"Demand predictions: {demand_pred.index[0]} → {demand_pred.index[-1]}")
print("Annual mean demand (MW):")
for yr in YEARS:
    m = demand_pred[demand_pred.index.year == yr].mean()
    print(f"  {yr}: {m:,.0f} MW")

# %% [markdown]
# ## Installed renewable capacities from SMARD
#
# Monthly installed capacities (MW) for solar PV, onshore wind and offshore wind,
# as downloaded from SMARD (script 01). The full time series is shown below.

# %%
cap = pd.read_parquet(paths.smard_capacities_file)
cap_re = cap[["solar", "wind_onshore", "wind_offshore"]].copy()
cap_re.index.name = "month"

print("Installed capacities (MW) — solar / wind_onshore / wind_offshore:")
print(cap_re.to_string())

# %% [markdown]
# ## Installed capacity time series

# %%
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(cap_re.index, cap_re["solar"] / 1e3,        color="#f4b942", linewidth=1.8, label="Solar PV")
ax.plot(cap_re.index, cap_re["wind_onshore"] / 1e3,  color="#4a90d9", linewidth=1.8, label="Wind onshore")
ax.plot(cap_re.index, cap_re["wind_offshore"] / 1e3, color="#1a5fa8", linewidth=1.8, label="Wind offshore")
ax.set_ylabel("Installed capacity (GW)")
ax.set_title("Germany — installed renewable capacity (SMARD, monthly)", fontsize=11)
ax.legend()
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(paths.images_path / "40_installed_capacities.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/40_installed_capacities.png
# :name: fig-40-installed-capacities
# Monthly installed renewable capacities in Germany (SMARD). Solar PV has grown most
# rapidly; offshore wind expanded significantly after 2020.
# ```

# %% [markdown]
# ## PECD capacity factors for Germany
#
# PECD ERA5 capacity factors are converted from hour-starting UTC to the SMARD
# hour-ending CET convention (same transform as script 39), then multiplied by
# the forward-filled monthly installed capacities.

# %%
pecd = pd.read_parquet(paths.pecd_processed_file)

cf_solar_raw    = pecd["solar_photovoltaic_power_generation"]["capacity_factor_ratio"]["DE"]
cf_onshore_raw  = pecd["wind_power_generation_onshore"]["capacity_factor_ratio"]["DE"]
cf_offshore_raw = pecd["wind_power_generation_offshore"]["capacity_factor_ratio"].get(
    "DE", pd.Series(dtype=float)
)


def pecd_to_smard_index(s: pd.Series) -> pd.Series:
    """Convert PECD hour-starting UTC → SMARD hour-ending CET (naive index)."""
    return (
        s.tz_localize("UTC")
         .tz_convert("Europe/Berlin")
         .shift(1, freq="h")
         .tz_localize(None)
    )


demand_index = demand_pred.index

def convert_and_clean(raw: pd.Series) -> pd.Series:
    """Convert PECD timestamps and remove DST-transition duplicates (keep first)."""
    s = pecd_to_smard_index(raw)
    return s[~s.index.duplicated(keep="first")]


cf_solar_s   = convert_and_clean(cf_solar_raw)
cf_solar_s   = cf_solar_s[cf_solar_s.index.year.isin(YEARS)]

cf_onshore_s = convert_and_clean(cf_onshore_raw)
cf_onshore_s = cf_onshore_s[cf_onshore_s.index.year.isin(YEARS)]

if not cf_offshore_raw.empty:
    cf_offshore_s = convert_and_clean(cf_offshore_raw)
    cf_offshore_s = cf_offshore_s[cf_offshore_s.index.year.isin(YEARS)]
else:
    cf_offshore_s = pd.Series(0.0, index=demand_index, name="cf_offshore")

print(f"PECD solar    CF: mean={cf_solar_s.mean():.4f}  "
      f"range {cf_solar_s.index[0]} → {cf_solar_s.index[-1]}")
print(f"PECD onshore  CF: mean={cf_onshore_s.mean():.4f}  "
      f"range {cf_onshore_s.index[0]} → {cf_onshore_s.index[-1]}")
print(f"PECD offshore CF: mean={cf_offshore_s.mean():.4f}")

# %% [markdown]
# ## Renewable generation estimates
#
# Monthly capacities are forward-filled to the hourly demand index, then multiplied
# by the PECD capacity factors.

# %%
# Forward-fill monthly capacities to hourly resolution
cap_h = cap_re.reindex(demand_index, method="ffill")

gen_solar_mw    = (cf_solar_s.reindex(demand_index)   * cap_h["solar"]).rename("gen_solar_mw")
gen_onshore_mw  = (cf_onshore_s.reindex(demand_index) * cap_h["wind_onshore"]).rename("gen_onshore_mw")
gen_offshore_mw = (cf_offshore_s.reindex(demand_index)* cap_h["wind_offshore"]).rename("gen_offshore_mw")

print(f"Mean solar generation:    {gen_solar_mw.mean():,.0f} MW")
print(f"Mean onshore generation:  {gen_onshore_mw.mean():,.0f} MW")
print(f"Mean offshore generation: {gen_offshore_mw.mean():,.0f} MW")

# %% [markdown]
# ## Annual generation: PECD reconstruction vs SMARD actuals

# %%
solar_s   = pd.read_parquet(paths.smard_solar_file)
solar_s   = solar_s.rename(columns={solar_s.columns[0]: "solar_mw"})["solar_mw"]
on_s      = pd.read_parquet(paths.smard_wind_onshore_file)
on_s      = on_s.rename(columns={on_s.columns[0]: "wind_onshore_mw"})["wind_onshore_mw"]
off_s     = pd.read_parquet(paths.smard_wind_offshore_file)
off_s     = off_s.rename(columns={off_s.columns[0]: "wind_offshore_mw"})["wind_offshore_mw"]

# Annual sums in TWh
def annual_twh(series: pd.Series) -> pd.Series:
    s = series[series.index.year.isin(YEARS)]
    return s.groupby(s.index.year).sum() / 1e6


gen_compare = pd.DataFrame({
    "solar_pecd":    annual_twh(gen_solar_mw),
    "solar_smard":   annual_twh(solar_s),
    "onshore_pecd":  annual_twh(gen_onshore_mw),
    "onshore_smard": annual_twh(on_s),
    "offshore_pecd": annual_twh(gen_offshore_mw),
    "offshore_smard":annual_twh(off_s),
}).reindex(YEARS)

print("\nAnnual generation (TWh) — PECD reconstruction vs SMARD actuals:")
print(gen_compare.round(1).to_string())

# %%
x = np.arange(len(YEARS))
bar_w = 0.35

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, tech, pecd_col, smard_col, color in [
    (axes[0], "Solar PV",      "solar_pecd",    "solar_smard",   "#f4b942"),
    (axes[1], "Wind onshore",  "onshore_pecd",  "onshore_smard", "#4a90d9"),
    (axes[2], "Wind offshore", "offshore_pecd", "offshore_smard","#1a5fa8"),
]:
    ax.bar(x - bar_w / 2, gen_compare[pecd_col].values,  width=bar_w, color=color,
           alpha=0.75, label="PECD", edgecolor="white")
    ax.bar(x + bar_w / 2, gen_compare[smard_col].values, width=bar_w, color=color,
           alpha=1.0,  label="SMARD", edgecolor="white", hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(YEARS)
    ax.set_ylabel("Annual generation (TWh)")
    ax.set_title(tech, fontsize=11)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)

fig.suptitle("Annual generation — PECD-based estimates vs SMARD actuals", fontsize=12)
fig.tight_layout()
fig.savefig(paths.images_path / "40_annual_generation_comparison.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/40_annual_generation_comparison.png
# :name: fig-40-annual-generation-comparison
# Annual renewable generation (TWh) from PECD-ERA5 capacity factors scaled by SMARD
# installed capacities (solid bars) vs SMARD-measured actuals (hatched bars), for
# 2022–2025. Differences reflect ERA5 reanalysis bias versus real-world curtailment,
# grid constraints and measurement conventions.
# ```

# %% [markdown]
# ## Residual load construction
#
# ```
# RL_reconstructed = demand_predicted − gen_solar − gen_wind_onshore − gen_wind_offshore
# ```

# %%
rl_reconstructed = (
    demand_pred
    - gen_solar_mw
    - gen_onshore_mw
    - gen_offshore_mw
).rename("residual_load_mw")

print(f"Reconstructed RL — mean: {rl_reconstructed.mean():,.0f} MW  "
      f"min: {rl_reconstructed.min():,.0f} MW  max: {rl_reconstructed.max():,.0f} MW")
print(f"Hours with negative RL (surplus generation): {(rl_reconstructed < 0).sum():,}")

# %% [markdown]
# ## SMARD reference residual load
#
# ```
# RL_smard = actual_total_load − actual_solar − actual_wind_onshore − actual_wind_offshore
# ```

# %%
load_h = pd.read_parquet(paths.smard_total_load_file)
load_h = load_h.rename(columns={load_h.columns[0]: "load_mw"})
load_h = load_h[load_h.index.year.isin(YEARS)].copy()

smard_ref = (
    load_h[["load_mw"]]
    .join(solar_s.rename("solar_mw"))
    .join(on_s.rename("wind_onshore_mw"))
    .join(off_s.rename("wind_offshore_mw"))
)
smard_ref = smard_ref[smard_ref.index.year.isin(YEARS)].copy()
smard_ref["rl_smard_mw"] = (
    smard_ref["load_mw"]
    - smard_ref["solar_mw"].fillna(0)
    - smard_ref["wind_onshore_mw"].fillna(0)
    - smard_ref["wind_offshore_mw"].fillna(0)
)
smard_ref.dropna(subset=["load_mw"], inplace=True)

print(f"SMARD RL — mean: {smard_ref['rl_smard_mw'].mean():,.0f} MW  "
      f"min: {smard_ref['rl_smard_mw'].min():,.0f} MW  max: {smard_ref['rl_smard_mw'].max():,.0f} MW")

# %% [markdown]
# ## Metrics by year

# %%
common_idx = rl_reconstructed.dropna().index.intersection(
    smard_ref.dropna(subset=["rl_smard_mw"]).index
)
rl_rec = rl_reconstructed.loc[common_idx]
rl_smd = smard_ref.loc[common_idx, "rl_smard_mw"]

records = []
for yr in YEARS:
    mask = rl_rec.index.year == yr
    if mask.sum() == 0:
        continue
    r2   = r2_score(rl_smd[mask], rl_rec[mask])
    mae  = mean_absolute_error(rl_smd[mask], rl_rec[mask])
    rmse = np.sqrt(mean_squared_error(rl_smd[mask], rl_rec[mask]))
    mape = np.mean(np.abs((rl_smd[mask].values - rl_rec[mask].values) / rl_smd[mask].values)) * 100
    records.append(dict(year=yr, r2=round(r2, 4), mae=round(mae, 0),
                        rmse=round(rmse, 0), mape=round(mape, 2)))

metrics_df = pd.DataFrame(records).set_index("year")
print("\nResidual load reconstruction — metrics vs SMARD reference:")
print(metrics_df.to_string())

# %% [markdown]
# ## Scatter: reconstructed vs SMARD residual load (by year)

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, yr in zip(axes.flat, YEARS):
    mask = rl_rec.index.year == yr
    if mask.sum() == 0:
        ax.set_visible(False)
        continue
    y_true = rl_smd[mask]
    y_pred = rl_rec[mask]
    m = metrics_df.loc[yr]
    ax.scatter(y_true, y_pred, alpha=0.06, s=2, color="#4a90d9", rasterized=True)
    lims = [
        min(float(y_true.min()), float(y_pred.min())),
        max(float(y_true.max()), float(y_pred.max())),
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="1:1")
    ax.set_xlabel("SMARD residual load (MW)")
    ax.set_ylabel("Reconstructed residual load (MW)")
    ax.set_title(
        f"{yr}  R²={m['r2']:.4f}  MAE={m['mae']:,.0f} MW  MAPE={m['mape']:.2f}%",
        fontsize=10,
    )
    ax.legend(fontsize=8)

fig.suptitle("Reconstructed vs SMARD residual load — 2022–2025", fontsize=12)
fig.tight_layout()
fig.savefig(paths.images_path / "40_scatter_residual_load.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/40_scatter_residual_load.png
# :name: fig-40-scatter-residual-load
# Scatter plots of reconstructed vs SMARD residual load for 2022–2025 (one panel per
# year). Two error sources contribute to deviations: (1) BDEW+temporal demand model
# error and (2) PECD capacity-factor mismatch with actual measured generation.
# ```

# %% [markdown]
# ## Sample week: time series comparison

# %%
week_data = {
    yr: pd.DataFrame({
        "rl_rec":   rl_rec[rl_rec.index.year == yr],
        "rl_smard": rl_smd[rl_smd.index.year == yr],
    })
    for yr in [2022, 2025]
}

for yr, data in week_data.items():
    for season, start_date in [("winter", f"{yr}-01-13"), ("summer", f"{yr}-07-07")]:
        start = pd.Timestamp(start_date)
        end = start + pd.Timedelta(days=7)
        sl = data.loc[(data.index >= start) & (data.index < end)]

        if sl.empty:
            print(f"No data for {season} {yr} — skipping")
            continue

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(sl.index, sl["rl_smard"], color="#333333", linewidth=1.0, label="SMARD")
        ax.plot(sl.index, sl["rl_rec"],   color="#e6734a", linewidth=1.0,
                label="Reconstructed", alpha=0.85)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_ylabel("Residual load (MW)")
        ax.set_title(f"Residual load — {season} week {yr} ({start_date})", fontsize=11)
        ax.legend()
        ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(
            paths.images_path / f"40_sample_week_{yr}_{season}.png",
            dpi=150, bbox_inches="tight",
        )
        show()

# %% [markdown]
# ```{figure} ../../output/images/40_sample_week_2022_winter.png
# :name: fig-40-week-2022-winter
# Reconstructed vs SMARD residual load for a representative winter week (January 2022).
# The BDEW demand model captures the intraday and day-type pattern; deviations arise
# mainly from actual renewable output differing from the ERA5-based PECD estimate.
# ```
#
# ```{figure} ../../output/images/40_sample_week_2022_summer.png
# :name: fig-40-week-2022-summer
# Reconstructed vs SMARD residual load for a representative summer week (July 2022).
# Solar generation is the dominant variable-generation source in summer; PECD CFs
# capture the daytime dip but ERA5 smoothing may understate cloud-cover variability.
# ```
#
# ```{figure} ../../output/images/40_sample_week_2025_winter.png
# :name: fig-40-week-2025-winter
# Reconstructed vs SMARD residual load for a representative winter week (January 2025).
# ```
#
# ```{figure} ../../output/images/40_sample_week_2025_summer.png
# :name: fig-40-week-2025-summer
# Reconstructed vs SMARD residual load for a representative summer week (July 2025).
# The higher installed solar and wind capacity in 2025 compared to 2022 leads to more
# frequent low or negative residual load events in summer afternoons.
# ```

# %% [markdown]
# ## Monthly mean residual load comparison

# %%
monthly_rec   = rl_rec.groupby([rl_rec.index.year, rl_rec.index.month]).mean().unstack(level=0)
monthly_smard = rl_smd.groupby([rl_smd.index.year, rl_smd.index.month]).mean().unstack(level=0)

month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, yr in zip(axes.flat, YEARS):
    if yr not in monthly_rec.columns or yr not in monthly_smard.columns:
        ax.set_visible(False)
        continue
    ax.plot(month_labels, monthly_smard[yr].values, color="#333333",
            linewidth=1.5, marker="o", markersize=4, label="SMARD")
    ax.plot(month_labels, monthly_rec[yr].values,   color="#e6734a",
            linewidth=1.5, marker="o", markersize=4, label="Reconstructed", alpha=0.9)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_title(str(yr), fontsize=11)
    ax.set_ylabel("Mean residual load (MW)")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)

fig.suptitle("Monthly mean residual load — SMARD vs reconstructed (2022–2025)", fontsize=12)
fig.tight_layout()
fig.savefig(paths.images_path / "40_monthly_residual_load.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/40_monthly_residual_load.png
# :name: fig-40-monthly-residual-load
# Monthly mean residual load for 2022–2025: SMARD actuals (black) vs reconstruction
# (orange). The winter peak and summer trough are well reproduced; systematic offsets
# in summer reflect ERA5 over- or under-estimating solar output relative to measured
# SMARD values.
# ```

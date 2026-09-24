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
# # Renewable Demand Coverage Analysis
#
# Uses the BDEW+temporal XGBoost demand model (script 39, baseline — no weather
# features) and PECD-ERA5 capacity factors scaled by **fixed** (latest available)
# SMARD installed capacities to stress-test the *current* German fleet against
# the full PECD reanalysis weather history (~1979 to present).
#
# Capacities are held constant at the most recent SMARD record for all years.
# This isolates the weather effect: how would today's installed fleet have
# performed under every historical weather year?
#
# **Per-hour quantities**
#
# | Variable | Definition |
# |---|---|
# | `gen_mw` | Solar + wind onshore + wind offshore generation |
# | `demand_mw` | Reconstructed demand (baseline model) |
# | `residual_load_mw` | `demand − gen` (negative = surplus) |
# | `residual_demand_mw` | `max(0, residual_load_mw)` — uncovered demand |
# | `excess_mw` | `max(0, gen − demand_mw)` — surplus generation |
#
# **Coverage metrics (averages)**
#
# - *Generation share*: `mean(gen) / mean(demand)` — fraction of average
#   demand met by average generation.
# - *Still needed*: `mean(residual_demand) / mean(demand)` — fraction that
#   cannot be covered because surpluses and deficits don't coincide in time.
# - *Excess share*: `mean(excess) / mean(gen)` — fraction of renewable output
#   generated during surplus hours.
#
# **Installed capacity note** — Capacities are fixed at the most recent SMARD
# monthly record for all years. Every year therefore represents the same
# hypothetical question: *how would the current fleet have performed under that
# year's weather?*

# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from woe.paths import ProjPaths


def show():
    """plt.show() wrapper: no-op when matplotlib uses a non-interactive backend."""
    try:
        plt.show()
    except Exception:
        pass


paths = ProjPaths()

# %% [markdown]
# ## Demand predictions (baseline model, no weather features)
#
# Script 39 saves predictions for the full PECD reanalysis range.

# %%
demand_preds = pd.read_parquet(paths.de_demand_predictions_file)
demand_pred = demand_preds["demand_baseline_mw"].rename("demand_mw")

print(f"Demand predictions span: {demand_pred.index[0]} → {demand_pred.index[-1]}")
print(f"Total rows: {len(demand_pred):,}")

# %% [markdown]
# ## PECD capacity factors and installed capacities
#
# Identical index conversion as script 40. Installed capacities are
# forward-filled for years beyond SMARD data and **backfilled** for years
# before the first SMARD record.

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


def convert_and_clean(raw: pd.Series) -> pd.Series:
    s = pecd_to_smard_index(raw)
    return s[~s.index.duplicated(keep="first")]


cf_solar_s   = convert_and_clean(cf_solar_raw)
cf_onshore_s = convert_and_clean(cf_onshore_raw)
cf_offshore_s = (
    convert_and_clean(cf_offshore_raw)
    if not cf_offshore_raw.empty
    else pd.Series(dtype=float)
)

cap = pd.read_parquet(paths.smard_capacities_file)
cap_re = cap[["solar", "wind_onshore", "wind_offshore"]].copy()

# Shared index: intersection of demand predictions and all three CF series
shared_idx = demand_pred.index
for cf in [cf_solar_s, cf_onshore_s]:
    shared_idx = shared_idx.intersection(cf.index)
if not cf_offshore_s.empty:
    shared_idx = shared_idx.intersection(cf_offshore_s.index)

# Fix capacities at the most recent available SMARD record
cap_latest = cap_re.iloc[-1]

gen_solar_mw    = (cf_solar_s.reindex(shared_idx)   * cap_latest["solar"]).rename("gen_solar_mw")
gen_onshore_mw  = (cf_onshore_s.reindex(shared_idx) * cap_latest["wind_onshore"]).rename("gen_onshore_mw")
gen_offshore_mw = (
    cf_offshore_s.reindex(shared_idx) * cap_latest["wind_offshore"]
    if not cf_offshore_s.empty
    else pd.Series(0.0, index=shared_idx)
).rename("gen_offshore_mw")

print(f"Shared index: {shared_idx[0]} → {shared_idx[-1]}  ({len(shared_idx):,} rows)")
print(f"Fixed capacities from SMARD record: {cap_re.index[-1].date()}")
print(f"  Solar: {cap_latest['solar']:,.0f} MW  |  Onshore: {cap_latest['wind_onshore']:,.0f} MW  |  Offshore: {cap_latest['wind_offshore']:,.0f} MW")

# %% [markdown]
# ## Hourly coverage quantities

# %%
df = pd.DataFrame({
    "demand_mw":    demand_pred.reindex(shared_idx),
    "solar_mw":     gen_solar_mw,
    "onshore_mw":   gen_onshore_mw,
    "offshore_mw":  gen_offshore_mw,
}).dropna()

df["gen_mw"]             = df["solar_mw"] + df["onshore_mw"] + df["offshore_mw"]
df["residual_load_mw"]   = df["demand_mw"] - df["gen_mw"]
df["residual_demand_mw"] = df["residual_load_mw"].clip(lower=0)
df["excess_mw"]          = (-df["residual_load_mw"]).clip(lower=0)

# Drop the most recent (partial) year so only complete years are analysed
df = df[df.index.year < df.index.year.max()]

AVAIL_YEARS = sorted(df.index.year.unique().tolist())
print(f"Analysis period: {AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]}  ({len(AVAIL_YEARS)} years)")
print(f"Total rows: {len(df):,}")
print(f"Hours with surplus generation: {(df['residual_load_mw'] < 0).sum():,}  "
      f"({(df['residual_load_mw'] < 0).mean()*100:.1f}%)")

# %% [markdown]
# ## Annual summary

# %%
records = []
for yr in AVAIL_YEARS:
    sub = df[df.index.year == yr]
    if len(sub) < 100:
        continue
    avg_demand  = sub["demand_mw"].mean()
    avg_gen     = sub["gen_mw"].mean()
    avg_res_dem = sub["residual_demand_mw"].mean()
    avg_excess  = sub["excess_mw"].mean()
    records.append({
        "year":                   yr,
        "avg_demand_mw":          round(avg_demand,  0),
        "avg_gen_mw":             round(avg_gen,     0),
        "avg_residual_demand_mw": round(avg_res_dem, 0),
        "avg_excess_mw":          round(avg_excess,  0),
        "gen_pct_of_demand":      round(avg_gen      / avg_demand * 100, 1),
        "still_needed_pct":       round(avg_res_dem  / avg_demand * 100, 1),
        "excess_pct_of_gen":      round(avg_excess   / avg_gen    * 100, 1),
        "surplus_hours_pct":      round((sub["residual_load_mw"] < 0).mean() * 100, 1),
    })

summary = pd.DataFrame(records).set_index("year")
print(f"\nAnnual coverage summary ({summary.index[0]}–{summary.index[-1]}):")
print(summary.to_string())

# %% [markdown]
# ## Annual average power — full period time series

# %%
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(summary.index, summary["avg_demand_mw"]         / 1e3, color="#333333", linewidth=1.5,
        label="Avg demand")
ax.plot(summary.index, summary["avg_gen_mw"]             / 1e3, color="#4a90d9", linewidth=1.5,
        label="Avg renewable generation")
ax.plot(summary.index, summary["avg_residual_demand_mw"] / 1e3, color="#e6734a", linewidth=1.5,
        label="Avg residual demand (unfilled)")
ax.plot(summary.index, summary["avg_excess_mw"]          / 1e3, color="#5ab55e", linewidth=1.5,
        label="Avg excess generation")

ax.set_ylabel("Average power (GW)")
ax.set_title("Germany — annual average power: demand, generation, residual demand, excess", fontsize=11)
ax.legend(fontsize=9)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(paths.images_path / "41_annual_coverage_bars.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/41_annual_coverage_bars.png
# :name: fig-41-annual-coverage-bars
# Annual average power (GW) for demand, total renewable generation (solar + wind
# onshore + offshore), residual demand (hours where generation falls short,
# averaged over all hours), and excess generation (hours where generation exceeds
# demand). Demand (black) is reconstructed from BDEW structural profiles and is
# roughly flat. Growing renewable capacity pushes average generation (blue) up
# while excess (green) grows disproportionately faster as peak solar/wind output
# increasingly outstrips demand. Years before the first SMARD capacity record
# use the earliest available installed capacity as a lower-bound proxy.
# ```

# %% [markdown]
# ## Annual coverage percentages — full period

# %%
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(summary.index, summary["gen_pct_of_demand"], color="#4a90d9", linewidth=1.8,
        label="Avg generation / avg demand (%)")
ax.plot(summary.index, summary["still_needed_pct"],  color="#e6734a", linewidth=1.8,
        label="Still needed / avg demand (%) — residual demand")
ax.plot(summary.index, summary["excess_pct_of_gen"], color="#5ab55e", linewidth=1.8,
        label="Excess / avg generation (%) — wasted surplus")

ax.axhline(100, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_ylabel("Percentage (%)")
ax.set_title("Germany — renewable coverage percentages (full PECD period)", fontsize=11)
ax.legend(fontsize=9)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
fig.tight_layout()
fig.savefig(paths.images_path / "41_annual_coverage_pct.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/41_annual_coverage_pct.png
# :name: fig-41-annual-coverage-pct
# Three coverage percentages across the full reanalysis period. *Generation
# share* (blue) is average renewable generation as a fraction of average demand;
# it rises steeply as capacity grows. *Still needed* (orange) is average
# residual demand as a fraction of average demand — the share of demand that
# renewables cannot cover because their output and the demand deficit do not
# coincide in time. *Excess share* (green) is average surplus generation as a
# fraction of average renewable output — the fraction "wasted" in the absence of
# storage or flexible demand. The widening gap between generation share and
# still-needed share illustrates the growing temporal mismatch as installed
# capacity increases.
# ```

# %% [markdown]
# ## Monthly climatology (averaged across all years)
#
# Average power by calendar month, pooling all available years.

# %%
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

clim = df.groupby(df.index.month)[
    ["demand_mw", "gen_mw", "residual_demand_mw", "excess_mw"]
].mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(clim.index, clim["demand_mw"]         / 1e3, color="#333333", linewidth=1.8,
        marker="o", markersize=5, label="Demand")
ax.plot(clim.index, clim["gen_mw"]             / 1e3, color="#4a90d9", linewidth=1.8,
        marker="o", markersize=5, label="Generation")
ax.plot(clim.index, clim["residual_demand_mw"] / 1e3, color="#e6734a", linewidth=1.8,
        marker="s", markersize=5, label="Residual demand")
ax.plot(clim.index, clim["excess_mw"]          / 1e3, color="#5ab55e", linewidth=1.8,
        marker="^", markersize=5, label="Excess")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS)
ax.set_ylabel("Average power (GW)")
ax.set_title(f"Monthly climatology — Germany, {AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]}", fontsize=11)
ax.legend(fontsize=9)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(paths.images_path / "41_monthly_coverage.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/41_monthly_coverage.png
# :name: fig-41-monthly-coverage
# Monthly climatology of average power (GW) pooled across the full reanalysis
# period. The strong summer peak in solar generation drives a seasonal pattern
# where excess peaks in summer while residual demand peaks in winter when solar
# output is low and heating demand is high.
# ```

# %% [markdown]
# ## Year × month jitter charts
#
# Each dot is one calendar year's monthly average. The x-axis is the calendar
# month; the black line shows the median across all years.

# %%
df_ym = df.copy()
df_ym["year"]  = df_ym.index.year
df_ym["month"] = df_ym.index.month

ym_avg = df_ym.groupby(["year", "month"])[
    ["gen_mw", "demand_mw", "residual_demand_mw"]
].mean()
ym_avg["gen_pct"]    = ym_avg["gen_mw"]             / ym_avg["demand_mw"] * 100
ym_avg["needed_pct"] = ym_avg["residual_demand_mw"] / ym_avg["demand_mw"] * 100

rng = np.random.default_rng(42)

for col, title, fname_suffix, color in [
    ("gen_pct",    "Generation / demand (%)",      "gen",    "#4a90d9"),
    ("needed_pct", "Residual demand / demand (%)", "needed", "#e6734a"),
]:
    data    = ym_avg[col].reset_index()
    jitter  = rng.uniform(-0.25, 0.25, size=len(data))
    medians = data.groupby("month")[col].median()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.scatter(data["month"] + jitter, data[col], color=color, alpha=0.5, s=18,
               edgecolors="black", linewidths=0.4)
    ax.plot(medians.index, medians.values, color="black", linewidth=1.5,
            marker="o", markersize=5, label="Median")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel(title)
    ax.set_title(
        f"Germany — {title} by calendar month\n"
        f"(one dot per year, {AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]})",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    fig.tight_layout()
    fig.savefig(paths.images_path / f"41_monthly_coverage_pct_{fname_suffix}.png",
                dpi=150, bbox_inches="tight")
    show()

# %% [markdown]
# ```{figure} ../../output/images/41_monthly_coverage_pct_gen.png
# :name: fig-41-jitter-gen
# Jitter chart of monthly generation as a share of demand (%). Each dot is one
# calendar year; the black line is the monthly median. The wide spread within
# each month reflects year-to-year weather variability; the strong seasonal
# pattern is driven by the solar resource.
# ```
#
# ```{figure} ../../output/images/41_monthly_coverage_pct_needed.png
# :name: fig-41-jitter-needed
# Jitter chart of monthly residual demand as a share of total demand (%).
# Winter months show both higher median residual demand and wider spread,
# reflecting the combined effect of low solar output and higher heating demand.
# ```

# %% [markdown]
# ## Interpretation summary

# %%
recent = summary[summary.index >= summary.index[-5]]
print("=" * 70)
print("RENEWABLE DEMAND COVERAGE SUMMARY — Germany (most recent 5 years)")
print("=" * 70)
print(f"\n{'Year':<6} {'Gen/Demand':>12} {'Still needed':>14} {'Excess/Gen':>12} {'Surplus hrs':>13}")
print("-" * 70)
for yr, row in recent.iterrows():
    print(
        f"{yr:<6} "
        f"{row['gen_pct_of_demand']:>11.1f}%"
        f"{row['still_needed_pct']:>13.1f}%"
        f"{row['excess_pct_of_gen']:>11.1f}%"
        f"{row['surplus_hours_pct']:>12.1f}%"
    )
print("-" * 70)
print("\nFull period averages:")
print(f"  Gen/Demand    : {summary['gen_pct_of_demand'].mean():.1f}%")
print(f"  Still needed  : {summary['still_needed_pct'].mean():.1f}%")
print(f"  Excess/Gen    : {summary['excess_pct_of_gen'].mean():.1f}%")
print(f"  Surplus hrs   : {summary['surplus_hours_pct'].mean():.1f}%")
print("\nNotes:")
print("  Gen/Demand    : avg renewable gen as % of avg demand (nameplate coverage)")
print("  Still needed  : avg residual demand as % of avg demand (temporal gap)")
print("  Excess/Gen    : avg surplus as % of avg generation (curtailment proxy)")
print("  Surplus hrs   : share of hours where renewables exceed demand")

# %% [markdown]
# ## Capacity vs. effective output vs. demand — summary bar chart
#
# Compares four quantities (all in GW) on a single axis:
# - **Installed capacity**: nameplate sum of solar + wind onshore + offshore
# - **Avg generation**: mean hourly output across the full analysis period
# - **Avg useful generation**: avg generation minus avg excess (surplus that exceeds demand)
# - **Avg demand**: mean hourly demand across the full analysis period

# %%
cap_total_gw        = (cap_latest["solar"] + cap_latest["wind_onshore"] + cap_latest["wind_offshore"]) / 1e3
avg_gen_gw          = df["gen_mw"].mean() / 1e3
avg_useful_gen_gw   = (df["gen_mw"] - df["excess_mw"]).mean() / 1e3
avg_demand_gw       = df["demand_mw"].mean() / 1e3

labels = [
    "Installed\ncapacity",
    "Avg\ngeneration",
    "Avg useful\ngeneration",
    "Avg\ndemand",
]
values = [cap_total_gw, avg_gen_gw, avg_useful_gen_gw, avg_demand_gw]
colors = ["#9b59b6", "#4a90d9", "#5ab55e", "#333333"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=0.8)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f} GW", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylabel("Average power (GW)")
ax.set_title(
    f"Germany — installed capacity vs. generation vs. demand\n"
    f"(weather avg {AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]}, capacities fixed at {cap_re.index[-1].date()})",
    fontsize=10,
)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
ax.set_ylim(0, max(values) * 1.18)
fig.tight_layout()
fig.savefig(paths.images_path / "41_capacity_vs_output_bar.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/41_capacity_vs_output_bar.png
# :name: fig-41-capacity-vs-output-bar
# Summary bar chart comparing installed nameplate capacity, average renewable
# generation, average useful generation (generation minus surplus that exceeds
# demand), and average electricity demand — all in GW. The gap between installed
# capacity and average generation reflects the capacity factor; the gap between
# average generation and useful generation reflects temporal mismatch (surplus
# hours); and the gap between useful generation and demand reflects the residual
# demand still to be covered by dispatchable sources.
# ```

# %% [markdown]
# ## Battery storage simulation
#
# Storage capacity: **1 × average hourly demand** (MWh) — enough to satisfy
# mean demand for one hour if fully charged.
#
# Each hour in sequence:
# - **Surplus** (gen > demand): store the excess up to remaining capacity.
# - **Deficit** (gen < demand): discharge stored energy to offset residual load.
#
# `storage_delta_mw` > 0 means charging, < 0 means discharging.

# %%
gen_arr    = df["gen_mw"].values
demand_arr = df["demand_mw"].values

storage_mwh = demand_arr.mean()          # 1-hour average-demand capacity
n = len(gen_arr)

soc_arr   = np.empty(n)
delta_arr = np.zeros(n)
resid_arr = np.zeros(n)

soc_val = 0.0
for i in range(n):
    raw_residual = demand_arr[i] - gen_arr[i]
    if raw_residual <= 0:                # surplus → charge
        charge = min(-raw_residual, storage_mwh - soc_val)
        delta_arr[i] = charge
        soc_val += charge
    else:                                # deficit → discharge
        discharge = min(raw_residual, soc_val)
        delta_arr[i] = -discharge
        soc_val -= discharge
        resid_arr[i] = raw_residual - discharge
    soc_arr[i] = soc_val

df["storage_delta_mw"]  = delta_arr
df["soc_mwh"]           = soc_arr
df["residual_storage_mw"] = resid_arr

avg_useful_storage_gw = (demand_arr - resid_arr).mean() / 1e3

print(f"Storage capacity : {storage_mwh:,.0f} MWh  ({storage_mwh/1e3:.2f} GWh)")
print(f"\nAvg useful gen  (no storage): {avg_useful_gen_gw:.2f} GW")
print(f"Avg useful gen  (w/ storage): {avg_useful_storage_gw:.2f} GW")
print(f"Improvement     : +{avg_useful_storage_gw - avg_useful_gen_gw:.2f} GW"
      f"  ({(avg_useful_storage_gw - avg_useful_gen_gw) / avg_demand_gw * 100:.1f}% of avg demand)")
print(f"\nHours charged   : {(delta_arr > 0).sum():,}  ({(delta_arr > 0).mean()*100:.1f}%)")
print(f"Hours discharged: {(delta_arr < 0).sum():,}  ({(delta_arr < 0).mean()*100:.1f}%)")
print(f"Hours idle      : {(delta_arr == 0).sum():,}  ({(delta_arr == 0).mean()*100:.1f}%)")
print(f"Avg SoC         : {soc_arr.mean():,.0f} MWh  ({soc_arr.mean()/storage_mwh*100:.1f}% of capacity)")

# %% [markdown]
# ### Capacity vs. output bar chart with storage

# %%
bat_labels = [
    "Installed\ncapacity\n(GW)",
    "Battery\ncapacity\n(GWh)",
    "Avg\ngeneration\n(GW)",
    "Avg useful gen\nw/ storage\n(GW)",
    "Avg\ndemand\n(GW)",
]
bat_values = [cap_total_gw, storage_mwh / 1e3, avg_gen_gw, avg_useful_storage_gw, avg_demand_gw]
bat_colors = ["#9b59b6", "#f39c12", "#4a90d9", "#5ab55e", "#333333"]
bat_units  = ["GW", "GWh", "GW", "GW", "GW"]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(bat_labels, bat_values, color=bat_colors, width=0.5, edgecolor="white", linewidth=0.8)
for bar, val, unit in zip(bars, bat_values, bat_units):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f} {unit}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("GW  (battery: GWh)")
ax.set_title(
    f"Germany — capacity vs. generation vs. demand with 1-hour battery\n"
    f"(weather avg {AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]}, capacities fixed at {cap_re.index[-1].date()})",
    fontsize=10,
)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
ax.set_ylim(0, max(bat_values) * 1.18)
fig.tight_layout()
fig.savefig(paths.images_path / "41_capacity_vs_output_storage_bar.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/41_capacity_vs_output_storage_bar.png
# :name: fig-41-capacity-vs-output-storage-bar
# Summary bar chart with 1-hour battery storage. Installed nameplate capacity
# (purple) and battery capacity in GWh (orange) are shown alongside average
# generation (blue), average useful generation after battery dispatch (green),
# and average demand (black). Comparing the green bar here with the one in the
# no-storage chart shows the incremental benefit of the battery.
# ```

# %% [markdown]
# ### Monthly climatology with battery storage
#
# Same layout as the earlier monthly climatology, extended with the residual
# demand after battery dispatch and the average state of charge (SoC) per month.

# %%
df["abs_delta_mw"] = df["storage_delta_mw"].abs()

clim_s = df.groupby(df.index.month)[
    ["demand_mw", "gen_mw", "residual_demand_mw", "excess_mw",
     "residual_storage_mw", "abs_delta_mw"]
].mean()

# Monthly average throughput per year (GWh/month): sum |delta| over each
# year-month, then average across years
monthly_throughput = (
    df.groupby([df.index.year, df.index.month])["abs_delta_mw"]
    .sum()                       # MWh total per year-month
    .groupby(level=1)
    .mean()                      # average across years
    / 1e3                        # → GWh
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# — top panel: power flows —
ax1.plot(clim_s.index, clim_s["demand_mw"]          / 1e3, color="#333333", linewidth=1.8,
         marker="o", markersize=5, label="Demand")
ax1.plot(clim_s.index, clim_s["gen_mw"]              / 1e3, color="#4a90d9", linewidth=1.8,
         marker="o", markersize=5, label="Generation")
ax1.plot(clim_s.index, clim_s["residual_demand_mw"]  / 1e3, color="#e6734a", linewidth=1.8,
         marker="s", markersize=5, label="Residual demand (no storage)")
ax1.plot(clim_s.index, clim_s["residual_storage_mw"] / 1e3, color="#c0392b", linewidth=1.8,
         marker="s", markersize=5, linestyle="--", label="Residual demand (with storage)")
ax1.plot(clim_s.index, clim_s["excess_mw"]           / 1e3, color="#5ab55e", linewidth=1.8,
         marker="^", markersize=5, label="Excess")
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(MONTH_LABELS)
ax1.set_ylabel("Average power (GW)")
ax1.set_title(
    f"Monthly climatology with battery storage — Germany, {AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]}",
    fontsize=11,
)
ax1.legend(fontsize=9)
ax1.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax1.set_axisbelow(True)

# — bottom panel: average monthly battery throughput —
ax2.bar(monthly_throughput.index, monthly_throughput.values,
        color="#f39c12", alpha=0.8, width=0.6)
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(MONTH_LABELS)
ax2.set_ylabel("Avg throughput (GWh/month)")
ax2.set_title("Average battery throughput (|charge| + |discharge|) per month", fontsize=10)
ax2.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax2.set_axisbelow(True)

fig.tight_layout()
fig.savefig(paths.images_path / "41_monthly_coverage_storage.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/41_monthly_coverage_storage.png
# :name: fig-41-monthly-coverage-storage
# **Top**: Monthly climatology of average power (GW) extended with battery
# dispatch. The dashed red line shows residual demand after the 1-hour battery
# reduces it; compare with the solid orange line (no storage) to see where
# storage provides the most relief.
# **Bottom**: Average total battery throughput (GWh/month) — the sum of absolute
# charge and discharge flows per month, averaged across all years. Higher bars
# indicate months where the battery cycles more actively.
# ```

# %% [markdown]
# ## Capacity scaling — useful power with and without storage

# %%
scales = np.linspace(1, 20, 500)
avg_useful_gw = np.array([
    np.minimum(k * gen_arr, demand_arr).mean() / 1e3
    for k in scales
])

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(scales, avg_useful_gw, color="#5ab55e", linewidth=1.8, label="Avg useful generation")
ax.axhline(avg_demand_gw, color="#333333", linewidth=1.5, linestyle="--", label="Avg demand")

tick_factors = range(1, 21)
ax.set_xticks(list(tick_factors))
ax.set_xticklabels([f"{k}x" for k in tick_factors])
ax.set_xlabel("Capacity scaling factor (relative to current fleet)")
ax.set_ylabel("Average power (GW)")
ax.set_title(
    "Germany — useful renewable output vs. capacity scaling factor\n"
    "(fixed fleet ratio, weather avg {first}–{last})".format(
        first=AVAIL_YEARS[0], last=AVAIL_YEARS[-1]
    ),
    fontsize=10,
)
ax.legend(fontsize=9)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(paths.images_path / "41_capacity_scaling.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/41_capacity_scaling.png
# :name: fig-41-capacity-scaling
# Average generation (blue) and average useful generation (green) as a function
# of total installed capacity, scaling the current solar/wind fleet ratio from
# 2× to 20× its present size. Average generation grows linearly with capacity;
# useful generation (the share that actually covers demand without going to
# surplus) grows sub-linearly and asymptotically approaches the average demand
# level (dashed black line) as more and more output falls in surplus hours.
# ```

# %% [markdown]
# ## Renewable × storage scaling heatmap
#
# Renewable capacity is scaled 1×–5× and battery storage 0×–5× of average
# hourly demand. For each combination the battery simulation is re-run and the
# resulting average useful generation (GW) is shown.

# %%
RE_SCALES  = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
BAT_SCALES = list(range(0, 11))
storage_base_mwh = demand_arr.mean()   # 1× = avg hourly demand


def simulate_useful_gen(gen_k: np.ndarray, demand: np.ndarray, s_cap: float) -> float:
    """Return avg useful generation (GW) for given hourly gen, demand and storage capacity."""
    if s_cap == 0:
        return float(np.minimum(gen_k, demand).mean() / 1e3)
    resid = np.zeros(len(gen_k))
    soc = 0.0
    for i in range(len(gen_k)):
        raw = demand[i] - gen_k[i]
        if raw <= 0:
            soc = min(soc + (-raw), s_cap)
        else:
            discharge = min(raw, soc)
            soc -= discharge
            resid[i] = raw - discharge
    return float((demand - resid).mean() / 1e3)


grid = np.zeros((len(BAT_SCALES), len(RE_SCALES)))
for i, b in enumerate(BAT_SCALES):
    for j, r in enumerate(RE_SCALES):
        grid[i, j] = simulate_useful_gen(gen_arr * r, demand_arr, b * storage_base_mwh)

grid_plot = grid[::-1]   # flip rows so 0x battery is at the bottom

vmin = grid[0, 0]        # 0x storage, 1x renewables — baseline
vmax = avg_demand_gw     # fully covered demand

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(grid_plot, cmap="RdYlGn", aspect="auto",
               vmin=vmin, vmax=vmax, interpolation="nearest")

for i in range(len(BAT_SCALES)):
    for j in range(len(RE_SCALES)):
        ax.text(j, i, f"{grid_plot[i, j]:.1f}", ha="center", va="center", fontsize=9)

ax.set_xticks(range(len(RE_SCALES)))
ax.set_xticklabels([f"{r}x" for r in RE_SCALES])
ax.set_yticks(range(len(BAT_SCALES)))
ax.set_yticklabels([f"{b}x" for b in BAT_SCALES[::-1]])
ax.set_xlabel("Renewable capacity scaling factor")
ax.set_ylabel("Battery capacity scaling factor\n(× avg hourly demand)")
ax.set_title(
    f"Avg useful generation (GW) — renewable × storage scaling\n"
    f"(weather avg {AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]}, avg demand = {avg_demand_gw:.1f} GW)",
    fontsize=10,
)
plt.colorbar(im, ax=ax, label="Avg useful generation (GW)", shrink=0.85)
fig.tight_layout()
fig.savefig(paths.images_path / "41_scaling_heatmap.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/41_scaling_heatmap.png
# :name: fig-41-scaling-heatmap
# Heatmap of average useful generation (GW) across combinations of renewable
# capacity scaling (x-axis, 1×–5× current fleet) and battery storage capacity
# (y-axis, 0×–5× average hourly demand). Green cells approach average demand;
# red cells indicate large residual demand remains. Moving right adds renewable
# capacity; moving up adds storage. The interaction between the two reveals
# diminishing returns and trade-off curves.
# ```

# %% [markdown]
# ## Useful generation vs. renewable scaling — three storage scenarios

# %%
scales_line = np.linspace(1, 20, 500)
useful_no_bat  = np.array([simulate_useful_gen(gen_arr * k, demand_arr, 0) for k in scales_line])
useful_bat_1x  = np.array([simulate_useful_gen(gen_arr * k, demand_arr, 1  * storage_base_mwh) for k in scales_line])
useful_bat_10x = np.array([simulate_useful_gen(gen_arr * k, demand_arr, 10 * storage_base_mwh) for k in scales_line])

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(scales_line, useful_no_bat,  color="#e6734a", linewidth=1.8, label="No storage (0x)")
ax.plot(scales_line, useful_bat_1x,  color="#4a90d9", linewidth=1.8, label="1x battery storage")
ax.plot(scales_line, useful_bat_10x, color="#5ab55e", linewidth=1.8, label="10x battery storage")
ax.axhline(avg_demand_gw, color="#333333", linewidth=1.5, linestyle="--", label="Avg demand")

tick_factors = range(1, 21)
ax.set_xticks(list(tick_factors))
ax.set_xticklabels([f"{k}x" for k in tick_factors])
ax.set_xlabel("Renewable capacity scaling factor (relative to current fleet)")
ax.set_ylabel("Average useful generation (GW)")
ax.set_title(
    "Germany — useful generation vs. renewable scaling, by storage scenario\n"
    "(weather avg {first}–{last})".format(first=AVAIL_YEARS[0], last=AVAIL_YEARS[-1]),
    fontsize=10,
)
ax.legend(fontsize=9)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(paths.images_path / "41_scaling_with_storage.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/41_scaling_with_storage.png
# :name: fig-41-scaling-with-storage
# Avg useful generation (GW) as a function of renewable capacity scaling for
# three storage scenarios: no storage (orange), 1× avg hourly demand battery
# (blue), and 10× battery (green). All three curves asymptote towards average
# demand (dashed black); storage raises the curve and shifts the point at which
# demand is nearly fully covered to a lower renewable scaling factor.
# ```

# %%

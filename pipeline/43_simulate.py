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
# # Greedy Storage Dispatch Simulation — Germany 2022–2023
#
# Uses the optimal capacities from script 42 (copper-plate LP) and simulates
# how battery and hydrogen storage evolve over the same 2022–2023 period
# under a **greedy "fill first" dispatch rule**:
#
# * **Surplus hours** (RE available > demand): charge battery to capacity, then
#   hydrogen; anything that cannot be stored is curtailed.
# * **Deficit hours** (RE available < demand): discharge battery first, then
#   hydrogen; any residual unmet demand is recorded as lost load.
#
# No look-ahead or optimisation — purely causal, hour-by-hour.

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
# ## Load optimal capacities (script 42 output)

# %%
cap = pd.read_parquet(paths.copper_plate_opt_capacities_de_file)
print(cap.to_string())

cap_solar_mw   = cap.loc["solar",        "p_nom_opt_mw"]
cap_onshore_mw = cap.loc["wind_onshore", "p_nom_opt_mw"]
cap_bat_mw     = cap.loc["battery",      "p_nom_opt_mw"]
cap_h2_mw      = cap.loc["hydrogen",     "p_nom_opt_mw"]

# Storage technology parameters (must match script 42)
BAT_MAX_HOURS    = 4
BAT_EFF_STORE    = 0.90   # charging efficiency (power → stored energy)
BAT_EFF_DISPATCH = 0.90   # discharging efficiency (stored energy → power)

H2_MAX_HOURS     = 168
H2_EFF_STORE     = 0.80   # electrolyser
H2_EFF_DISPATCH  = 0.60   # H2-CCGT

bat_energy_cap = cap_bat_mw * BAT_MAX_HOURS   # MWh
h2_energy_cap  = cap_h2_mw  * H2_MAX_HOURS    # MWh

print(f"\nCapacities loaded:")
print(f"  Solar PV        : {cap_solar_mw / 1e3:>8.2f} GW")
print(f"  Wind onshore    : {cap_onshore_mw / 1e3:>8.2f} GW")
print(f"  Battery         : {cap_bat_mw / 1e3:>8.2f} GW  ({bat_energy_cap / 1e3:.1f} GWh)")
print(f"  Hydrogen        : {cap_h2_mw / 1e3:>8.2f} GW  ({h2_energy_cap / 1e3:.1f} GWh)")

# %% [markdown]
# ## Load PECD capacity factors and demand — same period as script 42

# %%
pecd = pd.read_parquet(paths.pecd_processed_file)
cf_solar_raw   = pecd["solar_photovoltaic_power_generation"]["capacity_factor_ratio"]["DE"]
cf_onshore_raw = pecd["wind_power_generation_onshore"]["capacity_factor_ratio"]["DE"]


def pecd_to_smard_index(s: pd.Series) -> pd.Series:
    """Convert PECD hour-starting UTC to SMARD hour-ending CET (naive index)."""
    return (
        s.tz_localize("UTC")
         .tz_convert("Europe/Berlin")
         .shift(1, freq="h")
         .tz_localize(None)
    )


def convert_and_clean(raw: pd.Series) -> pd.Series:
    s = pecd_to_smard_index(raw)
    return s[~s.index.duplicated(keep="first")]


cf_solar   = convert_and_clean(cf_solar_raw)
cf_onshore = convert_and_clean(cf_onshore_raw)

demand_preds = pd.read_parquet(paths.de_demand_predictions_file)
demand_pred  = demand_preds["demand_baseline_mw"].rename("demand_mw")

# Select 2022–2023
start_dt = pd.Timestamp("2022-01-01")
end_dt   = pd.Timestamp("2024-01-01")
all_idx  = cf_solar.index.intersection(cf_onshore.index).intersection(demand_pred.index)
snapshots = all_idx[(all_idx >= start_dt) & (all_idx < end_dt)]

cf_s   = cf_solar.reindex(snapshots).clip(0, 1).fillna(0)
cf_on  = cf_onshore.reindex(snapshots).clip(0, 1).fillna(0)
demand = demand_pred.reindex(snapshots).interpolate(limit=2)

valid     = demand.notna()
snapshots = snapshots[valid]
cf_s      = cf_s[valid]
cf_on     = cf_on[valid]
demand    = demand[valid]

print(f"Simulation period: {snapshots[0]} → {snapshots[-1]}  ({len(snapshots):,} hours)")

# %% [markdown]
# ## Greedy simulation

# %%
T = len(snapshots)

# Available RE at each hour (MW)
re_avail = cf_s.values * cap_solar_mw + cf_on.values * cap_onshore_mw
demand_v = demand.values

# State-of-charge arrays (MWh); index 0..T, where soc[t] is state at start of hour t
bat_soc = np.zeros(T + 1)
h2_soc  = np.zeros(T + 1)
# Start at zero — storage fills naturally from first surpluses
bat_soc[0] = 0.0
h2_soc[0]  = 0.0

# Per-hour dispatch records (MW)
bat_charge   = np.zeros(T)
bat_dispatch = np.zeros(T)
h2_charge    = np.zeros(T)
h2_dispatch  = np.zeros(T)
curtailment  = np.zeros(T)
lost_load    = np.zeros(T)

for t in range(T):
    surplus = re_avail[t] - demand_v[t]

    if surplus >= 0:
        # --- Surplus: charge battery first, then hydrogen, curtail remainder ---
        bat_space     = bat_energy_cap - bat_soc[t]
        bat_ch_max    = min(cap_bat_mw, bat_space / BAT_EFF_STORE)
        bat_ch        = min(surplus, bat_ch_max)
        bat_charge[t] = bat_ch
        remaining     = surplus - bat_ch

        h2_space     = h2_energy_cap - h2_soc[t]
        h2_ch_max    = min(cap_h2_mw, h2_space / H2_EFF_STORE)
        h2_ch        = min(remaining, h2_ch_max)
        h2_charge[t] = h2_ch
        remaining   -= h2_ch

        curtailment[t] = remaining

        bat_soc[t + 1] = bat_soc[t] + bat_ch * BAT_EFF_STORE
        h2_soc[t + 1]  = h2_soc[t]  + h2_ch  * H2_EFF_STORE

    else:
        # --- Deficit: discharge battery first, then hydrogen ---
        deficit = -surplus

        bat_avail_power = min(cap_bat_mw, bat_soc[t] * BAT_EFF_DISPATCH)
        bat_dis         = min(deficit, bat_avail_power)
        bat_dispatch[t] = bat_dis
        deficit        -= bat_dis

        h2_avail_power  = min(cap_h2_mw, h2_soc[t] * H2_EFF_DISPATCH)
        h2_dis          = min(deficit, h2_avail_power)
        h2_dispatch[t]  = h2_dis
        deficit        -= h2_dis

        lost_load[t] = deficit

        bat_soc[t + 1] = bat_soc[t] - (bat_dis / BAT_EFF_DISPATCH)
        h2_soc[t + 1]  = h2_soc[t]  - (h2_dis  / H2_EFF_DISPATCH)

# Trim SoC series to simulation period (drop the extra t+1 tail element)
bat_soc_ts = pd.Series(bat_soc[:T], index=snapshots)
h2_soc_ts  = pd.Series(h2_soc[:T], index=snapshots)

# Summary statistics
total_demand_mwh  = demand_v.sum()
total_re_mwh      = re_avail.sum()
total_curtail_mwh = curtailment.sum()
total_lost_mwh    = lost_load.sum()
total_bat_dis_mwh = bat_dispatch.sum()
total_h2_dis_mwh  = h2_dispatch.sum()
N_YEARS           = 2

print(f"\nSimulation summary ({N_YEARS} years):")
print(f"  RE available    : {total_re_mwh / 1e6:.2f} TWh")
print(f"  Demand          : {total_demand_mwh / 1e6:.2f} TWh")
print(f"  Curtailment     : {total_curtail_mwh / 1e6:.2f} TWh  ({total_curtail_mwh / total_re_mwh * 100:.1f}% of avail. RE)")
print(f"  Battery dispatch: {total_bat_dis_mwh / 1e6:.2f} TWh")
print(f"  H2 dispatch     : {total_h2_dis_mwh / 1e6:.2f} TWh")
print(f"  Lost load       : {total_lost_mwh / 1e6:.4f} TWh  ({total_lost_mwh / total_demand_mwh * 100:.3f}% of demand)")

# %% [markdown]
# ## Chart 1 — Storage state of charge over time

# %%
TECH_COLORS = {
    "battery":  "#3498db",
    "hydrogen": "#9b59b6",
}
TECH_LABELS = {
    "battery":  f"Battery ({BAT_MAX_HOURS}h)",
    "hydrogen": f"Hydrogen ({H2_MAX_HOURS}h)",
}

storage_info = [
    ("battery",  bat_soc_ts, cap_bat_mw, bat_energy_cap),
    ("hydrogen", h2_soc_ts,  cap_h2_mw,  h2_energy_cap),
]

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

for ax, (tech, soc_ts, p_nom, e_cap) in zip(axes, storage_info):
    color = TECH_COLORS[tech]
    label = TECH_LABELS[tech]

    soc_pct = soc_ts / e_cap * 100

    daily_min  = soc_pct.resample("D").min()
    daily_max  = soc_pct.resample("D").max()
    daily_mean = soc_pct.resample("D").mean()

    ax.fill_between(daily_min.index, daily_min.values, daily_max.values,
                    alpha=0.25, color=color, label="Daily min–max")
    ax.plot(daily_mean.index, daily_mean.values, color=color,
            linewidth=1.2, label="Daily mean")

    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylabel("State of charge (%)")
    ax.set_title(
        f"{label} — state of charge  "
        f"(capacity: {e_cap / 1e3:.1f} GWh  |  power: {p_nom / 1e3:.2f} GW)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)

fig.suptitle(
    "Storage state of charge — greedy fill-first simulation, Germany 2022–2023",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(paths.images_path / "43_storage_soc.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/43_storage_soc.png
# :name: fig-43-storage-soc
# Daily state of charge (%) for battery (top) and hydrogen (bottom) under the
# greedy fill-first dispatch rule. Battery cycles daily; hydrogen accumulates
# through summer and is drawn down over winter low-wind periods.
# ```

# %% [markdown]
# ## Chart 2 — Curtailment over time

# %%
curtail_series = pd.Series(curtailment, index=snapshots)
curtail_daily  = curtail_series.resample("D").mean() / 1e3   # GW daily average
curtail_monthly = curtail_series.resample("ME").sum() / 1e6  # TWh per month

fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=False)

# Top: daily average curtailed power
ax = axes[0]
ax.fill_between(curtail_daily.index, curtail_daily.values, alpha=0.7,
                color="#e74c3c", linewidth=0)
ax.plot(curtail_daily.index, curtail_daily.values, color="#c0392b", linewidth=0.5)
ax.set_ylabel("Curtailed power (GW, daily avg)")
ax.set_title(
    f"Daily average curtailment  "
    f"(total: {total_curtail_mwh / 1e6:.1f} TWh, "
    f"{total_curtail_mwh / total_re_mwh * 100:.1f}% of available RE)",
    fontsize=10,
)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)

# Bottom: monthly curtailment totals
ax = axes[1]
ax.bar(curtail_monthly.index, curtail_monthly.values, width=25,
       color="#e74c3c", alpha=0.8, edgecolor="white", linewidth=0.5)
ax.set_ylabel("Curtailment (TWh/month)")
ax.set_title("Monthly curtailment totals", fontsize=10)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)

fig.suptitle(
    "Curtailment — greedy fill-first simulation, Germany 2022–2023",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(paths.images_path / "43_curtailment.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/43_curtailment.png
# :name: fig-43-curtailment
# Top: daily average curtailed power (GW). Bottom: monthly curtailment totals (TWh).
# Curtailment peaks in summer when solar surplus exceeds storage capacity.
# ```

# %% [markdown]
# ## Chart 3 — Energy generation vs consumption

# %%
# Build monthly energy balance (TWh)
re_avail_series    = pd.Series(re_avail,    index=snapshots)
demand_series      = pd.Series(demand_v,    index=snapshots)
bat_dis_series     = pd.Series(bat_dispatch, index=snapshots)
h2_dis_series      = pd.Series(h2_dispatch, index=snapshots)
bat_charge_series  = pd.Series(bat_charge,  index=snapshots)
h2_charge_series   = pd.Series(h2_charge,   index=snapshots)
lost_load_series   = pd.Series(lost_load,   index=snapshots)

TWH = 1e6

# --- Build per-hour derived quantities ---
# RE direct to load: surplus hours → RE covers demand, rest goes to storage/curtailment
#                    deficit hours → all RE goes to load
# Formula: re_avail − bat_charge − h2_charge − curtailment
re_direct_series = (
    re_avail_series - bat_charge_series - h2_charge_series - curtail_series
)

# Charging losses: energy drawn from the RE bus that is wasted entering storage
bat_charge_loss_series = bat_charge_series * (1 - BAT_EFF_STORE)
h2_charge_loss_series  = h2_charge_series  * (1 - H2_EFF_STORE)
# Energy that actually enters the storage medium (= charging power × efficiency)
bat_stored_series = bat_charge_series * BAT_EFF_STORE
h2_stored_series  = h2_charge_series  * H2_EFF_STORE

TWH = 1e6

monthly = pd.DataFrame({
    "re_avail":         re_avail_series.resample("ME").sum()          / TWH,
    # --- Fate of RE ---
    "re_direct":        re_direct_series.resample("ME").sum()         / TWH,
    "bat_stored":       bat_stored_series.resample("ME").sum()        / TWH,
    "h2_stored":        h2_stored_series.resample("ME").sum()         / TWH,
    "bat_charge_loss":  bat_charge_loss_series.resample("ME").sum()   / TWH,
    "h2_charge_loss":   h2_charge_loss_series.resample("ME").sum()    / TWH,
    "curtailment":      curtail_series.resample("ME").sum()           / TWH,
    # --- Sources of demand ---
    "bat_dispatch":     bat_dis_series.resample("ME").sum()           / TWH,
    "h2_dispatch":      h2_dis_series.resample("ME").sum()            / TWH,
    "demand":           demand_series.resample("ME").sum()            / TWH,
    "lost_load":        lost_load_series.resample("ME").sum()         / TWH,
})

# Sanity checks (should be ~0):
# re_direct + bat_stored + bat_charge_loss + h2_stored + h2_charge_loss + curtailment = re_avail
fate_sum = (
    monthly["re_direct"] + monthly["bat_stored"] + monthly["bat_charge_loss"]
    + monthly["h2_stored"] + monthly["h2_charge_loss"] + monthly["curtailment"]
)
# re_direct + bat_dispatch + h2_dispatch + lost_load = demand
demand_sum = monthly["re_direct"] + monthly["bat_dispatch"] + monthly["h2_dispatch"] + monthly["lost_load"]

print("\nEnergy balance checks (max absolute error per month):")
print(f"  Fate-of-RE residual : {(fate_sum - monthly['re_avail']).abs().max() * TWH:.0f} MWh")
print(f"  Demand-source residual: {(demand_sum - monthly['demand']).abs().max() * TWH:.0f} MWh")

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

x = np.arange(len(monthly))
month_labels = [d.strftime("%b %Y") for d in monthly.index]
BW = 0.7  # bar width

# --- Top panel: fate of all generated RE ---
# Stacked: re_direct | bat_stored | bat_charge_loss | h2_stored | h2_charge_loss | curtailment
# Sum = re_avail
ax = axes[0]
b0 = monthly["re_direct"]
b1 = b0 + monthly["bat_stored"]
b2 = b1 + monthly["bat_charge_loss"]
b3 = b2 + monthly["h2_stored"]
b4 = b3 + monthly["h2_charge_loss"]

ax.bar(x, monthly["re_direct"],       width=BW, color="#f1c40f",           label="Direct to demand",          edgecolor="white", linewidth=0.3)
ax.bar(x, monthly["bat_stored"],      width=BW, bottom=b0, color="#3498db", label="Stored — battery",          edgecolor="white", linewidth=0.3)
ax.bar(x, monthly["bat_charge_loss"], width=BW, bottom=b1, color="#aed6f1", label="Charging loss — battery",   edgecolor="white", linewidth=0.3)
ax.bar(x, monthly["h2_stored"],       width=BW, bottom=b2, color="#9b59b6", label="Stored — hydrogen",         edgecolor="white", linewidth=0.3)
ax.bar(x, monthly["h2_charge_loss"],  width=BW, bottom=b3, color="#d2b4de", label="Charging loss — hydrogen",  edgecolor="white", linewidth=0.3)
ax.bar(x, monthly["curtailment"],     width=BW, bottom=b4, color="#e74c3c", label="Curtailed",                 edgecolor="white", linewidth=0.3)

ax.plot(x, monthly["re_avail"], color="#d35400", linewidth=1.5,
        marker="o", markersize=4, label="RE available (total)", zorder=5)
ax.plot(x, monthly["demand"], color="#2c3e50", linewidth=1.5,
        marker="s", markersize=4, label="Demand", zorder=5)

ax.set_ylabel("Energy (TWh/month)")
ax.set_title(
    "Fate of available RE generation  "
    "(direct demand + stored net + charging losses + curtailment = total available)",
    fontsize=10,
)
ax.legend(fontsize=9, loc="upper right", ncol=2)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)

# --- Bottom panel: sources of demand ---
# Stacked: re_direct | bat_dispatch | h2_dispatch | lost_load
# Sum = demand
ax = axes[1]
d1 = monthly["re_direct"] + monthly["bat_dispatch"]

ax.bar(x, monthly["re_direct"],   width=BW, color="#f1c40f",           label="RE direct",         edgecolor="white", linewidth=0.3)
ax.bar(x, monthly["bat_dispatch"], width=BW, bottom=monthly["re_direct"],
       color="#3498db", label="Battery discharge", edgecolor="white", linewidth=0.3)
ax.bar(x, monthly["h2_dispatch"],  width=BW, bottom=d1,
       color="#9b59b6", label="H2 discharge", edgecolor="white", linewidth=0.3)
if monthly["lost_load"].sum() > 0:
    ax.bar(x, monthly["lost_load"], width=BW, bottom=d1 + monthly["h2_dispatch"],
           color="#c0392b", label="Lost load", edgecolor="white", linewidth=0.3)

ax.plot(x, monthly["demand"], color="#2c3e50", linewidth=1.5,
        marker="s", markersize=4, label="Demand", zorder=5)

ax.set_ylabel("Energy (TWh/month)")
ax.set_title("Sources of demand  (RE direct + storage discharge = demand)", fontsize=10)
ax.legend(fontsize=9, loc="upper right")
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
ax.set_xticks(x)
ax.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=8)

fig.suptitle(
    "Energy generation vs consumption — greedy fill-first simulation, Germany 2022–2023",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(paths.images_path / "43_generation_vs_demand.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/43_generation_vs_demand.png
# :name: fig-43-generation-vs-demand
# Top: monthly energy supply (stacked: direct RE + battery + hydrogen discharge) vs
# demand. Bottom: monthly RE availability broken down into consumed and curtailed
# fractions. Summer months show large curtailment as solar surplus exceeds combined
# storage capacity; winter months show the draw-down of hydrogen reserves built up
# over summer.
# ```

# %% [markdown]
# ## Chart 4 — Installed capacities vs average demand

# %%
avg_demand_mw     = demand_v.mean()          # MW
avg_solar_out_mw  = (cf_s.values * cap_solar_mw).mean()
avg_onshore_out_mw = (cf_on.values * cap_onshore_mw).mean()
daily_demand_mwh  = avg_demand_mw * 24       # MWh
weekly_demand_mwh = avg_demand_mw * 24 * 7   # MWh

# Power overbuild ratios (nameplate vs average demand)
re_nameplate_total = cap_solar_mw + cap_onshore_mw
re_avg_output      = avg_solar_out_mw + avg_onshore_out_mw

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left panel: power capacities (GW) ---
ax = axes[0]

categories = ["Solar PV\n(nameplate)", "Wind onshore\n(nameplate)", "Battery\n(power)", "H2\n(power)"]
values_gw   = [cap_solar_mw / 1e3, cap_onshore_mw / 1e3, cap_bat_mw / 1e3, cap_h2_mw / 1e3]
colors      = ["#f1c40f", "#27ae60", "#3498db", "#9b59b6"]

bars = ax.bar(categories, values_gw, color=colors, edgecolor="white", linewidth=0.5, width=0.55)

# Average demand reference line
ax.axhline(avg_demand_mw / 1e3, color="#2c3e50", linewidth=2, linestyle="--", zorder=5,
           label=f"Avg demand ({avg_demand_mw / 1e3:.1f} GW)")

# Average RE output reference line
ax.axhline(re_avg_output / 1e3, color="#d35400", linewidth=1.5, linestyle=":",
           label=f"Avg RE output ({re_avg_output / 1e3:.1f} GW)")

# Annotate overbuild ratios on bars
for bar, val in zip(bars, values_gw):
    ratio = val * 1e3 / avg_demand_mw
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"×{ratio:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Power capacity (GW)")
ax.set_title("Installed power capacities vs average demand\n(×N = multiple of avg demand)", fontsize=10)
ax.legend(fontsize=9)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)

# --- Right panel: energy capacities (GWh) relative to demand timeframes ---
ax = axes[1]

categories_e = ["Battery\nenergy", "Hydrogen\nenergy"]
values_gwh   = [bat_energy_cap / 1e3, h2_energy_cap / 1e3]
colors_e     = ["#3498db", "#9b59b6"]

bars_e = ax.bar(categories_e, values_gwh, color=colors_e, edgecolor="white", linewidth=0.5, width=0.4)

# Reference lines for demand timeframes
ax.axhline(daily_demand_mwh / 1e3, color="#2c3e50", linewidth=1.5, linestyle="--",
           label=f"1 day of avg demand ({daily_demand_mwh / 1e3:.0f} GWh)")
ax.axhline(weekly_demand_mwh / 1e3, color="#7f8c8d", linewidth=1.5, linestyle=":",
           label=f"1 week of avg demand ({weekly_demand_mwh / 1e3:.0f} GWh)")

# Annotate as hours / days of average demand
for bar, val in zip(bars_e, values_gwh):
    hours = val * 1e3 / avg_demand_mw
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values_gwh) * 0.01,
            f"{hours:.1f} h\n({hours / 24:.1f} days)",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Energy capacity (GWh)")
ax.set_title("Storage energy capacities\n(relative to average demand)", fontsize=10)
ax.legend(fontsize=9)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)

fig.suptitle(
    "Installed capacities vs average demand — Germany 2022–2023 optimum",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(paths.images_path / "43_capacity_vs_demand.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/43_capacity_vs_demand.png
# :name: fig-43-capacity-vs-demand
# Left: installed power capacities (GW) for each technology with the ×N multiple
# of average demand annotated. Right: storage energy capacities (GWh) expressed
# in hours and days of average demand, with daily and weekly demand reference lines.
# ```

# %% [markdown]
# ## Summary

# %%
print("=" * 72)
print("GREEDY FILL-FIRST SIMULATION — Germany 2022–2023")
print("=" * 72)
print(f"\n  RE available      : {total_re_mwh / 1e6:.2f} TWh")
print(f"  RE consumed       : {(total_re_mwh - total_curtail_mwh) / 1e6:.2f} TWh")
print(f"  Curtailment       : {total_curtail_mwh / 1e6:.2f} TWh  "
      f"({total_curtail_mwh / total_re_mwh * 100:.1f}% of avail. RE)")
print(f"\n  Battery dispatch  : {total_bat_dis_mwh / 1e6:.2f} TWh")
print(f"  H2 dispatch       : {total_h2_dis_mwh / 1e6:.2f} TWh")
print(f"  Total demand      : {total_demand_mwh / 1e6:.2f} TWh")
print(f"  Lost load         : {total_lost_mwh / 1e3:.1f} GWh  "
      f"({total_lost_mwh / total_demand_mwh * 100:.3f}% of demand)")
print(f"\n  Final battery SoC : {bat_soc[T] / 1e3:.1f} GWh  "
      f"({bat_soc[T] / bat_energy_cap * 100:.1f}%)")
print(f"  Final H2 SoC      : {h2_soc[T] / 1e3:.1f} GWh  "
      f"({h2_soc[T] / h2_energy_cap * 100:.1f}%)")

# %%

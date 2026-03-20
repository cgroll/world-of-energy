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
# # Optimal Copper-Plate Energy Mix — Germany
#
# Uses **PyPSA** to find the cost-minimising mix of solar PV, onshore wind,
# short-duration battery storage, and long-duration hydrogen
# storage for Germany. The country is treated as a single node (copper-plate /
# transport model) — no internal transmission constraints.
#
# **Capacity factors**: PECD ERA5 reanalysis for Germany, most recent complete
# April–March meteorological year (avoids splitting a natural high-demand winter).
# **Demand**: BDEW + XGBoost baseline model predictions (script 39).
#
# The linear programme minimises total annualised system cost (€/yr) by choosing
# the optimal installed capacity for each technology.  A high-VOLL backup
# generator ensures feasibility in any hour that renewables + storage cannot
# cover demand.
#
# **Technology cost assumptions** — WHOBS 2030 (annualised at 3 % WACC):
# Source: PyPSA/WHOBS run_single_simulation.ipynb (2030 scenario)
#
# | Technology | Overnight capex | Lifetime | Ann. factor | FOM | Total (k€/MW/yr) |
# |---|---|---|---|---|---|
# | Solar PV | 600 €/kW | 25 yr | 5.74 % | 3 % capex | **52** |
# | Wind onshore | 1 182 €/kW | 25 yr | 5.74 % | 3 % capex | **103** |
# | Battery (4 h, η=81 % rt) | 400 €/kW + 200 €/kWh·4h | 25 yr | 5.74 % | 3 % capex | **105** |
# | Hydrogen (168 h, η=48 % rt) | 750 €/kW elec + 800 €/kW CCGT + 11 €/kWh steel tank | mixed | — | 3 % power | **292** |
#
# Battery: inverter 400 €/kW (≈ USD at parity) + energy 200 €/kWh × 4 h cells; η=0.90 one-way.
# Hydrogen: 750 €/kW electrolyser (η=0.80, 20 yr) + 800 €/kW H2-CCGT (η=0.60, 25 yr)
# + 11 €/kWh steel-tank storage × (168 h ÷ 0.60) MWh/MW (30 yr).
# Offshore wind excluded (not modelled in WHOBS single-country runs).

# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pypsa

from woe.paths import ProjPaths


def show():
    """plt.show() wrapper: no-op when matplotlib uses a non-interactive backend."""
    try:
        plt.show()
    except Exception:
        pass


paths = ProjPaths()

# %% [markdown]
# ## Load PECD capacity factors (Germany)

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

print(f"CF solar   : {cf_solar.index[0]} → {cf_solar.index[-1]}  ({len(cf_solar):,} rows)")
print(f"CF onshore : {cf_onshore.index[0]} → {cf_onshore.index[-1]}")

# %% [markdown]
# ## Load demand predictions (baseline model, no weather features)

# %%
demand_preds = pd.read_parquet(paths.de_demand_predictions_file)
demand_pred  = demand_preds["demand_baseline_mw"].rename("demand_mw")
print(f"Demand      : {demand_pred.index[0]} → {demand_pred.index[-1]}  ({len(demand_pred):,} rows)")

# %% [markdown]
# ## Select most recent complete April–March meteorological year

# %%
shared_idx = cf_solar.index.intersection(cf_onshore.index).intersection(demand_pred.index)

# Walk backwards through available April years; take the first with ≥ 8 000 hours
opt_year: int | None = None
for start_yr in sorted(shared_idx.year.unique(), reverse=True):
    start_dt  = pd.Timestamp(f"{start_yr}-04-01")
    end_dt    = pd.Timestamp(f"{start_yr + 1}-04-01")
    period_idx = shared_idx[(shared_idx >= start_dt) & (shared_idx < end_dt)]
    if len(period_idx) >= 8_000:
        opt_year  = start_yr
        snapshots = period_idx
        break

assert opt_year is not None, "No complete April–March year found in the shared data index"

start_dt = pd.Timestamp(f"{opt_year}-04-01")
end_dt   = pd.Timestamp(f"{opt_year + 1}-04-01")
print(f"Optimisation period : {start_dt.date()} → {(end_dt - pd.Timedelta(hours=1)).date()}")
print(f"Snapshots           : {len(snapshots):,} hours")

# Align all series to the chosen snapshots
cf_s   = cf_solar.reindex(snapshots).clip(0, 1).fillna(0)
cf_on  = cf_onshore.reindex(snapshots).clip(0, 1).fillna(0)
demand = demand_pred.reindex(snapshots).interpolate(limit=2)

# Drop any residual NaN rows and re-align
valid     = demand.notna()
snapshots = snapshots[valid]
cf_s      = cf_s[valid]
cf_on     = cf_on[valid]
demand    = demand[valid]

print(f"Demand range: {demand.min():,.0f} – {demand.max():,.0f} MW  (mean {demand.mean():,.0f} MW)")

# %% [markdown]
# ## Technology cost assumptions

# %%
WACC = 0.03       # WHOBS 2030 discount rate
FOM_RATE = 0.03   # fixed O&M: 3 % of overnight capex/yr (all technologies)


def annuity(lifetime: int) -> float:
    """Annuity factor for the given integer lifetime and project WACC."""
    return WACC * (1 + WACC) ** lifetime / ((1 + WACC) ** lifetime - 1)


# Annualised capital costs in €/MW/yr  (WHOBS 2030)
# Solar PV: 600 €/kW, 25 yr
solar_ann_cost   = 600e3   * (annuity(25) + FOM_RATE)
# Wind onshore: 1 182 €/kW, 25 yr
onshore_ann_cost = 1_182e3 * (annuity(25) + FOM_RATE)
# Battery: 4 h duration
#   inverter 400 €/kW (USD≈EUR) + energy cells 200 €/kWh × 4 h; one-way η = 0.90
BAT_MAX_HOURS = 4
BAT_EFF       = 0.90   # one-way (WHOBS: 0.9)
_bat_capex    = 400e3 + 200e3 * BAT_MAX_HOURS   # €/MW total
bat_ann_cost  = _bat_capex * (annuity(25) + FOM_RATE)

# Hydrogen: 168 h (1 week) duration
#   750 €/kW electrolyser (η=0.80, 20 yr) + 800 €/kW H2-CCGT (η=0.60, 25 yr)
#   + 11 €/kWh steel-tank × (168 h ÷ 0.60) MWh/MW (30 yr, no FOM)
H2_MAX_HOURS    = 168
H2_EFF_STORE    = 0.80   # electrolysis (WHOBS 2030)
H2_EFF_DISPATCH = 0.60   # H2-CCGT
_h2_tank_kwh_per_mw = H2_MAX_HOURS / H2_EFF_DISPATCH   # kWh H2 per kW output
h2_ann_cost = (
    750e3 * (annuity(20) + FOM_RATE)            # electrolyser
    + 800e3 * (annuity(25) + FOM_RATE)          # H2-CCGT
    + 11e3 * _h2_tank_kwh_per_mw * annuity(30)  # steel-tank storage
)

# Backup / value of lost load
BACKUP_VOLL = 300_000.  # €/MWh — very high cost to deter use; not extendable

print("Annualised technology costs (€/MW/yr):")
for label, cost in [
    ("Solar PV",         solar_ann_cost),
    ("Wind onshore",     onshore_ann_cost),
    (f"Battery  ({BAT_MAX_HOURS}h, η={BAT_EFF**2:.0%} rt)",  bat_ann_cost),
    (f"Hydrogen ({H2_MAX_HOURS}h, η={H2_EFF_STORE * H2_EFF_DISPATCH:.0%} rt)", h2_ann_cost),
]:
    print(f"  {label:<35}: {cost:>10,.0f}")

# %% [markdown]
# ## Build PyPSA network

# %%
n = pypsa.Network()
n.set_snapshots(snapshots)

n.add("Bus", "DE")

# Inelastic hourly load
n.add("Load", "load", bus="DE", p_set=demand)

# --- Generators (extendable capacity) ---
n.add("Generator", "solar", bus="DE",
      p_nom_extendable=True,
      p_max_pu=cf_s,
      capital_cost=solar_ann_cost,
      marginal_cost=0.)

n.add("Generator", "wind_onshore", bus="DE",
      p_nom_extendable=True,
      p_max_pu=cf_on,
      capital_cost=onshore_ann_cost,
      marginal_cost=0.)

# --- Storage units (extendable) ---
n.add("StorageUnit", "battery", bus="DE",
      p_nom_extendable=True,
      max_hours=BAT_MAX_HOURS,
      capital_cost=bat_ann_cost,
      efficiency_store=BAT_EFF,
      efficiency_dispatch=BAT_EFF,
      cyclic_state_of_charge=True,
      marginal_cost=0.)

n.add("StorageUnit", "hydrogen", bus="DE",
      p_nom_extendable=True,
      max_hours=H2_MAX_HOURS,
      capital_cost=h2_ann_cost,
      efficiency_store=H2_EFF_STORE,
      efficiency_dispatch=H2_EFF_DISPATCH,
      cyclic_state_of_charge=True,
      marginal_cost=0.)

# --- High-cost backup generator (ensures feasibility; not extendable) ---
n.add("Generator", "backup", bus="DE",
      p_nom=demand.max() * 1.05,
      p_nom_extendable=False,
      marginal_cost=BACKUP_VOLL,
      capital_cost=0.)

# %% [markdown]
# ## Solve

# %%
print("Solving with HiGHS …")
status, condition = n.optimize(solver_name="highs")
print(f"Status: {status}  |  Condition: {condition}")

# %% [markdown]
# ## Results overview

# %%
GEN_TECH  = [g for g in n.generators.index if g != "backup"]
STO_TECH  = list(n.storage_units.index)

# Optimal capacities
cap_mw = pd.Series({
    **{g: n.generators.loc[g, "p_nom_opt"] for g in GEN_TECH},
    **{s: n.storage_units.loc[s, "p_nom_opt"] for s in STO_TECH},
})
cap_energy_gwh = pd.Series({
    s: n.storage_units.loc[s, "p_nom_opt"] * n.storage_units.loc[s, "max_hours"] / 1e3
    for s in STO_TECH
})

# Annualised costs
ann_cost_by_tech = pd.Series({
    **{g: n.generators.loc[g, "p_nom_opt"] * n.generators.loc[g, "capital_cost"]
       for g in GEN_TECH},
    **{s: n.storage_units.loc[s, "p_nom_opt"] * n.storage_units.loc[s, "capital_cost"]
       for s in STO_TECH},
})

# Energy delivered per technology
gen_mwh  = n.generators_t.p[GEN_TECH].sum()
sto_dispatch_mwh = n.storage_units_t.p[STO_TECH].clip(lower=0).sum()
backup_mwh = n.generators_t.p["backup"].sum()
total_demand_mwh = demand.sum()

# Curtailment: available RE minus actual dispatch
avail_re = cf_s * cap_mw.get("solar", 0) + cf_on * cap_mw.get("wind_onshore", 0)
curtail_mwh = (avail_re - n.generators_t.p[GEN_TECH].sum(axis=1)).clip(lower=0).sum()

# System LCOE
system_lcoe = n.objective / total_demand_mwh

print(f"\nTotal system cost : {n.objective / 1e9:.3f} B€/yr")
print(f"System LCOE       : {system_lcoe:.1f} €/MWh")
print(f"Total demand      : {total_demand_mwh / 1e6:.2f} TWh")
print(f"Backup energy     : {backup_mwh / 1e3:.1f} GWh  ({backup_mwh / total_demand_mwh * 100:.3f}% of demand)")
print(f"Curtailment       : {curtail_mwh / 1e6:.2f} TWh  ({curtail_mwh / total_demand_mwh * 100:.1f}% of demand)")

print("\nOptimal capacities:")
for tech, mw in cap_mw.items():
    line = f"  {tech:<20}: {mw / 1e3:>8.2f} GW"
    if tech in cap_energy_gwh.index:
        line += f"   ({cap_energy_gwh[tech]:>7.1f} GWh energy)"
    print(line)

print("\nAnnualised costs by technology:")
for tech, cost in ann_cost_by_tech.items():
    share = cost / n.objective * 100 if n.objective > 0 else 0
    print(f"  {tech:<20}: {cost / 1e6:>7.1f} M€/yr  ({share:.1f}%)")

# %% [markdown]
# ## Chart 1 — Optimal installed capacities

# %%
TECH_COLORS = {
    "solar":        "#f1c40f",
    "wind_onshore": "#27ae60",
    "battery":      "#3498db",
    "hydrogen":     "#9b59b6",
}
TECH_LABELS = {
    "solar":        "Solar PV",
    "wind_onshore": "Wind onshore",
    "battery":      f"Battery ({BAT_MAX_HOURS}h)",
    "hydrogen":     f"Hydrogen ({H2_MAX_HOURS}h)",
}

techs_ordered = [t for t in TECH_LABELS if t in cap_mw.index]
cap_gw = cap_mw[techs_ordered] / 1e3
colors  = [TECH_COLORS[t] for t in techs_ordered]
labels  = [TECH_LABELS[t] for t in techs_ordered]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(labels, cap_gw.values, color=colors, edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, cap_gw.values):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} GW", va="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Installed capacity (GW)")
ax.set_title(
    f"Optimal copper-plate energy mix — Germany  ({opt_year}-04 → {opt_year + 1}-03)\n"
    f"System LCOE: {system_lcoe:.0f} €/MWh | Total cost: {n.objective / 1e9:.2f} B€/yr",
    fontsize=10,
)
ax.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
ax.set_xlim(0, cap_gw.max() * 1.22)
fig.tight_layout()
fig.savefig(paths.images_path / "42_optimal_capacities.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/42_optimal_capacities.png
# :name: fig-42-optimal-capacities
# Optimal installed capacities (GW) for the copper-plate system cost minimisation.
# The mix balances the low capital cost of solar against its seasonal intermittency,
# the higher but steadier output of wind, and the cost of storing excess generation
# to cover periods of low renewable output.
# ```

# %% [markdown]
# ## Chart 2 — Annual cost breakdown

# %%
# WHOBS colour scheme (matches PyPSA/WHOBS run_single_simulation.ipynb)
WHOBS_COLORS = {
    "onshore wind":          "#1f77b4",  # tab:blue
    "utility solar PV":      "#bcbd22",  # tab:olive
    "battery storage":       "#7f7f7f",  # tab:gray
    "battery inverter":      "#212121",  # near-black
    "hydrogen storage":      "#e377c2",  # tab:pink
    "hydrogen electrolysis": "#17becf",  # tab:cyan
    "hydrogen turbine":      "#d62728",  # tab:red
}

# Split annualised costs into WHOBS sub-components; normalise to €/MWh of demand
bat_p_nom = n.storage_units.loc["battery", "p_nom_opt"]
h2_p_nom  = n.storage_units.loc["hydrogen", "p_nom_opt"]

lcoe_components = {
    # order matches WHOBS stacking (bottom → top)
    "onshore wind":          onshore_ann_cost * cap_mw.get("wind_onshore", 0),
    "utility solar PV":      solar_ann_cost   * cap_mw.get("solar", 0),
    "battery inverter":      400e3 * (annuity(25) + FOM_RATE) * bat_p_nom,
    "battery storage":       200e3 * BAT_MAX_HOURS * (annuity(25) + FOM_RATE) * bat_p_nom,
    "hydrogen electrolysis": 750e3 * (annuity(20) + FOM_RATE) * h2_p_nom,
    "hydrogen storage":      11e3 * _h2_tank_kwh_per_mw * annuity(30) * h2_p_nom,
    "hydrogen turbine":      800e3 * (annuity(25) + FOM_RATE) * h2_p_nom,
}
lcoe_comp = {k: v / total_demand_mwh for k, v in lcoe_components.items()}

fig, ax = plt.subplots(figsize=(5, 6))
bottom = 0.
for label, val in lcoe_comp.items():
    ax.bar("Germany", val, bottom=bottom, color=WHOBS_COLORS[label], label=label, width=0.5)
    bottom += val

ax.set_ylabel("Average system cost [EUR/MWh]")
ax.set_title(
    f"github.com/PyPSA/WHOBS cost breakdown\nGermany  ({opt_year}-04 → {opt_year + 1}-03)",
    fontsize=10,
)
ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(paths.images_path / "42_annual_costs.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/42_annual_costs.png
# :name: fig-42-annual-costs
# WHOBS-style stacked bar showing the LCOE contribution (€/MWh) of each
# technology sub-component for Germany, using the same colour scheme as the
# PyPSA/WHOBS run_single_simulation.ipynb notebook.
# ```

# %% [markdown]
# ## Chart 3 — Generation mix: sample summer and winter week

# %%
# Assemble hourly dispatch DataFrame (all technologies)
dispatch = n.generators_t.p[GEN_TECH].copy()
dispatch.columns.name = None
for s in STO_TECH:
    p_s = n.storage_units_t.p[s]
    dispatch[s + "_dispatch"] = p_s.clip(lower=0)
    dispatch[s + "_charge"]   = p_s.clip(upper=0)

# Choose representative weeks
summer_start = pd.Timestamp(f"{opt_year}-07-10")
winter_start = pd.Timestamp(f"{opt_year + 1}-01-13")

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)

for ax, week_start, season in zip(axes, [summer_start, winter_start], ["Summer", "Winter"]):
    week_end = week_start + pd.Timedelta(days=7)
    mask = (snapshots >= week_start) & (snapshots < week_end)
    if mask.sum() < 24:
        ax.set_title(f"{season} week not available in data")
        continue

    t  = snapshots[mask]
    dm = demand[mask] / 1e3  # GW

    # Positive supply stack
    stack_cols = [g for g in GEN_TECH if g in dispatch.columns]
    stack_cols += [s + "_dispatch" for s in STO_TECH]
    stack_data = {c: dispatch.loc[t, c].values / 1e3 for c in stack_cols if c in dispatch.columns}

    bottom = np.zeros(mask.sum())
    for col, vals in stack_data.items():
        if vals.max() < 0.01:
            continue
        tech   = col.replace("_dispatch", "")
        color  = TECH_COLORS.get(tech, "#aaaaaa")
        label  = TECH_LABELS.get(tech, tech)
        ax.fill_between(t, bottom, bottom + vals, alpha=0.85,
                        color=color, label=label, linewidth=0)
        bottom += vals

    # Charging shown as negative area (absorbs supply)
    bottom_neg = np.zeros(mask.sum())
    for s in STO_TECH:
        col   = s + "_charge"
        vals  = dispatch.loc[t, col].values / 1e3  # negative
        color = TECH_COLORS.get(s, "#aaaaaa")
        ax.fill_between(t, bottom_neg + vals, bottom_neg, alpha=0.5,
                        color=color, linewidth=0, hatch="///")
        bottom_neg += vals

    ax.plot(t, dm.values, color="black", linewidth=1.5, label="Demand", zorder=5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("Power (GW)")
    ax.set_title(f"{season} week ({week_start.date()} – {(week_end - pd.Timedelta(days=1)).date()})",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)

fig.suptitle(
    f"Hourly generation mix — copper-plate optimum, Germany {opt_year}–{opt_year + 1}",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(paths.images_path / "42_generation_week.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/42_generation_week.png
# :name: fig-42-generation-week
# Hourly generation mix for a representative summer week (top) and winter week
# (bottom). Stacked coloured areas show actual dispatch by technology (positive =
# supplying the grid). Hatched areas below zero represent storage charging
# (absorbing surplus generation). The black line is the demand profile.
# In summer, solar dominates daytime hours and battery storage shifts the surplus
# to evening. In winter, wind carries a larger share and hydrogen covers
# multi-day low-wind periods.
# ```

# %% [markdown]
# ## Chart 4 — Monthly average generation mix

# %%
MONTH_LABELS = ["Apr", "May", "Jun", "Jul", "Aug", "Sep",
                "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
# Map calendar months to the meteorological year order (Apr=1 … Mar=12)
met_month_order = [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3]

avg_by_month: dict[str, list[float]] = {}
for col in [g for g in GEN_TECH] + [s + "_dispatch" for s in STO_TECH]:
    if col not in dispatch.columns:
        continue
    monthly = dispatch[col].groupby(dispatch.index.month).mean() / 1e3
    avg_by_month[col] = [monthly.get(m, 0.0) for m in met_month_order]

demand_monthly = demand.groupby(demand.index.month).mean() / 1e3
demand_met = [demand_monthly.get(m, 0.0) for m in met_month_order]

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(12)
bottom = np.zeros(12)

for col, vals in avg_by_month.items():
    tech  = col.replace("_dispatch", "")
    color = TECH_COLORS.get(tech, "#aaaaaa")
    label = TECH_LABELS.get(tech, tech)
    vals_arr = np.array(vals)
    if vals_arr.max() < 0.01:
        continue
    ax.bar(x, vals_arr, bottom=bottom, color=color, label=label,
           width=0.7, edgecolor="white", linewidth=0.5)
    bottom += vals_arr

ax.plot(x, demand_met, color="black", linewidth=2, marker="o",
        markersize=5, label="Avg demand", zorder=5)

ax.set_xticks(x)
ax.set_xticklabels(MONTH_LABELS)
ax.set_ylabel("Average power (GW)")
ax.set_title(
    f"Monthly average generation mix — copper-plate optimum, Germany {opt_year}–{opt_year + 1}",
    fontsize=11,
)
ax.legend(fontsize=9, loc="upper right")
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(paths.images_path / "42_monthly_mix.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/42_monthly_mix.png
# :name: fig-42-monthly-mix
# Monthly average generation mix (GW) across the optimisation year. Stacked bars
# show average dispatch by technology; the black line is average demand. The gap
# between the stack top and the demand line (when negative) reflects storage
# charging or curtailment; when the stack exceeds demand, the difference is
# absorbed by storage.
# ```

# %% [markdown]
# ## Chart 5 — Storage state of charge (daily envelope)

# %%
soc = n.storage_units_t.state_of_charge.copy()

fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

for ax, s, color in zip(axes, STO_TECH, ["#3498db", "#9b59b6"]):
    cap_gwh = cap_mw[s] * n.storage_units.loc[s, "max_hours"] / 1e3
    if cap_gwh < 0.001:
        ax.set_title(f"{TECH_LABELS.get(s, s)}: not installed")
        continue

    soc_pct = soc[s] / (cap_gwh * 1e3) * 100  # SoC as % of energy capacity

    daily_min  = soc_pct.resample("D").min()
    daily_max  = soc_pct.resample("D").max()
    daily_mean = soc_pct.resample("D").mean()

    ax.fill_between(daily_min.index, daily_min.values, daily_max.values,
                    alpha=0.25, color=color, label="Daily min–max")
    ax.plot(daily_mean.index, daily_mean.values, color=color, linewidth=1.2, label="Daily mean")

    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylabel("State of charge (%)")
    ax.set_title(
        f"{TECH_LABELS.get(s, s)} — state of charge  "
        f"(capacity: {cap_gwh:.1f} GWh  |  power: {cap_mw[s] / 1e3:.2f} GW)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)

fig.suptitle(
    f"Storage state of charge — copper-plate optimum, Germany {opt_year}–{opt_year + 1}",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(paths.images_path / "42_storage_soc.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/42_storage_soc.png
# :name: fig-42-storage-soc
# Daily state of charge (%) for battery (top) and hydrogen (bottom) storage.
# The shaded band shows the daily min–max range; the solid line is the daily
# mean. Battery storage cycles rapidly (daily solar pattern), while hydrogen
# shows a slower seasonal pattern: charging in summer when solar surplus is
# large and discharging in winter to cover the renewable deficit.
# ```

# %% [markdown]
# ## Summary

# %%
print("=" * 72)
print(f"COPPER-PLATE OPTIMAL MIX — Germany  {opt_year}-04 → {opt_year + 1}-03")
print("=" * 72)
print(f"\n{'Technology':<22} {'Capacity':>10} {'Energy cap':>12} {'Ann. cost':>12} {'Share':>7}")
print("-" * 72)
for t in techs_ordered:
    mw   = cap_mw[t]
    cost = ann_cost_by_tech[t]
    share = cost / n.objective * 100 if n.objective > 0 else 0
    ecap  = f"{cap_energy_gwh[t]:.0f} GWh" if t in cap_energy_gwh.index else ""
    print(f"  {TECH_LABELS.get(t, t):<20} {mw/1e3:>8.2f} GW  {ecap:>10}  "
          f"{cost/1e6:>9.1f} M€  {share:>5.1f}%")
print("-" * 72)
print(f"  {'Total':<20} {'':>10}  {'':>10}  {n.objective/1e6:>9.1f} M€  100.0%")
print(f"\nSystem LCOE       : {system_lcoe:.1f} €/MWh")
print(f"Backup used       : {backup_mwh/1e3:.2f} GWh  ({backup_mwh/total_demand_mwh*100:.4f}% of demand)")
print(f"Curtailment       : {curtail_mwh/1e6:.2f} TWh  ({curtail_mwh/total_demand_mwh*100:.1f}%)")
print("\nGeneration mix (MWh delivered):")
for g in GEN_TECH:
    mwh = gen_mwh[g]
    print(f"  {TECH_LABELS.get(g, g):<22}: {mwh/1e6:>7.2f} TWh  ({mwh/total_demand_mwh*100:>5.1f}% of demand)")
print(f"  {'Storage dispatch':<22}: {sto_dispatch_mwh.sum()/1e6:>7.2f} TWh")

# %%

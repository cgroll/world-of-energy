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
# # Germany Energy Mix Optimization with Battery Storage
#
# Optimises the least-cost mix of solar PV, onshore wind, offshore wind,
# lithium-ion battery storage and gas backup to meet a constant 1 MW load,
# using PECD capacity factors scaled to Fraunhofer ISE 2024 full-load hours.
#
# **Key design choices:**
#
# - Gas CAPEX is excluded — the optimizer is free to use gas at marginal cost
#   only, reflecting a world where existing gas capacity is available as backup.
# - A global constraint caps total gas output to at most `GAS_SHARE_LIMIT`
#   of annual demand (default 10 %).
# - All RE and battery capacities are greenfield (extendable from zero).
# - Battery duration is fixed at `BAT_DURATION_H` hours; the optimizer
#   chooses the power (MW) capacity.
#
# **Solver:** [HiGHS](https://highs.dev/) via `pypsa.optimize`.
#
# ## Modules
#
# 1. **Cost and technology assumptions** — CAPEX, OPEX, efficiencies
# 2. **Data loading** — PECD capacity factors + Fraunhofer scaling
# 3. **PyPSA network** — buses, load, generators, storage unit
# 4. **Optimisation** — LP solve with gas-share constraint
# 5. **Results extraction** — capacities, dispatch, costs, LCOE
# 6. **Visualisations** — capacity bar, dispatch week, SOC, costs, LCOE

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa

from woe.paths import ProjPaths

paths = ProjPaths()


def show() -> None:
    """plt.show() wrapper: no-op when matplotlib uses a non-interactive backend."""
    try:
        plt.show()
    except Exception:
        pass


HOURS_PER_YEAR = 8760
COUNTRY = "DE"
SIM_YEARS = ["2024", "2025"]
N_YEARS = len(SIM_YEARS)

# Fraction of annual demand that may be covered by gas
GAS_SHARE_LIMIT = 0.10

# Battery duration (hours) — optimizer chooses MW capacity, duration is fixed
BAT_DURATION_H = 4

# %% [markdown]
# ## Cost and technology assumptions
#
# ### Renewable energy
#
# From Fraunhofer ISE LCOE 2024, Tables 1–2.  CAPEX midpoints are used.
#
# ### Gas (CCGT)
#
# | Parameter                | Value            | Source               |
# |--------------------------|----------------:|:---------------------|
# | Lifetime / CAPEX         | excluded         | open-cycle dispatch  |
# | Variable OPEX (non-fuel) | 0.5 ct/kWh       | Table 2              |
# | Electrical efficiency    | 60 %             | Table 2              |
# | Natural gas price (2025) | 36 €/MWh\_th     | interpolated         |
# | CO₂ price (2025)         | 90 €/t           | mid of 79–100        |
# | CO₂ intensity            | 0.202 t/MWh\_th  | stoichiometric       |
#
# ### Battery storage (Li-ion, utility-scale)
#
# | Parameter                | Value            | Source               |
# |--------------------------|----------------:|:---------------------|
# | CAPEX                    | 500 €/kWh       | mid of 400–600       |
# | Annualised fixed costs   | 6.8 % of CAPEX  | 3.5 % capital + 2.0 % O&M + 1.3 % degradation |
# | Round-trip efficiency    | 90 %             | Table 2              |
# | Duration                 | 4 h              | fixed assumption     |

# %%
RE_COSTS = {
    "solar_pv_utility": {
        "label": "Solar PV",
        "capex_mid": 800,          # EUR/kW (midpoint of 700–900)
        "lifetime": 30,            # years
        "wacc_real": 0.035,        # 3.5 %
        "opex_fix": 13.3,         # EUR/kW/yr
        "opex_var": 0.0,          # EUR/kWh
    },
    "wind_onshore": {
        "label": "Wind onshore",
        "capex_mid": 1600,         # midpoint of 1300–1900
        "lifetime": 25,
        "wacc_real": 0.039,        # 3.9 %
        "opex_fix": 32.0,
        "opex_var": 0.007,
    },
    "wind_offshore": {
        "label": "Wind offshore",
        "capex_mid": 2800,         # midpoint of 2200–3400
        "lifetime": 25,
        "wacc_real": 0.060,        # 6.0 %
        "opex_fix": 39.0,
        "opex_var": 0.008,
    },
}

GAS = {
    "label": "Gas (CCGT)",
    "opex_var": 0.005,            # EUR/kWh (non-fuel variable O&M)
    "efficiency": 0.60,           # electrical efficiency
    "gas_price": 36.0,            # EUR/MWh_th (2025 interpolated)
    "co2_price": 90.0,            # EUR/t CO₂ (midpoint of 79–100)
    "co2_intensity": 0.202,       # t CO₂ / MWh_th (stoichiometric)
}

BAT = {
    "label": "Battery (Li-ion)",
    "capex_kwh": 500,              # EUR/kWh usable capacity (mid of 400–600)
    "annuity_rate": 0.068,         # 6.8 % of CAPEX per year
    "rt_efficiency": 0.90,
}

PECD_VARIABLES = {
    "solar_pv_utility": "solar_photovoltaic_power_generation",
    "wind_onshore":     "wind_power_generation_onshore",
    "wind_offshore":    "wind_power_generation_offshore",
}

TECH_COLORS = {
    "solar_pv_utility": "#f4b942",
    "wind_onshore":     "#4a90d9",
    "wind_offshore":    "#1a5fa8",
    "Battery":          "#7fbc41",
    "Gas":              "#8B4513",
    "curtailment":      "#cc4444",
}

DEMAND_MW = 1.0

# Derived battery efficiencies
EFF_IN = np.sqrt(BAT["rt_efficiency"])
EFF_OUT = np.sqrt(BAT["rt_efficiency"])

# Derived gas marginal cost (EUR/kWh_el → EUR/MWh for PyPSA)
fuel_per_kwh = GAS["gas_price"] / 1000 / GAS["efficiency"]
co2_per_kwh = GAS["co2_intensity"] * GAS["co2_price"] / 1000 / GAS["efficiency"]
GAS_MARGINAL_EUR_MWH = (fuel_per_kwh + co2_per_kwh + GAS["opex_var"]) * 1000

print(f"Gas marginal cost: {GAS_MARGINAL_EUR_MWH:.2f} EUR/MWh")
print(f"Battery η_in={EFF_IN:.4f}, η_out={EFF_OUT:.4f}")

# %% [markdown]
# ## Load PECD data and apply Fraunhofer scaling factors

# %%
scaling_df = pd.read_csv(paths.processed_data_path / "pecd_fraunhofer_scaling_factors.csv")
scaling = dict(zip(scaling_df["tech_key"], scaling_df["scaling_factor"]))

print("PECD → Fraunhofer scaling factors:")
for tech, factor in scaling.items():
    print(f"  {tech:20s}  {factor:.4f}")

# %%
pecd = pd.read_parquet(paths.pecd_processed_file)
pecd_sim = pd.concat([pecd.loc[y] for y in SIM_YEARS])
total_hours = len(pecd_sim)
N_YEARS_actual = total_hours / HOURS_PER_YEAR
print(f"Simulation period: {SIM_YEARS[0]}–{SIM_YEARS[-1]}  ({total_hours:,} hours, {N_YEARS_actual:.1f} years)")

hourly_cf = pd.DataFrame(index=pecd_sim.index)
for tech, variable in PECD_VARIABLES.items():
    raw_cf = pecd_sim[variable]["capacity_factor_ratio"][COUNTRY].values.astype(float)
    hourly_cf[tech] = np.clip(np.nan_to_num(raw_cf) * scaling[tech], 0, 1)

print(f"Hourly CF shape: {hourly_cf.shape}")
print(hourly_cf.describe().round(4))


# %% [markdown]
# ---
# ## Helper: capital cost conversion for PyPSA
#
# PyPSA's objective adds `capital_cost × p_nom` (once) plus
# `sum_t(marginal_cost × p_t × weight_t)` over all snapshots.
#
# With a multi-year time series we set the objective snapshot weight to
# `1 / N_YEARS` so that the marginal-cost sum reflects **one year** of
# operation, matching the annualised capital costs.

# %%
def capital_recovery_factor(wacc: float, lifetime: int) -> float:
    if wacc == 0:
        return 1.0 / lifetime
    return wacc * (1 + wacc) ** lifetime / ((1 + wacc) ** lifetime - 1)


def re_capital_cost_pypsa(p: dict) -> float:
    """Annualised capital + fixed OPEX per MW (EUR/MW/yr), for PyPSA capital_cost."""
    crf = capital_recovery_factor(p["wacc_real"], p["lifetime"])
    return (p["capex_mid"] * crf + p["opex_fix"]) * 1000  # kW → MW


re_techs = list(RE_COSTS.keys())

print("Annualised RE capital costs (EUR/MW/yr):")
for tech, p in RE_COSTS.items():
    print(f"  {p['label']:20s}  {re_capital_cost_pypsa(p):>10,.0f}")

bat_capital_cost_pypsa = BAT["annuity_rate"] * BAT["capex_kwh"] * BAT_DURATION_H * 1000  # EUR/MW-power/yr
print(f"\n  {'Battery (4h)':20s}  {bat_capital_cost_pypsa:>10,.0f}  (EUR/MW-power/yr)")
print(f"  {'Gas (no CAPEX)':20s}  {'0':>10s}")


# %% [markdown]
# ---
# ## Build PyPSA network

# %%
n = pypsa.Network()
n.set_snapshots(hourly_cf.index)

# Scale objective snapshot weights so marginal-cost sums represent 1 year
n.snapshot_weightings.loc[:, "objective"] = 1.0 / N_YEARS_actual

n.add("Bus", "Germany")
n.add("Load", "Total Demand", bus="Germany", p_set=DEMAND_MW)

# RE generators (extendable, no p_nom constraint)
for tech in re_techs:
    p = RE_COSTS[tech]
    n.add(
        "Generator", tech,
        bus="Germany",
        carrier=tech,
        p_max_pu=hourly_cf[tech],
        capital_cost=re_capital_cost_pypsa(p),
        marginal_cost=p["opex_var"] * 1000,   # EUR/MWh
        p_nom_extendable=True,
    )

# Battery storage unit (extendable power, fixed duration)
n.add(
    "StorageUnit", "Battery",
    bus="Germany",
    carrier="battery",
    capital_cost=bat_capital_cost_pypsa,
    marginal_cost=0.0,
    efficiency_store=EFF_IN,
    efficiency_dispatch=EFF_OUT,
    max_hours=BAT_DURATION_H,
    cyclic_state_of_charge=True,
    p_nom_extendable=True,
)

# Gas generator — zero CAPEX, only marginal cost, carrier tracked for constraint
n.add("Carrier", "gas")
n.add(
    "Generator", "Gas",
    bus="Germany",
    carrier="gas",
    capital_cost=0.0,
    marginal_cost=GAS_MARGINAL_EUR_MWH,
    p_nom_extendable=True,
)

print("Network components:")
print(f"  Generators:    {list(n.generators.index)}")
print(f"  StorageUnits:  {list(n.storage_units.index)}")
print(f"  Snapshots:     {len(n.snapshots):,}")
print(f"  Objective weights (total): {n.snapshot_weightings['objective'].sum():.2f} (target: 1.0)")


# %% [markdown]
# ---
# ## Optimise with gas-share constraint
#
# The global constraint limits total electrical output from gas to
# `GAS_SHARE_LIMIT × annual_demand`.  The constraint is expressed in
# MWh over the full simulation horizon (unweighted sum of hourly values).

# %%
gas_limit_total_mwh = DEMAND_MW * total_hours * GAS_SHARE_LIMIT

print(f"Gas share limit: {GAS_SHARE_LIMIT:.0%} of demand")
print(f"  Annual demand:       {DEMAND_MW * HOURS_PER_YEAR:,.0f} MWh")
print(f"  Annual gas limit:    {DEMAND_MW * HOURS_PER_YEAR * GAS_SHARE_LIMIT:,.0f} MWh")
print(f"  Total gas limit ({N_YEARS_actual:.0f}yr): {gas_limit_total_mwh:,.0f} MWh")


def gas_limit_constraint(n, snapshots):
    """Add constraint: sum of hourly gas generation <= gas_limit_total_mwh."""
    lhs = n.model["Generator-p"].sel(name=["Gas"]).sum()
    n.model.add_constraints(lhs <= gas_limit_total_mwh, name="GasShareLimit")


opt_status, opt_cond = n.optimize(solver_name="highs", extra_functionality=gas_limit_constraint)

print(f"\nOptimisation status: {opt_status} / {opt_cond}")
print(f"Objective value: {n.objective:,.0f} EUR/yr (annualised, snapshot weights = 1/N_YEARS)")


# %% [markdown]
# ---
# ## Extract results

# %%
# Optimal capacities
opt_caps = {}
for tech in re_techs:
    opt_caps[tech] = n.generators.at[tech, "p_nom_opt"]
opt_caps["Battery"] = n.storage_units.at["Battery", "p_nom_opt"]
opt_caps["Gas"] = n.generators.at["Gas", "p_nom_opt"]

print("Optimal capacities:")
for name, cap in opt_caps.items():
    label = RE_COSTS[name]["label"] if name in RE_COSTS else name
    print(f"  {label:20s}  {cap:.4f} MW")

# %%
# Annual dispatch (MWh/yr)
gen_t = n.generators_t.p         # DataFrame: snapshots × generators
bat_t = n.storage_units_t.p      # positive = dispatch, negative = charge
soc_t = n.storage_units_t.state_of_charge

bat_dispatch = bat_t["Battery"].clip(lower=0)
bat_charge = (-bat_t["Battery"]).clip(lower=0)

annual = {}
for tech in re_techs:
    annual[tech] = gen_t[tech].sum() / N_YEARS_actual
annual["Battery"] = bat_dispatch.sum() / N_YEARS_actual
annual["Gas"] = gen_t["Gas"].sum() / N_YEARS_actual

total_delivered = DEMAND_MW * HOURS_PER_YEAR
bat_delivered_annual = bat_dispatch.sum() / N_YEARS_actual
gas_annual = gen_t["Gas"].sum() / N_YEARS_actual

# Curtailment = RE available (CF × p_nom_opt) − RE actually dispatched
re_available_total = sum(
    (hourly_cf[t] * opt_caps[t]).sum() / N_YEARS_actual for t in re_techs
)
re_dispatched_total = sum(gen_t[t].sum() / N_YEARS_actual for t in re_techs)
curtailment_annual = max(0.0, re_available_total - re_dispatched_total)

print("\nAnnual energy (MWh/yr):")
for tech in re_techs:
    label = RE_COSTS[tech]["label"]
    cf_mean = float(hourly_cf[tech].mean())
    flh = annual[tech] / opt_caps[tech] if opt_caps[tech] > 1e-6 else 0
    print(f"  {label:20s}  {annual[tech]:>8,.1f} MWh  ({flh:,.0f} FLH)")
print(f"  {'Battery dispatch':20s}  {bat_delivered_annual:>8,.1f} MWh")
print(f"  {'Gas':20s}  {gas_annual:>8,.1f} MWh  ({gas_annual / total_delivered:.1%} of demand)")
print(f"  {'Curtailment':20s}  {curtailment_annual:>8,.1f} MWh")


# %% [markdown]
# ---
# ## Cost breakdown

# %%
cost_rows = []

for tech in re_techs:
    p = RE_COSTS[tech]
    cap_mw = opt_caps[tech]
    cap_cost_yr = re_capital_cost_pypsa(p) * cap_mw  # EUR/yr
    var_cost_yr = p["opex_var"] * annual[tech] * 1000  # EUR/yr
    total_cost_yr = cap_cost_yr + var_cost_yr
    produced = annual[tech]
    lcoe = total_cost_yr / (produced * 1000) if produced > 1e-9 else float("inf")  # EUR/kWh

    cost_rows.append({
        "technology": p["label"],
        "tech_key": tech,
        "installed_mw": cap_mw,
        "annual_mwh": produced,
        "annual_cost_eur": total_cost_yr,
        "lcoe_ct": lcoe * 100,
    })

# Battery
bat_cost_yr = bat_capital_cost_pypsa * opt_caps["Battery"]
bat_lcos = bat_cost_yr / (bat_delivered_annual * 1000) if bat_delivered_annual > 1e-9 else float("inf")
cost_rows.append({
    "technology": BAT["label"],
    "tech_key": "Battery",
    "installed_mw": opt_caps["Battery"],
    "annual_mwh": bat_delivered_annual,
    "annual_cost_eur": bat_cost_yr,
    "lcoe_ct": bat_lcos * 100,
})

# Gas (marginal cost only)
gas_var_cost_yr = GAS_MARGINAL_EUR_MWH * gas_annual  # EUR/yr  (EUR/MWh × MWh)
gas_lcoe = gas_var_cost_yr / (gas_annual * 1000) if gas_annual > 1e-9 else 0
cost_rows.append({
    "technology": GAS["label"],
    "tech_key": "Gas",
    "installed_mw": opt_caps["Gas"],
    "annual_mwh": gas_annual,
    "annual_cost_eur": gas_var_cost_yr,
    "lcoe_ct": gas_lcoe * 100,
})

cost_df = pd.DataFrame(cost_rows)

total_annual_cost = cost_df["annual_cost_eur"].sum()
system_lcoe = total_annual_cost / (total_delivered * 1000)  # EUR/kWh

print("Cost summary:")
pd.set_option("display.float_format", lambda v: f"{v:.2f}")
print(cost_df[["technology", "installed_mw", "annual_mwh", "lcoe_ct", "annual_cost_eur"]].to_string(index=False))
print(f"\nTotal annual cost:  {total_annual_cost:>12,.0f} EUR")
print(f"System LCOE:        {system_lcoe * 100:>12.2f} ct/kWh")

print("\nCost attribution to system LCOE:")
for _, row in cost_df.iterrows():
    attr_ct = row["annual_cost_eur"] / (total_delivered * 1000) * 100
    share = row["annual_mwh"] / total_delivered
    print(f"  {row['technology']:20s}  {share:>6.1%}  {row['lcoe_ct']:>6.2f} ct own  →  {attr_ct:>6.2f} ct system")


# %% [markdown]
# ---
# ## Visualisations

# %% [markdown]
# ### Optimal installed capacities

# %%
fig, ax = plt.subplots(figsize=(8, 5))

labels = []
caps = []
colors_bar = []
for tech in re_techs:
    labels.append(RE_COSTS[tech]["label"])
    caps.append(opt_caps[tech])
    colors_bar.append(TECH_COLORS[tech])
labels += ["Battery", "Gas"]
caps += [opt_caps["Battery"], opt_caps["Gas"]]
colors_bar += [TECH_COLORS["Battery"], TECH_COLORS["Gas"]]

bars = ax.bar(labels, caps, color=colors_bar, edgecolor="white", linewidth=0.8)
ax.bar_label(bars, fmt="{:.3f}", padding=3, fontsize=9)
ax.set_ylabel("Optimal capacity (MW per MW demand)")
ax.set_title(f"Optimal installed capacities\n(gas share ≤ {GAS_SHARE_LIMIT:.0%}, battery {BAT_DURATION_H}h)")
ax.set_ylim(0, max(caps) * 1.15)
ax.tick_params(axis="x", rotation=15)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(paths.images_path / "55_optimal_capacities.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/55_optimal_capacities.png
# :name: fig-55-optimal-capacities
# Optimal installed capacities per MW of constant demand, minimising
# annualised system costs subject to a gas share limit of 10 %.
# ```

# %% [markdown]
# ### Sample dispatch — first week of January

# %%
week_slice = slice(0, 168)
snap_week = n.snapshots[week_slice]
t_idx = range(168)

re_gen_week = {tech: gen_t[tech].iloc[week_slice].values for tech in re_techs}
bat_dis_week = bat_dispatch.iloc[week_slice].values
bat_chg_week = bat_charge.iloc[week_slice].values
gas_week = gen_t["Gas"].iloc[week_slice].values
soc_week = soc_t["Battery"].iloc[week_slice].values

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]

# Stacked area: RE → battery dispatch → gas
bottom = np.zeros(168)
for tech in re_techs:
    ax.bar(t_idx, re_gen_week[tech], bottom=bottom,
           color=TECH_COLORS[tech], label=RE_COSTS[tech]["label"], width=1.0)
    bottom += re_gen_week[tech]
ax.bar(t_idx, bat_dis_week, bottom=bottom,
       color=TECH_COLORS["Battery"], label="Battery dispatch", width=1.0)
bottom += bat_dis_week
ax.bar(t_idx, gas_week, bottom=bottom,
       color=TECH_COLORS["Gas"], label="Gas", width=1.0)

# Battery charging shown as negative
ax.bar(t_idx, -bat_chg_week, color=TECH_COLORS["Battery"], alpha=0.4,
       label="Battery charge", width=1.0)

ax.axhline(DEMAND_MW, color="black", lw=1.5, ls="--", label="Demand")
ax.set_ylabel("Power (MW)")
ax.set_title("Hourly dispatch — first week of January")
ax.legend(loc="upper right", ncol=3, fontsize=8)
ax.grid(alpha=0.3)

# SOC
axes[1].fill_between(t_idx, soc_week, alpha=0.6, color=TECH_COLORS["Battery"])
axes[1].set_ylabel("Battery SOC (MWh)")
axes[1].set_xlabel("Hour of week")
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(paths.images_path / "55_dispatch_week.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/55_dispatch_week.png
# :name: fig-55-dispatch-week
# Stacked hourly dispatch for the first 168 hours of the simulation.
# Negative bars indicate battery charging.  Bottom panel shows battery
# state of charge.
# ```

# %% [markdown]
# ### Battery state of charge — full year (first year)

# %%
first_year_hours = HOURS_PER_YEAR
soc_full = soc_t["Battery"].iloc[:first_year_hours].values
t_full = np.arange(first_year_hours)

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(t_full, soc_full, alpha=0.7, color=TECH_COLORS["Battery"])
ax.axhline(opt_caps["Battery"] * BAT_DURATION_H, color="grey", ls="--", lw=1,
           label=f"Max capacity ({opt_caps['Battery'] * BAT_DURATION_H:.2f} MWh)")
ax.set_xlabel("Hour of year")
ax.set_ylabel("State of charge (MWh)")
ax.set_title("Battery state of charge — first simulated year")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(paths.images_path / "55_soc.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/55_soc.png
# :name: fig-55-soc
# Battery state of charge over the first simulated year.  The dashed line
# marks the maximum usable capacity determined by the optimiser.
# ```

# %% [markdown]
# ### Annual cost breakdown

# %%
fig, ax = plt.subplots(figsize=(8, 5))

bar_colors = []
for _, row in cost_df.iterrows():
    bar_colors.append(TECH_COLORS.get(row["tech_key"], "#aaaaaa"))

bars = ax.bar(cost_df["technology"], cost_df["annual_cost_eur"] / 1e3,
              color=bar_colors, edgecolor="white", linewidth=0.8)
ax.bar_label(bars, fmt="{:.0f}", padding=3, fontsize=9, label_type="edge")
ax.set_ylabel("Annual cost (k EUR/yr)")
ax.set_title(f"Annual cost by technology\nSystem LCOE: {system_lcoe * 100:.2f} ct/kWh")
ax.tick_params(axis="x", rotation=15)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(paths.images_path / "55_annual_costs.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/55_annual_costs.png
# :name: fig-55-annual-costs
# Annualised costs per technology.  Gas costs here are marginal costs only
# (no CAPEX), reflecting the assumption that dispatchable backup capacity
# is available at no additional fixed cost.
# ```

# %% [markdown]
# ### LCOE comparison

# %%
fig, ax = plt.subplots(figsize=(8, 5))

lcoe_colors = [TECH_COLORS.get(row["tech_key"], "#aaaaaa") for _, row in cost_df.iterrows()]
bars = ax.bar(cost_df["technology"], cost_df["lcoe_ct"],
              color=lcoe_colors, edgecolor="white", linewidth=0.8)
ax.bar_label(bars, fmt="{:.1f}", padding=3, fontsize=9)
ax.axhline(system_lcoe * 100, color="black", ls="--", lw=1.5,
           label=f"System LCOE {system_lcoe * 100:.2f} ct/kWh")
ax.set_ylabel("LCOE / LCOS (ct/kWh)")
ax.set_title("Levelised cost per technology")
ax.tick_params(axis="x", rotation=15)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(paths.images_path / "55_lcoe_comparison.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/55_lcoe_comparison.png
# :name: fig-55-lcoe-comparison
# Own LCOE / LCOS for each technology alongside the system LCOE (dashed line).
# ```

# %% [markdown]
# ### Demand fulfilment by source

# %%
# Share of delivered demand from each source
sources = re_techs + ["Battery", "Gas"]
delivered_mwh = [annual[t] for t in re_techs] + [bat_delivered_annual, gas_annual]
# Note: RE production > RE delivered because some goes to battery or is curtailed
# For demand fulfilment we show what reaches the load
re_direct_annual = [
    max(0, annual[t] - gen_t[t][bat_charge > 0].sum() / N_YEARS_actual)
    for t in re_techs
]
# Simpler: demand = sum(re direct) + battery dispatch + gas → use load balance
# Total load = DEMAND_MW * HOURS_PER_YEAR
# Delivered from RE (direct) = total_demand - gas - battery_dispatch
re_direct_total_annual = total_delivered - gas_annual - bat_delivered_annual
re_shares = [annual[t] / sum(annual[t2] for t2 in re_techs) * re_direct_total_annual
             for t in re_techs]

delivered_by_source = re_shares + [bat_delivered_annual, gas_annual]
source_labels = [RE_COSTS[t]["label"] for t in re_techs] + ["Battery", "Gas"]
source_colors = [TECH_COLORS[t] for t in re_techs] + [TECH_COLORS["Battery"], TECH_COLORS["Gas"]]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(source_labels, [v / total_delivered * 100 for v in delivered_by_source],
              color=source_colors, edgecolor="white", linewidth=0.8)
ax.bar_label(bars, fmt="{:.1f}%", padding=3, fontsize=9)
ax.set_ylabel("Share of annual demand (%)")
ax.set_title(f"Demand fulfilment by source\n(gas ≤ {GAS_SHARE_LIMIT:.0%} constraint active)")
ax.set_ylim(0, max(v / total_delivered * 100 for v in delivered_by_source) * 1.15)
ax.tick_params(axis="x", rotation=15)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(paths.images_path / "55_demand_fulfilment.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/55_demand_fulfilment.png
# :name: fig-55-demand-fulfilment
# Fraction of annual demand met by each source.  The gas share is at or
# below the `GAS_SHARE_LIMIT` ceiling.
# ```

# %% [markdown]
# ### RE utilisation rates

# %%
fig, ax = plt.subplots(figsize=(8, 5))

util_labels = [RE_COSTS[t]["label"] for t in re_techs]
produced_arr = [annual[t] for t in re_techs]
max_possible = [opt_caps[t] * float(hourly_cf[t].sum()) / N_YEARS_actual for t in re_techs]
utilisation = [p / m * 100 if m > 1e-9 else 0 for p, m in zip(produced_arr, max_possible)]

bars = ax.bar(util_labels, utilisation,
              color=[TECH_COLORS[t] for t in re_techs],
              edgecolor="white", linewidth=0.8)
ax.bar_label(bars, fmt="{:.1f}%", padding=3, fontsize=9)
ax.set_ylabel("Utilisation of potential generation (%)")
ax.set_title("RE utilisation (curtailment shown implicitly)")
ax.set_ylim(0, 115)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(paths.images_path / "55_re_utilisation.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/55_re_utilisation.png
# :name: fig-55-re-utilisation
# Fraction of each technology's potential generation (capacity × CF hours)
# that is actually utilised.  Values below 100 % reflect curtailment losses.
# ```

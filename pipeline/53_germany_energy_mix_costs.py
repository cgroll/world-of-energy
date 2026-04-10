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
# # Germany Energy Mix Costs: Renewables + Gas Backup
#
# Simulates an hourly dispatch for Germany in 2025 using PECD capacity factors
# (scaled to align with Fraunhofer ISE 2024 midpoint full-load hours) with a
# simple merit order: all available renewable generation is dispatched first;
# any surplus above demand is curtailed pro-rata; the residual shortfall is
# met by gas-fired combined-cycle (CCGT) plants.
#
# Installed renewable capacities are set proportional to Germany's actual
# end-2024 fleet relative to average demand. Demand is normalised to a
# constant 1 MW (= 1 MWh per hour, 8 760 MWh per year).
#
# All cost and financing assumptions follow the Fraunhofer ISE *Levelized
# Cost of Electricity — Renewable Energy Technologies* (July 2024) study.
#
# ## Key outputs
#
# - Energy share per source (solar PV, wind onshore, wind offshore, gas)
# - Curtailed energy
# - Annualised system cost and system LCOE
# - Technology-level LCOE (with and without curtailment effect)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# %% [markdown]
# ## Cost assumptions
#
# ### Renewable energy
#
# From Fraunhofer ISE LCOE 2024, Tables 1–2. CAPEX midpoints are used.
#
# ### Gas (CCGT)
#
# From the same study, with fuel and CO₂ prices interpolated to 2025:
#
# | Parameter                  | Value              | Source                |
# |----------------------------|-------------------:|:----------------------|
# | CAPEX                      | 1 100 €/kW         | mid of 900–1 300      |
# | Lifetime                   | 30 years            | Table 2               |
# | WACC (real)                | 7.5 %               | assumed (carbon risk) |
# | Fixed OPEX                 | 20 €/kW/yr          | Table 2               |
# | Variable OPEX (non-fuel)   | 0.5 ct/kWh          | Table 2               |
# | Electrical efficiency      | 60 %                | Table 2               |
# | Natural gas price (2025)   | 36 €/MWh\_th        | interpolated          |
# | CO₂ price (2025)           | 90 €/t              | mid of 79–100         |
# | CO₂ intensity (nat. gas)   | 0.202 t/MWh\_th     | stoichiometric        |

# %%
RE_COSTS = {
    "solar_pv_utility": {
        "label": "Solar PV",
        "capex_mid": 800,          # EUR/kW (midpoint of 700–900)
        "lifetime": 30,            # years
        "wacc_real": 0.035,        # 3.5 %
        "opex_fix": 13.3,         # EUR/kW/yr
        "opex_var": 0.0,          # EUR/kWh
        "degradation": 0.0025,    # 0.25 %/yr
    },
    "wind_onshore": {
        "label": "Wind onshore",
        "capex_mid": 1600,         # midpoint of 1300–1900
        "lifetime": 25,
        "wacc_real": 0.039,        # 3.9 %
        "opex_fix": 32.0,
        "opex_var": 0.007,
        "degradation": 0.0,
    },
    "wind_offshore": {
        "label": "Wind offshore",
        "capex_mid": 2800,         # midpoint of 2200–3400
        "lifetime": 25,
        "wacc_real": 0.060,        # 6.0 %
        "opex_fix": 39.0,
        "opex_var": 0.008,
        "degradation": 0.0,
    },
}

GAS = {
    "label": "Gas (CCGT)",
    "capex_mid": 1100,             # EUR/kW (midpoint of 900–1300)
    "lifetime": 30,                # years
    "wacc_real": 0.075,            # 7.5 % (higher due to carbon/transition risk)
    "opex_fix": 20.0,             # EUR/kW/yr
    "opex_var": 0.005,            # EUR/kWh (non-fuel variable O&M)
    "efficiency": 0.60,           # electrical efficiency
    "gas_price": 36.0,            # EUR/MWh_th (2025 interpolated)
    "co2_price": 90.0,            # EUR/t CO₂ (midpoint of 79–100)
    "co2_intensity": 0.202,       # t CO₂ / MWh_th (stoichiometric for nat. gas)
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
    "gas":              "#888888",
    "curtailment":      "#cc4444",
}

# %% [markdown]
# ## Installed capacities
#
# Germany end-2024 approximate installed capacities normalised to 1 MW of
# constant demand (~57 GW average load, ~500 TWh/yr):
#
# | Technology    | Installed (DE) | Per 1 MW demand |
# |---------------|---------------:|----------------:|
# | Solar PV      |       96 GW    |        1.68 MW  |
# | Wind onshore  |       62 GW    |        1.09 MW  |
# | Wind offshore |        9 GW    |        0.16 MW  |

# %%
AVG_DEMAND_GW = 57.0
RE_SCALING = 2.0  # multiplier for all installed RE capacities

INSTALLED_CAP = {
    "solar_pv_utility": 96.0 / AVG_DEMAND_GW * RE_SCALING,
    "wind_onshore":     62.0 / AVG_DEMAND_GW * RE_SCALING,
    "wind_offshore":     9.0 / AVG_DEMAND_GW * RE_SCALING,
}
DEMAND_MW = 1.0

print(f"RE scaling factor: {RE_SCALING:.1f}x\n")
for tech, cap in INSTALLED_CAP.items():
    print(f"{RE_COSTS[tech]['label']:20s}  {cap:.3f} MW per MW demand")
print(f"{'Total RE':20s}  {sum(INSTALLED_CAP.values()):.3f} MW per MW demand")


# %% [markdown]
# ## Load PECD data and apply Fraunhofer scaling factors
#
# The scaling factors (computed in script 52) align the PECD long-run mean
# capacity factors with Fraunhofer ISE 2024 midpoint full-load-hour
# assumptions.

# %%
scaling_df = pd.read_csv(paths.processed_data_path / "pecd_fraunhofer_scaling_factors.csv")
scaling = dict(zip(scaling_df["tech_key"], scaling_df["scaling_factor"]))

print("PECD -> Fraunhofer scaling factors:")
for tech, factor in scaling.items():
    print(f"  {tech:20s}  {factor:.4f}")

# %%
pecd = pd.read_parquet(paths.pecd_processed_file)
pecd_sim = pd.concat([pecd.loc[y] for y in SIM_YEARS])
total_hours = len(pecd_sim)
print(f"Simulation period: {SIM_YEARS[0]}–{SIM_YEARS[-1]}  ({total_hours:,} hours, {N_YEARS} years)")

hourly_cf = pd.DataFrame(index=pecd_sim.index)
for tech, variable in PECD_VARIABLES.items():
    raw_cf = pecd_sim[variable]["capacity_factor_ratio"][COUNTRY].values.astype(float)
    hourly_cf[tech] = np.clip(np.nan_to_num(raw_cf) * scaling[tech], 0, 1)

print(f"Hourly CF (scaled), {SIM_YEARS[0]}–{SIM_YEARS[-1]}, DE — shape: {hourly_cf.shape}")
print(hourly_cf.describe().round(4))


# %% [markdown]
# ## Hourly dispatch simulation
#
# Merit order: all available RE is dispatched first. If total RE exceeds
# demand, the excess is curtailed pro-rata across RE sources. Remaining
# demand is met by gas (CCGT).

# %%
dispatch = pd.DataFrame(index=hourly_cf.index)

for tech, cap in INSTALLED_CAP.items():
    dispatch[f"{tech}_raw"] = hourly_cf[tech] * cap

dispatch["re_total_raw"] = sum(dispatch[f"{tech}_raw"] for tech in INSTALLED_CAP)
dispatch["demand"] = DEMAND_MW

# Pro-rata utilisation: when RE > demand, each source is scaled down equally
dispatch["re_util"] = np.where(
    dispatch["re_total_raw"] > 0,
    np.minimum(1.0, dispatch["demand"] / dispatch["re_total_raw"]),
    1.0,
)

for tech in INSTALLED_CAP:
    dispatch[f"{tech}_useful"] = dispatch[f"{tech}_raw"] * dispatch["re_util"]

dispatch["re_total_useful"] = sum(dispatch[f"{tech}_useful"] for tech in INSTALLED_CAP)
dispatch["curtailment"] = (dispatch["re_total_raw"] - dispatch["re_total_useful"]).clip(lower=0)
dispatch["gas"] = (dispatch["demand"] - dispatch["re_total_useful"]).clip(lower=0)

# Sanity check: useful RE + gas = demand
balance = dispatch["re_total_useful"] + dispatch["gas"] - dispatch["demand"]
assert balance.abs().max() < 1e-9, f"Energy balance error: {balance.abs().max():.2e}"

total_demand_mwh = dispatch["demand"].sum() / N_YEARS  # annual average

print(f"Annual energy summary (MWh/yr, averaged over {N_YEARS} years):")
print(f"  Demand:            {total_demand_mwh:>10,.1f}")
print(f"  RE produced (raw): {dispatch['re_total_raw'].sum() / N_YEARS:>10,.1f}")
print(f"  RE useful:         {dispatch['re_total_useful'].sum() / N_YEARS:>10,.1f}")
print(f"  Curtailment:       {dispatch['curtailment'].sum() / N_YEARS:>10,.1f}")
print(f"  Gas:               {dispatch['gas'].sum() / N_YEARS:>10,.1f}")


# %% [markdown]
# ## Energy mix breakdown

# %%
annual_energy = {}
for tech in INSTALLED_CAP:
    produced = dispatch[f"{tech}_raw"].sum() / N_YEARS
    useful = dispatch[f"{tech}_useful"].sum() / N_YEARS
    annual_energy[tech] = {"produced": produced, "useful": useful,
                           "curtailed": produced - useful}

gas_energy_mwh = dispatch["gas"].sum() / N_YEARS
annual_energy["gas"] = {"produced": gas_energy_mwh, "useful": gas_energy_mwh,
                        "curtailed": 0.0}

mix_rows = []
for tech, vals in annual_energy.items():
    label = RE_COSTS[tech]["label"] if tech in RE_COSTS else GAS["label"]
    mix_rows.append({
        "technology": label,
        "tech_key": tech,
        "produced_mwh": vals["produced"],
        "useful_mwh": vals["useful"],
        "curtailed_mwh": vals["curtailed"],
        "share_pct": vals["useful"] / total_demand_mwh * 100,
        "curtailment_pct": (vals["curtailed"] / vals["produced"] * 100
                            if vals["produced"] > 0 else 0),
    })

mix_df = pd.DataFrame(mix_rows)
print(mix_df.to_string(index=False))

# %%
gas_capacity_mw = dispatch["gas"].max()
gas_cf = (gas_energy_mwh / (gas_capacity_mw * total_hours / N_YEARS)
          if gas_capacity_mw > 0 else 0)

print(f"\nRequired gas capacity: {gas_capacity_mw:.4f} MW"
      f"  (= {gas_capacity_mw / DEMAND_MW:.1%} of peak demand)")
print(f"Gas capacity factor:  {gas_cf:.4f}  ({gas_cf * HOURS_PER_YEAR:,.0f} FLH)")


# %% [markdown]
# ## Cost computation
#
# For each technology we compute an **equivalent annual cost** (EAC) that
# distributes the total discounted lifecycle cost evenly across the plant
# lifetime. Multiplying EAC per kW by installed capacity gives the annual
# system cost attributable to that technology.
#
# - **RE LCOE with curtailment**: annual cost stays the same (full capacity
#   is paid for) but useful energy shrinks, raising the effective LCOE.
# - **Gas LCOE**: fixed costs (CAPEX annuity + fixed OPEX) are spread over
#   actual output; variable costs (fuel, CO₂, var OPEX) scale with
#   generation.
# - **System LCOE**: sum of all annual costs ÷ total demand. Equivalently,
#   the demand-share-weighted average of per-technology LCOEs.
#
# The capital recovery factor (CRF) converts CAPEX to an equivalent annual
# payment:
#
# $$\text{CRF} = \frac{r\,(1+r)^N}{(1+r)^N - 1}$$

# %%
def capital_recovery_factor(wacc: float, lifetime: int) -> float:
    """CRF: converts lump-sum CAPEX to an equivalent annual payment."""
    if wacc == 0:
        return 1.0 / lifetime
    return wacc * (1 + wacc) ** lifetime / ((1 + wacc) ** lifetime - 1)


def re_equivalent_annual_cost(p: dict, cf: float) -> float:
    """Equivalent annual cost per kW for a RE technology (EUR/kW/yr).

    Uses the same discounting approach as compute_lcoe in script 52 —
    accounts for degradation and both fixed and variable OPEX — but returns
    the annualised total cost rather than cost per unit of energy.
    """
    years = np.arange(1, p["lifetime"] + 1)
    discount = (1 + p["wacc_real"]) ** years
    annual_gen = cf * HOURS_PER_YEAR * (1 - p["degradation"]) ** (years - 1)
    annual_opex = p["opex_fix"] + p["opex_var"] * annual_gen
    total_discounted_cost = p["capex_mid"] + np.sum(annual_opex / discount)
    annuity_factor = np.sum(1.0 / discount)
    return total_discounted_cost / annuity_factor


# %%
cost_rows = []

for tech, p in RE_COSTS.items():
    cap_mw = INSTALLED_CAP[tech]
    cf = float(hourly_cf[tech].mean())
    produced = annual_energy[tech]["produced"]
    useful = annual_energy[tech]["useful"]
    curt_frac = 1 - useful / produced if produced > 0 else 0

    eac_per_kw = re_equivalent_annual_cost(p, cf)
    # Total annual cost: EUR/kW/yr × MW × 1000 kW/MW = EUR/yr
    annual_cost = eac_per_kw * cap_mw * 1000

    # LCOE: EUR/kW/yr × MW cancels with MWh to give EUR/kWh
    # (since EUR/kW × MW / MWh = EUR × 1000 / (1000 kWh) = EUR/kWh)
    lcoe_no_curt = eac_per_kw * cap_mw / produced if produced > 0 else 0
    lcoe_with_curt = eac_per_kw * cap_mw / useful if useful > 0 else float("inf")

    cost_rows.append({
        "technology": p["label"],
        "tech_key": tech,
        "installed_mw": cap_mw,
        "useful_mwh": useful,
        "curtailment_pct": curt_frac * 100,
        "lcoe_ct": lcoe_with_curt * 100,
        "lcoe_no_curt_ct": lcoe_no_curt * 100,
        "annual_cost_eur": annual_cost,
    })

# Gas costs
g = GAS
crf = capital_recovery_factor(g["wacc_real"], g["lifetime"])
gas_fixed_per_kw = g["capex_mid"] * crf + g["opex_fix"]   # EUR/kW/yr

fuel_per_kwh = g["gas_price"] / 1000 / g["efficiency"]     # EUR/kWh_el
co2_per_kwh = (g["co2_intensity"] * g["co2_price"]
               / 1000 / g["efficiency"])                    # EUR/kWh_el
gas_marginal = fuel_per_kwh + co2_per_kwh + g["opex_var"]  # EUR/kWh_el

gas_total_fixed = gas_fixed_per_kw * gas_capacity_mw * 1000     # EUR/yr
gas_total_variable = gas_marginal * gas_energy_mwh * 1000       # EUR/yr
gas_annual_cost = gas_total_fixed + gas_total_variable
gas_lcoe = gas_annual_cost / (gas_energy_mwh * 1000) if gas_energy_mwh > 0 else 0

print("Gas cost detail:")
print(f"  CRF (WACC={g['wacc_real']:.1%}, N={g['lifetime']}yr):  {crf:.4f}")
print(f"  Annual fixed cost:   {gas_fixed_per_kw:>8.2f} EUR/kW/yr")
print(f"  Fuel cost:           {fuel_per_kwh * 100:>8.2f} ct/kWh_el")
print(f"  CO2 cost:            {co2_per_kwh * 100:>8.2f} ct/kWh_el")
print(f"  Var OPEX:            {g['opex_var'] * 100:>8.2f} ct/kWh_el")
print(f"  Marginal total:      {gas_marginal * 100:>8.2f} ct/kWh_el")

cost_rows.append({
    "technology": g["label"],
    "tech_key": "gas",
    "installed_mw": gas_capacity_mw,
    "useful_mwh": gas_energy_mwh,
    "curtailment_pct": 0.0,
    "lcoe_ct": gas_lcoe * 100,
    "lcoe_no_curt_ct": gas_lcoe * 100,
    "annual_cost_eur": gas_annual_cost,
})

cost_df = pd.DataFrame(cost_rows)

total_annual_cost = cost_df["annual_cost_eur"].sum()
system_lcoe = total_annual_cost / (total_demand_mwh * 1000)  # EUR/kWh

print("\nCost summary:")
pd.set_option("display.float_format", lambda v: f"{v:.2f}")
print(cost_df[["technology", "installed_mw", "useful_mwh", "curtailment_pct",
               "lcoe_no_curt_ct", "lcoe_ct", "annual_cost_eur"]].to_string(index=False))
print(f"\nTotal annual cost:  {total_annual_cost:>12,.0f} EUR")
print(f"System LCOE:        {system_lcoe * 100:>12.2f} ct/kWh")


# %%
# --- Individual annualised costs and LCOE per source ---
print("Annualised costs and LCOE per source:")
print(f"  {'Source':20s}  {'Annual cost':>14s}  {'Useful MWh':>12s}  {'LCOE':>10s}")
print(f"  {'':20s}  {'[EUR/yr]':>14s}  {'[MWh/yr]':>12s}  {'[ct/kWh]':>10s}")
print(f"  {'-' * 60}")
for _, row in cost_df.iterrows():
    print(f"  {row['technology']:20s}  {row['annual_cost_eur']:>14,.0f}"
          f"  {row['useful_mwh']:>12,.1f}  {row['lcoe_ct']:>10.2f}")
print(f"  {'-' * 60}")
print(f"  {'TOTAL SYSTEM':20s}  {total_annual_cost:>14,.0f}"
      f"  {total_demand_mwh:>12,.1f}  {system_lcoe * 100:>10.2f}")

total_re_produced = sum(annual_energy[t]["produced"] for t in INSTALLED_CAP)
total_curtailed = dispatch["curtailment"].sum() / N_YEARS
print(f"\n  Total RE produced:  {total_re_produced:>10,.1f} MWh/yr")
print(f"  Total curtailed:    {total_curtailed:>10,.1f} MWh/yr"
      f"  ({total_curtailed / total_re_produced:.1%} of RE production)")

# %%
# --- Cost attribution: each source's contribution to system LCOE ---
print("\nCost attribution to system LCOE:")
attr_sum = 0
for _, row in cost_df.iterrows():
    # Attribution = annualised cost of source / total system demand
    attr_ct = row["annual_cost_eur"] / (total_demand_mwh * 1000) * 100  # ct/kWh
    attr_sum += attr_ct
    share = row["useful_mwh"] / total_demand_mwh
    print(f"  {row['technology']:20s}  share={share:>5.1%}  "
          f"LCOE={row['lcoe_ct']:>6.2f} ct  ->  attribution={attr_ct:>5.2f} ct/kWh")
print(f"  {'Sum':20s}  {attr_sum:>42.2f} ct/kWh")
print(f"  {'System LCOE':20s}  {system_lcoe * 100:>42.2f} ct/kWh")


# %% [markdown]
# ## Visualisations

# %% [markdown]
# ### Energy mix and curtailment

# %%
fig_mix, (ax_share, ax_curt) = plt.subplots(1, 2, figsize=(12, 5))

# Left: share of demand
techs = list(annual_energy.keys())
labels = [RE_COSTS[t]["label"] if t in RE_COSTS else GAS["label"] for t in techs]
shares = [annual_energy[t]["useful"] / total_demand_mwh * 100 for t in techs]
colors = [TECH_COLORS[t] for t in techs]
bars = ax_share.barh(labels, shares, color=colors)
for bar, s in zip(bars, shares):
    ax_share.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                  f"{s:.1f} %", va="center", fontsize=9)
ax_share.set_xlabel("Share of demand [%]")
ax_share.set_title("Energy mix — share of annual demand")
ax_share.set_xlim(0, max(shares) * 1.2)
ax_share.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_share.set_axisbelow(True)

# Right: produced vs useful vs curtailed
re_techs = [t for t in techs if t != "gas"]
re_labels = [RE_COSTS[t]["label"] for t in re_techs]
produced = [annual_energy[t]["produced"] for t in re_techs]
useful = [annual_energy[t]["useful"] for t in re_techs]
curtailed = [annual_energy[t]["curtailed"] for t in re_techs]
re_colors = [TECH_COLORS[t] for t in re_techs]

y = np.arange(len(re_techs))
ax_curt.barh(y, useful, color=re_colors, alpha=0.9, label="Useful")
ax_curt.barh(y, curtailed, left=useful, color=TECH_COLORS["curtailment"],
             alpha=0.6, label="Curtailed")
for yi, u, c in zip(y, useful, curtailed):
    if c > 0:
        pct = c / (u + c) * 100
        ax_curt.text(u + c + 10, yi, f"{pct:.1f} % curtailed",
                     va="center", fontsize=8, color=TECH_COLORS["curtailment"])
ax_curt.set_yticks(y)
ax_curt.set_yticklabels(re_labels)
ax_curt.set_xlabel("Energy [MWh / yr]")
ax_curt.set_title("RE production: useful vs curtailed")
ax_curt.legend(loc="lower right", fontsize=9)
ax_curt.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_curt.set_axisbelow(True)

fig_mix.tight_layout()
fig_mix.savefig(paths.images_path / "53_energy_mix.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/53_energy_mix.png
# :name: fig-53-energy-mix
# Left: share of annual demand served by each technology. Right: total
# renewable production split into useful energy and curtailed excess.
# Curtailment occurs when combined RE output exceeds the constant 1 MW
# demand.
# ```

# %% [markdown]
# ### Sample dispatch week
#
# The week with the highest cumulative gas usage illustrates the worst-case
# Dunkelflaute dynamics.

# %%
# Find week with most gas usage
dispatch["week"] = dispatch.index.isocalendar().week.values
weekly_gas = dispatch.groupby("week")["gas"].sum()
worst_week = int(weekly_gas.idxmax())

week_mask = dispatch["week"] == worst_week
dw = dispatch.loc[week_mask].copy()
hours = np.arange(len(dw))

fig_week, ax_w = plt.subplots(figsize=(14, 5))
# Stack: solar, onshore, offshore, gas (bottom to top)
stack_techs = ["solar_pv_utility", "wind_onshore", "wind_offshore", "gas"]
stack_labels = ["Solar PV", "Wind onshore", "Wind offshore", "Gas (CCGT)"]
stack_colors = [TECH_COLORS[t] for t in stack_techs]
stack_data = np.array([dw[f"{t}_useful" if t != "gas" else t].values
                       for t in stack_techs])

# Add curtailment on top of the dispatch stack (above the demand line)
curt = dw["curtailment"].values
all_stack_data = np.vstack([stack_data, [curt]])
all_labels = stack_labels + ["Curtailment"]
all_colors = stack_colors + [TECH_COLORS["curtailment"]]
all_alpha = [0.85] * len(stack_techs) + [0.5]

# stackplot doesn't support per-layer alpha, so draw manually
bottoms = np.zeros(len(hours))
for data_row, label, color, alpha in zip(all_stack_data, all_labels,
                                          all_colors, all_alpha):
    ax_w.fill_between(hours, bottoms, bottoms + data_row, color=color,
                      alpha=alpha, label=label)
    bottoms += data_row

ax_w.plot(hours, dw["demand"].values, "k--", linewidth=1.2, label="Demand")

ax_w.set_xlabel("Hour of week")
ax_w.set_ylabel("Dispatch [MWh/h]")
start_date = dw.index[0].strftime("%d %b")
end_date = dw.index[-1].strftime("%d %b %Y")
ax_w.set_title(f"Hourly dispatch — week {worst_week} ({start_date} – {end_date}), "
               f"highest gas usage")
ax_w.legend(loc="upper left", fontsize=9)
ax_w.set_xlim(0, len(dw) - 1)
ax_w.set_ylim(0)

fig_week.tight_layout()
fig_week.savefig(paths.images_path / "53_dispatch_week.png", dpi=150,
                 bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/53_dispatch_week.png
# :name: fig-53-dispatch-week
# Hourly dispatch during the week with the highest gas usage in 2025.
# Stacked areas show useful generation by source; the dashed line marks
# constant demand (1 MW). The secondary axis shows curtailment, if any.
# ```

# %% [markdown]
# ### Monthly generation mix

# %%
dispatch["month"] = dispatch.index.month
monthly = dispatch.groupby("month").agg({
    "solar_pv_utility_useful": "sum",
    "wind_onshore_useful": "sum",
    "wind_offshore_useful": "sum",
    "gas": "sum",
    "curtailment": "sum",
    "demand": "sum",
}) / N_YEARS  # average across simulation years

fig_month, ax_m = plt.subplots(figsize=(10, 5))
months = monthly.index.values
bottom = np.zeros(len(months))
for tech, label in zip(["solar_pv_utility", "wind_onshore", "wind_offshore", "gas"],
                       ["Solar PV", "Wind onshore", "Wind offshore", "Gas (CCGT)"]):
    col = f"{tech}_useful" if tech != "gas" else tech
    vals = monthly[col].values
    ax_m.bar(months, vals, bottom=bottom, color=TECH_COLORS[tech],
             label=label, width=0.7)
    bottom += vals

# Curtailment markers
curt_monthly = monthly["curtailment"].values
for m, c in zip(months, curt_monthly):
    if c > 1:
        ax_m.annotate(f"{c:.0f}", (m, bottom[m - 1] + 5), fontsize=7,
                      ha="center", color=TECH_COLORS["curtailment"])

ax_m.set_xticks(months)
ax_m.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax_m.set_ylabel("Energy [MWh]")
ax_m.set_title(f"Monthly generation mix ({SIM_YEARS[0]}–{SIM_YEARS[-1]} avg)")
ax_m.legend(loc="upper right", fontsize=9)
ax_m.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_m.set_axisbelow(True)

fig_month.tight_layout()
fig_month.savefig(paths.images_path / "53_monthly_mix.png", dpi=150,
                  bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/53_monthly_mix.png
# :name: fig-53-monthly-mix
# Monthly generation by source. Winter months are dominated by wind and gas;
# summer months see higher solar contributions and potentially some
# curtailment.
# ```

# %% [markdown]
# ### LCOE per technology and system LCOE

# %%
fig_lcoe, (ax_lcoe, ax_cost) = plt.subplots(1, 2, figsize=(13, 5))

# Left: LCOE bars (with and without curtailment)
techs_ordered = ["solar_pv_utility", "wind_onshore", "wind_offshore", "gas"]
lcoe_labels = [cost_df.loc[cost_df["tech_key"] == t, "technology"].iloc[0]
               for t in techs_ordered]
lcoe_no_curt = [cost_df.loc[cost_df["tech_key"] == t, "lcoe_no_curt_ct"].iloc[0]
                for t in techs_ordered]
lcoe_curt = [cost_df.loc[cost_df["tech_key"] == t, "lcoe_ct"].iloc[0]
             for t in techs_ordered]
bar_colors = [TECH_COLORS[t] for t in techs_ordered]

y = np.arange(len(techs_ordered))
bars_nc = ax_lcoe.barh(y + 0.18, lcoe_no_curt, height=0.32, color=bar_colors,
                       alpha=0.5, label="LCOE (no curtailment)")
bars_c = ax_lcoe.barh(y - 0.18, lcoe_curt, height=0.32, color=bar_colors,
                      alpha=0.9, label="LCOE (with curtailment)")
ax_lcoe.axvline(system_lcoe * 100, color="black", linewidth=1.5, linestyle="--",
                label=f"System LCOE: {system_lcoe * 100:.1f} ct/kWh")

for yi, nc, c in zip(y, lcoe_no_curt, lcoe_curt):
    ax_lcoe.text(nc + 0.2, yi + 0.18, f"{nc:.1f}", va="center", fontsize=8,
                 alpha=0.6)
    ax_lcoe.text(c + 0.2, yi - 0.18, f"{c:.1f}", va="center", fontsize=8)

ax_lcoe.set_yticks(y)
ax_lcoe.set_yticklabels(lcoe_labels)
ax_lcoe.set_xlabel("LCOE [ct/kWh]")
ax_lcoe.set_title("LCOE per technology")
ax_lcoe.legend(loc="lower right", fontsize=8)
ax_lcoe.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_lcoe.set_axisbelow(True)
ax_lcoe.set_xlim(0)

# Right: annual cost breakdown (stacked bar)
cost_vals = [cost_df.loc[cost_df["tech_key"] == t, "annual_cost_eur"].iloc[0]
             for t in techs_ordered]
cost_labels = lcoe_labels

ax_cost.barh(cost_labels, cost_vals, color=bar_colors, alpha=0.85)
for bar, val in zip(ax_cost.patches, cost_vals):
    ax_cost.text(bar.get_width() + total_annual_cost * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:,.0f} EUR", va="center", fontsize=8)
ax_cost.set_xlabel("Annual cost [EUR / yr]")
ax_cost.set_title(f"Annual cost breakdown (total: {total_annual_cost:,.0f} EUR)")
ax_cost.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_cost.set_axisbelow(True)
ax_cost.set_xlim(0)

fig_lcoe.tight_layout()
fig_lcoe.savefig(paths.images_path / "53_cost_lcoe.png", dpi=150,
                 bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/53_cost_lcoe.png
# :name: fig-53-cost-lcoe
# Left: LCOE per technology with and without the curtailment penalty. The
# dashed line marks the system-wide average LCOE. Right: total annualised
# cost attributable to each technology (CAPEX annuity + OPEX + fuel/CO₂ for
# gas). All values in real 2024 EUR.
# ```

# %% [markdown]
# ### System overview: energy mix, costs and curtailment

# %%
fig_ov, (ax_emix, ax_lcoe_s, ax_curt_s) = plt.subplots(1, 3, figsize=(15, 5))

all_techs = ["solar_pv_utility", "wind_onshore", "wind_offshore", "gas"]
all_labels = [RE_COSTS[t]["label"] if t in RE_COSTS else GAS["label"]
              for t in all_techs]
all_colors = [TECH_COLORS[t] for t in all_techs]

# --- Left: stacked energy mix bar ---
useful_vals = [annual_energy[t]["useful"] for t in all_techs]
bottom_e = 0
for val, label, color in zip(useful_vals, all_labels, all_colors):
    ax_emix.bar(0, val, bottom=bottom_e, color=color, label=label, width=0.5)
    if val / total_demand_mwh > 0.04:
        ax_emix.text(0, bottom_e + val / 2, f"{val / total_demand_mwh:.1%}",
                     ha="center", va="center", fontsize=9, fontweight="bold",
                     color="white")
    bottom_e += val
ax_emix.set_ylabel("Energy [MWh / yr]")
ax_emix.set_title("Energy mix\n(after curtailment)")
ax_emix.set_xticks([0])
ax_emix.set_xticklabels(["Demand\nserved"])
ax_emix.legend(loc="upper right", fontsize=8)
ax_emix.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_emix.set_axisbelow(True)

# --- Middle: individual LCOEs, stacked cost attributions, system LCOE ---
lcoe_contribs = []
lcoe_individual = []
for t in all_techs:
    row = cost_df.loc[cost_df["tech_key"] == t].iloc[0]
    # Attribution = annualised cost / total system demand
    attr_ct = row["annual_cost_eur"] / (total_demand_mwh * 1000) * 100  # ct/kWh
    lcoe_contribs.append(attr_ct)
    lcoe_individual.append(row["lcoe_ct"])

lcoe_total = sum(lcoe_contribs)

# Individual LCOE bars (one per technology)
x_ind = np.arange(len(all_techs))
for xi, (val, label, color) in enumerate(zip(lcoe_individual, all_labels, all_colors)):
    ax_lcoe_s.bar(xi, val, color=color, width=0.6, alpha=0.85)
    ax_lcoe_s.text(xi, val + 0.2, f"{val:.1f}", ha="center", va="bottom",
                   fontsize=9)

# Stacked weighted contributions
x_stack = len(all_techs) + 0.8
bottom_l = 0
for val, label, color in zip(lcoe_contribs, all_labels, all_colors):
    ax_lcoe_s.bar(x_stack, val, bottom=bottom_l, color=color, width=0.6)
    if val > 0.3:
        ax_lcoe_s.text(x_stack, bottom_l + val / 2, f"{val:.1f}",
                       ha="center", va="center", fontsize=9, fontweight="bold",
                       color="white")
    bottom_l += val
ax_lcoe_s.text(x_stack, bottom_l + 0.2,
               f"{lcoe_total:.2f}", ha="center", va="bottom",
               fontsize=10, fontweight="bold")

# System LCOE bar
x_sys = len(all_techs) + 1.8
ax_lcoe_s.bar(x_sys, system_lcoe * 100, color="black", alpha=0.25, width=0.6)
ax_lcoe_s.text(x_sys, system_lcoe * 100 + 0.2,
               f"{system_lcoe * 100:.2f}", ha="center", va="bottom",
               fontsize=10, fontweight="bold")

ax_lcoe_s.set_ylabel("LCOE [ct / kWh]")
all_x = list(x_ind) + [x_stack, x_sys]
all_xlabels = all_labels + ["Cost\nattribution", "System\nLCOE"]
ax_lcoe_s.set_xticks(all_x)
ax_lcoe_s.set_xticklabels(all_xlabels, fontsize=8)
ax_lcoe_s.set_title("LCOE per source and system LCOE")
ax_lcoe_s.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_lcoe_s.set_axisbelow(True)
ax_lcoe_s.set_ylim(0)

# --- Right: curtailed energy per RE source ---
re_techs_o = [t for t in all_techs if t != "gas"]
re_labels_o = [RE_COSTS[t]["label"] for t in re_techs_o]
re_colors_o = [TECH_COLORS[t] for t in re_techs_o]
curt_vals = [annual_energy[t]["curtailed"] for t in re_techs_o]
curt_pcts = [annual_energy[t]["curtailed"] / annual_energy[t]["produced"] * 100
             if annual_energy[t]["produced"] > 0 else 0
             for t in re_techs_o]

bars_curt = ax_curt_s.bar(re_labels_o, curt_vals, color=re_colors_o, alpha=0.85)
for bar, pct in zip(bars_curt, curt_pcts):
    ax_curt_s.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                   f"{pct:.1f} %", ha="center", va="bottom", fontsize=9,
                   color=TECH_COLORS["curtailment"])
ax_curt_s.set_ylabel("Curtailed energy [MWh / yr]")
ax_curt_s.set_title("Curtailment\nper RE source")
ax_curt_s.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_curt_s.set_axisbelow(True)

total_curt = sum(curt_vals)
total_re_prod = sum(annual_energy[t]["produced"] for t in re_techs_o)
ax_curt_s.text(0.95, 0.95, f"Total: {total_curt:.1f} MWh\n({total_curt/total_re_prod:.1%} of RE)",
               transform=ax_curt_s.transAxes, ha="right", va="top", fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

fig_ov.suptitle(f"System overview — RE + Gas backup (Germany {SIM_YEARS[0]}–{SIM_YEARS[-1]}, 1 MW demand)",
                fontsize=12, y=1.02)
fig_ov.tight_layout()
fig_ov.savefig(paths.images_path / "53_system_overview.png", dpi=150,
               bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/53_system_overview.png
# :name: fig-53-system-overview
# System overview combining the energy mix (after curtailment), stacked LCOE
# contributions and annual cost breakdown, and curtailed energy per renewable
# source. Percentage labels on the curtailment bars show the fraction of each
# technology's total production that was curtailed.
# ```

# %% [markdown]
# ### RE production vs demand — aggregate comparison
#
# Before looking at the temporal dynamics, check whether total RE production
# is even sufficient to cover demand in aggregate. If RE < demand on an
# annual basis, the cumulative balance will drift downward structurally.

# %%
re_prod_annual = sum(annual_energy[t]["produced"] for t in INSTALLED_CAP)
re_useful_annual = sum(annual_energy[t]["useful"] for t in INSTALLED_CAP)

fig_agg, ax_agg = plt.subplots(figsize=(7, 5))
bar_labels = ["RE produced\n(potential)", "RE useful\n(after curtailment)", "Demand"]
bar_vals = [re_prod_annual, re_useful_annual, total_demand_mwh]
bar_colors = ["#2ca02c", "#4a90d9", "#333333"]
bars_agg = ax_agg.bar(bar_labels, bar_vals, color=bar_colors, alpha=0.85, width=0.55)
for bar, val in zip(bars_agg, bar_vals):
    ax_agg.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax_agg.axhline(total_demand_mwh, color="#333333", linewidth=1, linestyle="--", alpha=0.5)
ax_agg.set_ylabel("Energy [MWh / yr]")
re_ratio = re_prod_annual / total_demand_mwh
ax_agg.set_title(f"RE potential vs demand — RE covers {re_ratio:.0%} of demand\n"
                 f"(curtailment wastes {re_prod_annual - re_useful_annual:,.0f} MWh/yr)")
ax_agg.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_agg.set_axisbelow(True)
ax_agg.set_ylim(0, max(bar_vals) * 1.15)

fig_agg.tight_layout()
fig_agg.savefig(paths.images_path / "53_re_vs_demand.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/53_re_vs_demand.png
# :name: fig-53-re-vs-demand
# Aggregate annual comparison of potential RE production, useful RE (after
# curtailment), and demand. If RE produced < demand, the cumulative balance
# will trend downward over time and drawdowns will grow structurally.
# ```

# %% [markdown]
# ## Cumulative RE balance and drawdown analysis
#
# The cumulative balance series tracks `cumsum(RE_raw - demand)` over the full
# simulation period. When this series rises, RE output exceeds demand (surplus
# that would be curtailed or stored); when it falls, demand exceeds RE
# (requiring gas backup). The maximum drawdown — the largest peak-to-trough
# decline — quantifies the worst sustained period where renewables
# progressively fall behind demand.

# %%
re_balance = (dispatch["re_total_raw"] - dispatch["demand"]).cumsum()

running_max = re_balance.cummax()
drawdown = re_balance - running_max  # ≤ 0

trough_time = drawdown.idxmin()
trough_val = re_balance[trough_time]
peak_val = running_max[trough_time]
peak_time = re_balance[:trough_time].idxmax()
magnitude = peak_val - trough_val
duration_hours = (trough_time - peak_time) / pd.Timedelta("1h")

print("Cumulative RE balance — maximum drawdown:")
print(f"  Peak:      {peak_time}  (balance = {peak_val:+,.1f} MWh)")
print(f"  Trough:    {trough_time}  (balance = {trough_val:+,.1f} MWh)")
print(f"  Magnitude: {magnitude:,.1f} MWh  ({duration_hours / 24:.1f} days)")

# %%
fig_bal, (ax_bal, ax_dd) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# --- Top: cumulative balance ---
ax_bal.plot(re_balance.index, re_balance.values, color="#2ca02c", linewidth=0.7)
ax_bal.fill_between(re_balance.index, re_balance.values, 0,
                    where=(re_balance.values >= 0), color="#2ca02c", alpha=0.15)
ax_bal.fill_between(re_balance.index, re_balance.values, 0,
                    where=(re_balance.values < 0), color="tomato", alpha=0.15)
ax_bal.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax_bal.axvline(peak_time, color="#2ca02c", linewidth=1.0, linestyle=":", alpha=0.8)
ax_bal.axvline(trough_time, color="tomato", linewidth=1.0, linestyle=":", alpha=0.8)
ax_bal.annotate(f"Peak\n{peak_time.strftime('%d %b %Y')}",
                xy=(peak_time, peak_val), xytext=(10, 6),
                textcoords="offset points", fontsize=8, color="#2ca02c")
ax_bal.annotate(f"Trough\n{trough_time.strftime('%d %b %Y')}\n"
                f"−{magnitude:,.0f} MWh ({duration_hours / 24:.0f} d)",
                xy=(trough_time, trough_val), xytext=(10, -30),
                textcoords="offset points", fontsize=8, color="tomato")
ax_bal.set_ylabel("Cumulative balance [MWh]")
ax_bal.set_title(f"Cumulative RE production minus demand "
                 f"({SIM_YEARS[0]}–{SIM_YEARS[-1]})")
ax_bal.yaxis.grid(True, linewidth=0.4, alpha=0.5)
ax_bal.set_axisbelow(True)

# --- Bottom: drawdown from running peak ---
ax_dd.fill_between(drawdown.index, drawdown.values, 0,
                   color="#4a90d9", alpha=0.4, linewidth=0)
ax_dd.plot(drawdown.index, drawdown.values, color="#4a90d9", linewidth=0.6)
ax_dd.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax_dd.scatter([trough_time], [drawdown[trough_time]], color="tomato", zorder=5, s=40)
ax_dd.annotate(f"Max drawdown: {magnitude:,.0f} MWh\n"
               f"{peak_time.strftime('%b %Y')} → {trough_time.strftime('%b %Y')}"
               f" ({duration_hours / 24:.0f} d)",
               xy=(trough_time, drawdown[trough_time]),
               xytext=(12, -4), textcoords="offset points",
               fontsize=8, color="tomato", va="top")
ax_dd.set_ylabel("Drawdown [MWh]")
ax_dd.set_xlabel("Date")
ax_dd.set_title("RE drawdown from cumulative peak (shortfall vs demand)")
ax_dd.yaxis.grid(True, linewidth=0.4, alpha=0.5)
ax_dd.set_axisbelow(True)

fig_bal.tight_layout()
fig_bal.savefig(paths.images_path / "53_re_cumulative_balance.png", dpi=150,
                bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/53_re_cumulative_balance.png
# :name: fig-53-re-cumulative-balance
# Top: cumulative balance of total RE production minus demand. Positive
# (green) means RE has been ahead of demand cumulatively; negative (red)
# means demand has outpaced RE. Bottom: drawdown from the running
# cumulative peak — the maximum drawdown quantifies the worst sustained
# period where renewables progressively failed to keep up with demand.
# ```

# %% [markdown]
# ### Top-5 RE shortfall episodes
#
# Each episode is a contiguous period where the cumulative balance stays
# below its running maximum. The magnitude is the peak-to-trough energy
# deficit (MWh per MW of demand).

# %%
TOP_N = 5


def find_top_drawdowns(balance: pd.Series, n: int = TOP_N) -> pd.DataFrame:
    """Return top-N drawdown episodes ranked by magnitude."""
    running_max = balance.cummax()
    dd = balance - running_max  # ≤ 0

    in_dd = dd < 0
    ep_start = in_dd & ~in_dd.shift(1, fill_value=False)
    ep_end = ~in_dd & in_dd.shift(1, fill_value=False)

    starts = balance.index[ep_start].tolist()
    ends = balance.index[ep_end].tolist()
    if len(starts) > len(ends):  # series ends mid-drawdown
        ends.append(balance.index[-1])

    rows = []
    for s, e in zip(starts, ends):
        trough_t = dd[s:e].idxmin()
        peak_v = running_max[s]
        trough_v = balance[trough_t]
        mag = peak_v - trough_v

        candidates = balance[:s][balance[:s] >= peak_v]
        peak_t = candidates.index[-1] if len(candidates) else s

        rows.append({
            "peak_time": peak_t,
            "trough_time": trough_t,
            "recovery_time": e,
            "magnitude_mwh": mag,
            "peak_to_trough_days": (trough_t - peak_t) / pd.Timedelta("1D"),
            "total_episode_days": (e - peak_t) / pd.Timedelta("1D"),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("magnitude_mwh", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


top_dd = find_top_drawdowns(re_balance)

print(f"\nTop-{TOP_N} RE shortfall episodes:")
for i, row in top_dd.iterrows():
    print(f"  #{i+1}  {row['peak_time'].strftime('%d %b %Y')} → "
          f"{row['trough_time'].strftime('%d %b %Y')}  "
          f"magnitude {row['magnitude_mwh']:,.0f} MWh  "
          f"duration {row['peak_to_trough_days']:.0f} d  "
          f"(recovery after {row['total_episode_days']:.0f} d)")

# %%
fig_dd, (ax_mag, ax_dur) = plt.subplots(1, 2, figsize=(14, 4.5))

labels = [
    f"#{i+1}  {r['peak_time'].strftime('%b %Y')} → {r['trough_time'].strftime('%b %Y')}"
    for i, (_, r) in enumerate(top_dd.iterrows())
]
y = range(len(top_dd))

# Magnitude panel
ax_mag.barh(list(y), top_dd["magnitude_mwh"], color="#4a90d9",
            edgecolor="white", linewidth=0.5)
ax_mag.set_yticks(list(y))
ax_mag.set_yticklabels(labels, fontsize=8)
ax_mag.invert_yaxis()
ax_mag.set_xlabel("Magnitude [MWh]")
ax_mag.set_title("RE shortfall depth (energy deficit)")
ax_mag.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_mag.set_axisbelow(True)
for yi, v in zip(y, top_dd["magnitude_mwh"]):
    ax_mag.text(v + top_dd["magnitude_mwh"].max() * 0.01, yi,
                f"{v:,.0f}", va="center", fontsize=8)

# Duration panel
ax_dur.barh(list(y), top_dd["peak_to_trough_days"], color="#4a90d9",
            edgecolor="white", linewidth=0.5, alpha=0.7)
ax_dur.set_yticks(list(y))
ax_dur.set_yticklabels(labels, fontsize=8)
ax_dur.invert_yaxis()
ax_dur.set_xlabel("Peak-to-trough duration [days]")
ax_dur.set_title("RE shortfall duration")
ax_dur.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_dur.set_axisbelow(True)
for yi, v in zip(y, top_dd["peak_to_trough_days"]):
    ax_dur.text(v + top_dd["peak_to_trough_days"].max() * 0.01, yi,
                f"{v:.0f} d", va="center", fontsize=8)

fig_dd.suptitle(f"Top-{TOP_N} RE shortfall episodes "
                f"({SIM_YEARS[0]}–{SIM_YEARS[-1]}, {RE_SCALING:.0f}× RE capacity)",
                fontsize=11)
fig_dd.tight_layout()
fig_dd.savefig(paths.images_path / "53_re_shortfall_episodes.png", dpi=150,
               bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/53_re_shortfall_episodes.png
# :name: fig-53-re-shortfall-episodes
# Top-5 episodes where cumulative RE production fell furthest behind
# cumulative demand. Left: energy deficit magnitude. Right: duration from
# the onset of the deficit to the deepest point. These episodes represent
# the worst sustained periods where renewables alone could not keep up with
# demand, requiring either storage or dispatchable backup.
# ```

# %%

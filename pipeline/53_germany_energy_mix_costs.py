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
# # Germany Energy Mix Costs: Renewables + Battery + Gas Backup
#
# Simulates an hourly dispatch for Germany using PECD capacity factors
# (scaled to align with Fraunhofer ISE 2024 midpoint full-load hours) with a
# simple merit order: all available renewable generation is dispatched first;
# any surplus is stored in a lithium-ion battery (until full); remaining
# surplus is curtailed pro-rata; shortfalls are covered first by the battery,
# then by gas-fired combined-cycle (CCGT) plants.
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
# - Energy share per source (solar PV, wind onshore, wind offshore, battery, gas)
# - Per-source energy accounting (direct use, stored, delivered from battery,
#   round-trip loss, curtailed)
# - Annualised system cost and system LCOE
# - Technology-level LCOE (with and without curtailment/storage effect)

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
#
# ### Battery storage (Li-ion, utility-scale)
#
# From the same study (PV + battery utility-scale component, Tables 1–2).
# Fraunhofer reports battery costs only in the context of PV+battery systems;
# we extract the battery component for standalone use.
#
# | Parameter                  | Value              | Source                |
# |----------------------------|-------------------:|:----------------------|
# | CAPEX                      | 500 €/kWh          | mid of 400–600        |
# | Lifetime                   | 15 years            | Table 2               |
# | WACC (real)                | 2.5 %               | Table 2               |
# | Fixed OPEX                 | 10 €/kWh/yr         | ~2 % of CAPEX         |
# | Round-trip efficiency      | 90 %                | Table 2               |

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

BAT = {
    "label": "Battery (Li-ion)",
    "capex_kwh": 500,              # EUR/kWh usable capacity (mid of 400–600)
    "lifetime": 15,                # years
    "wacc_real": 0.025,            # 2.5 %
    "opex_fix_kwh": 10.0,         # EUR/kWh/yr (~2 % of CAPEX)
    "rt_efficiency": 0.90,        # round-trip efficiency
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
    "battery":          "#7fbc41",
    "gas":              "#888888",
    "curtailment":      "#cc4444",
}

# %% [markdown]
# ## Installed capacities and battery sizing
#
# Germany end-2024 approximate installed capacities normalised to 1 MW of
# constant demand (~57 GW average load, ~500 TWh/yr):
#
# | Technology    | Installed (DE) | Per 1 MW demand |
# |---------------|---------------:|----------------:|
# | Solar PV      |       96 GW    |        1.68 MW  |
# | Wind onshore  |       62 GW    |        1.09 MW  |
# | Wind offshore |        9 GW    |        0.16 MW  |
#
# Battery sizing: 4-hour duration at 1 MW rated power (matching demand),
# giving 4 MWh of usable capacity. This is a common reference for
# grid-scale storage studies and captures typical daily cycling patterns.

# %%
AVG_DEMAND_GW = 57.0
RE_SCALING = 2.0  # multiplier for all installed RE capacities

INSTALLED_CAP = {
    "solar_pv_utility": 96.0 / AVG_DEMAND_GW * RE_SCALING,
    "wind_onshore":     62.0 / AVG_DEMAND_GW * RE_SCALING,
    "wind_offshore":     9.0 / AVG_DEMAND_GW * RE_SCALING,
}
DEMAND_MW = 1.0

BAT_POWER_MW = 1.0                              # max charge/discharge rate
BAT_DURATION_H = 4                               # hours at rated power
BAT_CAPACITY_MWH = BAT_POWER_MW * BAT_DURATION_H  # usable energy capacity

print(f"RE scaling factor: {RE_SCALING:.1f}x\n")
for tech, cap in INSTALLED_CAP.items():
    print(f"{RE_COSTS[tech]['label']:20s}  {cap:.3f} MW per MW demand")
print(f"{'Total RE':20s}  {sum(INSTALLED_CAP.values()):.3f} MW per MW demand")
print(f"\nBattery: {BAT_POWER_MW:.1f} MW / {BAT_CAPACITY_MWH:.1f} MWh "
      f"({BAT_DURATION_H}h, η_rt={BAT['rt_efficiency']:.0%})")


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
# demand, the surplus charges the battery (pro-rata across RE sources).
# Once the battery is full, remaining surplus is curtailed pro-rata.
# When demand exceeds RE, the battery discharges first; any remaining
# shortfall is met by gas (CCGT).
#
# Round-trip losses are applied at charge time: for every 1 MWh entering
# the battery from the grid, only η\_rt MWh is stored (and later delivered
# 1:1). This simplifies attribution — the loss is immediately assigned
# to the source that produced it.

# %%
re_techs = list(INSTALLED_CAP.keys())
n_hours = total_hours
η = BAT["rt_efficiency"]

# --- Pre-compute raw generation arrays ---
gen_arrays = {}
for tech in re_techs:
    gen_arrays[tech] = hourly_cf[tech].values * INSTALLED_CAP[tech]

# --- Hourly result arrays ---
soc = np.zeros(n_hours + 1)            # SOC at start of each hour (MWh)
bat_charge_grid = np.zeros(n_hours)    # grid-side energy into battery
bat_discharge = np.zeros(n_hours)      # energy from battery to grid
bat_rt_loss = np.zeros(n_hours)        # round-trip loss (at charge time)
gas_hourly = np.zeros(n_hours)
curt_hourly = np.zeros(n_hours)

# Per-source hourly tracking
direct_h = {t: np.zeros(n_hours) for t in re_techs}
stored_h = {t: np.zeros(n_hours) for t in re_techs}     # grid-side into battery
from_bat_h = {t: np.zeros(n_hours) for t in re_techs}   # delivered from battery
loss_h = {t: np.zeros(n_hours) for t in re_techs}       # RT loss
curt_src_h = {t: np.zeros(n_hours) for t in re_techs}   # curtailed

# SOC composition: absolute MWh in SOC per source (running state)
soc_src = {t: 0.0 for t in re_techs}

for h in range(n_hours):
    # Raw generation per source
    gen = {t: gen_arrays[t][h] for t in re_techs}
    total_gen = sum(gen.values())
    demand = DEMAND_MW

    if total_gen >= demand and total_gen > 0:
        # --- Surplus hour: RE covers demand ---
        frac_demand = demand / total_gen
        for t in re_techs:
            direct_h[t][h] = gen[t] * frac_demand

        excess = total_gen - demand

        # Charge battery: limited by power, remaining capacity, and excess
        space = BAT_CAPACITY_MWH - soc[h]
        # Grid-side energy that would fill remaining space: space / η
        max_charge_grid = space / η if η > 0 else 0
        charge_grid = min(excess, BAT_POWER_MW, max_charge_grid)
        charge_to_soc = charge_grid * η
        loss = charge_grid - charge_to_soc

        bat_charge_grid[h] = charge_grid
        bat_rt_loss[h] = loss

        # Attribute charge, loss, curtailment pro-rata to sources
        frac_charge = charge_grid / excess if excess > 0 else 0
        for t in re_techs:
            src_excess = gen[t] - direct_h[t][h]
            src_charge = src_excess * frac_charge
            stored_h[t][h] = src_charge
            loss_h[t][h] = src_charge * (1 - η)
            curt_src_h[t][h] = src_excess - src_charge
            soc_src[t] += src_charge * η  # deliverable portion

        soc[h + 1] = soc[h] + charge_to_soc
        curt_hourly[h] = excess - charge_grid

    else:
        # --- Deficit hour: RE < demand ---
        for t in re_techs:
            direct_h[t][h] = gen[t]

        shortfall = demand - total_gen

        # Discharge battery
        max_discharge = min(soc[h], BAT_POWER_MW)
        discharge = min(shortfall, max_discharge)
        bat_discharge[h] = discharge

        # Attribute discharge pro-rata to SOC composition
        if discharge > 0 and soc[h] > 0:
            for t in re_techs:
                frac = soc_src[t] / soc[h]
                from_bat_h[t][h] = discharge * frac
                soc_src[t] -= discharge * frac

        soc[h + 1] = soc[h] - discharge
        gas_hourly[h] = shortfall - discharge

# %% [markdown]
# ### Dispatch summary

# %%
# --- Build dispatch DataFrame for plotting ---
dispatch = pd.DataFrame(index=hourly_cf.index)
for tech in re_techs:
    dispatch[f"{tech}_raw"] = gen_arrays[tech]
    dispatch[f"{tech}_direct"] = direct_h[tech]
    dispatch[f"{tech}_useful"] = direct_h[tech] + from_bat_h[tech]
    dispatch[f"{tech}_curtailed"] = curt_src_h[tech]

dispatch["re_total_raw"] = sum(gen_arrays[t] for t in re_techs)
dispatch["re_total_direct"] = sum(direct_h[t] for t in re_techs)
dispatch["bat_charge"] = bat_charge_grid
dispatch["bat_discharge"] = bat_discharge
dispatch["bat_rt_loss"] = bat_rt_loss
dispatch["soc"] = soc[:-1]
dispatch["gas"] = gas_hourly
dispatch["curtailment"] = curt_hourly
dispatch["demand"] = DEMAND_MW

# Sanity check: supply = demand at every hour
supply = dispatch["re_total_direct"] + dispatch["bat_discharge"] + dispatch["gas"]
balance = supply - dispatch["demand"]
assert balance.abs().max() < 1e-9, f"Energy balance error: {balance.abs().max():.2e}"

total_demand_mwh = dispatch["demand"].sum() / N_YEARS  # annual average

print(f"Annual energy summary (MWh/yr, averaged over {N_YEARS} years):")
print(f"  Demand:              {total_demand_mwh:>10,.1f}")
print(f"  RE produced (raw):   {dispatch['re_total_raw'].sum() / N_YEARS:>10,.1f}")
print(f"  RE direct use:       {dispatch['re_total_direct'].sum() / N_YEARS:>10,.1f}")
print(f"  Battery charged:     {bat_charge_grid.sum() / N_YEARS:>10,.1f}  (grid-side)")
print(f"  Battery discharged:  {bat_discharge.sum() / N_YEARS:>10,.1f}")
print(f"  RT loss:             {bat_rt_loss.sum() / N_YEARS:>10,.1f}")
print(f"  Curtailment:         {dispatch['curtailment'].sum() / N_YEARS:>10,.1f}")
print(f"  Gas:                 {dispatch['gas'].sum() / N_YEARS:>10,.1f}")
print(f"\n  Battery utilisation: {bat_discharge.sum() / N_YEARS / BAT_CAPACITY_MWH:.0f} "
      f"equiv. full cycles/yr")


# %% [markdown]
# ## Energy mix breakdown
#
# For each RE source we track:
# - **Direct use** — generation dispatched straight to demand
# - **Stored** — energy sent to battery (grid-side, before RT loss)
# - **Delivered from battery** — energy returned from battery to demand
# - **Round-trip loss** — energy lost in storage (= stored × (1 − η\_rt))
# - **Still in battery** — remaining SOC attributed to this source
# - **Curtailed** — excess that could not be stored
# - **Useful** — directly used + delivered from battery

# %%
annual_energy = {}
for tech in re_techs:
    produced = gen_arrays[tech].sum() / N_YEARS
    direct = direct_h[tech].sum() / N_YEARS
    stored = stored_h[tech].sum() / N_YEARS
    delivered = from_bat_h[tech].sum() / N_YEARS
    rt_loss = loss_h[tech].sum() / N_YEARS
    still_in_bat = soc_src[tech] / N_YEARS
    curtailed = curt_src_h[tech].sum() / N_YEARS
    useful = direct + delivered

    annual_energy[tech] = {
        "produced": produced,
        "direct": direct,
        "stored": stored,
        "delivered_from_bat": delivered,
        "rt_loss": rt_loss,
        "still_in_bat": still_in_bat,
        "curtailed": curtailed,
        "useful": useful,
    }

gas_energy_mwh = gas_hourly.sum() / N_YEARS
bat_delivered_mwh = bat_discharge.sum() / N_YEARS

annual_energy["gas"] = {
    "produced": gas_energy_mwh, "direct": gas_energy_mwh,
    "stored": 0, "delivered_from_bat": 0, "rt_loss": 0,
    "still_in_bat": 0, "curtailed": 0, "useful": gas_energy_mwh,
}

mix_rows = []
for tech, vals in annual_energy.items():
    label = RE_COSTS[tech]["label"] if tech in RE_COSTS else GAS["label"]
    mix_rows.append({
        "technology": label,
        "tech_key": tech,
        "produced_mwh": vals["produced"],
        "direct_mwh": vals["direct"],
        "stored_mwh": vals["stored"],
        "delivered_bat_mwh": vals["delivered_from_bat"],
        "rt_loss_mwh": vals["rt_loss"],
        "curtailed_mwh": vals["curtailed"],
        "useful_mwh": vals["useful"],
        "share_pct": vals["useful"] / total_demand_mwh * 100,
    })

mix_df = pd.DataFrame(mix_rows)
print(mix_df[["technology", "produced_mwh", "direct_mwh", "stored_mwh",
              "delivered_bat_mwh", "rt_loss_mwh", "curtailed_mwh",
              "useful_mwh", "share_pct"]].to_string(index=False))

# %%
gas_capacity_mw = gas_hourly.max()
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
# - **RE LCOE with curtailment/storage**: annual cost stays the same (full
#   capacity is paid for) but useful energy now includes direct use plus
#   energy delivered through the battery, reducing the effective LCOE
#   compared to pure curtailment.
# - **Battery**: annualised CAPEX + OPEX spread over delivered energy gives
#   the levelised cost of storage (LCOS).
# - **Gas LCOE**: fixed costs (CAPEX annuity + fixed OPEX) are spread over
#   actual output; variable costs (fuel, CO₂, var OPEX) scale with
#   generation.
# - **System LCOE**: sum of all annual costs ÷ total demand.

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
    annual_cost = eac_per_kw * cap_mw * 1000  # EUR/yr

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

# --- Battery costs ---
bat_crf = capital_recovery_factor(BAT["wacc_real"], BAT["lifetime"])
bat_annual_capex = BAT["capex_kwh"] * BAT_CAPACITY_MWH * 1000 * bat_crf
bat_annual_opex = BAT["opex_fix_kwh"] * BAT_CAPACITY_MWH * 1000
bat_annual_cost = bat_annual_capex + bat_annual_opex
bat_lcos = (bat_annual_cost / (bat_delivered_mwh * 1000)
            if bat_delivered_mwh > 0 else float("inf"))

print("Battery cost detail:")
print(f"  CAPEX:       {BAT['capex_kwh']} EUR/kWh × {BAT_CAPACITY_MWH * 1000:.0f} kWh"
      f" = {BAT['capex_kwh'] * BAT_CAPACITY_MWH * 1000:,.0f} EUR")
print(f"  CRF (WACC={BAT['wacc_real']:.1%}, N={BAT['lifetime']}yr):  {bat_crf:.4f}")
print(f"  Annual CAPEX:  {bat_annual_capex:>10,.0f} EUR/yr")
print(f"  Annual OPEX:   {bat_annual_opex:>10,.0f} EUR/yr")
print(f"  Total annual:  {bat_annual_cost:>10,.0f} EUR/yr")
print(f"  Delivered:     {bat_delivered_mwh:>10,.1f} MWh/yr")
print(f"  LCOS:          {bat_lcos * 100:>10.2f} ct/kWh")

cost_rows.append({
    "technology": BAT["label"],
    "tech_key": "battery",
    "installed_mw": BAT_POWER_MW,
    "useful_mwh": bat_delivered_mwh,
    "curtailment_pct": 0.0,
    "lcoe_ct": bat_lcos * 100,
    "lcoe_no_curt_ct": bat_lcos * 100,
    "annual_cost_eur": bat_annual_cost,
})

# --- Gas costs ---
g = GAS
crf = capital_recovery_factor(g["wacc_real"], g["lifetime"])
gas_fixed_per_kw = g["capex_mid"] * crf + g["opex_fix"]

fuel_per_kwh = g["gas_price"] / 1000 / g["efficiency"]
co2_per_kwh = g["co2_intensity"] * g["co2_price"] / 1000 / g["efficiency"]
gas_marginal = fuel_per_kwh + co2_per_kwh + g["opex_var"]

gas_total_fixed = gas_fixed_per_kw * gas_capacity_mw * 1000
gas_total_variable = gas_marginal * gas_energy_mwh * 1000
gas_annual_cost = gas_total_fixed + gas_total_variable
gas_lcoe = gas_annual_cost / (gas_energy_mwh * 1000) if gas_energy_mwh > 0 else 0

print("\nGas cost detail:")
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

total_re_produced = sum(annual_energy[t]["produced"] for t in re_techs)
total_curtailed = dispatch["curtailment"].sum() / N_YEARS
total_rt_loss = bat_rt_loss.sum() / N_YEARS
print(f"\n  Total RE produced:  {total_re_produced:>10,.1f} MWh/yr")
print(f"  Total curtailed:    {total_curtailed:>10,.1f} MWh/yr"
      f"  ({total_curtailed / total_re_produced:.1%} of RE production)")
print(f"  Total RT loss:      {total_rt_loss:>10,.1f} MWh/yr")

# %%
# --- Cost attribution: each source's contribution to system LCOE ---
print("\nCost attribution to system LCOE:")
attr_sum = 0
for _, row in cost_df.iterrows():
    attr_ct = row["annual_cost_eur"] / (total_demand_mwh * 1000) * 100
    attr_sum += attr_ct
    share = row["useful_mwh"] / total_demand_mwh if row["tech_key"] != "battery" else 0
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
all_techs_mix = list(re_techs) + ["gas"]
labels_mix = [RE_COSTS[t]["label"] if t in RE_COSTS else GAS["label"]
              for t in all_techs_mix]
shares = [annual_energy[t]["useful"] / total_demand_mwh * 100 for t in all_techs_mix]
colors_mix = [TECH_COLORS[t] for t in all_techs_mix]
bars = ax_share.barh(labels_mix, shares, color=colors_mix)
for bar, s in zip(bars, shares):
    ax_share.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                  f"{s:.1f} %", va="center", fontsize=9)
ax_share.set_xlabel("Share of demand [%]")
ax_share.set_title("Energy mix — share of annual demand")
ax_share.set_xlim(0, max(shares) * 1.2)
ax_share.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_share.set_axisbelow(True)

# Right: produced vs useful (direct + battery) vs curtailed + loss
re_labels = [RE_COSTS[t]["label"] for t in re_techs]
re_colors = [TECH_COLORS[t] for t in re_techs]
direct_vals = [annual_energy[t]["direct"] for t in re_techs]
delivered_vals = [annual_energy[t]["delivered_from_bat"] for t in re_techs]
curtailed_vals = [annual_energy[t]["curtailed"] for t in re_techs]
loss_vals = [annual_energy[t]["rt_loss"] for t in re_techs]

y = np.arange(len(re_techs))
ax_curt.barh(y, direct_vals, color=re_colors, alpha=0.9, label="Direct use")
ax_curt.barh(y, delivered_vals, left=direct_vals, color=re_colors, alpha=0.5,
             hatch="//", label="Via battery")
left_2 = [d + v for d, v in zip(direct_vals, delivered_vals)]
ax_curt.barh(y, loss_vals, left=left_2, color="#ff9900", alpha=0.6,
             label="RT loss")
left_3 = [l + v for l, v in zip(left_2, loss_vals)]
ax_curt.barh(y, curtailed_vals, left=left_3, color=TECH_COLORS["curtailment"],
             alpha=0.6, label="Curtailed")
for yi, d, b, lo, c in zip(y, direct_vals, delivered_vals, loss_vals, curtailed_vals):
    total = d + b + lo + c
    if c > 0:
        pct = c / total * 100
        ax_curt.text(total + 10, yi, f"{pct:.1f}% curt",
                     va="center", fontsize=7, color=TECH_COLORS["curtailment"])
ax_curt.set_yticks(y)
ax_curt.set_yticklabels(re_labels)
ax_curt.set_xlabel("Energy [MWh / yr]")
ax_curt.set_title("RE production: direct, battery, loss, curtailed")
ax_curt.legend(loc="lower right", fontsize=8)
ax_curt.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_curt.set_axisbelow(True)

fig_mix.tight_layout()
fig_mix.savefig(paths.images_path / "53_energy_mix.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/53_energy_mix.png
# :name: fig-53-energy-mix
# Left: share of annual demand served by each technology. Right: total
# renewable production split into direct use, battery-mediated delivery,
# round-trip losses, and curtailed excess.
# ```

# %% [markdown]
# ### Sample dispatch week
#
# The week with the highest cumulative gas usage illustrates the worst-case
# Dunkelflaute dynamics. The chart uses a supply/demand layout: sources of
# power (RE generation, battery discharge, gas) are stacked above zero;
# consumption (demand, battery charging, curtailment) is stacked below zero.

# %%
# Find week with most gas usage
dispatch["week"] = dispatch.index.isocalendar().week.values
weekly_gas = dispatch.groupby("week")["gas"].sum()
worst_week = int(weekly_gas.idxmax())

week_mask = dispatch["week"] == worst_week
dw = dispatch.loc[week_mask].copy()
hours = np.arange(len(dw))

fig_week, ax_w = plt.subplots(figsize=(14, 6))

# --- Positive side: supply sources ---
pos_layers = [
    ("solar_pv_utility_raw", "Solar PV", TECH_COLORS["solar_pv_utility"], 0.85),
    ("wind_onshore_raw", "Wind onshore", TECH_COLORS["wind_onshore"], 0.85),
    ("wind_offshore_raw", "Wind offshore", TECH_COLORS["wind_offshore"], 0.85),
    ("bat_discharge", "Battery discharge", TECH_COLORS["battery"], 0.75),
    ("gas", "Gas (CCGT)", TECH_COLORS["gas"], 0.75),
]
pos_bottom = np.zeros(len(hours))
for col, label, color, alpha in pos_layers:
    vals = dw[col].values
    ax_w.fill_between(hours, pos_bottom, pos_bottom + vals, color=color,
                      alpha=alpha, label=label, linewidth=0)
    pos_bottom += vals

# --- Negative side: consumption ---
neg_layers = [
    ("demand", "Demand", "#333333", 0.35),
    ("bat_charge", "Battery charging", TECH_COLORS["battery"], 0.35),
    ("curtailment", "Curtailment", TECH_COLORS["curtailment"], 0.45),
]
neg_bottom = np.zeros(len(hours))
for col, label, color, alpha in neg_layers:
    vals = dw[col].values
    ax_w.fill_between(hours, -neg_bottom, -(neg_bottom + vals), color=color,
                      alpha=alpha, label=label, linewidth=0)
    neg_bottom += vals

ax_w.axhline(0, color="black", linewidth=0.8)
ax_w.set_xlabel("Hour of week")
ax_w.set_ylabel("Power [MW]")
start_date = dw.index[0].strftime("%d %b")
end_date = dw.index[-1].strftime("%d %b %Y")
ax_w.set_title(f"Hourly dispatch — week {worst_week} ({start_date} – {end_date}), "
               f"highest gas usage")
ax_w.legend(loc="upper left", fontsize=8, ncol=2)
ax_w.set_xlim(0, len(dw) - 1)

fig_week.tight_layout()
fig_week.savefig(paths.images_path / "53_dispatch_week.png", dpi=150,
                 bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/53_dispatch_week.png
# :name: fig-53-dispatch-week
# Hourly dispatch during the week with the highest gas usage. Above zero:
# supply (RE generation, battery discharge, gas). Below zero: consumption
# (demand, battery charging, curtailment). The areas balance at every hour.
# ```

# %% [markdown]
# ### Monthly generation mix

# %%
dispatch["month"] = dispatch.index.month
monthly = dispatch.groupby("month").agg({
    "solar_pv_utility_direct": "sum",
    "wind_onshore_direct": "sum",
    "wind_offshore_direct": "sum",
    "bat_discharge": "sum",
    "gas": "sum",
    "curtailment": "sum",
    "demand": "sum",
}) / N_YEARS

fig_month, ax_m = plt.subplots(figsize=(10, 5))
months = monthly.index.values
bottom = np.zeros(len(months))

month_techs = [
    ("solar_pv_utility_direct", "Solar PV", TECH_COLORS["solar_pv_utility"]),
    ("wind_onshore_direct", "Wind onshore", TECH_COLORS["wind_onshore"]),
    ("wind_offshore_direct", "Wind offshore", TECH_COLORS["wind_offshore"]),
    ("bat_discharge", "Battery", TECH_COLORS["battery"]),
    ("gas", "Gas (CCGT)", TECH_COLORS["gas"]),
]
for col, label, color in month_techs:
    vals = monthly[col].values
    ax_m.bar(months, vals, bottom=bottom, color=color, label=label, width=0.7)
    bottom += vals

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
# Monthly generation by source including battery discharge. Winter months
# are dominated by wind and gas; summer months see higher solar
# contributions.
# ```

# %% [markdown]
# ### LCOE per technology and system LCOE

# %%
fig_lcoe, (ax_lcoe, ax_cost) = plt.subplots(1, 2, figsize=(13, 5))

# Left: LCOE bars (with and without curtailment)
techs_ordered = ["solar_pv_utility", "wind_onshore", "wind_offshore", "battery", "gas"]
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

# Right: annual cost breakdown
cost_vals = [cost_df.loc[cost_df["tech_key"] == t, "annual_cost_eur"].iloc[0]
             for t in techs_ordered]

ax_cost.barh(lcoe_labels, cost_vals, color=bar_colors, alpha=0.85)
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
# Left: LCOE per technology with and without the curtailment penalty. For
# battery, both bars show the LCOS. The dashed line marks the system-wide
# average LCOE. Right: total annualised cost attributable to each
# technology. All values in real 2024 EUR.
# ```

# %% [markdown]
# ### System overview: energy mix, costs and curtailment

# %%
fig_ov, (ax_emix, ax_lcoe_s, ax_curt_s) = plt.subplots(1, 3, figsize=(15, 5))

all_techs = ["solar_pv_utility", "wind_onshore", "wind_offshore", "battery", "gas"]
all_labels = []
all_colors = []
for t in all_techs:
    if t in RE_COSTS:
        all_labels.append(RE_COSTS[t]["label"])
    elif t == "battery":
        all_labels.append(BAT["label"])
    else:
        all_labels.append(GAS["label"])
    all_colors.append(TECH_COLORS[t])

# --- Left: stacked energy mix bar ---
# Battery delivers energy but shouldn't double-count (it stores RE).
# Show RE direct, battery discharge (separately), and gas.
energy_vals = [
    annual_energy["solar_pv_utility"]["direct"],
    annual_energy["wind_onshore"]["direct"],
    annual_energy["wind_offshore"]["direct"],
    bat_delivered_mwh,
    gas_energy_mwh,
]
bottom_e = 0
for val, label, color in zip(energy_vals, all_labels, all_colors):
    ax_emix.bar(0, val, bottom=bottom_e, color=color, label=label, width=0.5)
    if val / total_demand_mwh > 0.04:
        ax_emix.text(0, bottom_e + val / 2, f"{val / total_demand_mwh:.1%}",
                     ha="center", va="center", fontsize=9, fontweight="bold",
                     color="white")
    bottom_e += val
ax_emix.set_ylabel("Energy [MWh / yr]")
ax_emix.set_title("Energy mix\n(demand served by)")
ax_emix.set_xticks([0])
ax_emix.set_xticklabels(["Demand\nserved"])
ax_emix.legend(loc="upper right", fontsize=8)
ax_emix.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_emix.set_axisbelow(True)

# --- Middle: LCOEs and stacked cost attributions ---
lcoe_contribs = []
lcoe_individual = []
for t in all_techs:
    row = cost_df.loc[cost_df["tech_key"] == t].iloc[0]
    attr_ct = row["annual_cost_eur"] / (total_demand_mwh * 1000) * 100
    lcoe_contribs.append(attr_ct)
    lcoe_individual.append(row["lcoe_ct"])

lcoe_total = sum(lcoe_contribs)

x_ind = np.arange(len(all_techs))
for xi, (val, label, color) in enumerate(zip(lcoe_individual, all_labels, all_colors)):
    ax_lcoe_s.bar(xi, val, color=color, width=0.6, alpha=0.85)
    ax_lcoe_s.text(xi, val + 0.2, f"{val:.1f}", ha="center", va="bottom",
                   fontsize=8)

x_stack = len(all_techs) + 0.8
bottom_l = 0
for val, label, color in zip(lcoe_contribs, all_labels, all_colors):
    ax_lcoe_s.bar(x_stack, val, bottom=bottom_l, color=color, width=0.6)
    if val > 0.3:
        ax_lcoe_s.text(x_stack, bottom_l + val / 2, f"{val:.1f}",
                       ha="center", va="center", fontsize=8, fontweight="bold",
                       color="white")
    bottom_l += val
ax_lcoe_s.text(x_stack, bottom_l + 0.2,
               f"{lcoe_total:.2f}", ha="center", va="bottom",
               fontsize=10, fontweight="bold")

x_sys = len(all_techs) + 1.8
ax_lcoe_s.bar(x_sys, system_lcoe * 100, color="black", alpha=0.25, width=0.6)
ax_lcoe_s.text(x_sys, system_lcoe * 100 + 0.2,
               f"{system_lcoe * 100:.2f}", ha="center", va="bottom",
               fontsize=10, fontweight="bold")

ax_lcoe_s.set_ylabel("LCOE [ct / kWh]")
all_x = list(x_ind) + [x_stack, x_sys]
all_xlabels = all_labels + ["Cost\nattribution", "System\nLCOE"]
ax_lcoe_s.set_xticks(all_x)
ax_lcoe_s.set_xticklabels(all_xlabels, fontsize=7, rotation=30, ha="right")
ax_lcoe_s.set_title("LCOE per source and system LCOE")
ax_lcoe_s.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_lcoe_s.set_axisbelow(True)
ax_lcoe_s.set_ylim(0)

# --- Right: curtailed energy per RE source ---
re_labels_o = [RE_COSTS[t]["label"] for t in re_techs]
re_colors_o = [TECH_COLORS[t] for t in re_techs]
curt_vals = [annual_energy[t]["curtailed"] for t in re_techs]
curt_pcts = [annual_energy[t]["curtailed"] / annual_energy[t]["produced"] * 100
             if annual_energy[t]["produced"] > 0 else 0
             for t in re_techs]

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
total_re_prod = sum(annual_energy[t]["produced"] for t in re_techs)
ax_curt_s.text(0.95, 0.95, f"Total: {total_curt:.1f} MWh\n({total_curt/total_re_prod:.1%} of RE)",
               transform=ax_curt_s.transAxes, ha="right", va="top", fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

fig_ov.suptitle(f"System overview — RE + Battery + Gas "
                f"(Germany {SIM_YEARS[0]}–{SIM_YEARS[-1]}, 1 MW demand)",
                fontsize=12, y=1.02)
fig_ov.tight_layout()
fig_ov.savefig(paths.images_path / "53_system_overview.png", dpi=150,
               bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/53_system_overview.png
# :name: fig-53-system-overview
# System overview combining the energy mix (after curtailment and storage),
# stacked LCOE contributions and annual cost breakdown, and curtailed energy
# per renewable source. Battery costs are included in the system LCOE.
# ```

# %% [markdown]
# ### Battery state of charge

# %%
fig_soc, (ax_soc_full, ax_soc_week) = plt.subplots(2, 1, figsize=(16, 7))

# --- Top: full simulation period ---
ax_soc_full.fill_between(dispatch.index, dispatch["soc"].values,
                         color=TECH_COLORS["battery"], alpha=0.4, linewidth=0)
ax_soc_full.plot(dispatch.index, dispatch["soc"].values,
                 color=TECH_COLORS["battery"], linewidth=0.5)
ax_soc_full.axhline(BAT_CAPACITY_MWH, color="black", linewidth=0.8,
                     linestyle="--", alpha=0.5, label=f"Capacity: {BAT_CAPACITY_MWH} MWh")
ax_soc_full.set_ylabel("SOC [MWh]")
ax_soc_full.set_title(f"Battery state of charge — {BAT_POWER_MW:.0f} MW / "
                       f"{BAT_CAPACITY_MWH:.0f} MWh ({BAT_DURATION_H}h)")
ax_soc_full.legend(loc="upper right", fontsize=9)
ax_soc_full.set_ylim(0, BAT_CAPACITY_MWH * 1.1)
ax_soc_full.yaxis.grid(True, linewidth=0.4, alpha=0.5)
ax_soc_full.set_axisbelow(True)

# --- Bottom: worst gas week ---
soc_week = dw["soc"].values
ax_soc_week.fill_between(hours, soc_week, color=TECH_COLORS["battery"],
                         alpha=0.4, linewidth=0)
ax_soc_week.plot(hours, soc_week, color=TECH_COLORS["battery"], linewidth=1.0)
ax_soc_week.axhline(BAT_CAPACITY_MWH, color="black", linewidth=0.8,
                     linestyle="--", alpha=0.5)
ax_soc_week.set_xlabel("Hour of week")
ax_soc_week.set_ylabel("SOC [MWh]")
ax_soc_week.set_title(f"SOC during week {worst_week} ({start_date} – {end_date})")
ax_soc_week.set_xlim(0, len(dw) - 1)
ax_soc_week.set_ylim(0, BAT_CAPACITY_MWH * 1.1)
ax_soc_week.yaxis.grid(True, linewidth=0.4, alpha=0.5)
ax_soc_week.set_axisbelow(True)

fig_soc.tight_layout()
fig_soc.savefig(paths.images_path / "53_battery_soc.png", dpi=150,
                bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/53_battery_soc.png
# :name: fig-53-battery-soc
# Battery state of charge over the full simulation period (top) and during
# the worst gas usage week (bottom). The dashed line marks full capacity.
# ```

# %% [markdown]
# ### RE production vs demand — aggregate comparison

# %%
re_prod_annual = sum(annual_energy[t]["produced"] for t in re_techs)
re_useful_annual = sum(annual_energy[t]["useful"] for t in re_techs)

fig_agg, ax_agg = plt.subplots(figsize=(7, 5))
bar_labels = ["RE produced\n(potential)", "RE useful\n(direct + battery)", "Demand"]
bar_vals = [re_prod_annual, re_useful_annual, total_demand_mwh]
bar_colors_agg = ["#2ca02c", "#4a90d9", "#333333"]
bars_agg = ax_agg.bar(bar_labels, bar_vals, color=bar_colors_agg, alpha=0.85, width=0.55)
for bar, val in zip(bars_agg, bar_vals):
    ax_agg.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax_agg.axhline(total_demand_mwh, color="#333333", linewidth=1, linestyle="--", alpha=0.5)
ax_agg.set_ylabel("Energy [MWh / yr]")
re_ratio = re_prod_annual / total_demand_mwh
ax_agg.set_title(f"RE potential vs demand — RE covers {re_ratio:.0%} of demand\n"
                 f"(curtailment: {total_curtailed:,.0f} MWh, "
                 f"RT loss: {total_rt_loss:,.0f} MWh)")
ax_agg.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_agg.set_axisbelow(True)
ax_agg.set_ylim(0, max(bar_vals) * 1.15)

fig_agg.tight_layout()
fig_agg.savefig(paths.images_path / "53_re_vs_demand.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/53_re_vs_demand.png
# :name: fig-53-re-vs-demand
# Aggregate annual comparison of potential RE production, useful RE (direct
# use plus battery-mediated delivery), and demand. Curtailment and
# round-trip losses reduce the useful fraction.
# ```

# %% [markdown]
# ## Cumulative RE balance and shortfall analysis
#
# The cumulative balance series tracks `cumsum(RE_raw - demand)` over the full
# simulation period to show the structural relationship between renewable
# production and demand. With battery storage, shortfall episodes are defined
# as contiguous periods where gas backup is needed (RE + battery insufficient).

# %%
re_balance = (dispatch["re_total_raw"] - dispatch["demand"]).cumsum()

running_max = re_balance.cummax()
drawdown = re_balance - running_max

trough_time = drawdown.idxmin()
trough_val = re_balance[trough_time]
peak_val = running_max[trough_time]
peak_time = re_balance[:trough_time].idxmax()
magnitude = peak_val - trough_val
duration_hours = (trough_time - peak_time) / pd.Timedelta("1h")

print("Cumulative RE balance — maximum drawdown (structural, before battery):")
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
                 f"({SIM_YEARS[0]}–{SIM_YEARS[-1]}, structural)")
ax_bal.yaxis.grid(True, linewidth=0.4, alpha=0.5)
ax_bal.set_axisbelow(True)

# --- Bottom: drawdown from running peak ---
ax_dd.fill_between(drawdown.index, drawdown.values, 0,
                   color="#4a90d9", alpha=0.4, linewidth=0)
ax_dd.plot(drawdown.index, drawdown.values, color="#4a90d9", linewidth=0.6)
ax_dd.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax_dd.axhline(-BAT_CAPACITY_MWH, color=TECH_COLORS["battery"], linewidth=1.2,
              linestyle="--", alpha=0.8,
              label=f"Battery capacity: {BAT_CAPACITY_MWH:.0f} MWh")
ax_dd.scatter([trough_time], [drawdown[trough_time]], color="tomato", zorder=5, s=40)
ax_dd.annotate(f"Max drawdown: {magnitude:,.0f} MWh\n"
               f"{peak_time.strftime('%b %Y')} → {trough_time.strftime('%b %Y')}"
               f" ({duration_hours / 24:.0f} d)",
               xy=(trough_time, drawdown[trough_time]),
               xytext=(12, -4), textcoords="offset points",
               fontsize=8, color="tomato", va="top")
ax_dd.set_ylabel("Drawdown [MWh]")
ax_dd.set_xlabel("Date")
ax_dd.set_title("RE drawdown from cumulative peak — battery covers "
                f"drawdowns ≤ {BAT_CAPACITY_MWH:.0f} MWh")
ax_dd.legend(loc="lower left", fontsize=9)
ax_dd.yaxis.grid(True, linewidth=0.4, alpha=0.5)
ax_dd.set_axisbelow(True)

fig_bal.tight_layout()
fig_bal.savefig(paths.images_path / "53_re_cumulative_balance.png", dpi=150,
                bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/53_re_cumulative_balance.png
# :name: fig-53-re-cumulative-balance
# Top: cumulative balance of total RE production minus demand (structural).
# Bottom: drawdown from the running cumulative peak. The dashed green line
# marks the battery capacity — drawdowns within this range can be buffered
# by the battery; deeper drawdowns require gas backup.
# ```

# %% [markdown]
# ### Top-5 gas shortfall episodes
#
# Each episode is a contiguous period where gas backup is needed (RE + battery
# cannot cover demand). The magnitude is the cumulative gas energy in each
# episode.

# %%
TOP_N = 5


def find_top_gas_episodes(gas_series: np.ndarray, index: pd.DatetimeIndex,
                          n: int = TOP_N) -> pd.DataFrame:
    """Return top-N contiguous gas usage episodes ranked by cumulative gas."""
    is_gas = gas_series > 1e-9
    ep_start = is_gas & ~np.roll(is_gas, 1)
    ep_start[0] = is_gas[0]
    ep_end = ~is_gas & np.roll(is_gas, 1)
    ep_end[0] = False

    starts = np.where(ep_start)[0].tolist()
    ends = np.where(ep_end)[0].tolist()
    if is_gas[-1] and (len(ends) == 0 or ends[-1] <= starts[-1]):
        ends.append(len(gas_series))

    rows = []
    for s_idx, e_idx in zip(starts, ends):
        cum_gas = gas_series[s_idx:e_idx].sum()
        peak_gas = gas_series[s_idx:e_idx].max()
        duration_h = e_idx - s_idx
        rows.append({
            "start_time": index[s_idx],
            "end_time": index[min(e_idx - 1, len(index) - 1)],
            "cumulative_gas_mwh": cum_gas,
            "peak_gas_mw": peak_gas,
            "duration_hours": duration_h,
            "duration_days": duration_h / 24,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("cumulative_gas_mwh", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


top_gas = find_top_gas_episodes(gas_hourly, dispatch.index)

print(f"\nTop-{TOP_N} gas shortfall episodes (RE + battery insufficient):")
for i, row in top_gas.iterrows():
    print(f"  #{i+1}  {row['start_time'].strftime('%d %b %Y')} → "
          f"{row['end_time'].strftime('%d %b %Y')}  "
          f"gas {row['cumulative_gas_mwh']:,.1f} MWh  "
          f"peak {row['peak_gas_mw']:.3f} MW  "
          f"duration {row['duration_days']:.1f} d")

# %%
fig_dd, (ax_mag, ax_dur) = plt.subplots(1, 2, figsize=(14, 4.5))

labels = [
    f"#{i+1}  {r['start_time'].strftime('%b %Y')} → {r['end_time'].strftime('%b %Y')}"
    for i, (_, r) in enumerate(top_gas.iterrows())
]
y = range(len(top_gas))

# Magnitude panel
ax_mag.barh(list(y), top_gas["cumulative_gas_mwh"], color=TECH_COLORS["gas"],
            edgecolor="white", linewidth=0.5)
ax_mag.set_yticks(list(y))
ax_mag.set_yticklabels(labels, fontsize=8)
ax_mag.invert_yaxis()
ax_mag.set_xlabel("Cumulative gas [MWh]")
ax_mag.set_title("Gas shortfall depth (after battery)")
ax_mag.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_mag.set_axisbelow(True)
for yi, v in zip(y, top_gas["cumulative_gas_mwh"]):
    ax_mag.text(v + top_gas["cumulative_gas_mwh"].max() * 0.01, yi,
                f"{v:,.1f}", va="center", fontsize=8)

# Duration panel
ax_dur.barh(list(y), top_gas["duration_days"], color=TECH_COLORS["gas"],
            edgecolor="white", linewidth=0.5, alpha=0.7)
ax_dur.set_yticks(list(y))
ax_dur.set_yticklabels(labels, fontsize=8)
ax_dur.invert_yaxis()
ax_dur.set_xlabel("Episode duration [days]")
ax_dur.set_title("Gas shortfall duration")
ax_dur.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_dur.set_axisbelow(True)
for yi, v in zip(y, top_gas["duration_days"]):
    ax_dur.text(v + top_gas["duration_days"].max() * 0.01, yi,
                f"{v:.1f} d", va="center", fontsize=8)

fig_dd.suptitle(f"Top-{TOP_N} gas shortfall episodes "
                f"({SIM_YEARS[0]}–{SIM_YEARS[-1]}, {RE_SCALING:.0f}× RE, "
                f"{BAT_CAPACITY_MWH:.0f} MWh battery)",
                fontsize=11)
fig_dd.tight_layout()
fig_dd.savefig(paths.images_path / "53_re_shortfall_episodes.png", dpi=150,
               bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/53_re_shortfall_episodes.png
# :name: fig-53-re-shortfall-episodes
# Top-5 episodes where gas backup was needed despite battery storage. Left:
# cumulative gas energy in each episode. Right: episode duration. These
# represent the worst sustained periods where renewables plus the 4-hour
# battery could not keep up with demand.
# ```

# %%

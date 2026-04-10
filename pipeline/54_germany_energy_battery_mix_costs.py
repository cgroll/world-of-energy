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
# # Germany Energy Mix with Battery Storage
#
# Simulates an hourly dispatch for Germany using PECD capacity factors
# (scaled to align with Fraunhofer ISE 2024 midpoint full-load hours).
#
# **Merit order:** all available renewable generation is dispatched first;
# surplus charges a lithium-ion battery (losses split equally between
# charge and discharge via √η); remaining surplus is curtailed pro-rata;
# shortfalls are covered first by the battery, then by gas (CCGT).
#
# Demand is normalised to a constant 1 MW (= 8 760 MWh/yr).
#
# ## Modules
#
# 1. **RE generation** — hourly power from capacity factors × installed capacity
# 2. **Battery simulation** — physical dispatch with split charge/discharge losses
# 3. **Power attribution** — pro-rata assignment of direct use, storage, curtailment
# 4. **Cost computation** — annualised costs for RE, battery, gas
# 5. **System LCOE** — overall and per-source levelised costs
# 6. **Drawdown analysis** — cumulative RE balance, top gas shortfall episodes

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
# ## Cost and technology assumptions
#
# ### Renewable energy
#
# From Fraunhofer ISE LCOE 2024, Tables 1–2. CAPEX midpoints are used.
#
# ### Gas (CCGT)
#
# | Parameter                | Value            | Source               |
# |--------------------------|----------------:|:---------------------|
# | CAPEX                    | 1 100 €/kW      | mid of 900–1 300     |
# | Lifetime                 | 30 years         | Table 2              |
# | WACC (real)              | 7.5 %            | assumed (carbon risk)|
# | Fixed OPEX               | 20 €/kW/yr       | Table 2              |
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
#
# The 6.8 % annuity includes a notional capacity-replacement component
# (1.3 %) that offsets degradation, so the simulation assumes constant
# capacity over an infinite horizon.

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
    "annuity_rate": 0.068,         # 6.8 % of CAPEX per year (capital + O&M + degradation)
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
# Battery: 4-hour duration at 1 MW rated power → 4 MWh usable capacity.

# %%
AVG_DEMAND_GW = 57.0
RE_SCALING = 3.0

INSTALLED_CAP = {
    "solar_pv_utility": 96.0 / AVG_DEMAND_GW * RE_SCALING,
    "wind_onshore":     62.0 / AVG_DEMAND_GW * RE_SCALING,
    "wind_offshore":     9.0 / AVG_DEMAND_GW * RE_SCALING,
}
DEMAND_MW = 1.0

BAT_POWER_MW = 1.0
BAT_DURATION_H = 4
BAT_CAPACITY_MWH = BAT_POWER_MW * BAT_DURATION_H

# Derived battery efficiencies: split RT losses equally between charge/discharge
EFF_IN = np.sqrt(BAT["rt_efficiency"])
EFF_OUT = np.sqrt(BAT["rt_efficiency"])

print(f"RE scaling factor: {RE_SCALING:.1f}x\n")
for tech, cap in INSTALLED_CAP.items():
    print(f"{RE_COSTS[tech]['label']:20s}  {cap:.3f} MW per MW demand")
print(f"{'Total RE':20s}  {sum(INSTALLED_CAP.values()):.3f} MW per MW demand")
print(f"\nBattery: {BAT_POWER_MW:.1f} MW / {BAT_CAPACITY_MWH:.1f} MWh "
      f"({BAT_DURATION_H}h, η_rt={BAT['rt_efficiency']:.0%}, "
      f"η_in={EFF_IN:.4f}, η_out={EFF_OUT:.4f})")

# %% [markdown]
# ## Load PECD data and apply Fraunhofer scaling factors

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

print(f"Hourly CF shape: {hourly_cf.shape}")
print(hourly_cf.describe().round(4))


# %% [markdown]
# ---
# ## Module 1 — Renewable power generation
#
# Hourly generation per RE source = capacity factor × installed capacity.

# %%
re_techs = list(INSTALLED_CAP.keys())

gen_arrays = {}
for tech in re_techs:
    gen_arrays[tech] = hourly_cf[tech].values * INSTALLED_CAP[tech]

total_gen = sum(gen_arrays[t] for t in re_techs)

print("Annual generation per source (MWh/yr):")
for tech in re_techs:
    annual = gen_arrays[tech].sum() / N_YEARS
    flh = annual / INSTALLED_CAP[tech]
    print(f"  {RE_COSTS[tech]['label']:20s}  {annual:>8,.1f} MWh  ({flh:,.0f} FLH)")
print(f"  {'Total RE':20s}  {total_gen.sum() / N_YEARS:>8,.1f} MWh")


# %% [markdown]
# ---
# ## Module 2 — Battery dispatch simulation
#
# Physical simulation with split charge/discharge efficiencies.
# For every hour the function computes the energy flows at the "battery
# gate" and updates the state of charge (SOC).
#
# - **Surplus (gen > demand):** excess charges the battery.  Grid-side
#   intake is limited by available excess, max power, and free SOC
#   (adjusted for charge losses).  Of the grid-side intake, only
#   `intake × η_in` is stored; the rest is charge loss.  Un-storable
#   surplus is curtailed.
#
# - **Deficit (gen < demand):** the battery discharges.  Gross energy
#   removed from SOC is limited by the deficit (adjusted for discharge
#   losses), max power, and available SOC.  The grid receives
#   `gross × η_out`; the rest is discharge loss.  Remaining deficit
#   is residual load (gas).

# %%
def simulate_battery(
    total_gen: np.ndarray,
    demand: float,
    cap_mwh: float,
    max_p_mw: float,
    eff_in: float,
    eff_out: float,
) -> pd.DataFrame:
    """Simulate battery physics over an hourly time series.

    Returns arrays (not a DataFrame) for: soc, curtailment, charge/discharge
    losses, grid-side power into/from battery, and residual load.
    """
    n = len(total_gen)
    soc = 0.0

    soc_arr = np.zeros(n)
    curtailment = np.zeros(n)
    loss_in = np.zeros(n)
    loss_out = np.zeros(n)
    power_to_bat = np.zeros(n)      # >0 = grid→bat (charge), <0 = bat→grid (discharge gross)
    residual_load = np.zeros(n)

    for t in range(n):
        net = total_gen[t] - demand

        if net > 0:
            # --- Surplus: charge battery ---
            space_gross = (cap_mwh - soc) / eff_in
            can_take = min(net, max_p_mw, space_gross)

            stored = can_take * eff_in
            soc += stored

            power_to_bat[t] = can_take
            loss_in[t] = can_take * (1 - eff_in)
            curtailment[t] = net - can_take

        else:
            # --- Deficit: discharge battery ---
            deficit = -net
            needed_gross = deficit / eff_out
            max_p_gross = max_p_mw / eff_out
            can_give = min(needed_gross, max_p_gross, soc)

            provided = can_give * eff_out
            soc -= can_give

            power_to_bat[t] = -can_give
            loss_out[t] = can_give * (1 - eff_out)
            residual_load[t] = deficit - provided

        soc_arr[t] = soc

    return {
        "soc": soc_arr,
        "curtailment": curtailment,
        "loss_in": loss_in,
        "loss_out": loss_out,
        "power_to_bat": power_to_bat,
        "residual_load": residual_load,
    }


# %%
sim = simulate_battery(total_gen, DEMAND_MW, BAT_CAPACITY_MWH, BAT_POWER_MW,
                       EFF_IN, EFF_OUT)

# Derived convenience arrays
bat_charge_grid = np.maximum(sim["power_to_bat"], 0)
bat_discharge_gross = np.maximum(-sim["power_to_bat"], 0)
bat_discharge_net = bat_discharge_gross * EFF_OUT

# Energy balance check: RE direct + battery discharge + gas = demand
re_direct_total = np.minimum(total_gen, DEMAND_MW)
# In surplus hours RE covers all demand; in deficit hours all RE is used
# but the "direct" share is total_gen (< demand).  Battery + gas fill gap.
supply = np.where(
    total_gen >= DEMAND_MW,
    DEMAND_MW,
    total_gen,
) + bat_discharge_net + sim["residual_load"]
balance_err = np.abs(supply - DEMAND_MW).max()
assert balance_err < 1e-9, f"Energy balance error: {balance_err:.2e}"

print("Battery simulation complete — energy balance OK")
print(f"  Annual curtailment:   {sim['curtailment'].sum() / N_YEARS:>10,.1f} MWh")
print(f"  Annual charge loss:   {sim['loss_in'].sum() / N_YEARS:>10,.1f} MWh")
print(f"  Annual discharge loss:{sim['loss_out'].sum() / N_YEARS:>10,.1f} MWh")
print(f"  Annual gas (residual):{sim['residual_load'].sum() / N_YEARS:>10,.1f} MWh")
print(f"  Battery cycles/yr:    {bat_discharge_gross.sum() / N_YEARS / BAT_CAPACITY_MWH:,.0f}")


# %% [markdown]
# ---
# ## Module 3 — Power attribution
#
# Pro-rata assignment of energy flows to individual RE sources.
#
# - **Surplus hours:** demand is met pro-rata by all sources.  The excess
#   is split pro-rata into battery charging and curtailment.  Charge
#   losses are attributed to the charging sources.
# - **Deficit hours:** all RE generation goes to direct use.  Battery
#   discharge is attributed to sources proportional to their share of
#   the current SOC (SOC composition tracking).  Discharge losses are
#   similarly attributed.

# %%
def attribute_power(
    gen_per_source: dict[str, np.ndarray],
    total_gen: np.ndarray,
    demand: float,
    sim: dict[str, np.ndarray],
    eff_in: float,
    eff_out: float,
) -> dict:
    """Pro-rata attribution of all energy flows to individual RE sources.

    Returns a dict of dicts: {tech: {direct, to_battery, from_battery,
    loss_charge, loss_discharge, curtailed}} with numpy arrays, plus
    soc_composition (final SOC per source).
    """
    techs = list(gen_per_source.keys())
    n = len(total_gen)

    direct = {t: np.zeros(n) for t in techs}
    to_battery = {t: np.zeros(n) for t in techs}
    from_battery = {t: np.zeros(n) for t in techs}
    loss_charge = {t: np.zeros(n) for t in techs}
    loss_discharge = {t: np.zeros(n) for t in techs}
    curtailed = {t: np.zeros(n) for t in techs}

    soc_comp = {t: 0.0 for t in techs}

    ptb = sim["power_to_bat"]
    curt = sim["curtailment"]

    for h in range(n):
        tg = total_gen[h]
        net = tg - demand

        if net > 0:
            # Surplus: pro-rata direct use, then battery / curtailment
            demand_frac = demand / tg if tg > 0 else 0.0
            charge_grid = ptb[h]   # grid-side into battery
            curt_h = curt[h]

            for t in techs:
                g = gen_per_source[t][h]
                direct[t][h] = g * demand_frac
                excess = g - direct[t][h]
                frac = excess / net if net > 0 else 0.0
                to_battery[t][h] = charge_grid * frac
                curtailed[t][h] = curt_h * frac
                loss_charge[t][h] = to_battery[t][h] * (1 - eff_in)
                soc_comp[t] += to_battery[t][h] * eff_in
        else:
            # Deficit: all gen is direct use
            for t in techs:
                direct[t][h] = gen_per_source[t][h]

            discharge_gross = -ptb[h]
            if discharge_gross > 0:
                total_soc = sum(soc_comp.values())
                if total_soc > 0:
                    for t in techs:
                        frac = soc_comp[t] / total_soc
                        from_battery[t][h] = discharge_gross * eff_out * frac
                        loss_discharge[t][h] = discharge_gross * (1 - eff_out) * frac
                        soc_comp[t] -= discharge_gross * frac

    return {
        "direct": direct,
        "to_battery": to_battery,
        "from_battery": from_battery,
        "loss_charge": loss_charge,
        "loss_discharge": loss_discharge,
        "curtailed": curtailed,
        "soc_composition": soc_comp,
    }


# %%
attr = attribute_power(gen_arrays, total_gen, DEMAND_MW, sim, EFF_IN, EFF_OUT)

# %% [markdown]
# ### Attribution summary

# %%
annual_energy = {}
for tech in re_techs:
    produced = gen_arrays[tech].sum() / N_YEARS
    direct_use = attr["direct"][tech].sum() / N_YEARS
    stored = attr["to_battery"][tech].sum() / N_YEARS
    delivered = attr["from_battery"][tech].sum() / N_YEARS
    loss_ch = attr["loss_charge"][tech].sum() / N_YEARS
    loss_dis = attr["loss_discharge"][tech].sum() / N_YEARS
    still_in = attr["soc_composition"][tech] / N_YEARS
    curt = attr["curtailed"][tech].sum() / N_YEARS
    useful = direct_use + delivered

    annual_energy[tech] = {
        "produced": produced,
        "direct": direct_use,
        "stored": stored,
        "delivered_from_bat": delivered,
        "loss_charge": loss_ch,
        "loss_discharge": loss_dis,
        "total_loss": loss_ch + loss_dis,
        "still_in_bat": still_in,
        "curtailed": curt,
        "useful": useful,
    }

    # Accounting check: produced ≈ useful + total_loss + still_in_bat + curtailed
    accounted = useful + loss_ch + loss_dis + still_in + curt
    err = abs(produced - accounted)
    assert err < 1e-6, f"{tech}: accounting error {err:.2e}"

gas_energy = sim["residual_load"].sum() / N_YEARS
bat_delivered = bat_discharge_net.sum() / N_YEARS
total_demand = DEMAND_MW * total_hours / N_YEARS

annual_energy["gas"] = {
    "produced": gas_energy, "direct": gas_energy,
    "stored": 0, "delivered_from_bat": 0,
    "loss_charge": 0, "loss_discharge": 0, "total_loss": 0,
    "still_in_bat": 0, "curtailed": 0, "useful": gas_energy,
}

print(f"{'Source':20s}  {'Produced':>10s}  {'Direct':>10s}  {'Via bat':>10s}  "
      f"{'Loss(ch)':>10s}  {'Loss(dis)':>10s}  {'Curtailed':>10s}  {'Useful':>10s}  {'Share':>7s}")
for tech, v in annual_energy.items():
    label = RE_COSTS[tech]["label"] if tech in RE_COSTS else GAS["label"]
    share = v["useful"] / total_demand * 100
    print(f"{label:20s}  {v['produced']:>10,.1f}  {v['direct']:>10,.1f}  "
          f"{v['delivered_from_bat']:>10,.1f}  {v['loss_charge']:>10,.1f}  "
          f"{v['loss_discharge']:>10,.1f}  {v['curtailed']:>10,.1f}  "
          f"{v['useful']:>10,.1f}  {share:>6.1f}%")


# %% [markdown]
# ---
# ## Module 4 — Cost computation
#
# ### RE costs
# Equivalent annual cost (EAC) per kW, accounting for degradation and
# both fixed and variable OPEX.
#
# ### Battery costs
# Annual cost = 6.8 % of CAPEX (includes capital, O&M, degradation
# replacement).  LCOS = annual cost ÷ energy delivered.
#
# ### Gas costs
# Fixed component (CAPEX annuity + fixed OPEX) sized to peak gas demand,
# plus variable component (fuel, CO₂, variable OPEX) proportional to
# generation.

# %%
def capital_recovery_factor(wacc: float, lifetime: int) -> float:
    if wacc == 0:
        return 1.0 / lifetime
    return wacc * (1 + wacc) ** lifetime / ((1 + wacc) ** lifetime - 1)


def re_equivalent_annual_cost(p: dict, cf: float) -> float:
    """Equivalent annual cost per kW for a RE technology (EUR/kW/yr)."""
    years = np.arange(1, p["lifetime"] + 1)
    discount = (1 + p["wacc_real"]) ** years
    annual_gen = cf * HOURS_PER_YEAR * (1 - p["degradation"]) ** (years - 1)
    annual_opex = p["opex_fix"] + p["opex_var"] * annual_gen
    total_discounted_cost = p["capex_mid"] + np.sum(annual_opex / discount)
    annuity_factor = np.sum(1.0 / discount)
    return total_discounted_cost / annuity_factor


# %%
cost_rows = []

# --- RE costs ---
for tech, p in RE_COSTS.items():
    cap_mw = INSTALLED_CAP[tech]
    cf = float(hourly_cf[tech].mean())
    produced = annual_energy[tech]["produced"]
    useful = annual_energy[tech]["useful"]
    curt_frac = 1 - useful / produced if produced > 0 else 0

    eac_per_kw = re_equivalent_annual_cost(p, cf)
    annual_cost = eac_per_kw * cap_mw * 1000  # EUR/yr (cap_mw in MW, eac in EUR/kW)

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
bat_annual_cost = BAT["annuity_rate"] * BAT["capex_kwh"] * BAT_CAPACITY_MWH * 1000
bat_lcos = bat_annual_cost / (bat_delivered * 1000) if bat_delivered > 0 else float("inf")

print("Battery cost:")
print(f"  CAPEX:       {BAT['capex_kwh']} EUR/kWh × {BAT_CAPACITY_MWH * 1000:.0f} kWh"
      f" = {BAT['capex_kwh'] * BAT_CAPACITY_MWH * 1000:,.0f} EUR")
print(f"  Annuity rate:  {BAT['annuity_rate']:.1%}")
print(f"  Annual cost:   {bat_annual_cost:>10,.0f} EUR/yr")
print(f"  Delivered:     {bat_delivered:>10,.1f} MWh/yr")
print(f"  LCOS:          {bat_lcos * 100:>10.2f} ct/kWh")

cost_rows.append({
    "technology": BAT["label"],
    "tech_key": "battery",
    "installed_mw": BAT_POWER_MW,
    "useful_mwh": bat_delivered,
    "curtailment_pct": 0.0,
    "lcoe_ct": bat_lcos * 100,
    "lcoe_no_curt_ct": bat_lcos * 100,
    "annual_cost_eur": bat_annual_cost,
})

# --- Gas costs ---
g = GAS
gas_capacity_mw = sim["residual_load"].max()
gas_cf = (gas_energy / (gas_capacity_mw * HOURS_PER_YEAR)
          if gas_capacity_mw > 0 else 0)

crf = capital_recovery_factor(g["wacc_real"], g["lifetime"])
gas_fixed_per_kw = g["capex_mid"] * crf + g["opex_fix"]

fuel_per_kwh = g["gas_price"] / 1000 / g["efficiency"]
co2_per_kwh = g["co2_intensity"] * g["co2_price"] / 1000 / g["efficiency"]
gas_marginal = fuel_per_kwh + co2_per_kwh + g["opex_var"]

gas_total_fixed = gas_fixed_per_kw * gas_capacity_mw * 1000
gas_total_variable = gas_marginal * gas_energy * 1000
gas_annual_cost = gas_total_fixed + gas_total_variable
gas_lcoe = gas_annual_cost / (gas_energy * 1000) if gas_energy > 0 else 0

print(f"\nGas cost:")
print(f"  Required capacity: {gas_capacity_mw:.4f} MW  ({gas_capacity_mw / DEMAND_MW:.1%} of demand)")
print(f"  Capacity factor:   {gas_cf:.4f}  ({gas_cf * HOURS_PER_YEAR:,.0f} FLH)")
print(f"  Marginal cost:     {gas_marginal * 100:.2f} ct/kWh_el")

cost_rows.append({
    "technology": g["label"],
    "tech_key": "gas",
    "installed_mw": gas_capacity_mw,
    "useful_mwh": gas_energy,
    "curtailment_pct": 0.0,
    "lcoe_ct": gas_lcoe * 100,
    "lcoe_no_curt_ct": gas_lcoe * 100,
    "annual_cost_eur": gas_annual_cost,
})


# %% [markdown]
# ---
# ## Module 5 — System LCOE and per-source breakdown

# %%
cost_df = pd.DataFrame(cost_rows)

total_annual_cost = cost_df["annual_cost_eur"].sum()
system_lcoe = total_annual_cost / (total_demand * 1000)

print("Cost summary:")
pd.set_option("display.float_format", lambda v: f"{v:.2f}")
print(cost_df[["technology", "installed_mw", "useful_mwh", "curtailment_pct",
               "lcoe_no_curt_ct", "lcoe_ct", "annual_cost_eur"]].to_string(index=False))
print(f"\nTotal annual cost:  {total_annual_cost:>12,.0f} EUR")
print(f"System LCOE:        {system_lcoe * 100:>12.2f} ct/kWh")

# %%
print("\nCost attribution to system LCOE:")
print(f"  {'Source':20s}  {'Share':>7s}  {'Own LCOE':>10s}  {'→ System':>10s}")
attr_sum = 0
for _, row in cost_df.iterrows():
    attr_ct = row["annual_cost_eur"] / (total_demand * 1000) * 100
    attr_sum += attr_ct
    share = row["useful_mwh"] / total_demand if row["tech_key"] != "battery" else 0
    print(f"  {row['technology']:20s}  {share:>6.1%}  {row['lcoe_ct']:>8.2f} ct  "
          f"  {attr_ct:>8.2f} ct")
print(f"  {'-' * 55}")
print(f"  {'System LCOE':20s}  {'':>7s}  {'':>10s}    {attr_sum:>8.2f} ct")

# %%
total_re_produced = sum(annual_energy[t]["produced"] for t in re_techs)
total_curtailed = sim["curtailment"].sum() / N_YEARS
total_loss_in = sim["loss_in"].sum() / N_YEARS
total_loss_out = sim["loss_out"].sum() / N_YEARS

print(f"\nEnergy losses:")
print(f"  RE produced:    {total_re_produced:>10,.1f} MWh/yr")
print(f"  Curtailed:      {total_curtailed:>10,.1f} MWh/yr  ({total_curtailed / total_re_produced:.1%})")
print(f"  Charge loss:    {total_loss_in:>10,.1f} MWh/yr")
print(f"  Discharge loss: {total_loss_out:>10,.1f} MWh/yr")
print(f"  Total RT loss:  {total_loss_in + total_loss_out:>10,.1f} MWh/yr")


# %% [markdown]
# ---
# ## Module 6 — Drawdown analysis
#
# Cumulative RE balance tracks `cumsum(RE_raw − demand)`.  Gas shortfall
# episodes are contiguous periods where the battery cannot cover the
# deficit and gas backup is needed.

# %%
re_balance = pd.Series(total_gen - DEMAND_MW, index=hourly_cf.index).cumsum()
running_max = re_balance.cummax()
drawdown = re_balance - running_max

trough_time = drawdown.idxmin()
trough_val = re_balance[trough_time]
peak_val = running_max[trough_time]
peak_time = re_balance[:trough_time].idxmax()
magnitude = peak_val - trough_val
duration_hours = (trough_time - peak_time) / pd.Timedelta("1h")

print("Maximum cumulative RE drawdown (structural, before battery):")
print(f"  Peak:      {peak_time}  ({peak_val:+,.1f} MWh)")
print(f"  Trough:    {trough_time}  ({trough_val:+,.1f} MWh)")
print(f"  Magnitude: {magnitude:,.1f} MWh  ({duration_hours / 24:.1f} days)")


# %%
def find_top_gas_episodes(gas_arr: np.ndarray, index: pd.DatetimeIndex,
                          n: int = 5) -> pd.DataFrame:
    is_gas = gas_arr > 1e-9
    ep_start = is_gas & ~np.roll(is_gas, 1)
    ep_start[0] = is_gas[0]
    ep_end = ~is_gas & np.roll(is_gas, 1)
    ep_end[0] = False

    starts = np.where(ep_start)[0].tolist()
    ends = np.where(ep_end)[0].tolist()
    if is_gas[-1] and (len(ends) == 0 or ends[-1] <= starts[-1]):
        ends.append(len(gas_arr))

    rows = []
    for s, e in zip(starts, ends):
        rows.append({
            "start_time": index[s],
            "end_time": index[min(e - 1, len(index) - 1)],
            "cumulative_gas_mwh": gas_arr[s:e].sum(),
            "peak_gas_mw": gas_arr[s:e].max(),
            "duration_hours": e - s,
            "duration_days": (e - s) / 24,
        })

    return (pd.DataFrame(rows)
            .sort_values("cumulative_gas_mwh", ascending=False)
            .head(n).reset_index(drop=True))


top_gas = find_top_gas_episodes(sim["residual_load"], hourly_cf.index)

print(f"\nTop-5 gas shortfall episodes:")
for i, row in top_gas.iterrows():
    print(f"  #{i+1}  {row['start_time'].strftime('%d %b %Y')} → "
          f"{row['end_time'].strftime('%d %b %Y')}  "
          f"gas {row['cumulative_gas_mwh']:,.1f} MWh  "
          f"peak {row['peak_gas_mw']:.3f} MW  "
          f"duration {row['duration_days']:.1f} d")

# %% [markdown]
# ---
# ## Visualisations
#
# ### Dispatch balance — 14-day window
#
# Stacked area chart showing grid-side energy flows.  At every hour,
# positive and negative areas are symmetric (equal magnitude):
#
# - **Above zero (supply):** RE generation per source, battery discharge
#   (net to grid), gas.
# - **Below zero (consumption):** demand, battery charging (grid-side),
#   curtailment.

# %%
# Build a dispatch DataFrame for plotting
dispatch = pd.DataFrame(index=hourly_cf.index)
for tech in re_techs:
    dispatch[tech] = gen_arrays[tech]
dispatch["bat_discharge"] = bat_discharge_net
dispatch["gas"] = sim["residual_load"]
dispatch["demand"] = DEMAND_MW
dispatch["bat_charge"] = bat_charge_grid
dispatch["curtailment"] = sim["curtailment"]
dispatch["soc"] = sim["soc"]

# Pick 14-day window around the worst gas episode
ep_start = top_gas.loc[0, "start_time"]
window_start = ep_start - pd.Timedelta(days=1)
window_end = window_start + pd.Timedelta(days=14)
dw = dispatch.loc[window_start:window_end].copy()

# Sanity: supply = consumption at every hour
pos = sum(dw[t] for t in re_techs) + dw["bat_discharge"] + dw["gas"]
neg = dw["demand"] + dw["bat_charge"] + dw["curtailment"]
assert (pos - neg).abs().max() < 1e-9, "Dispatch balance broken"

fig, ax = plt.subplots(figsize=(16, 6))

hours = dw.index

# --- Positive side: supply ---
pos_layers = [
    ("solar_pv_utility", RE_COSTS["solar_pv_utility"]["label"], TECH_COLORS["solar_pv_utility"]),
    ("wind_onshore", RE_COSTS["wind_onshore"]["label"], TECH_COLORS["wind_onshore"]),
    ("wind_offshore", RE_COSTS["wind_offshore"]["label"], TECH_COLORS["wind_offshore"]),
    ("bat_discharge", "Battery discharge", TECH_COLORS["battery"]),
    ("gas", "Gas (CCGT)", TECH_COLORS["gas"]),
]
pos_bottom = np.zeros(len(dw))
for col, label, color in pos_layers:
    vals = dw[col].values
    ax.fill_between(hours, pos_bottom, pos_bottom + vals,
                    color=color, alpha=0.85, label=label, linewidth=0)
    pos_bottom += vals

# --- Negative side: consumption ---
neg_layers = [
    ("demand", "Demand", "#333333"),
    ("bat_charge", "Battery charging", TECH_COLORS["battery"]),
    ("curtailment", "Curtailment", TECH_COLORS["curtailment"]),
]
neg_bottom = np.zeros(len(dw))
for col, label, color in neg_layers:
    vals = dw[col].values
    ax.fill_between(hours, -neg_bottom, -(neg_bottom + vals),
                    color=color, alpha=0.4, label=label, linewidth=0)
    neg_bottom += vals

ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Power [MW]")
ax.set_title(f"Hourly dispatch balance — {window_start.strftime('%d %b')} – "
             f"{window_end.strftime('%d %b %Y')}")
ax.legend(loc="upper left", fontsize=8, ncol=2)

fig.tight_layout()
fig.savefig(paths.images_path / "54_dispatch_balance.png", dpi=150,
            bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/54_dispatch_balance.png
# :name: fig-54-dispatch-balance
# Hourly dispatch during a 14-day window around the worst gas shortfall.
# Above zero: supply (RE generation, battery discharge, gas).
# Below zero: consumption (demand, battery charging, curtailment).
# The areas balance at every hour.
# ```

# %% [markdown]
# ### Battery state of charge — time series and distribution

# %%
soc_pct = sim["soc"] / BAT_CAPACITY_MWH * 100  # 0–100 %
soc_zoom_pct = dw["soc"] / BAT_CAPACITY_MWH * 100

fig_soc, (ax_soc_full, ax_soc_zoom, ax_soc_hist) = plt.subplots(
    3, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [2, 2, 1.5]}
)

# --- Full simulation SOC ---
ax_soc_full.fill_between(hourly_cf.index, soc_pct,
                          color=TECH_COLORS["battery"], alpha=0.5, linewidth=0)
ax_soc_full.set_ylabel("SOC [%]")
ax_soc_full.set_ylim(0, 110)
ax_soc_full.set_title("Battery state of charge — full simulation period")
ax_soc_full.axhline(100, color=TECH_COLORS["battery"], linewidth=0.8, linestyle="--", alpha=0.4)
ax_soc_full.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_soc_full.set_axisbelow(True)

# --- Zoom: 14-day window around worst gas episode ---
gas_mask_zoom = dw["gas"] > 1e-9
ax_soc_zoom.fill_between(dw.index, 0, 110, where=gas_mask_zoom.values,
                          color=TECH_COLORS["gas"], alpha=0.12, label="Gas burning hours")
ax_soc_zoom.fill_between(dw.index, soc_zoom_pct,
                          color=TECH_COLORS["battery"], alpha=0.6, linewidth=0)
ax_soc_zoom.set_ylabel("SOC [%]")
ax_soc_zoom.set_ylim(0, 110)
ax_soc_zoom.set_title(
    f"Zoom: 14-day window around worst gas episode "
    f"({window_start.strftime('%d %b')}–{window_end.strftime('%d %b %Y')})"
)
ax_soc_zoom.axhline(100, color=TECH_COLORS["battery"], linewidth=0.8, linestyle="--", alpha=0.4)
ax_soc_zoom.legend(fontsize=8, loc="upper right")
ax_soc_zoom.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_soc_zoom.set_axisbelow(True)

# --- SOC histogram ---
freq_weights = np.ones(len(soc_pct)) / len(soc_pct) * 100
n_hist, _, _ = ax_soc_hist.hist(
    soc_pct, bins=50, weights=freq_weights,
    color=TECH_COLORS["battery"], alpha=0.7, edgecolor="none"
)
ax_soc_hist.set_xlabel("State of charge [%]")
ax_soc_hist.set_ylabel("Frequency [% of hours]")
ax_soc_hist.set_title("SOC distribution")
ax_soc_hist.set_xlim(0, 100)
ax_soc_hist.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_soc_hist.set_axisbelow(True)

pct_full = (soc_pct >= 99.9).mean() * 100
pct_empty = (soc_pct <= 0.1).mean() * 100
ymax_hist = n_hist.max()
ax_soc_hist.text(1, ymax_hist * 0.85, f"Empty: {pct_empty:.1f}% of hours",
                  fontsize=8, color=TECH_COLORS["gas"])
ax_soc_hist.text(70, ymax_hist * 0.85, f"Full: {pct_full:.1f}% of hours",
                  fontsize=8, color=TECH_COLORS["battery"])

fig_soc.tight_layout()
fig_soc.savefig(paths.images_path / "54_soc.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/54_soc.png
# :name: fig-54-soc
# Top: battery SOC (% of capacity) over the full simulation period.
# Middle: 14-day zoom around the worst gas episode; grey shading marks hours
# where the battery is depleted and gas must run.
# Bottom: SOC distribution — peaks near 0 % (empty) or 100 % (full) indicate
# the battery is chronically undersized or oversized for the task.
# ```

# %% [markdown]
# ### Annualised system costs — stacked bar chart

# %%
fig_cost, ax_cost = plt.subplots(figsize=(8, 5))

# Individual bars for each source + one stacked "System" bar
all_techs = ["solar_pv_utility", "wind_onshore", "wind_offshore", "battery", "gas"]
all_labels = [cost_df.loc[cost_df["tech_key"] == t, "technology"].iloc[0]
              for t in all_techs]
all_colors = [TECH_COLORS[t] for t in all_techs]
all_costs = [cost_df.loc[cost_df["tech_key"] == t, "annual_cost_eur"].iloc[0]
             for t in all_techs]

x_ind = np.arange(len(all_techs))
bars = ax_cost.bar(x_ind, all_costs, color=all_colors, alpha=0.85, width=0.6)
for bar, val in zip(bars, all_costs):
    ax_cost.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total_annual_cost * 0.01,
                 f"{val:,.0f}", ha="center", va="bottom", fontsize=8)

# Stacked system bar
x_sys = len(all_techs) + 0.8
bottom_s = 0
for val, color in zip(all_costs, all_colors):
    ax_cost.bar(x_sys, val, bottom=bottom_s, color=color, width=0.6)
    if val / total_annual_cost > 0.06:
        ax_cost.text(x_sys, bottom_s + val / 2, f"{val / total_annual_cost:.0%}",
                     ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    bottom_s += val
ax_cost.text(x_sys, bottom_s + total_annual_cost * 0.01,
             f"{total_annual_cost:,.0f}", ha="center", va="bottom",
             fontsize=9, fontweight="bold")

ax_cost.set_ylabel("Annual cost [EUR / yr]")
ax_cost.set_title(f"Annualised system costs (1 MW demand, {RE_SCALING:.0f}× RE)")
ax_cost.set_xticks(list(x_ind) + [x_sys])
ax_cost.set_xticklabels(all_labels + ["System\ntotal"], fontsize=8, rotation=25, ha="right")
ax_cost.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_cost.set_axisbelow(True)
ax_cost.set_ylim(0, total_annual_cost * 1.15)

fig_cost.tight_layout()
fig_cost.savefig(paths.images_path / "54_annual_costs.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/54_annual_costs.png
# :name: fig-54-annual-costs
# Annualised cost per technology and stacked system total.
# ```

# %% [markdown]
# ### RE energy fate — produced vs used / curtailed / battery loss

# %%
fig_fate, axes_fate = plt.subplots(1, len(re_techs), figsize=(4 * len(re_techs), 5),
                                   sharey=True)

for ax_f, tech in zip(axes_fate, re_techs):
    v = annual_energy[tech]
    label = RE_COSTS[tech]["label"]

    categories = ["Useful\n(direct)", "Via\nbattery", "Battery\nloss", "Curtailed"]
    values = [v["direct"], v["delivered_from_bat"], v["total_loss"], v["curtailed"]]
    colors = [TECH_COLORS[tech], TECH_COLORS["battery"], "#ff9900", TECH_COLORS["curtailment"]]

    bottom_f = 0
    for cat, val, col in zip(categories, values, colors):
        bar = ax_f.bar(0, val, bottom=bottom_f, color=col, width=0.5, alpha=0.85)
        if val / v["produced"] > 0.04:
            ax_f.text(0, bottom_f + val / 2, f"{val:,.0f}\n({val / v['produced']:.0%})",
                      ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bottom_f += val

    ax_f.set_title(f"{label}\n({v['produced']:,.0f} MWh/yr)", fontsize=10)
    ax_f.set_xticks([0])
    ax_f.set_xticklabels([""])
    ax_f.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax_f.set_axisbelow(True)

axes_fate[0].set_ylabel("Energy [MWh / yr]")

# Shared legend
legend_items = [
    Patch(facecolor=TECH_COLORS["solar_pv_utility"], alpha=0.85, label="Direct use"),
    Patch(facecolor=TECH_COLORS["battery"], alpha=0.85, label="Via battery"),
    Patch(facecolor="#ff9900", alpha=0.85, label="Battery loss"),
    Patch(facecolor=TECH_COLORS["curtailment"], alpha=0.85, label="Curtailed"),
]
fig_fate.legend(handles=legend_items, loc="upper center", ncol=4, fontsize=9,
                bbox_to_anchor=(0.5, 1.02))

fig_fate.suptitle("RE energy fate — where does each MWh end up?", fontsize=12, y=1.08)
fig_fate.tight_layout()
fig_fate.savefig(paths.images_path / "54_energy_fate.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/54_energy_fate.png
# :name: fig-54-energy-fate
# For each RE source: stacked breakdown of produced energy into direct use,
# battery-mediated delivery, round-trip losses, and curtailed surplus.
# ```

# %% [markdown]
# ### LCOE comparison — per source, gas, and system

# %%
fig_lcoe, ax_l = plt.subplots(figsize=(10, 5))

# RE: LCOE on produced energy vs LCOE on useful energy only
re_labels_l = [RE_COSTS[t]["label"] for t in re_techs]
re_colors_l = [TECH_COLORS[t] for t in re_techs]

lcoe_produced = []
lcoe_useful = []
for tech in re_techs:
    row = cost_df.loc[cost_df["tech_key"] == tech].iloc[0]
    lcoe_produced.append(row["lcoe_no_curt_ct"])
    lcoe_useful.append(row["lcoe_ct"])

x = np.arange(len(re_techs))
w = 0.35
bars_prod = ax_l.bar(x - w / 2, lcoe_produced, width=w, color=re_colors_l, alpha=0.45,
                     label="LCOE (all produced)")
bars_useful = ax_l.bar(x + w / 2, lcoe_useful, width=w, color=re_colors_l, alpha=0.9,
                       label="LCOE (useful only)")

for bar, val in zip(bars_prod, lcoe_produced):
    ax_l.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
              f"{val:.1f}", ha="center", va="bottom", fontsize=8, alpha=0.6)
for bar, val in zip(bars_useful, lcoe_useful):
    ax_l.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
              f"{val:.1f}", ha="center", va="bottom", fontsize=8)

# Battery LCOS bar (cost per MWh delivered from storage)
x_bat_lcos = len(re_techs) + 0.5
ax_l.bar(x_bat_lcos, bat_lcos * 100, width=0.5,
         color=TECH_COLORS["battery"], alpha=0.85, label="Battery LCOS")
ax_l.text(x_bat_lcos, bat_lcos * 100 + 0.2, f"{bat_lcos * 100:.1f}",
          ha="center", va="bottom", fontsize=8)

# Gas bar
x_gas = len(re_techs) + 1.3
ax_l.bar(x_gas, gas_lcoe * 100, width=0.5, color=TECH_COLORS["gas"], alpha=0.85,
         label="Gas LCOE")
ax_l.text(x_gas, gas_lcoe * 100 + 0.2, f"{gas_lcoe * 100:.1f}",
          ha="center", va="bottom", fontsize=8)

# System LCOE bar
x_sys_l = len(re_techs) + 2.1
ax_l.bar(x_sys_l, system_lcoe * 100, width=0.5, color="black", alpha=0.25,
         label="System LCOE")
ax_l.text(x_sys_l, system_lcoe * 100 + 0.2, f"{system_lcoe * 100:.1f}",
          ha="center", va="bottom", fontsize=9, fontweight="bold")

# System LCOE reference line
ax_l.axhline(system_lcoe * 100, color="black", linewidth=1, linestyle="--", alpha=0.4)

ax_l.set_ylabel("LCOE / LCOS [ct / kWh]")
ax_l.set_title("LCOE/LCOS comparison — RE (produced vs useful), battery, gas, and system")
all_x_l = list(x) + [x_bat_lcos, x_gas, x_sys_l]
all_xlabels_l = re_labels_l + ["Battery\n(LCOS)", "Gas\n(CCGT)", "System\nLCOE"]
ax_l.set_xticks(all_x_l)
ax_l.set_xticklabels(all_xlabels_l, fontsize=9)
ax_l.legend(loc="upper left", fontsize=8)
ax_l.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_l.set_axisbelow(True)
ax_l.set_ylim(0)

fig_lcoe.tight_layout()
fig_lcoe.savefig(paths.images_path / "54_lcoe_comparison.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/54_lcoe_comparison.png
# :name: fig-54-lcoe-comparison
# LCOE per RE source on all produced energy (light) vs useful only (dark,
# penalised by curtailment and storage losses).  Battery LCOS (cost per MWh
# delivered from storage) sits alongside gas LCOE — if LCOS < gas LCOE the
# battery is the cheaper gap-filler.  System LCOE shown for reference.
# ```

# %% [markdown]
# ### Installed capacities

# %%
fig_cap, (ax_pow, ax_stor) = plt.subplots(1, 2, figsize=(11, 5),
                                           gridspec_kw={"width_ratios": [3, 1]})

# --- Left: power capacity (MW) ---
cap_techs = list(re_techs) + ["battery", "gas"]
cap_labels = [RE_COSTS[t]["label"] for t in re_techs] + [BAT["label"], GAS["label"]]
cap_colors = [TECH_COLORS[t] for t in cap_techs]
cap_values = [INSTALLED_CAP[t] for t in re_techs] + [BAT_POWER_MW, gas_capacity_mw]

bars_cap = ax_pow.bar(cap_labels, cap_values, color=cap_colors, alpha=0.85, width=0.6)
for bar, val in zip(bars_cap, cap_values):
    ax_pow.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f} MW", ha="center", va="bottom", fontsize=9)

ax_pow.set_ylabel("Power capacity [MW]")
ax_pow.set_title("Installed power capacity")
ax_pow.set_xticklabels(cap_labels, fontsize=8, rotation=25, ha="right")
ax_pow.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_pow.set_axisbelow(True)
ax_pow.set_ylim(0, max(cap_values) * 1.15)

# --- Right: storage capacity (MWh) ---
ax_stor.bar(["Battery"], [BAT_CAPACITY_MWH], color=TECH_COLORS["battery"],
            alpha=0.85, width=0.4)
ax_stor.text(0, BAT_CAPACITY_MWH + 0.05,
             f"{BAT_CAPACITY_MWH:.1f} MWh\n({BAT_DURATION_H}h @ {BAT_POWER_MW:.0f} MW)",
             ha="center", va="bottom", fontsize=9)
ax_stor.set_ylabel("Storage capacity [MWh]")
ax_stor.set_title("Storage capacity")
ax_stor.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_stor.set_axisbelow(True)
ax_stor.set_ylim(0, BAT_CAPACITY_MWH * 1.35)

fig_cap.suptitle(f"System sizing (1 MW demand, {RE_SCALING:.0f}× RE)", fontsize=12)
fig_cap.tight_layout()
fig_cap.savefig(paths.images_path / "54_installed_capacities.png", dpi=150,
                bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/54_installed_capacities.png
# :name: fig-54-installed-capacities
# Left: installed power capacity per technology (MW).  Battery shows the
# max charge/discharge rate; gas is sized to the peak residual load.
# Right: battery storage capacity (MWh).
# ```

# %% [markdown]
# ### Demand fulfilment — annual and monthly
#
# Where does each MWh of served demand come from?  Direct RE use per
# source, battery discharge, and gas.

# %%
# Build hourly demand-fulfilment DataFrame
# Direct use per RE source comes from the attribution; battery discharge
# and gas are system-level.
fulfilment = pd.DataFrame(index=hourly_cf.index)
for tech in re_techs:
    fulfilment[tech] = attr["direct"][tech]
fulfilment["battery"] = bat_discharge_net
fulfilment["gas"] = sim["residual_load"]

supply_labels = [RE_COSTS[t]["label"] for t in re_techs] + [BAT["label"], GAS["label"]]
supply_keys = list(re_techs) + ["battery", "gas"]
supply_colors = [TECH_COLORS[t] for t in supply_keys]

# --- Annual stacked bar ---
fig_ful, (ax_ann, ax_mon) = plt.subplots(1, 2, figsize=(14, 5),
                                          gridspec_kw={"width_ratios": [1, 3]})

annual_vals = np.array([fulfilment[k].sum() / N_YEARS for k in supply_keys])
annual_pct = annual_vals / annual_vals.sum() * 100
bottom_a = 0
for pct, label, color in zip(annual_pct, supply_labels, supply_colors):
    ax_ann.bar(0, pct, bottom=bottom_a, color=color, width=0.5, label=label)
    if pct > 4:
        ax_ann.text(0, bottom_a + pct / 2, f"{pct:.1f} %",
                    ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    bottom_a += pct

ax_ann.set_ylabel("Share of demand [%]")
ax_ann.set_title("Annual")
ax_ann.set_xticks([0])
ax_ann.set_xticklabels([""])
ax_ann.legend(loc="upper right", fontsize=8)
ax_ann.set_ylim(0, 100)
ax_ann.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_ann.set_axisbelow(True)

# --- Monthly stacked bars (normalised to 100 %) ---
fulfilment["month"] = fulfilment.index.month
monthly_ful = fulfilment.groupby("month")[supply_keys].sum() / N_YEARS
monthly_total = monthly_ful.sum(axis=1)
monthly_pct = monthly_ful.div(monthly_total, axis=0) * 100
months = monthly_pct.index.values
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

bottom_m = np.zeros(len(months))
for key, label, color in zip(supply_keys, supply_labels, supply_colors):
    vals = monthly_pct[key].values
    ax_mon.bar(months, vals, bottom=bottom_m, color=color, width=0.7, label=label)
    bottom_m += vals

ax_mon.set_xticks(months)
ax_mon.set_xticklabels(month_names)
ax_mon.set_ylabel("Share of demand [%]")
ax_mon.set_ylim(0, 100)
ax_mon.set_title(f"Monthly ({SIM_YEARS[0]}–{SIM_YEARS[-1]} avg)")
ax_mon.legend(loc="upper right", fontsize=8)
ax_mon.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_mon.set_axisbelow(True)

fig_ful.suptitle("Demand fulfilment by source", fontsize=12)
fig_ful.tight_layout()
fig_ful.savefig(paths.images_path / "54_demand_fulfilment.png", dpi=150,
                bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/54_demand_fulfilment.png
# :name: fig-54-demand-fulfilment
# Left: annual demand fulfilment by source.  Right: monthly breakdown
# showing seasonal shifts — solar dominates summer, wind and gas cover
# winter demand.
# ```

# %% [markdown]
# ### Battery monthly charge/discharge throughput

# %%
dispatch["month"] = dispatch.index.month
monthly_charge = dispatch.groupby("month")["bat_charge"].sum() / N_YEARS
monthly_discharge = dispatch.groupby("month")["bat_discharge"].sum() / N_YEARS

fig_tp, ax_tp = plt.subplots(figsize=(10, 4))

x_months = np.arange(1, 13)
w = 0.35
ax_tp.bar(x_months - w / 2, monthly_charge.values, width=w,
          color=TECH_COLORS["battery"], alpha=0.45, label="Charged (grid → battery)")
ax_tp.bar(x_months + w / 2, monthly_discharge.values, width=w,
          color=TECH_COLORS["battery"], alpha=0.9, label="Discharged (battery → grid)")

ax_tp.set_xticks(x_months)
ax_tp.set_xticklabels(month_names)
ax_tp.set_ylabel("Energy [MWh / month]")
ax_tp.set_title(f"Battery monthly throughput ({SIM_YEARS[0]}–{SIM_YEARS[-1]} avg)")
ax_tp.legend(fontsize=9)
ax_tp.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_tp.set_axisbelow(True)

fig_tp.tight_layout()
fig_tp.savefig(paths.images_path / "54_battery_throughput.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/54_battery_throughput.png
# :name: fig-54-battery-throughput
# Monthly battery throughput: energy charged (light) vs discharged (dark).
# Roughly symmetric months indicate daily solar arbitrage; months where
# one direction dominates signal multi-day or seasonal energy shifting.
# ```

# %% [markdown]
# ### RE utilisation overview

# %%
re_produced = sum(annual_energy[t]["produced"] for t in re_techs)
re_direct = sum(annual_energy[t]["direct"] for t in re_techs)
re_useful = sum(annual_energy[t]["useful"] for t in re_techs)  # direct + via battery

bar_labels = ["Demand", "RE\nproduced", "RE direct\nuse", "RE direct\n+ battery"]
bar_vals = [total_demand, re_produced, re_direct, re_useful]
bar_colors = ["#333333", "#2ca02c", "#4a90d9", "#1a5fa8"]

fig_util, ax_u = plt.subplots(figsize=(8, 5))
bars_u = ax_u.bar(bar_labels, bar_vals, color=bar_colors, alpha=0.85, width=0.55)
for bar, val in zip(bars_u, bar_vals):
    ax_u.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total_demand * 0.01,
              f"{val:,.0f} MWh", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Reference line at demand level
ax_u.axhline(total_demand, color="#333333", linewidth=1, linestyle="--", alpha=0.4)

# Annotate gaps
ax_u.annotate(f"curtailed + losses\n{re_produced - re_useful:,.0f} MWh "
              f"({(re_produced - re_useful) / re_produced:.0%})",
              xy=(1.5, (re_produced + re_useful) / 2),
              ha="center", fontsize=8, color="#cc4444")
ax_u.annotate(f"gap filled by gas\n{total_demand - re_useful:,.0f} MWh "
              f"({(total_demand - re_useful) / total_demand:.0%})",
              xy=(3, (total_demand + re_useful) / 2),
              ha="center", fontsize=8, color=TECH_COLORS["gas"])

ax_u.set_ylabel("Energy [MWh / yr]")
ax_u.set_title(f"RE utilisation overview ({RE_SCALING:.0f}× RE, {BAT_CAPACITY_MWH:.0f} MWh battery)")
ax_u.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_u.set_axisbelow(True)
ax_u.set_ylim(0, max(bar_vals) * 1.12)

fig_util.tight_layout()
fig_util.savefig(paths.images_path / "54_re_utilisation.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/54_re_utilisation.png
# :name: fig-54-re-utilisation
# Comparison of annual demand against total RE production, RE used
# directly, and RE used after battery storage.  The gap between RE
# produced and RE useful is curtailment + RT losses; the gap between
# RE useful and demand is filled by gas.
# ```

# %% [markdown]
# ### System drawdown — RE + battery vs demand
#
# The hourly system balance is
# `(RE_gen + bat_discharge − bat_charge) − demand = curtailment − gas`.
# Its cumulative sum rises during surplus hours (curtailment) and falls
# during deficit hours (gas usage).  Drawdown from the running peak
# quantifies the worst sustained gas-dependent episodes.

# %%
# Hourly system balance: positive when curtailing, negative when burning gas
sys_balance_hourly = sim["curtailment"] - sim["residual_load"]
sys_balance = pd.Series(sys_balance_hourly, index=hourly_cf.index).cumsum()

running_max_sb = sys_balance.cummax()
sys_drawdown = sys_balance - running_max_sb  # ≤ 0


def find_top_drawdowns(balance: pd.Series, n: int = 5) -> pd.DataFrame:
    """Return top-N drawdown episodes ranked by magnitude.

    Each episode is a contiguous period where the balance stays below its
    running maximum.
    """
    running_max = balance.cummax()
    dd = balance - running_max

    in_dd = dd < 0
    ep_start = in_dd & ~in_dd.shift(1, fill_value=False)
    ep_end = ~in_dd & in_dd.shift(1, fill_value=False)

    starts = balance.index[ep_start].tolist()
    ends = balance.index[ep_end].tolist()
    if len(starts) > len(ends):
        ends.append(balance.index[-1])

    rows = []
    for s, e in zip(starts, ends):
        segment = dd[s:e]
        trough_t = segment.idxmin()
        peak_val = running_max[s]
        trough_val = balance[trough_t]
        magnitude = peak_val - trough_val

        # Actual peak: last time the running-max level was set
        candidates = balance[:s][balance[:s] >= peak_val]
        peak_t = candidates.index[-1] if len(candidates) else s

        dur_h = (trough_t - peak_t) / pd.Timedelta("1h")
        rows.append({
            "peak_time": peak_t,
            "trough_time": trough_t,
            "recovery_time": e,
            "magnitude_mwh": magnitude,
            "peak_to_trough_hours": dur_h,
            "peak_to_trough_days": dur_h / 24,
        })

    return (pd.DataFrame(rows)
            .sort_values("magnitude_mwh", ascending=False)
            .head(n).reset_index(drop=True))


top_sys_dd = find_top_drawdowns(sys_balance)

print("Top-5 system drawdown episodes (RE + battery vs demand):")
for i, row in top_sys_dd.iterrows():
    print(f"  #{i+1}  {row['peak_time'].strftime('%d %b %Y')} → "
          f"{row['trough_time'].strftime('%d %b %Y')}  "
          f"depth {row['magnitude_mwh']:,.1f} MWh  "
          f"duration {row['peak_to_trough_days']:.1f} d")

# %%
fig_sdd, (ax_bal_s, ax_dd_s) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# --- Top: cumulative system balance ---
ax_bal_s.plot(sys_balance.index, sys_balance.values, color="#2ca02c", linewidth=0.7)
ax_bal_s.fill_between(sys_balance.index, sys_balance.values, 0,
                      where=(sys_balance.values >= 0), color="#2ca02c", alpha=0.15)
ax_bal_s.fill_between(sys_balance.index, sys_balance.values, 0,
                      where=(sys_balance.values < 0), color="tomato", alpha=0.15)
ax_bal_s.axhline(0, color="black", linewidth=0.6, linestyle="--")

# Mark worst drawdown
worst = top_sys_dd.iloc[0]
ax_bal_s.axvline(worst["peak_time"], color="#2ca02c", linewidth=1, linestyle=":", alpha=0.8)
ax_bal_s.axvline(worst["trough_time"], color="tomato", linewidth=1, linestyle=":", alpha=0.8)
ax_bal_s.annotate(f"Peak\n{worst['peak_time'].strftime('%d %b %Y')}",
                  xy=(worst["peak_time"], sys_balance[worst["peak_time"]]),
                  xytext=(10, 6), textcoords="offset points", fontsize=8, color="#2ca02c")
ax_bal_s.annotate(f"Trough\n{worst['trough_time'].strftime('%d %b %Y')}\n"
                  f"−{worst['magnitude_mwh']:,.0f} MWh ({worst['peak_to_trough_days']:.0f} d)",
                  xy=(worst["trough_time"], sys_balance[worst["trough_time"]]),
                  xytext=(10, -30), textcoords="offset points", fontsize=8, color="tomato")

ax_bal_s.set_ylabel("Cumulative balance [MWh]")
ax_bal_s.set_title(f"Cumulative system balance: (RE + battery) − demand "
                   f"({SIM_YEARS[0]}–{SIM_YEARS[-1]})")
ax_bal_s.yaxis.grid(True, linewidth=0.4, alpha=0.5)
ax_bal_s.set_axisbelow(True)

# --- Bottom: drawdown from running peak ---
ax_dd_s.fill_between(sys_drawdown.index, sys_drawdown.values, 0,
                     color="#4a90d9", alpha=0.4, linewidth=0)
ax_dd_s.plot(sys_drawdown.index, sys_drawdown.values, color="#4a90d9", linewidth=0.6)
ax_dd_s.axhline(0, color="black", linewidth=0.6, linestyle="--")

# Mark top-5 troughs
for i, row in top_sys_dd.iterrows():
    tt = row["trough_time"]
    dd_val = sys_drawdown[tt]
    color_dot = "tomato" if i == 0 else "#4a90d9"
    ax_dd_s.scatter([tt], [dd_val], color=color_dot, zorder=5, s=30)
    ax_dd_s.annotate(f"#{i+1} −{row['magnitude_mwh']:,.0f} MWh\n"
                     f"{row['peak_to_trough_days']:.0f} d",
                     xy=(tt, dd_val), xytext=(8, -6), textcoords="offset points",
                     fontsize=7, color=color_dot, va="top")

ax_dd_s.set_ylabel("Drawdown [MWh]")
ax_dd_s.set_xlabel("Date")
ax_dd_s.set_title("System drawdown from running peak — "
                  "depth = cumulative gas needed in episode")
ax_dd_s.yaxis.grid(True, linewidth=0.4, alpha=0.5)
ax_dd_s.set_axisbelow(True)

fig_sdd.tight_layout()
fig_sdd.savefig(paths.images_path / "54_system_drawdown.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/54_system_drawdown.png
# :name: fig-54-system-drawdown
# Top: cumulative balance of the RE + battery system vs demand.  Green
# shading = cumulative surplus (curtailment); red = cumulative deficit
# (gas usage).  Bottom: drawdown from running peak with the top-5
# episodes marked.  Each drawdown's depth equals the cumulative gas
# energy consumed during the episode.
# ```

# %% [markdown]
# ### Top-5 system drawdowns — depth and duration

# %%
fig_ep, (ax_ep_mag, ax_ep_dur) = plt.subplots(1, 2, figsize=(14, 4.5))

ep_labels = [
    f"#{i+1}  {r['peak_time'].strftime('%b %Y')} → {r['trough_time'].strftime('%b %Y')}"
    for i, (_, r) in enumerate(top_sys_dd.iterrows())
]
y_ep = range(len(top_sys_dd))

# Magnitude
ax_ep_mag.barh(list(y_ep), top_sys_dd["magnitude_mwh"], color="#4a90d9",
               edgecolor="white", linewidth=0.5)
ax_ep_mag.set_yticks(list(y_ep))
ax_ep_mag.set_yticklabels(ep_labels, fontsize=8)
ax_ep_mag.invert_yaxis()
ax_ep_mag.set_xlabel("Cumulative gas [MWh]")
ax_ep_mag.set_title("Drawdown depth")
ax_ep_mag.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_ep_mag.set_axisbelow(True)
for yi, v in zip(y_ep, top_sys_dd["magnitude_mwh"]):
    ax_ep_mag.text(v + top_sys_dd["magnitude_mwh"].max() * 0.01, yi,
                   f"{v:,.1f}", va="center", fontsize=8)

# Duration
ax_ep_dur.barh(list(y_ep), top_sys_dd["peak_to_trough_days"], color="#4a90d9",
               edgecolor="white", linewidth=0.5, alpha=0.7)
ax_ep_dur.set_yticks(list(y_ep))
ax_ep_dur.set_yticklabels(ep_labels, fontsize=8)
ax_ep_dur.invert_yaxis()
ax_ep_dur.set_xlabel("Peak → trough [days]")
ax_ep_dur.set_title("Drawdown duration")
ax_ep_dur.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_ep_dur.set_axisbelow(True)
for yi, v in zip(y_ep, top_sys_dd["peak_to_trough_days"]):
    ax_ep_dur.text(v + top_sys_dd["peak_to_trough_days"].max() * 0.01, yi,
                   f"{v:.1f} d", va="center", fontsize=8)

fig_ep.suptitle(f"Top-5 system drawdown episodes — RE + battery vs demand "
                f"({SIM_YEARS[0]}–{SIM_YEARS[-1]}, {RE_SCALING:.0f}× RE, "
                f"{BAT_CAPACITY_MWH:.0f} MWh battery)", fontsize=11)
fig_ep.tight_layout()
fig_ep.savefig(paths.images_path / "54_system_drawdown_episodes.png", dpi=150,
               bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/54_system_drawdown_episodes.png
# :name: fig-54-system-drawdown-episodes
# Top-5 system drawdown episodes ranked by cumulative gas consumption.
# Left: depth (total gas energy in the episode).  Right: duration from
# peak to trough.
# ```

# %%

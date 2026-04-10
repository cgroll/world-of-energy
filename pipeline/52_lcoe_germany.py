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
# # LCOE for Solar PV, Onshore and Offshore Wind in Germany
#
# Combines long-run capacity factors from the Pan-European Climate Database
# (PECD ERA5, NUTS 0) with techno-economic assumptions from the Fraunhofer ISE
# study *Levelized Cost of Electricity — Renewable Energy Technologies*
# (July 2024, `data/EN2024_ISE_LCOE.pdf`) to compute a real-terms LCOE for
# Germany as a whole.
#
# This is a NUTS 0 starting point; sub-national granularity (NUTS 1 / NUTS 2)
# can be added later by extending the PECD download step.
#
# ## Method
#
# The standard discounted LCOE, computed per 1 kW of installed capacity:
#
# $$\text{LCOE} = \frac{I_0 + \sum_{t=1}^{N} \dfrac{A_t}{(1+r)^t}}
#                      {\sum_{t=1}^{N} \dfrac{E_t}{(1+r)^t}}$$
#
# with
#
# - $I_0$ — specific investment (CAPEX), EUR / kW
# - $N$ — economic lifetime in years
# - $r$ — real weighted average cost of capital (WACC)
# - $E_t = \text{CF} \cdot 8760 \cdot (1-d)^{\,t-1}$ — annual generation
#   per kW in year $t$, with annual degradation $d$
# - $A_t = \text{OPEX}_\text{fix} + \text{OPEX}_\text{var} \cdot E_t$ —
#   annual operating cost per kW in year $t$
#
# All costs and the WACC are in **real** 2024 terms (Fraunhofer applies an
# assumed 1.8 %/yr inflation rate to convert the nominal WACC to real).
#
# ## Cost assumptions
#
# Extracted from Tables 1 and 2 of the Fraunhofer ISE LCOE 2024 study
# (pages 11–13). CAPEX midpoints are used (low/high from Table 1); all other
# values are the single point estimates in Table 2.

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


# %%
# Fraunhofer ISE LCOE 2024 — Table 1 (CAPEX) + Table 2 (financials, OPEX, degradation)
# PV reference: "PV utility-scale" (>1 MWp), closest match to open-field PECD CF assumption.
COSTS = {
    "solar_pv_utility": {
        "label": "Solar PV (utility-scale)",
        "capex_eur_per_kw_low":  700,   # Table 1
        "capex_eur_per_kw_high": 900,   # Table 1
        "lifetime_years":   30,         # Table 2
        "wacc_real":        0.035,      # Table 2 (3.5%)
        "opex_fix_eur_per_kw_yr": 13.3, # Table 2
        "opex_var_eur_per_kwh":   0.0,  # Table 2
        "degradation_per_yr":     0.0025,  # Table 2 (0.25 %/yr)
    },
    "wind_onshore": {
        "label": "Wind onshore",
        "capex_eur_per_kw_low":  1300,  # Table 1
        "capex_eur_per_kw_high": 1900,  # Table 1
        "lifetime_years":   25,         # Table 2
        "wacc_real":        0.039,      # Table 2 (3.9%)
        "opex_fix_eur_per_kw_yr": 32.0, # Table 2
        "opex_var_eur_per_kwh":   0.007,
        "degradation_per_yr":     0.0,
    },
    "wind_offshore": {
        "label": "Wind offshore",
        "capex_eur_per_kw_low":  2200,  # Table 1
        "capex_eur_per_kw_high": 3400,  # Table 1
        "lifetime_years":   25,         # Table 2
        "wacc_real":        0.060,      # Table 2 (6.0%)
        "opex_fix_eur_per_kw_yr": 39.0, # Table 2
        "opex_var_eur_per_kwh":   0.008,
        "degradation_per_yr":     0.0,
    },
}

PECD_VARIABLES = {
    "solar_pv_utility": "solar_photovoltaic_power_generation",
    "wind_onshore":     "wind_power_generation_onshore",
    "wind_offshore":    "wind_power_generation_offshore",
}

COUNTRY = "DE"
HOURS_PER_YEAR = 8760

TECH_COLORS = {
    "solar_pv_utility": "#f4b942",
    "wind_onshore":     "#4a90d9",
    "wind_offshore":    "#1a5fa8",
}


# %%
def compute_lcoe(
    cf: float,
    capex: float,
    opex_fix: float,
    opex_var: float,
    lifetime: int,
    wacc: float,
    degradation: float,
) -> float:
    """Discounted LCOE in EUR/kWh, per 1 kW of installed capacity.

    Parameters
    ----------
    cf : float
        Long-run mean capacity factor (0–1).
    capex : float
        Specific investment in EUR/kW (I_0).
    opex_fix : float
        Fixed OPEX in EUR/kW/yr.
    opex_var : float
        Variable OPEX in EUR/kWh generated.
    lifetime : int
        Economic lifetime in years (N).
    wacc : float
        Real weighted average cost of capital (e.g. 0.035 for 3.5 %).
    degradation : float
        Annual output degradation (e.g. 0.0025 for 0.25 %/yr).
    """
    years = np.arange(1, lifetime + 1)
    discount = (1 + wacc) ** years
    annual_gen = cf * HOURS_PER_YEAR * (1 - degradation) ** (years - 1)  # kWh / kW
    annual_opex = opex_fix + opex_var * annual_gen                       # EUR / kW

    pv_gen  = np.sum(annual_gen  / discount)
    pv_cost = capex + np.sum(annual_opex / discount)
    return pv_cost / pv_gen


# %% [markdown]
# ## Replication check: Fraunhofer's own LCOE endpoints
#
# Before we mix PECD capacity factors into the calculation, verify that the
# `compute_lcoe` implementation above can reproduce the Fraunhofer ISE 2024
# headline LCOE ranges (Figure 1, page 2) when fed *their own* FLH and CAPEX
# assumptions. Fraunhofer pairs the low CAPEX with the high-FLH (best-case)
# site and the high CAPEX with the low-FLH (worst-case) site, giving a
# single-axis low/high LCOE band.
#
# Published reference ranges (read off Figure 1):
#
# | Technology              | Low  | High |
# |-------------------------|-----:|-----:|
# | PV utility-scale        |  4.1 |  6.9 |
# | Wind onshore            |  4.3 |  9.2 |
# | Wind offshore           |  5.5 | 10.3 |

# %%
FRAUNHOFER_FLH = {
    "solar_pv_utility": (935,  1280),   # kWh/kWp, northern → southern Germany
    "wind_onshore":     (1800, 3200),   # inland → coastal/high-wind sites
    "wind_offshore":    (3200, 4500),   # short → very-good distance-from-coast sites
}

FRAUNHOFER_PUBLISHED_LCOE = {   # €cent/kWh, read from Figure 1
    "solar_pv_utility": (4.1, 6.9),
    "wind_onshore":     (4.3, 9.2),
    "wind_offshore":    (5.5, 10.3),
}

repl_rows = []
for tech, p in COSTS.items():
    flh_low, flh_high = FRAUNHOFER_FLH[tech]
    # best case: cheap CAPEX × high FLH
    lcoe_best = compute_lcoe(
        cf=flh_high / HOURS_PER_YEAR,
        capex=p["capex_eur_per_kw_low"],
        opex_fix=p["opex_fix_eur_per_kw_yr"],
        opex_var=p["opex_var_eur_per_kwh"],
        lifetime=p["lifetime_years"],
        wacc=p["wacc_real"],
        degradation=p["degradation_per_yr"],
    )
    # worst case: expensive CAPEX × low FLH
    lcoe_worst = compute_lcoe(
        cf=flh_low / HOURS_PER_YEAR,
        capex=p["capex_eur_per_kw_high"],
        opex_fix=p["opex_fix_eur_per_kw_yr"],
        opex_var=p["opex_var_eur_per_kwh"],
        lifetime=p["lifetime_years"],
        wacc=p["wacc_real"],
        degradation=p["degradation_per_yr"],
    )
    pub_low, pub_high = FRAUNHOFER_PUBLISHED_LCOE[tech]
    repl_rows.append({
        "technology":        p["label"],
        "lcoe_low_computed":  lcoe_best * 100,
        "lcoe_low_published": pub_low,
        "lcoe_high_computed": lcoe_worst * 100,
        "lcoe_high_published": pub_high,
        "err_low_ct":   lcoe_best * 100 - pub_low,
        "err_high_ct":  lcoe_worst * 100 - pub_high,
    })

repl_df = pd.DataFrame(repl_rows)
print("Replication of Fraunhofer ISE 2024 LCOE ranges using their own FLH × CAPEX pairs:")
print(repl_df.to_string(index=False))


# %%
# Load PECD capacity factors and take long-run mean CF for Germany
df = pd.read_parquet(paths.pecd_processed_file)

cf_mean = {}
for tech, variable in PECD_VARIABLES.items():
    series = df[variable]["capacity_factor_ratio"][COUNTRY]
    cf_mean[tech] = float(series.mean())
    print(f"{tech:20s}  long-run mean CF (DE) = {cf_mean[tech]:.4f}  "
          f"→ {cf_mean[tech] * HOURS_PER_YEAR:,.0f} FLH/yr")


# %% [markdown]
# ## Full-load hours: PECD vs Fraunhofer assumptions
#
# Fraunhofer ISE Table 3 gives explicit FLH ranges per technology for typical
# German locations. Compare them against what the PECD NUTS 0 long-run mean
# implies for Germany as a whole.

# %%
# Fraunhofer ISE 2024 Table 3 FLH ranges are defined near the top of the script.
flh_rows = []
for tech, p in COSTS.items():
    flh_pecd = cf_mean[tech] * HOURS_PER_YEAR
    lo, hi = FRAUNHOFER_FLH[tech]
    if flh_pecd < lo:
        status = "below Fraunhofer range"
    elif flh_pecd > hi:
        status = "above Fraunhofer range"
    else:
        frac = (flh_pecd - lo) / (hi - lo)
        status = f"inside range ({frac:.0%} of the way from low→high)"
    flh_rows.append({
        "technology":      p["label"],
        "cf_mean_pecd":    cf_mean[tech],
        "flh_pecd":        flh_pecd,
        "flh_fraunhofer_low":  lo,
        "flh_fraunhofer_high": hi,
        "status":          status,
    })

flh_df = pd.DataFrame(flh_rows)
print(flh_df.to_string(index=False))

flh_out = paths.processed_data_path / "lcoe_germany_flh_comparison.parquet"
flh_df.to_parquet(flh_out, index=False)
print(f"\nSaved FLH comparison to {flh_out}")


# %% [markdown]
# ### PECD → Fraunhofer scaling factors
#
# For each technology, compute the ratio between Fraunhofer's midpoint FLH
# (average of low and high from Table 3) and the PECD NUTS 0 long-run mean
# FLH. Multiplying PECD capacity factors by this scaling factor aligns them
# with Fraunhofer's central assumption. A factor < 1 means PECD overestimates
# (solar PV); a factor > 1 means PECD underestimates (offshore wind).

# %%
scaling_rows = []
for tech, p in COSTS.items():
    flh_lo, flh_hi = FRAUNHOFER_FLH[tech]
    flh_fraunhofer_mid = 0.5 * (flh_lo + flh_hi)
    flh_pecd = cf_mean[tech] * HOURS_PER_YEAR
    factor = flh_fraunhofer_mid / flh_pecd
    scaling_rows.append({
        "technology":        p["label"],
        "tech_key":          tech,
        "flh_pecd":          flh_pecd,
        "flh_fraunhofer_low":  flh_lo,
        "flh_fraunhofer_mid":  flh_fraunhofer_mid,
        "flh_fraunhofer_high": flh_hi,
        "scaling_factor":    factor,
    })

scaling_df = pd.DataFrame(scaling_rows)
print("\nPECD → Fraunhofer scaling factors (NUTS 0, DE):")
print(scaling_df.to_string(index=False))

scaling_csv = paths.processed_data_path / "pecd_fraunhofer_scaling_factors.csv"
scaling_df.to_csv(scaling_csv, index=False)
print(f"\nSaved scaling factors to {scaling_csv}")

# %%
fig_flh, ax_flh = plt.subplots(figsize=(9, 4))
y = np.arange(len(flh_df))
for yi, row in enumerate(flh_df.itertuples(index=False)):
    tech_key = [k for k, v in COSTS.items() if v["label"] == row.technology][0]
    color = TECH_COLORS[tech_key]
    ax_flh.hlines(yi, row.flh_fraunhofer_low, row.flh_fraunhofer_high,
                  color=color, linewidth=14, alpha=0.35,
                  label="Fraunhofer range" if yi == 0 else None)
    ax_flh.plot(row.flh_pecd, yi, "o", color="black", markersize=8, zorder=3,
                label="PECD NUTS 0 mean" if yi == 0 else None)
    ax_flh.annotate(f"{row.flh_pecd:,.0f} h", (row.flh_pecd, yi),
                    xytext=(8, 8), textcoords="offset points", fontsize=9)

ax_flh.set_yticks(y)
ax_flh.set_yticklabels(flh_df["technology"])
ax_flh.set_xlabel("Full-load hours [h / yr]")
ax_flh.set_title(
    "Full-load hours: PECD ERA5 long-run mean for DE vs Fraunhofer ISE 2024 assumptions",
    fontsize=11,
)
ax_flh.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_flh.set_axisbelow(True)
ax_flh.set_xlim(left=0)
ax_flh.legend(loc="lower right", fontsize=9)
fig_flh.tight_layout()
fig_flh.savefig(paths.images_path / "52_flh_comparison.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/52_flh_comparison.png
# :name: fig-52-flh-comparison
# Full-load hours implied by the PECD ERA5 long-run mean capacity factor for
# Germany (NUTS 0, black markers) vs the FLH ranges assumed by Fraunhofer ISE
# 2024 (Table 3, shaded bars). Differences explain most of the LCOE gap
# between this script's output and the Fraunhofer headline figures: where the
# PECD mean is better than Fraunhofer's worst-case site assumption, LCOE sits
# below the published range; where it is worse than the best-case site
# assumption, LCOE sits above.
# ```

# %%
# Compute LCOE for each technology using CAPEX low, mid, high
rows = []
for tech, p in COSTS.items():
    cf = cf_mean[tech]
    capex_mid = 0.5 * (p["capex_eur_per_kw_low"] + p["capex_eur_per_kw_high"])
    for capex_label, capex in [
        ("low",  p["capex_eur_per_kw_low"]),
        ("mid",  capex_mid),
        ("high", p["capex_eur_per_kw_high"]),
    ]:
        lcoe = compute_lcoe(
            cf=cf,
            capex=capex,
            opex_fix=p["opex_fix_eur_per_kw_yr"],
            opex_var=p["opex_var_eur_per_kwh"],
            lifetime=p["lifetime_years"],
            wacc=p["wacc_real"],
            degradation=p["degradation_per_yr"],
        )
        rows.append({
            "technology":    p["label"],
            "tech_key":      tech,
            "capex_scenario": capex_label,
            "cf_mean":       cf,
            "flh":           cf * HOURS_PER_YEAR,
            "capex_eur_per_kw": capex,
            "wacc_real":     p["wacc_real"],
            "lifetime_years": p["lifetime_years"],
            "lcoe_eur_per_kwh":   lcoe,
            "lcoe_ct_per_kwh":    lcoe * 100,
        })

lcoe_df = pd.DataFrame(rows)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
print(lcoe_df[[
    "technology", "capex_scenario", "flh", "capex_eur_per_kw",
    "wacc_real", "lcoe_ct_per_kwh",
]].to_string(index=False))

# %%
# Save the full LCOE table
out_path = paths.processed_data_path / "lcoe_germany.parquet"
out_path.parent.mkdir(parents=True, exist_ok=True)
lcoe_df.to_parquet(out_path, index=False)
print(f"\nSaved LCOE table to {out_path}")


# %% [markdown]
# ## LCOE ranges per technology
#
# The bar extends from the LCOE at the low CAPEX end to the high CAPEX end;
# the marker shows the LCOE at the CAPEX midpoint. All figures use the
# long-run mean PECD capacity factor for Germany (NUTS 0, ERA5 1979–2026).

# %%
fig, ax = plt.subplots(figsize=(9, 5))
pivot = lcoe_df.pivot(index="tech_key", columns="capex_scenario",
                      values="lcoe_ct_per_kwh")
order = ["solar_pv_utility", "wind_onshore", "wind_offshore"]
labels = [COSTS[t]["label"] for t in order]
lows  = pivot.loc[order, "low"].values
mids  = pivot.loc[order, "mid"].values
highs = pivot.loc[order, "high"].values

y = np.arange(len(order))
for yi, lo, mi, hi, tech in zip(y, lows, mids, highs, order):
    ax.hlines(yi, lo, hi, color=TECH_COLORS[tech], linewidth=14, alpha=0.85)
    ax.plot(mi, yi, "o", color="black", markersize=6, zorder=3)
    ax.annotate(f"{mi:.1f} ct/kWh", (mi, yi),
                xytext=(8, 8), textcoords="offset points", fontsize=9)

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("LCOE [€cent$_{2024}$ / kWh]")
ax.set_title(
    "LCOE for Germany (PECD ERA5 CF × Fraunhofer ISE 2024 costs)\n"
    "bars: CAPEX low→high · marker: CAPEX midpoint",
    fontsize=11,
)
ax.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
ax.set_xlim(left=0)
fig.tight_layout()
fig.savefig(paths.images_path / "52_lcoe_germany.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/52_lcoe_germany.png
# :name: fig-52-lcoe-germany
# LCOE for solar PV (utility-scale), onshore wind and offshore wind in Germany,
# combining the long-run mean PECD ERA5 capacity factor for DE (NUTS 0, 1979–2026)
# with Fraunhofer ISE 2024 cost and financing assumptions (Tables 1–2 of the
# study). Bars span the LCOE implied by the low and high CAPEX values; the
# marker shows the CAPEX midpoint. All values are real 2024 €cent/kWh.
# ```

# %% [markdown]
# ## Sub-national analysis: per-region FLH and LCOE
#
# The PECD "level 1" download returns Germany split into 38 sub-national
# regions (Eurostat NUTS 2 codes — `DE11`, `DE12`, …, `DEG0` — since PECD's
# internal level numbering happens to land on Eurostat NUTS 2 for Germany).
# This gives finer spatial resolution than the single NUTS 0 average used
# above for solar PV and onshore wind. Offshore wind is kept at the NUTS 0
# value since PECD does not provide sub-national offshore aggregation.
#
# For each region we compute the long-run mean capacity factor, the implied
# FLH, and the mid-CAPEX LCOE; then compare the spread of per-region FLH
# against Fraunhofer's assumed FLH bands (Table 3).

# %%
nuts_df = pd.read_parquet(paths.pecd_nuts1_de_processed_file)
print(f"Loaded sub-national PECD: shape={nuts_df.shape}, "
      f"{nuts_df.index[0]} → {nuts_df.index[-1]}")

NUTS_VARIABLES = {
    "solar_pv_utility": "solar_photovoltaic_power_generation",
    "wind_onshore":     "wind_power_generation_onshore",
}

# Long-run mean CF per region, one column per technology
region_cf = pd.concat(
    {tech: nuts_df[var].mean() for tech, var in NUTS_VARIABLES.items()},
    axis=1,
)
region_cf.index.name = "region"
region_flh = region_cf * HOURS_PER_YEAR
print("\nMin / mean / max FLH across sub-national regions:")
print(region_flh.agg(["min", "mean", "max"]).round(0).to_string())


# %%
# Per-region LCOE at mid CAPEX
region_rows = []
for tech, p in COSTS.items():
    if tech not in NUTS_VARIABLES:
        continue
    capex_mid = 0.5 * (p["capex_eur_per_kw_low"] + p["capex_eur_per_kw_high"])
    for region, cf in region_cf[tech].items():
        lcoe = compute_lcoe(
            cf=float(cf),
            capex=capex_mid,
            opex_fix=p["opex_fix_eur_per_kw_yr"],
            opex_var=p["opex_var_eur_per_kwh"],
            lifetime=p["lifetime_years"],
            wacc=p["wacc_real"],
            degradation=p["degradation_per_yr"],
        )
        region_rows.append({
            "technology":       p["label"],
            "tech_key":         tech,
            "region":           region,
            "cf_mean":          float(cf),
            "flh":              float(cf) * HOURS_PER_YEAR,
            "capex_eur_per_kw": capex_mid,
            "lcoe_ct_per_kwh":  lcoe * 100,
        })

region_lcoe_df = pd.DataFrame(region_rows)
region_out = paths.processed_data_path / "lcoe_germany_subnational.parquet"
region_lcoe_df.to_parquet(region_out, index=False)
print(f"\nSaved sub-national LCOE table to {region_out}")

# Summary: best / worst region per technology
for tech, p in COSTS.items():
    if tech not in NUTS_VARIABLES:
        continue
    sub = region_lcoe_df[region_lcoe_df["tech_key"] == tech]
    best = sub.loc[sub["lcoe_ct_per_kwh"].idxmin()]
    worst = sub.loc[sub["lcoe_ct_per_kwh"].idxmax()]
    print(f"\n{p['label']}:")
    print(f"  best  region: {best['region']}  FLH={best['flh']:>6,.0f}  LCOE={best['lcoe_ct_per_kwh']:.2f} ct/kWh")
    print(f"  worst region: {worst['region']}  FLH={worst['flh']:>6,.0f}  LCOE={worst['lcoe_ct_per_kwh']:.2f} ct/kWh")


# %% [markdown]
# ### Per-region FLH vs Fraunhofer bands

# %%
fig_sub, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
for ax, tech in zip(axes, ["solar_pv_utility", "wind_onshore"]):
    sub = region_lcoe_df[region_lcoe_df["tech_key"] == tech].sort_values("flh")
    color = TECH_COLORS[tech]
    lo, hi = FRAUNHOFER_FLH[tech]

    ypos = np.arange(len(sub))
    ax.barh(ypos, sub["flh"].values, color=color, alpha=0.9)
    ax.axvspan(lo, hi, color="grey", alpha=0.18,
               label=f"Fraunhofer range ({lo}–{hi} h)")
    ax.axvline(lo, color="grey", linewidth=0.8, linestyle="--")
    ax.axvline(hi, color="grey", linewidth=0.8, linestyle="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(sub["region"].values, fontsize=7)
    ax.set_xlabel("Full-load hours [h / yr]")
    ax.set_title(COSTS[tech]["label"])
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8)

fig_sub.suptitle(
    "Sub-national FLH per PECD region vs Fraunhofer ISE 2024 FLH bands",
    fontsize=12,
)
fig_sub.tight_layout()
fig_sub.savefig(paths.images_path / "52_flh_subnational.png",
                dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/52_flh_subnational.png
# :name: fig-52-flh-subnational
# Long-run mean full-load hours per PECD sub-national region in Germany (38
# regions, corresponding to Eurostat NUTS 2 codes) for solar PV and onshore
# wind. The shaded band marks the FLH envelope Fraunhofer ISE 2024 assumes
# for "typical German sites" (Table 3); bars inside the band are consistent
# with those assumptions.
# ```

# %% [markdown]
# ### Per-region LCOE (mid CAPEX)

# %%
fig_lcoe_sub, axes2 = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
for ax, tech in zip(axes2, ["solar_pv_utility", "wind_onshore"]):
    sub = region_lcoe_df[region_lcoe_df["tech_key"] == tech].sort_values("lcoe_ct_per_kwh")
    color = TECH_COLORS[tech]

    repl_row = repl_df[repl_df["technology"] == COSTS[tech]["label"]].iloc[0]
    frau_lo = float(repl_row["lcoe_low_computed"])
    frau_hi = float(repl_row["lcoe_high_computed"])

    ypos = np.arange(len(sub))
    ax.barh(ypos, sub["lcoe_ct_per_kwh"].values, color=color, alpha=0.9)
    ax.axvspan(frau_lo, frau_hi, color="grey", alpha=0.18,
               label=f"Fraunhofer range ({frau_lo:.1f}–{frau_hi:.1f} ct)")
    ax.axvline(frau_lo, color="grey", linewidth=0.8, linestyle="--")
    ax.axvline(frau_hi, color="grey", linewidth=0.8, linestyle="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(sub["region"].values, fontsize=7)
    ax.set_xlabel("LCOE [€cent$_{2024}$ / kWh]")
    ax.set_title(COSTS[tech]["label"])
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8)

fig_lcoe_sub.suptitle(
    "Sub-national LCOE per PECD region (mid CAPEX) vs Fraunhofer ISE 2024",
    fontsize=12,
)
fig_lcoe_sub.tight_layout()
fig_lcoe_sub.savefig(paths.images_path / "52_lcoe_subnational.png",
                     dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/52_lcoe_subnational.png
# :name: fig-52-lcoe-subnational
# LCOE per PECD sub-national region for solar PV and onshore wind in
# Germany, computed with CAPEX set to the Fraunhofer ISE 2024 midpoint and
# the per-region long-run mean capacity factor. The shaded band indicates
# the Fraunhofer headline LCOE range (best- and worst-case FLH × CAPEX
# pairs, successfully replicated earlier in this script).
# ```

# %%

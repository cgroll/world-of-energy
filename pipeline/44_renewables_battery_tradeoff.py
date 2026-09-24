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
# # Renewables & Battery Trade-off Analysis
#
# Uses the same PECD-ERA5 capacity factors and BDEW+XGBoost demand model as
# script 41, with capacities fixed at the most recent SMARD record.
#
# **Topics covered**
#
# 1. **Dunkelflaute** — worst 7-day low-generation window in the reanalysis
#    record: gap chart, per-hour scaling multiplier, cumulative shortfall with
#    battery and generation-scaling scenarios.
# 2. **Energy-matched scaling** — find the renewable scaling factor `r` such
#    that total generation equals total demand, then compute the minimum seasonal
#    battery required (peak-to-trough of the cumulative balance curve).
# 3. **Multi-scenario scaling 2×–10×** — required battery size, curtailment,
#    and SoC profiles for a range of overbuild factors.

# %%
import matplotlib.pyplot as plt
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
# ## Data loading
#
# Identical to script 41: demand predictions from script 39, PECD capacity
# factors converted to SMARD hour-ending CET index, capacities fixed at the
# most recent SMARD record.

# %%
demand_preds = pd.read_parquet(paths.de_demand_predictions_file)
demand_pred  = demand_preds["demand_baseline_mw"].rename("demand_mw")

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


cf_solar_s    = convert_and_clean(cf_solar_raw)
cf_onshore_s  = convert_and_clean(cf_onshore_raw)
cf_offshore_s = (
    convert_and_clean(cf_offshore_raw)
    if not cf_offshore_raw.empty
    else pd.Series(dtype=float)
)

cap     = pd.read_parquet(paths.smard_capacities_file)
cap_re  = cap[["solar", "wind_onshore", "wind_offshore"]].copy()

shared_idx = demand_pred.index
for cf in [cf_solar_s, cf_onshore_s]:
    shared_idx = shared_idx.intersection(cf.index)
if not cf_offshore_s.empty:
    shared_idx = shared_idx.intersection(cf_offshore_s.index)

cap_latest = cap_re.iloc[-1]

gen_solar_mw    = (cf_solar_s.reindex(shared_idx)   * cap_latest["solar"]).rename("gen_solar_mw")
gen_onshore_mw  = (cf_onshore_s.reindex(shared_idx) * cap_latest["wind_onshore"]).rename("gen_onshore_mw")
gen_offshore_mw = (
    cf_offshore_s.reindex(shared_idx) * cap_latest["wind_offshore"]
    if not cf_offshore_s.empty
    else pd.Series(0.0, index=shared_idx)
).rename("gen_offshore_mw")

df = pd.DataFrame({
    "demand_mw":   demand_pred.reindex(shared_idx),
    "solar_mw":    gen_solar_mw,
    "onshore_mw":  gen_onshore_mw,
    "offshore_mw": gen_offshore_mw,
}).dropna()

df["gen_mw"]             = df["solar_mw"] + df["onshore_mw"] + df["offshore_mw"]
df["residual_load_mw"]   = df["demand_mw"] - df["gen_mw"]
df["residual_demand_mw"] = df["residual_load_mw"].clip(lower=0)
df["excess_mw"]          = (-df["residual_load_mw"]).clip(lower=0)

# Drop the most recent (partial) year
df = df[df.index.year < df.index.year.max()]

AVAIL_YEARS      = sorted(df.index.year.unique().tolist())
storage_base_mwh = df["demand_mw"].mean()   # 1× = avg hourly demand (MWh)
RE_SCALES        = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]

print(f"Analysis period: {AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]}  ({len(AVAIL_YEARS)} years)")
print(f"Total rows: {len(df):,}")
print(f"Fixed capacities from SMARD record: {cap_re.index[-1].date()}")
print(f"  Solar: {cap_latest['solar']:,.0f} MW  |  Onshore: {cap_latest['wind_onshore']:,.0f} MW"
      f"  |  Offshore: {cap_latest['wind_offshore']:,.0f} MW")

# %% [markdown]
# ## Dunkelflaute — worst 7-day window
#
# Rolling 7-day (168 h) ratio of total renewable generation to demand. The
# minimum identifies the worst sustained low-generation period in the full
# reanalysis dataset (fixed modern capacities). The analysis window shown in
# the charts is the 7-day core event plus ±14 days of context.

# %%
DUNKELFLAUTE_W = 7 * 24  # 168 h

roll_cov = (
    df["gen_mw"].rolling(DUNKELFLAUTE_W, center=True).sum()
    / df["demand_mw"].rolling(DUNKELFLAUTE_W, center=True).sum()
)

dk_center = roll_cov.idxmin()
dk_start  = dk_center - pd.Timedelta(hours=DUNKELFLAUTE_W // 2)
dk_end    = dk_center + pd.Timedelta(hours=DUNKELFLAUTE_W // 2)

print("Worst 7-day Dunkelflaute")
print(f"  Window  : {dk_start.date()} – {dk_end.date()}")
print(f"  gen/demand (7-day): {roll_cov[dk_center]:.1%}")
print(f"  Avg gen    : {df.loc[dk_start:dk_end, 'gen_mw'].mean() / 1e3:.1f} GW")
print(f"  Avg demand : {df.loc[dk_start:dk_end, 'demand_mw'].mean() / 1e3:.1f} GW")

LEAD_DAYS = 14
TAIL_DAYS = 14
win_start = dk_start - pd.Timedelta(days=LEAD_DAYS)
win_end   = dk_end   + pd.Timedelta(days=TAIL_DAYS)

dk = df.loc[win_start:win_end].copy()

# %% [markdown]
# ### Demand vs. generation during the event

# %%
fig, ax = plt.subplots(figsize=(16, 5))

ax.fill_between(dk.index, 0, dk["solar_mw"] / 1e3,
                color="#f9c74f", alpha=0.85, label="Solar")
ax.fill_between(dk.index,
                dk["solar_mw"] / 1e3,
                (dk["solar_mw"] + dk["onshore_mw"]) / 1e3,
                color="#4cc9f0", alpha=0.85, label="Wind onshore")
ax.fill_between(dk.index,
                (dk["solar_mw"] + dk["onshore_mw"]) / 1e3,
                dk["gen_mw"] / 1e3,
                color="#0077b6", alpha=0.85, label="Wind offshore")
ax.plot(dk.index, dk["demand_mw"] / 1e3,
        color="#333333", linewidth=1.5, label="Demand")

ax.axvspan(dk_start, dk_end, color="red", alpha=0.10, label="Worst 7-day window")

ax.set_ylabel("Power (GW)")
ax.set_title(
    f"Dunkelflaute {dk_start.date()} – {dk_end.date()} "
    f"(7-day gen/demand = {roll_cov[dk_center]:.1%})  |  ±{LEAD_DAYS}d context",
    fontsize=11,
)
ax.legend(fontsize=9, loc="upper left")
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(paths.images_path / "44_dunkelflaute_gap.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/44_dunkelflaute_gap.png
# :name: fig-44-dunkelflaute-gap
# Stacked renewable generation (solar = yellow, wind onshore = light blue,
# wind offshore = dark blue) and electricity demand (black line) for the
# worst 7-day Dunkelflaute found in the reanalysis dataset (red shading),
# with ±14 days of context. The gap between the demand line and the top of
# the stack is electricity that must be covered by storage, imports, or
# dispatchable backup.
# ```

# %% [markdown]
# ### Generation multiplier needed to cover demand
#
# For each hour: how much would current renewable generation need to be scaled
# up to exactly meet demand? Clipped to a minimum of 1 (generation already
# covers demand) and capped at 20 (near-zero generation — no scaling helps).

# %%
dk["gen_multiplier"] = (
    (dk["demand_mw"] / dk["gen_mw"].clip(lower=1))
    .clip(lower=1, upper=20)
)

fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(dk.index, dk["gen_multiplier"],
        color="#e6734a", linewidth=0.8, alpha=0.85)
ax.fill_between(dk.index, 1, dk["gen_multiplier"],
                color="#e6734a", alpha=0.25)
ax.axhline(1, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7,
           label="1× — generation already covers demand")
ax.axvspan(dk_start, dk_end, color="red", alpha=0.10, label="Worst 7-day window")

ax.set_ylabel("Required generation multiplier (×)")
ax.set_title(
    "Multiplier needed to cover demand from renewables alone  "
    "(1 = already covered, 20× cap = near-zero generation)",
    fontsize=11,
)
ax.legend(fontsize=9)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(paths.images_path / "44_dunkelflaute_multiplier.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/44_dunkelflaute_multiplier.png
# :name: fig-44-dunkelflaute-multiplier
# Hour-by-hour multiplier by which current renewable capacity would need to be
# scaled to exactly cover demand — clipped to 1 (already sufficient) and capped
# at 20 (generation near zero, e.g. calm nights). Values of 2 mean doubling the
# fleet would suffice for that hour; values at the 20× cap indicate hours where
# no amount of capacity scaling replaces storage or dispatchable backup. The red
# shading marks the worst 7-day Dunkelflaute window.
# ```

# %% [markdown]
# ## Energy-matched scaling: required battery and curtailment
#
# Find the renewable scaling factor `r` such that total generated energy over
# the full period exactly equals total demand. The minimum battery size is the
# peak-to-trough range of `cumsum(gen_scaled − demand)` — the only formula
# that guarantees zero curtailment and zero unmet demand. Starting SoC is
# `-min(b)`, so `SoC(t) = initial_soc + b(t)` stays in `[0, capacity]`.

# %%
gen_full    = df["gen_mw"].values
demand_full = df["demand_mw"].values

r_match         = demand_full.sum() / gen_full.sum()
gen_scaled_full = gen_full * r_match

b                = np.cumsum(gen_scaled_full - demand_full)
battery_size_mwh = float(b.max() - b.min())
initial_soc      = float(-b.min())
soc_ts_mwh       = initial_soc + b

curtailment = float(np.maximum(0, soc_ts_mwh - battery_size_mwh).sum())
unmet       = float(np.maximum(0, -soc_ts_mwh).sum())

print(f"Energy-matched scaling over {AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]}")
print(f"  Total demand          : {demand_full.sum()/1e6:,.1f} TWh")
print(f"  Total generation (1×) : {gen_full.sum()/1e6:,.1f} TWh")
print(f"  Scaling factor r      : {r_match:.3f}×")
print(f"  Required battery size : {battery_size_mwh/1e6:,.2f} TWh  ({battery_size_mwh/1e3:,.0f} GWh)")
print(f"  Starting SoC          : {initial_soc/1e6:,.2f} TWh  ({initial_soc/battery_size_mwh*100:.1f}% of capacity)")
print(f"  Curtailment           : {curtailment/1e6:,.4f} TWh  (should be ~0)")
print(f"  Unmet demand          : {unmet/1e6:,.4f} TWh  (should be ~0)")

# %%
fig, ax = plt.subplots(figsize=(16, 4))

ax.plot(df.index, soc_ts_mwh / 1e3, color="#4a90d9", linewidth=0.6, alpha=0.8)
ax.axhline(battery_size_mwh / 1e3, color="#e6734a", linewidth=1.0, linestyle="--",
           alpha=0.7, label=f"Capacity ({battery_size_mwh/1e6:.2f} TWh)")
ax.axhline(initial_soc / 1e3, color="#5ab55e", linewidth=1.0, linestyle="--",
           alpha=0.7, label=f"Starting SoC ({initial_soc/1e6:.2f} TWh)")

ax.set_ylabel("State of charge (GWh)")
ax.set_title(
    f"Battery SoC — energy-matched scaling (r = {r_match:.3f}×), "
    f"capacity = {battery_size_mwh/1e6:.2f} TWh",
    fontsize=11,
)
ax.legend(fontsize=9)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(paths.images_path / "44_battery_soc_full.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/44_battery_soc_full.png
# :name: fig-44-battery-soc-full
# Battery state of charge (GWh) over the full reanalysis period under the
# energy-matched scaling scenario. The battery capacity (orange dashed) is the
# minimum peak-to-trough range that guarantees demand is always met with zero
# curtailment. The starting SoC (green dashed) is set to the level that
# prevents the battery from ever hitting zero or overflowing.
# ```

# %% [markdown]
# ## Multi-scenario: renewable scaling 2×–10×
#
# For each integer scaling factor the minimum battery size is the **maximum
# drawdown** of `cumsum(demand − gen_scaled)` — the worst sustained shortfall
# the battery must survive when curtailment of excess is allowed. This gives the
# correct trade-off: more overbuild → smaller required battery, more curtailment.
#
# Curtailment is computed from a forward simulation that starts with a full
# battery and clips SoC to `[0, battery_size]`.

# %%
RE_SCALE_RANGE = range(2, 11)  # 2× to 10×

scenario_records = []
soc_by_scale     = {}

for r in RE_SCALE_RANGE:
    gen_s = gen_full * r

    # Minimum battery (curtailment allowed) = max drawdown of cumulative deficit
    cum_deficit = np.cumsum(demand_full - gen_s)
    running_min = np.minimum.accumulate(cum_deficit)
    bat  = float(np.max(cum_deficit - running_min))
    isoc = bat  # start full — excess is curtailed, shortfalls draw down from bat

    # Forward simulation with floor=0 and ceiling=bat (curtail overflow)
    soc_ts = np.empty(len(gen_s))
    soc_ts[0] = isoc
    for t in range(len(gen_s) - 1):
        soc_ts[t + 1] = np.clip(soc_ts[t] + gen_s[t] - demand_full[t], 0.0, bat)

    # Curtailment: energy clipped at the ceiling during the simulation
    raw_next = soc_ts[:-1] + gen_s[:-1] - demand_full[:-1]
    curt = float(np.sum(np.maximum(0.0, raw_next - bat)))

    # Peak residual load: worst single hour that gas/backup must cover (no storage)
    peak_residual_mw = float(np.max(demand_full - gen_s))

    soc_by_scale[r] = soc_ts
    scenario_records.append({
        "scale":              f"{r}×",
        "total_gen_twh":      round(gen_s.sum()       / 1e6, 0),
        "total_demand_twh":   round(demand_full.sum() / 1e6, 0),
        "battery_twh":        round(bat  / 1e6, 2),
        "initial_soc_twh":    round(isoc / 1e6, 2),
        "curtailment_twh":    round(curt / 1e6, 1),
        "curtailment_pct":    round(curt / gen_s.sum() * 100, 1),
        "peak_residual_gw":   round(peak_residual_mw / 1e3, 1),
    })

scenarios_df = pd.DataFrame(scenario_records).set_index("scale")
print(f"\nRenewable scaling scenarios — {AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]}")
print(scenarios_df.to_string())

# %% [markdown]
# ## Cumulative excess generation and drawdown analysis
#
# For each scaling factor `r`, the cumulative balance `b(t) = cumsum(gen×r − demand)`
# shows whether the system is in a net-surplus or net-deficit state at every
# moment. A downward slope means demand outstrips generation; the deepest
# trough relative to the preceding peak is the **maximum drawdown** —
# equivalent to the minimum battery needed (curtailment allowed).
#
# An initial offset of `−min(b)` lifts the whole series so it never dips below
# zero, turning it into a physical "reservoir level" interpretation.

# %%
# Pre-compute cumsum balance, drawdown statistics, and offset for every scenario
cmap_sc   = plt.cm.plasma
colors_sc = [cmap_sc(i / (len(RE_SCALE_RANGE) - 1)) for i in range(len(RE_SCALE_RANGE))]

b_by_scale      = {}
dd_stats        = []   # (r, offset_twh, max_dd_twh, dd_days)

for r in RE_SCALE_RANGE:
    b = np.cumsum(gen_full * r - demand_full)
    b_by_scale[r] = b

    offset_mwh = float(max(0.0, -b.min()))

    # Max drawdown: largest drop from any running peak to a subsequent value
    running_peak = np.maximum.accumulate(b)
    drawdown     = running_peak - b
    max_dd_mwh   = float(drawdown.max())
    trough_idx   = int(drawdown.argmax())
    peak_idx     = int(b[:trough_idx + 1].argmax())
    dd_days      = (trough_idx - peak_idx) / 24

    dd_stats.append(dict(
        r          = r,
        label      = f"{r}×",
        offset_twh = offset_mwh / 1e6,
        max_dd_twh = max_dd_mwh / 1e6,
        dd_days    = dd_days,
    ))

dd_df = pd.DataFrame(dd_stats).set_index("label")
print("\nDrawdown statistics per scaling scenario")
print(dd_df.to_string(float_format=lambda x: f"{x:,.1f}"))

# %%  raw cumsum balance — can go negative
fig, ax = plt.subplots(figsize=(16, 5))

for r, color in zip(RE_SCALE_RANGE, colors_sc):
    ax.plot(df.index, b_by_scale[r] / 1e6, color=color, linewidth=0.7,
            alpha=0.85, label=f"{r}×")

ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_ylabel("Cumulative balance (TWh)")
ax.set_title(
    f"Cumulative excess generation  b(t) = cumsum(gen×r − demand)  "
    f"({AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]})",
    fontsize=11,
)
ax.legend(fontsize=9, ncol=3, title="Scaling factor")
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(paths.images_path / "44_cumsum_balance.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/44_cumsum_balance.png
# :name: fig-44-cumsum-balance
# Cumulative energy balance `b(t) = cumsum(gen×r − demand)` for renewable
# scaling factors 2×–10× (plasma colour scale). An upward slope means the
# system generates more than it consumes; a downward slope means the opposite.
# The lowest point of each curve (distance below zero) is the initial stored
# energy needed to ensure no deficit ever occurs.
# ```

# %% offset series — lifted so the minimum is exactly zero
fig, ax = plt.subplots(figsize=(16, 5))

for r, color, stat in zip(RE_SCALE_RANGE, colors_sc, dd_stats):
    b_off = b_by_scale[r] + stat["offset_twh"] * 1e6
    ax.plot(df.index, b_off / 1e6, color=color, linewidth=0.7,
            alpha=0.85, label=f"{r}×")

ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_ylabel("Energy reservoir level (TWh)")
ax.set_title(
    "Cumulative balance offset so reservoir never goes below zero\n"
    f"(initial stored energy = −min(b),  {AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]})",
    fontsize=11,
)
ax.legend(fontsize=9, ncol=3, title="Scaling factor")
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(paths.images_path / "44_cumsum_offset.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/44_cumsum_offset.png
# :name: fig-44-cumsum-offset
# Same cumulative balance as above but shifted up by the initial stored
# energy `−min(b)` so the reservoir floor is exactly zero. The peak of each
# curve is the total capacity the storage system must have to never overflow;
# that peak minus the floor is the maximum drawdown.
# ```

# %% summary: initial offset, max drawdown, drawdown duration per scenario
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

r_vals    = [s["r"]          for s in dd_stats]
off_vals  = [s["offset_twh"] for s in dd_stats]
dd_vals   = [s["max_dd_twh"] for s in dd_stats]
day_vals  = [s["dd_days"]    for s in dd_stats]

for ax, yvals, ylabel, title, color in zip(
    axes,
    [off_vals,  dd_vals,        day_vals],
    ["TWh",     "TWh",          "Days"],
    ["Initial offset\n(starting stored energy)",
     "Maximum drawdown\n(min battery needed)",
     "Drawdown duration\n(peak → trough)"],
    ["#4a90d9", "#e6734a",      "#5ab55e"],
):
    ax.scatter(r_vals, yvals, color=color, s=80, zorder=3,
               edgecolors="#333333", linewidths=0.5)
    ax.plot(r_vals, yvals, color=color, linewidth=0.8, zorder=2)
    for x, y in zip(r_vals, yvals):
        ax.annotate(f"{y:,.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color="#555555")
    ax.set_xlabel("Renewable scaling factor (×)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)

fig.suptitle(
    f"Drawdown summary — {AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]}",
    fontsize=11, y=1.02,
)
fig.tight_layout()
fig.savefig(paths.images_path / "44_drawdown_summary.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/44_drawdown_summary.png
# :name: fig-44-drawdown-summary
# Per-scenario drawdown statistics across scaling factors 2×–10×.
# Left: initial stored energy required so the cumulative balance never goes
# negative (equivalent to the minimum starting SoC). Centre: maximum
# drawdown — the minimum battery capacity that prevents any shortfall.
# Right: calendar duration of that worst drawdown from peak to trough (days).
# Higher overbuild reduces all three metrics.
# ```

# %%
cmap_sc   = plt.cm.plasma
colors_sc = [cmap_sc(i / (len(RE_SCALE_RANGE) - 1)) for i in range(len(RE_SCALE_RANGE))]

fig, ax = plt.subplots(figsize=(16, 5))

for r, color in zip(RE_SCALE_RANGE, colors_sc):
    ax.plot(df.index, soc_by_scale[r] / 1e6,
            color=color, linewidth=0.7, alpha=0.85, label=f"{r}×")

ax.set_ylabel("State of charge (TWh)")
ax.set_title(
    f"Battery SoC — renewable scaling 2×–10×  ({AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]})",
    fontsize=11,
)
ax.legend(fontsize=9, ncol=3, title="Scaling factor")
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(paths.images_path / "44_battery_soc_scenarios.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/44_battery_soc_scenarios.png
# :name: fig-44-battery-soc-scenarios
# Battery state of charge (TWh) over the full reanalysis period for renewable
# scaling factors 2×–10× (plasma colour scale). Each scenario uses the minimum
# battery needed when curtailment is allowed — the maximum drawdown of the
# cumulative demand deficit. Higher scaling factors reduce the required battery
# (shortfalls are shorter and shallower) at the cost of increased curtailment.
# ```

# %% [markdown]
# ## Renewable scaling vs. required battery size
#
# Each point is one scaling scenario. The trade-off is clear: more overbuild
# reduces the required seasonal battery, at the cost of higher curtailment.

# %%
fig, ax = plt.subplots(figsize=(8, 5))

scales      = [int(s.rstrip("×")) for s in scenarios_df.index]
battery_twh = scenarios_df["battery_twh"].values
curt_pct    = scenarios_df["curtailment_pct"].values

sc = ax.scatter(scales, battery_twh, c=curt_pct, cmap="YlOrRd",
                s=90, zorder=3, edgecolors="#333333", linewidths=0.5)
ax.plot(scales, battery_twh, color="#aaaaaa", linewidth=0.8, zorder=2)

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Curtailment (%)", fontsize=9)

for x, y, pct in zip(scales, battery_twh, curt_pct):
    ax.annotate(f"{pct:.0f}%", (x, y), textcoords="offset points",
                xytext=(6, 4), fontsize=8, color="#555555")

ax.set_xlabel("Renewable scaling factor (×)")
ax.set_ylabel("Required battery (TWh)")
ax.set_title(
    f"Renewable scaling vs. required battery  ({AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]})\n"
    "Colour = curtailment share; labels = curtailment %",
    fontsize=11,
)
ax.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(paths.images_path / "44_scaling_vs_battery.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/44_scaling_vs_battery.png
# :name: fig-44-scaling-vs-battery
# Required seasonal battery size (TWh) as a function of renewable overbuild
# factor. Each point is one integer scaling scenario (2×–10×); colour encodes
# curtailment share. More renewables reduce the battery needed to survive the
# worst shortfall periods, but at the cost of curtailing an increasing fraction
# of total generation.
# ```

# %% [markdown]
# ## Renewable scaling vs. required battery (in hours of average demand)
#
# Same trade-off as above, but battery size is expressed as a multiple of
# average hourly demand — so 1 = one hour of average demand stored, 8760 = one
# full year. This unit makes the scale immediately interpretable without knowing
# the absolute size of the German power system.

# %%
battery_hours = scenarios_df["battery_twh"].values * 1e6 / storage_base_mwh

fig, ax = plt.subplots(figsize=(8, 5))

sc = ax.scatter(scales, battery_hours, c=curt_pct, cmap="YlOrRd",
                s=90, zorder=3, edgecolors="#333333", linewidths=0.5)
ax.plot(scales, battery_hours, color="#aaaaaa", linewidth=0.8, zorder=2)

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Curtailment (%)", fontsize=9)

for x, y, pct in zip(scales, battery_hours, curt_pct):
    ax.annotate(f"{pct:.0f}%", (x, y), textcoords="offset points",
                xytext=(6, 4), fontsize=8, color="#555555")

ax.set_xlabel("Renewable scaling factor (×)")
ax.set_ylabel("Required battery (× avg hourly demand)")
ax.set_title(
    f"Renewable scaling vs. required battery  ({AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]})\n"
    "Battery in multiples of avg hourly demand; colour = curtailment share",
    fontsize=11,
)
ax.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(paths.images_path / "44_scaling_vs_battery_hours.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/44_scaling_vs_battery_hours.png
# :name: fig-44-scaling-vs-battery-hours
# Required seasonal battery size expressed as a multiple of average hourly
# demand (1 = one hour of average demand stored). Each point is one integer
# scaling scenario (2×–10×); colour encodes curtailment share. The normalised
# unit makes the result directly comparable across different system sizes: a
# value of 8 760 would equal one full year of storage.
# ```

# %% [markdown]
# ## Battery vs. peak gas capacity across scaling scenarios
#
# Two dimensions of the storage/backup trade-off on the same axes: required
# seasonal battery (hours of avg demand, left axis) and peak residual load that
# dispatchable generation must cover when no storage is available (GW, right
# axis). Both decline with more overbuild, but the gas capacity floor is set by
# the single worst hour in the reanalysis record.

# %%
peak_residual_gw = scenarios_df["peak_residual_gw"].values

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

ax1.scatter(scales, battery_hours, color="#4a90d9", s=80, zorder=3,
            edgecolors="#1a5a99", linewidths=0.5, label="Battery (h avg demand)")
ax1.plot(scales, battery_hours, color="#4a90d9", linewidth=0.8, zorder=2)

ax2.scatter(scales, peak_residual_gw, color="#e6734a", s=80, zorder=3,
            marker="D", edgecolors="#a03010", linewidths=0.5, label="Peak residual load (GW)")
ax2.plot(scales, peak_residual_gw, color="#e6734a", linewidth=0.8, zorder=2,
         linestyle="--")

ax1.set_xlabel("Renewable scaling factor (×)")
ax1.set_ylabel("Required battery (× avg hourly demand)", color="#4a90d9")
ax2.set_ylabel("Peak residual load — no storage (GW)", color="#e6734a")
ax1.tick_params(axis="y", labelcolor="#4a90d9")
ax2.tick_params(axis="y", labelcolor="#e6734a")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

ax1.set_title(
    f"Battery vs. peak gas capacity required  ({AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]})",
    fontsize=11,
)
ax1.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax1.yaxis.grid(True, linewidth=0.4, alpha=0.4)
ax1.set_axisbelow(True)
fig.tight_layout()
fig.savefig(paths.images_path / "44_battery_vs_gas.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/44_battery_vs_gas.png
# :name: fig-44-battery-vs-gas
# Two dimensions of the storage/backup trade-off across renewable scaling
# scenarios (2×–10×). Blue circles (left axis): minimum seasonal battery in
# multiples of average hourly demand. Orange diamonds (right axis): peak
# residual load in GW — the dispatchable capacity gas plants must provide in
# the single worst hour if no storage is available. Both decline with overbuild,
# but the gas capacity required remains substantial even at high scaling factors
# because calm, dark winter nights cannot be engineered away.
# ```

# %% [markdown]
# ## Reiche 2045 scenario: are the planned capacities and storage sufficient?
#
# Long-term 2045 capacity targets (source: Wirtschaftsministerium scenarios /
# public statements, March 2026):
#
# | Source | Planned capacity |
# |---|---|
# | Solar PV | 400 GW |
# | Wind onshore | 180 GW |
# | Wind offshore | 70 GW |
# | Battery storage | 600 GWh |
#
# The same PECD capacity factors are applied to the planned capacities.
# Demand is scaled from the historical profile to two annual totals:
# a conservative 700 TWh scenario and the high-electrification projection
# of 1 000 TWh (full coupling of transport, heat, and hydrogen by 2045).
# Gas capacity is held at the Reiche 2030 plan (47 GW) as a reference.

# %%
CAP_REICHE = {
    "solar":         400_000,   # MW
    "wind_onshore":  180_000,
    "wind_offshore":   70_000,
}
BAT_REICHE_MWH = 600_000  # 600 GWh

# Rescale hourly generation from current SMARD capacities to planned capacities
gen_solar_r   = df["solar_mw"].values   / cap_latest["solar"]        * CAP_REICHE["solar"]
gen_onshore_r = df["onshore_mw"].values / cap_latest["wind_onshore"] * CAP_REICHE["wind_onshore"]
gen_offshore_r = (
    df["offshore_mw"].values / cap_latest["wind_offshore"] * CAP_REICHE["wind_offshore"]
    if cap_latest["wind_offshore"] > 0
    else np.zeros(len(df))
)
gen_reiche = gen_solar_r + gen_onshore_r + gen_offshore_r

avg_annual_demand_mwh = df["demand_mw"].sum() / len(AVAIL_YEARS)

DEMAND_SCENARIOS = {
    "Conservative (700 TWh)":       700e6,
    "High electrification (1 000 TWh)": 1_000e6,
}

GAS_PLANNED_MW = 47_000  # 35 GW existing + 12 GW new build (Reiche plan)

reiche_results = {}
for label, target_mwh in DEMAND_SCENARIOS.items():
    d_scale  = target_mwh / avg_annual_demand_mwh
    demand_s = df["demand_mw"].values * d_scale

    # Forward simulation: renewables first, then battery, then gas (residual)
    soc_ts      = np.empty(len(demand_s))
    soc_ts[0]   = BAT_REICHE_MWH
    gas_mw      = np.zeros(len(demand_s))   # hourly gas dispatch (residual load)
    curt_total  = 0.0
    for t in range(len(demand_s) - 1):
        raw = soc_ts[t] + gen_reiche[t] - demand_s[t]
        if raw > BAT_REICHE_MWH:
            curt_total    += raw - BAT_REICHE_MWH
            soc_ts[t + 1]  = BAT_REICHE_MWH
        elif raw < 0.0:
            gas_mw[t]      = -raw           # shortfall dispatched by gas
            soc_ts[t + 1]  = 0.0
        else:
            soc_ts[t + 1]  = raw

    gas_twh_annual     = gas_mw.sum() / 1e6 / len(AVAIL_YEARS)
    demand_twh_annual  = demand_s.sum() / 1e6 / len(AVAIL_YEARS)
    re_twh_annual      = demand_twh_annual - gas_twh_annual

    reiche_results[label] = dict(
        demand_scale       = d_scale,
        gas_mw             = gas_mw,
        total_gen_twh      = gen_reiche.sum() / 1e6 / len(AVAIL_YEARS),
        total_demand_twh   = demand_twh_annual,
        re_served_twh      = re_twh_annual,
        gas_twh            = gas_twh_annual,
        gas_pct            = gas_twh_annual / demand_twh_annual * 100,
        re_pct             = re_twh_annual  / demand_twh_annual * 100,
        peak_gas_gw        = gas_mw.max() / 1e3,
        curtailment_twh    = curt_total / 1e6 / len(AVAIL_YEARS),
    )

print(f"\nReiche 2045 scenario — PECD capacity factors, {AVAIL_YEARS[0]}–{AVAIL_YEARS[-1]}")
print(f"Planned renewables:  solar {CAP_REICHE['solar']/1e3:.0f} GW  |  "
      f"onshore {CAP_REICHE['wind_onshore']/1e3:.0f} GW  |  "
      f"offshore {CAP_REICHE['wind_offshore']/1e3:.0f} GW")
print(f"Planned battery:     {BAT_REICHE_MWH/1e3:.0f} GWh")
print(f"Planned gas:         {GAS_PLANNED_MW/1e3:.0f} GW (35 GW existing + 12 GW new)\n")

for label, res in reiche_results.items():
    gap = res["peak_gas_gw"] - GAS_PLANNED_MW / 1e3
    print(f"  [{label}]")
    print(f"    Avg annual generation  : {res['total_gen_twh']:,.0f} TWh")
    print(f"    Avg annual demand      : {res['total_demand_twh']:,.0f} TWh")
    print(f"    RE share               : {res['re_pct']:.1f}%  ({res['re_served_twh']:,.0f} TWh/yr)")
    print(f"    Gas share              : {res['gas_pct']:.1f}%  ({res['gas_twh']:,.0f} TWh/yr)")
    print(f"    Peak gas dispatch      : {res['peak_gas_gw']:.1f} GW  "
          f"(planned capacity: {GAS_PLANNED_MW/1e3:.0f} GW  →  "
          f"{'surplus' if gap <= 0 else 'GAP'} {abs(gap):.1f} GW)")
    print(f"    Avg annual curtailment : {res['curtailment_twh']:,.0f} TWh/yr")
    print()

# %%
colors_reiche = {"Conservative (700 TWh)": "#4a90d9", "High electrification (1 000 TWh)": "#e6734a"}
hours_per_year = 8_760

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# --- Left: residual load duration curve ---
ax = axes[0]
for label, res in reiche_results.items():
    sorted_gas = np.sort(res["gas_mw"] / 1e3)[::-1]
    # average over years: show hours-per-year on x axis
    hours = np.linspace(0, hours_per_year, len(sorted_gas))
    ax.plot(hours, sorted_gas, color=colors_reiche[label], linewidth=1.2, label=label)

ax.axhline(GAS_PLANNED_MW / 1e3, color="#333333", linewidth=1.2, linestyle="--",
           label=f"Planned gas capacity ({GAS_PLANNED_MW/1e3:.0f} GW)")
ax.set_xlabel("Hours per year (sorted)")
ax.set_ylabel("Gas dispatch (GW)")
ax.set_title("Residual load duration curve\n(hours requiring gas backup, 2045 targets)", fontsize=11)
ax.legend(fontsize=9)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)

# --- Right: annual energy mix ---
ax = axes[1]
scenario_labels = list(reiche_results.keys())
re_vals   = [reiche_results[s]["re_pct"]  for s in scenario_labels]
gas_vals  = [reiche_results[s]["gas_pct"] for s in scenario_labels]
x = np.arange(len(scenario_labels))
w = 0.5

bars_re  = ax.bar(x, re_vals,  w, label="Renewables + storage", color="#4cc9f0")
bars_gas = ax.bar(x, gas_vals, w, bottom=re_vals, label="Gas (residual)", color="#e6734a")

for i, (rv, gv) in enumerate(zip(re_vals, gas_vals)):
    ax.text(i, rv / 2,         f"{rv:.1f}%",  ha="center", va="center", fontsize=9, color="#003049")
    ax.text(i, rv + gv / 2,    f"{gv:.1f}%",  ha="center", va="center", fontsize=9, color="#7a1a00")

ax.set_xticks(x)
ax.set_xticklabels(scenario_labels, fontsize=9)
ax.set_ylabel("Share of annual demand (%)")
ax.set_title("Annual energy mix\n(Reiche 2045 capacities)", fontsize=11)
ax.set_ylim(0, 110)
ax.legend(fontsize=9)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(paths.images_path / "44_reiche_scenario.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/44_reiche_scenario.png
# :name: fig-44-reiche-scenario
# Left: residual load duration curve for the Reiche 2045 capacity plan
# (400 GW solar, 180 GW wind onshore, 70 GW offshore, 600 GWh storage).
# Each point on the curve is one hour per year sorted by gas dispatch size.
# The dashed line is the planned gas capacity (47 GW). Hours above that line
# represent a capacity gap. Right: resulting annual energy mix — share met by
# renewables and storage vs. gas backup for the two demand scenarios.
# ```

# %%

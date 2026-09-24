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
# # Renewable Energy Drawdowns — Germany
#
# Analyses multi-year "energy droughts" using PECD ERA5 capacity factors for
# Germany. For each renewable source we compute the cumulative sum of the
# capacity factor minus its long-run mean, giving a signed balance series
# that shows whether we are currently ahead of or behind the climatological
# average. Maximum drawdown (peak-to-trough decline) in this balance series
# quantifies the worst sustained energy shortfall in the historical record.
#
# **Inputs**
# - `data/processed/pecd/pecd_regions.parquet` — PECD ERA5 capacity factors

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

# %%
df = pd.read_parquet(paths.pecd_processed_file)

cf_solar    = df["solar_photovoltaic_power_generation"]["capacity_factor_ratio"]
cf_onshore  = df["wind_power_generation_onshore"]["capacity_factor_ratio"]
cf_offshore = df["wind_power_generation_offshore"]["capacity_factor_ratio"]

cf_solar    = cf_solar.dropna(axis=1, how="all")
cf_onshore  = cf_onshore.dropna(axis=1, how="all")
cf_offshore = cf_offshore.dropna(axis=1, how="all")

# %% [markdown]
# ## Build per-source capacity factor series for Germany

# %%
COUNTRY = "DE"

sources = {}

if COUNTRY in cf_solar.columns:
    sources["Solar PV"] = (cf_solar[COUNTRY], "#f4b942")

if COUNTRY in cf_onshore.columns:
    sources["Wind onshore"] = (cf_onshore[COUNTRY], "#4a90d9")

if COUNTRY in cf_offshore.columns:
    sources["Wind offshore"] = (cf_offshore[COUNTRY], "#1a5fa8")

mean_cfs = {name: s.mean() for name, (s, _) in sources.items()}

print(f"Sources available for {COUNTRY}: {list(sources.keys())}")
for name, (s, _) in sources.items():
    print(f"  {name}: {s.index[0]} → {s.index[-1]}, mean CF = {s.mean():.4f}")

# %% [markdown]
# ## Compute cumulative balance series
#
# For each source:
# ```
# balance(t) = cumsum( CF(t) - quantile(CF, 0.20) )
# ```
# Using the 20th-percentile as reference ensures the series generally trends
# upward (most hours exceed q20), so sustained below-q20 periods stand out
# as clear drawdowns rather than being masked by offsetting surpluses.

# %%
QUANTILE = 0.20


def cumulative_balance(series: pd.Series) -> pd.Series:
    """Cumulative sum of (CF - p20), y-axis in equivalent full-load hours."""
    deviation = series - series.quantile(QUANTILE)
    return deviation.cumsum()


balances = {name: cumulative_balance(s) for name, (s, _) in sources.items()}

# %% [markdown]
# ## Maximum drawdown calculation
#
# The drawdown at time *t* is the decline from the running maximum up to *t*,
# stored as a **negative** value so the series is capped at zero from above:
# ```
# drawdown(t) = balance(t) - running_max(t)   ≤ 0
# ```
# Maximum drawdown magnitude = `max( running_max - balance )` over the full series.

# %%
def max_drawdown_info(balance: pd.Series) -> dict:
    """Return start, trough, and magnitude of the maximum drawdown.

    ``drawdown`` is stored as non-positive values (0 = at peak, negative = below peak).
    """
    running_max = balance.cummax()
    drawdown    = balance - running_max          # ≤ 0 always

    trough_time  = drawdown.idxmin()             # most negative = worst drawdown
    trough_val   = balance[trough_time]
    peak_val     = running_max[trough_time]
    peak_time    = balance[:trough_time].idxmax()
    magnitude    = peak_val - trough_val         # positive scalar for reporting

    return {
        "peak_time":   peak_time,
        "trough_time": trough_time,
        "peak_val":    peak_val,
        "trough_val":  trough_val,
        "magnitude":   magnitude,
        "drawdown":    drawdown,                 # non-positive series
        "running_max": running_max,
    }


drawdown_info = {name: max_drawdown_info(bal) for name, bal in balances.items()}

print("Maximum drawdowns (× mean CF  |  days peak→trough):")
for name, info in drawdown_info.items():
    mag_norm = info["magnitude"] / mean_cfs[name]
    dur_hours = (info["trough_time"] - info["peak_time"]) / pd.Timedelta("1h")
    print(
        f"  {name}: {mag_norm:.1f}× mean CF  "
        f"  peak {info['peak_time'].date()} → trough {info['trough_time'].date()}"
        f"  ({dur_hours / 24:.1f} days)"
    )

# %%
# Save maximum drawdown periods to disk for downstream scripts (46, 47, 48)
drawdown_records = []
for name, info in drawdown_info.items():
    drawdown_records.append({
        "source": name,
        "peak_time": info["peak_time"],
        "trough_time": info["trough_time"],
        "peak_val": info["peak_val"],
        "trough_val": info["trough_val"],
        "magnitude": info["magnitude"],
        "mean_cf": mean_cfs[name],
    })

drawdown_df = pd.DataFrame(drawdown_records)
drawdown_df.to_parquet(paths.re_drawdowns_file, index=False)
print(f"\nSaved drawdown info → {paths.re_drawdowns_file}")

# %% [markdown]
# ## Top-N drawdown episodes
#
# A drawdown *episode* is a contiguous period where the balance is below its
# running maximum (`balance < cummax`). Each episode has one trough (the deepest
# point) and ends when the series recovers to a new high. We rank all episodes by
# magnitude and keep the five worst.

# %%
TOP_N = 5


def find_top_drawdowns(balance: pd.Series, n: int = TOP_N) -> pd.DataFrame:
    """Return the top-N drawdown episodes ranked by magnitude (MWh/MW).

    Each row contains:
    - ``peak_time``/``trough_time``/``recovery_time`` — key timestamps
    - ``magnitude`` — peak-to-trough depth (MWh/MW)
    - ``peak_to_trough_hours`` — hours from peak to trough
    """
    running_max = balance.cummax()
    dd = balance - running_max  # ≤ 0

    in_dd   = dd < 0
    ep_start = in_dd &  ~in_dd.shift(1, fill_value=False)
    ep_end   = ~in_dd &  in_dd.shift(1, fill_value=False)

    starts = balance.index[ep_start].tolist()
    ends   = balance.index[ep_end].tolist()
    if len(starts) > len(ends):          # series ends mid-drawdown
        ends.append(balance.index[-1])

    rows = []
    for s, e in zip(starts, ends):
        trough_t  = dd[s:e].idxmin()
        peak_val  = running_max[s]
        trough_val = balance[trough_t]
        magnitude  = peak_val - trough_val

        # Actual peak: last timestamp where the running-max level was set
        candidates = balance[:s][balance[:s] >= peak_val]
        peak_t     = candidates.index[-1] if len(candidates) else s

        rows.append({
            "peak_time":            peak_t,
            "trough_time":          trough_t,
            "recovery_time":        e,
            "magnitude":            magnitude,
            "peak_to_trough_hours": (trough_t - peak_t) / pd.Timedelta("1h"),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("magnitude", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


top_drawdowns = {name: find_top_drawdowns(bal) for name, bal in balances.items()}

print(f"\nTop-{TOP_N} drawdown episodes per source:")
for name, tdf in top_drawdowns.items():
    print(f"\n  {name}:")
    for _, row in tdf.iterrows():
        mag_norm = row["magnitude"] / mean_cfs[name]
        dur_days = row["peak_to_trough_hours"] / 24
        print(
            f"    {row['peak_time'].strftime('%Y-%m')} → {row['trough_time'].strftime('%Y-%m')}"
            f"  magnitude {mag_norm:.1f}× mean CF  duration {dur_days:.1f} d"
        )

# %% [markdown]
# ## Cumulative balance time series

# %%
n = len(sources)
fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), sharex=True)
if n == 1:
    axes = [axes]

for ax, (name, bal) in zip(axes, balances.items()):
    color  = sources[name][1]
    info   = drawdown_info[name]
    mcf    = mean_cfs[name]
    bal_n  = bal / mcf

    ax.plot(bal_n.index, bal_n.values, color=color, linewidth=0.8, label=name)
    ax.fill_between(bal_n.index, bal_n.values, 0,
                    where=(bal_n.values >= 0), color=color, alpha=0.15, linewidth=0)
    ax.fill_between(bal_n.index, bal_n.values, 0,
                    where=(bal_n.values < 0),  color="tomato", alpha=0.15, linewidth=0)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")

    # Mark the maximum drawdown peak and trough
    ax.axvline(info["peak_time"],   color=color,    linewidth=1.0, linestyle=":", alpha=0.8)
    ax.axvline(info["trough_time"], color="tomato", linewidth=1.0, linestyle=":", alpha=0.8)
    ax.annotate(
        f"Peak\n{info['peak_time'].strftime('%b %Y')}",
        xy=(info["peak_time"], info["peak_val"] / mcf),
        xytext=(10, 6), textcoords="offset points",
        fontsize=7.5, color=color,
    )
    ax.annotate(
        f"Trough\n{info['trough_time'].strftime('%b %Y')}\n−{info['magnitude'] / mcf:.1f}× mean",
        xy=(info["trough_time"], info["trough_val"] / mcf),
        xytext=(10, -30), textcoords="offset points",
        fontsize=7.5, color="tomato",
    )

    ax.set_ylabel("Cumulative balance (× mean hourly generation)")
    ax.set_title(f"{name} — cumulative CF balance vs p{int(QUANTILE*100)}", fontsize=10)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

axes[-1].set_xlabel("Date")
fig.suptitle(
    f"Germany — cumulative renewable energy balance (ERA5 PECD)\n"
    "Shaded red = deficit vs long-run mean; blue/yellow = surplus",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(paths.images_path / "45_cumulative_balance.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/45_cumulative_balance.png
# :name: fig-45-cumulative-balance
# Cumulative capacity-factor balance for Germany (ERA5 PECD). For each
# renewable source the series shows the running sum of hourly CF deviations
# from the long-run mean. Surplus periods are shaded in the source colour;
# deficit periods in red. Vertical dotted lines mark the peak and trough of
# the maximum historical drawdown.
# ```

# %% [markdown]
# ## Drawdown series over time

# %%
fig2, axes2 = plt.subplots(n, 1, figsize=(16, 3.5 * n), sharex=True)
if n == 1:
    axes2 = [axes2]

for ax, (name, bal) in zip(axes2, balances.items()):
    color  = sources[name][1]
    info   = drawdown_info[name]
    mcf    = mean_cfs[name]
    dd_n   = info["drawdown"] / mcf   # ≤ 0; normalized by mean hourly generation

    ax.fill_between(dd_n.index, dd_n.values, 0, color=color, alpha=0.35, linewidth=0)
    ax.plot(dd_n.index, dd_n.values, color=color, linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")

    # Annotate the worst point (most negative value)
    trough_dd = -info["magnitude"] / mcf   # negative y-coordinate, normalized
    dur_hours = (info["trough_time"] - info["peak_time"]) / pd.Timedelta("1h")
    ax.scatter([info["trough_time"]], [trough_dd], color="tomato", zorder=5, s=40)
    ax.annotate(
        f"−{info['magnitude'] / mcf:.1f}× mean  ({dur_hours / 24:.1f} d)\n"
        f"({info['peak_time'].strftime('%b %Y')} → {info['trough_time'].strftime('%b %Y')})",
        xy=(info["trough_time"], trough_dd),
        xytext=(12, -4), textcoords="offset points",
        fontsize=7.5, color="tomato", va="top",
    )

    ax.set_ylabel("Drawdown (× mean hourly generation)")
    ax.set_title(f"{name} — drawdown from running peak", fontsize=10)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

axes2[-1].set_xlabel("Date")
fig2.suptitle(
    "Germany — renewable energy drawdown from running cumulative peak (ERA5 PECD)",
    fontsize=11,
)
fig2.tight_layout()
fig2.savefig(paths.images_path / "45_drawdown_series.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/45_drawdown_series.png
# :name: fig-45-drawdown-series
# Drawdown from the running cumulative-balance peak for each renewable source
# in Germany. A large drawdown indicates a prolonged period where capacity
# factors were persistently below average. The red dot marks the historical
# maximum drawdown; the annotation shows the peak-to-trough interval and
# magnitude in equivalent full-load hours.
# ```

# %% [markdown]
# ## Top-5 drawdown episodes — summary chart

# %%
n_sources = len(sources)
fig3, axes3 = plt.subplots(n_sources, 2, figsize=(14, 3.5 * n_sources))
if n_sources == 1:
    axes3 = [axes3]

for row_axes, (name, tdf) in zip(axes3, top_drawdowns.items()):
    color = sources[name][1]
    ax_mag, ax_dur = row_axes

    labels = [
        f"#{i+1}  {r['peak_time'].strftime('%b %Y')} → {r['trough_time'].strftime('%b %Y')}"
        for i, (_, r) in enumerate(tdf.iterrows())
    ]
    mcf  = mean_cfs[name]
    mags = (tdf["magnitude"] / mcf).tolist()
    durs = (tdf["peak_to_trough_hours"] / 24).tolist()
    y    = range(len(tdf))

    # Magnitude panel
    ax_mag.barh(list(y), mags, color=color, edgecolor="white", linewidth=0.5)
    ax_mag.set_yticks(list(y))
    ax_mag.set_yticklabels(labels, fontsize=8)
    ax_mag.invert_yaxis()
    ax_mag.set_xlabel("Magnitude (× mean CF)")
    ax_mag.set_title(f"{name} — depth", fontsize=10)
    ax_mag.xaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax_mag.set_axisbelow(True)
    for yi, v in zip(y, mags):
        ax_mag.text(v + max(mags) * 0.01, yi, f"{v:.1f}×", va="center", fontsize=8)

    # Duration panel
    ax_dur.barh(list(y), durs, color=color, edgecolor="white", linewidth=0.5, alpha=0.7)
    ax_dur.set_yticks(list(y))
    ax_dur.set_yticklabels(labels, fontsize=8)
    ax_dur.invert_yaxis()
    ax_dur.set_xlabel("Peak-to-trough duration (days)")
    ax_dur.set_title(f"{name} — duration", fontsize=10)
    ax_dur.xaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax_dur.set_axisbelow(True)
    for yi, v in zip(y, durs):
        ax_dur.text(v + max(durs) * 0.01, yi, f"{v:.1f} d", va="center", fontsize=8)

fig3.suptitle(
    f"Germany — top {TOP_N} renewable energy drawdown episodes (ERA5 PECD)",
    fontsize=11,
)
fig3.tight_layout()
fig3.savefig(paths.images_path / "45_drawdown_summary.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/45_drawdown_summary.png
# :name: fig-45-drawdown-summary
# Top-5 drawdown episodes per renewable source for Germany (ERA5 PECD).
# Each episode is a contiguous period where the cumulative balance stayed
# below its previous peak. Left panels show depth (× mean hourly generation); right panels
# show the calendar duration from peak to trough. Events are ranked by
# magnitude (worst first).
# ```

# %% [markdown]
# ## Drawdown magnitude vs duration

# %%
fig4, ax4 = plt.subplots(figsize=(10, 6))

for name, tdf in top_drawdowns.items():
    color = sources[name][1]
    mcf   = mean_cfs[name]
    mags  = tdf["magnitude"] / mcf
    durs  = tdf["peak_to_trough_hours"] / 24

    ax4.scatter(durs, mags, color=color, s=60, label=name, edgecolors="white", linewidth=0.5, zorder=3)

ax4.set_xlabel("Peak-to-trough duration (days)")
ax4.set_ylabel("Magnitude (× mean hourly generation)")
ax4.set_title("Germany — drawdown magnitude vs duration (ERA5 PECD)", fontsize=11)
ax4.legend()
ax4.grid(True, linewidth=0.4, alpha=0.6)
ax4.set_axisbelow(True)
fig4.tight_layout()
fig4.savefig(paths.images_path / "45_drawdown_scatter.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/45_drawdown_scatter.png
# :name: fig-45-drawdown-scatter
# Drawdown magnitude vs duration for the top-5 episodes per renewable source.
# ```

# %% [markdown]
# ## Rolling generation sums per source

# %%
ROLLING_WINDOW = 5 * 24  # 5 days in hourly observations
MAX_DAYS = 350
N_LOWEST = 3


def find_non_overlapping_lowest(series: pd.Series, window_hours: int, n: int) -> list:
    """Find n lowest values from non-overlapping windows."""
    result = []
    excluded = set()
    sorted_series = series.dropna().sort_values()
    for idx, val in sorted_series.items():
        pos = series.index.get_loc(idx)
        if pos in excluded:
            continue
        result.append(val)
        # Exclude all positions that overlap with this window
        for p in range(pos - window_hours + 1, pos + window_hours):
            excluded.add(p)
        if len(result) >= n:
            break
    # Pad with NaN if fewer than n non-overlapping windows exist
    while len(result) < n:
        result.append(np.nan)
    return result


# %%
def name_slug(n):
    return n.lower().replace(" ", "_")

for name, (cf_series, color) in sources.items():
    slug = name_slug(name)
    # Fill NaN with 0 for rolling sums (solar nighttime NaN = no generation)
    cf_filled = cf_series.fillna(0)

    # --- 5-day rolling sum time series ---
    rolling_sum = cf_filled.rolling(ROLLING_WINDOW).sum()

    fig_rs, ax_rs = plt.subplots(figsize=(14, 5))
    ax_rs.plot(rolling_sum.index, rolling_sum.values, color=color, linewidth=0.5)
    ax_rs.set_ylabel(f"Rolling {ROLLING_WINDOW // 24}-day sum of CF")
    ax_rs.set_xlabel("Time")
    ax_rs.set_title(f"{name} ({COUNTRY}) — {ROLLING_WINDOW // 24}-day rolling generation sum", fontsize=11)
    ax_rs.grid(True, linewidth=0.4, alpha=0.6)
    ax_rs.set_axisbelow(True)
    fig_rs.tight_layout()
    fig_rs.savefig(paths.images_path / f"45_{slug}_rolling.png", dpi=150, bbox_inches="tight")
    show()

    # --- Lowest non-overlapping rolling sums by window length ---
    lowest_values = {}
    for days in range(1, MAX_DAYS + 1):
        window_hours = days * 24
        rs = cf_filled.rolling(window_hours).sum()
        lowest_values[days] = find_non_overlapping_lowest(rs, window_hours, N_LOWEST)

    lowest_df = pd.DataFrame(lowest_values, index=[f"#{i+1}" for i in range(N_LOWEST)]).T
    lowest_df.index.name = "window (days)"

    mean_cf   = mean_cfs[name]
    median_cf = cf_series.median()
    days_range  = np.arange(1, MAX_DAYS + 1)
    avg_line    = mean_cf   * days_range * 24
    median_line = median_cf * days_range * 24

    fig_lo, ax_lo = plt.subplots(figsize=(12, 5))

    ax_lo.plot(days_range, avg_line, color="black", linewidth=1.5, linestyle="--",
               label=f"Mean (CF = {mean_cf:.3f})")
    ax_lo.plot(days_range, median_line, color="grey", linewidth=1.5, linestyle=":",
               label=f"Median (CF = {median_cf:.3f})")

    for i in range(N_LOWEST):
        ax_lo.plot(lowest_df.index, lowest_df.iloc[:, i], marker="o", markersize=4,
                   color=color, alpha=1.0 - i * 0.25, linewidth=1.5, label=f"#{i+1} lowest")

    ax_lo.set_xlabel("Rolling window (days)")
    ax_lo.set_ylabel("Rolling sum of CF")
    ax_lo.set_title(f"{name} ({COUNTRY}) — {N_LOWEST} lowest non-overlapping rolling sums vs average", fontsize=11)
    ax_lo.legend()
    ax_lo.grid(True, linewidth=0.4, alpha=0.6)
    ax_lo.set_axisbelow(True)
    ax_lo.set_xticks(range(0, MAX_DAYS + 1, 5))
    fig_lo.tight_layout()
    fig_lo.savefig(paths.images_path / f"45_{slug}_lowest_rolling.png", dpi=150, bbox_inches="tight")
    show()

# %% [markdown]
# ```{figure} ../../output/images/45_solar_pv_rolling.png
# :name: fig-45-solar-pv-rolling
# 5-day rolling sum of solar PV capacity factor for Germany (ERA5 PECD).
# ```
#
# ```{figure} ../../output/images/45_solar_pv_lowest_rolling.png
# :name: fig-45-solar-pv-lowest-rolling
# Three lowest non-overlapping rolling CF sums for solar PV across window
# lengths from 1 to 40 days.
# ```
#
# ```{figure} ../../output/images/45_wind_onshore_rolling.png
# :name: fig-45-wind-onshore-rolling
# 5-day rolling sum of wind onshore capacity factor for Germany (ERA5 PECD).
# ```
#
# ```{figure} ../../output/images/45_wind_onshore_lowest_rolling.png
# :name: fig-45-wind-onshore-lowest-rolling
# Three lowest non-overlapping rolling CF sums for wind onshore across window
# lengths from 1 to 40 days.
# ```
#
# ```{figure} ../../output/images/45_wind_offshore_rolling.png
# :name: fig-45-wind-offshore-rolling
# 5-day rolling sum of wind offshore capacity factor for Germany (ERA5 PECD).
# ```
#
# ```{figure} ../../output/images/45_wind_offshore_lowest_rolling.png
# :name: fig-45-wind-offshore-lowest-rolling
# Three lowest non-overlapping rolling CF sums for wind offshore across window
# lengths from 1 to 40 days.
# ```

# %%

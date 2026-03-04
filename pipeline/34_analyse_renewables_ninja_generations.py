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
# # Renewable Generation Analysis
#
# Analyses the hourly solar PV, onshore wind, and offshore wind capacity factors
# produced by `pipeline/32_dev_energy_generation.py` (renewables.ninja / MERRA-2).
#
# **Inputs**
# - `ninja_pv_cf.parquet`: hourly PV capacity factors (0–1) per Bundesland
# - `ninja_wind_onshore_cf.parquet`: hourly onshore wind capacity factors per Bundesland
# - `ninja_wind_offshore_cf.parquet`: hourly offshore wind capacity factors per sea region
# - `pv_state_aggregates.parquet`: PV installed capacity per state
# - `wind_onshore_state_aggregates.parquet`: onshore wind capacity per state
# - `wind_offshore_aggregates.parquet`: offshore wind capacity per sea region

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from woe.paths import ProjPaths

paths = ProjPaths()

YEAR = 2019

# %%
pv_cf = pd.read_parquet(paths.ninja_pv_cf_file)
wind_onshore_cf = pd.read_parquet(paths.ninja_wind_onshore_cf_file)
wind_offshore_cf = pd.read_parquet(paths.ninja_wind_offshore_cf_file)
pv_agg = pd.read_parquet(paths.pv_state_aggregates_file)
wind_onshore_agg = pd.read_parquet(paths.wind_onshore_state_aggregates_file)
wind_offshore_agg = pd.read_parquet(paths.wind_offshore_aggregates_file)

print(f"PV CF:            {pv_cf.shape}  (rows=hours, cols=states)")
print(f"Wind onshore CF:  {wind_onshore_cf.shape}")
print(f"Wind offshore CF: {wind_offshore_cf.shape}  (cols=Nordsee, Ostsee)")

# %% [markdown]
# ## Monthly generation summary

# %%
cap_pv = pv_agg.set_index("state")["capacity_MW"]
cap_onshore = wind_onshore_agg.set_index("state")["capacity_MW"]
cap_offshore = wind_offshore_agg.set_index("region")["capacity_MW"]

pv_de = (pv_cf * cap_pv).sum(axis=1)
wind_onshore_de = (wind_onshore_cf * cap_onshore).sum(axis=1)
wind_offshore_de = (wind_offshore_cf * cap_offshore).sum(axis=1)

monthly = pd.DataFrame({
    "PV": pv_de.resample("ME").sum() / 1e3,
    "Wind Onshore": wind_onshore_de.resample("ME").sum() / 1e3,
    "Wind Offshore": wind_offshore_de.resample("ME").sum() / 1e3,
})
monthly.index = monthly.index.strftime("%b")

fig, ax = plt.subplots(figsize=(12, 5))
x = range(len(monthly))
width = 0.38
ax.bar([i - width / 2 for i in x], monthly["PV"], width=width,
       label="Solar PV", color="#f4b942", edgecolor="white", linewidth=0.5)
ax.bar([i + width / 2 for i in x], monthly["Wind Onshore"], width=width,
       label="Wind onshore", color="#4a90d9", edgecolor="white", linewidth=0.5)
ax.bar([i + width / 2 for i in x], monthly["Wind Offshore"], width=width,
       bottom=monthly["Wind Onshore"],
       label="Wind offshore", color="#1a5fa8", edgecolor="white", linewidth=0.5)
ax.set_xticks(list(x))
ax.set_xticklabels(monthly.index)
ax.set_ylabel("Generation (GWh)")
ax.set_title(
    f"Monthly Solar PV and Wind Generation — Germany {YEAR}\n"
    "(renewables.ninja / MERRA-2, capacity-weighted centroids)",
    fontsize=11,
)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(paths.images_path / "34_monthly_generation.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/34_monthly_generation.png
# :name: fig-34-monthly-generation
# Monthly solar PV, onshore wind, and offshore wind electricity generation for
# Germany in 2019, modelled via renewables.ninja (MERRA-2 reanalysis).
# Wind bars are stacked: onshore (light blue) + offshore (dark blue).
# PV peaks sharply in summer; wind shows the inverse seasonal pattern with
# highest output in winter months.
# ```

# %% [markdown]
# ## Annual capacity factors per state (onshore wind and PV)

# %%
cf_pv = pv_cf.mean().rename("CF_PV")
cf_wind_onshore = wind_onshore_cf.mean().rename("CF_Wind_Onshore")

cf_summary = pd.concat([cf_pv, cf_wind_onshore], axis=1).dropna(how="all")
cf_summary = cf_summary.sort_values("CF_Wind_Onshore", ascending=True)

fig2, ax2 = plt.subplots(figsize=(10, 6))
y = range(len(cf_summary))
ax2.barh([i + 0.2 for i in y], cf_summary["CF_PV"], height=0.35,
         label="Solar PV", color="#f4b942", edgecolor="white", linewidth=0.5)
ax2.barh([i - 0.2 for i in y], cf_summary["CF_Wind_Onshore"], height=0.35,
         label="Wind (onshore)", color="#4a90d9", edgecolor="white", linewidth=0.5)
ax2.set_yticks(list(y))
ax2.set_yticklabels(cf_summary.index, fontsize=9)
ax2.set_xlabel("Annual capacity factor")
ax2.set_title(
    f"Annual Capacity Factors by Bundesland — {YEAR}\n"
    "(renewables.ninja / MERRA-2)",
    fontsize=11,
)
ax2.legend(fontsize=10)
ax2.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax2.set_axisbelow(True)
fig2.tight_layout()
fig2.savefig(paths.images_path / "34_capacity_factors.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/34_capacity_factors.png
# :name: fig-34-capacity-factors
# Annual capacity factors for solar PV and onshore wind by Bundesland in 2019.
# Northern states (Schleswig-Holstein, Mecklenburg-Vorpommern) show the highest
# wind capacity factors; southern states (Bayern, Baden-Württemberg) lead for PV.
# Offshore wind capacity factors (Nordsee, Ostsee) are substantially higher
# than any onshore state and are not shown here.
# ```

# %% [markdown]
# ## Capacity factor choropleths

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.colors as mcolors
import matplotlib.cm as cm

PROJ = ccrs.LambertConformal(central_longitude=10.5, central_latitude=51.5)
GERMANY_EXTENT = [5.5, 15.5, 47.0, 55.5]  # [W, E, S, N] in PlateCarree degrees

shpfilename = shpreader.natural_earth(
    resolution="10m", category="cultural", name="admin_1_states_provinces_lakes"
)
german_states = {
    r.attributes["name"]: r.geometry
    for r in shpreader.Reader(shpfilename).records()
    if r.attributes["admin"] == "Germany"
}

# %% [markdown]
# ### Solar PV and wind onshore capacity factor maps

# %%
fig_map, (ax_pv, ax_wind) = plt.subplots(
    1, 2, subplot_kw={"projection": PROJ}, figsize=(16, 9)
)

for ax, cf, cmap_name, title in [
    (ax_pv,   cf_pv,          "YlOrRd", f"Solar PV — annual CF ({YEAR})"),
    (ax_wind, cf_wind_onshore, "Blues",  f"Wind onshore — annual CF ({YEAR})"),
]:
    cmap = plt.colormaps[cmap_name]
    norm = mcolors.Normalize(vmin=cf.min(), vmax=cf.max())
    ax.set_extent(GERMANY_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="#c6def1")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
    for name, geom in german_states.items():
        val = cf.get(name, np.nan)
        face = cmap(norm(val)) if not np.isnan(val) else "lightgrey"
        ax.add_geometries([geom], ccrs.PlateCarree(),
                          facecolor=face, edgecolor="black", linewidth=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation="vertical",
                 fraction=0.03, pad=0.04).set_label("Annual capacity factor", fontsize=9)
    ax.set_title(title + "\n(renewables.ninja / MERRA-2)", fontsize=10, pad=8)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)

fig_map.tight_layout()
fig_map.savefig(paths.images_path / "34_cf_choropleths.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/34_cf_choropleths.png
# :name: fig-34-cf-choropleths
# Annual capacity factors for solar PV (left) and onshore wind (right) by
# Bundesland in 2019 (renewables.ninja / MERRA-2). PV efficiency increases
# southward; wind efficiency increases northward toward the coast.
# ```

# %% [markdown]
# ### Offshore wind capacity factors

# %%
cf_offshore = wind_offshore_cf.mean().sort_values(ascending=False)

fig_off, ax_off = plt.subplots(figsize=(5, 4))
colors_off = ["#1a5fa8", "#4a90d9"]
bars = ax_off.bar(cf_offshore.index, cf_offshore.values,
                  color=colors_off, edgecolor="white", linewidth=0.5, width=0.4)
for bar, val in zip(bars, cf_offshore.values):
    ax_off.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax_off.set_ylabel("Annual capacity factor")
ax_off.set_title(
    f"Offshore wind capacity factors — {YEAR}\n(renewables.ninja / MERRA-2)", fontsize=11
)
ax_off.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax_off.set_axisbelow(True)
fig_off.tight_layout()
fig_off.savefig(paths.images_path / "34_cf_offshore.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/34_cf_offshore.png
# :name: fig-34-cf-offshore
# Annual capacity factors for Nordsee and Ostsee offshore wind in 2019.
# Both sites substantially outperform all onshore Bundesländer; Nordsee is
# marginally stronger due to higher North Sea wind speeds.
# ```

# %% [markdown]
# ## Spatial correlation of capacity factors

# %%
def _corr_heatmap(ax, corr, cmap, title):
    im = ax.imshow(corr.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    n = len(corr)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson r", fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=4.5, color="white" if v > 0.75 else "black")


# %% [markdown]
# ### PV correlation (hourly and daily)

# %%
fig_cpv, (ax_ph, ax_pd) = plt.subplots(1, 2, figsize=(18, 8))
_corr_heatmap(ax_ph, pv_cf.corr(), "YlOrRd",
              f"Solar PV capacity factor — hourly correlation ({YEAR})")
_corr_heatmap(ax_pd, pv_cf.resample("D").mean().corr(), "YlOrRd",
              f"Solar PV capacity factor — daily correlation ({YEAR})")
fig_cpv.tight_layout()
fig_cpv.savefig(paths.images_path / "34_corr_pv.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/34_corr_pv.png
# :name: fig-34-corr-pv
# Pearson correlation matrix of solar PV capacity factors across all 16 Bundesländer,
# at hourly (left) and daily (right) resolution in 2019. Daily correlations are
# universally high (>0.95) reflecting the shared irradiance seasonality; hourly
# correlations reveal modest east–west and north–south cloud-cover differences.
# ```

# %% [markdown]
# ### Wind correlation (hourly and daily, onshore + offshore)

# %%
wind_combined_cf = pd.concat([wind_onshore_cf, wind_offshore_cf], axis=1)

fig_cw, (ax_wh, ax_wd) = plt.subplots(1, 2, figsize=(20, 9))
_corr_heatmap(ax_wh, wind_combined_cf.corr(), "Blues",
              f"Wind capacity factor — hourly correlation ({YEAR})")
_corr_heatmap(ax_wd, wind_combined_cf.resample("D").mean().corr(), "Blues",
              f"Wind capacity factor — daily correlation ({YEAR})")
fig_cw.tight_layout()
fig_cw.savefig(paths.images_path / "34_corr_wind.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/34_corr_wind.png
# :name: fig-34-corr-wind
# Pearson correlation matrix of wind capacity factors across all 16 onshore
# Bundesländer plus Nordsee and Ostsee offshore regions, at hourly (left) and
# daily (right) resolution in 2019. Coastal and northern states cluster together;
# southern inland states show weaker correlation with offshore and northern sites.
# ```

# %% [markdown]
# ## Validation: synthetic vs SMARD actual generation
#
# Compares renewables.ninja/MERRA-2 simulated hourly output against measured
# SMARD generation data (DE/LU bidding zone) for 2019.
#
# SMARD timestamps are in CET/CEST (Europe/Berlin local time); the ninja UTC
# index is converted to Europe/Berlin so both series share the same clock.

# %%
smard_solar = pd.read_parquet(paths.smard_solar_file)["SOLAR"]
smard_wind_on = pd.read_parquet(paths.smard_wind_onshore_file)["WIND_ONSHORE"]
smard_wind_off = pd.read_parquet(paths.smard_wind_offshore_file)["WIND_OFFSHORE"]

smard_solar = smard_solar[smard_solar.index.year == YEAR]
smard_wind_on = smard_wind_on[smard_wind_on.index.year == YEAR]
smard_wind_off = smard_wind_off[smard_wind_off.index.year == YEAR]


def to_berlin_naive(series: pd.Series) -> pd.Series:
    """Convert a UTC-aware Series to Europe/Berlin tz-naive."""
    new_idx = series.index.tz_convert("Europe/Berlin").tz_localize(None)
    return series.set_axis(new_idx)


ninja_solar = to_berlin_naive(pv_de)
ninja_wind_on = to_berlin_naive(wind_onshore_de)
ninja_wind_off = to_berlin_naive(wind_offshore_de)

# %% [markdown]
# ### Daily generation: synthetic vs SMARD (full year)

# %%
_specs_lines = [
    (ninja_solar, smard_solar, "Solar PV", "#f4b942"),
    (ninja_wind_on, smard_wind_on, "Wind Onshore", "#4a90d9"),
    (ninja_wind_off, smard_wind_off, "Wind Offshore", "#1a5fa8"),
]

fig3, axes3 = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for ax, (ninja, smard, label, color) in zip(axes3, _specs_lines):
    d_ninja = ninja.resample("D").sum() / 1e3   # GWh
    d_smard = smard.resample("D").sum() / 1e3
    common = d_smard.index.intersection(d_ninja.index)
    ax.plot(common, d_smard.loc[common], color="0.4", linewidth=0.9, label="SMARD (actual)")
    ax.plot(common, d_ninja.loc[common], color=color, linewidth=0.9, alpha=0.85,
            label="renewables.ninja")
    ax.set_ylabel("GWh / day")
    ax.set_title(f"{label}: daily generation — {YEAR}", fontsize=10)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
axes3[-1].set_xlabel("Date")
fig3.tight_layout()
fig3.savefig(paths.images_path / "34_daily_lines.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/34_daily_lines.png
# :name: fig-34-daily-lines
# Daily electricity generation (GWh) from renewables.ninja/MERRA-2 (coloured)
# versus SMARD actuals (grey) for Germany in 2019.
# The model broadly tracks the seasonal pattern for all three technologies but
# systematically over-estimates solar PV and, to a lesser extent, wind.
# ```

# %% [markdown]
# ### Daily absolute differences (full year)

# %%
fig4, axes4 = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for ax, (ninja, smard, label, color) in zip(axes4, _specs_lines):
    d_ninja = ninja.resample("D").sum() / 1e3   # GWh
    d_smard = smard.resample("D").sum() / 1e3
    common = d_smard.index.intersection(d_ninja.index)
    diff = d_ninja.loc[common] - d_smard.loc[common]
    med = diff.median()
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(diff.index, diff, 0,
                    where=diff >= 0, color=color, alpha=0.5, label="Over-estimate")
    ax.fill_between(diff.index, diff, 0,
                    where=diff < 0, color="tomato", alpha=0.5, label="Under-estimate")
    ax.axhline(med, color="black", linewidth=1.0, linestyle="--", label=f"median {med:+.0f} GWh")
    ax.set_ylabel("GWh / day")
    ax.set_title(f"{label}: renewables.ninja vs SMARD — daily difference ({YEAR})", fontsize=10)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
axes4[-1].set_xlabel("Date")
fig4.tight_layout()
fig4.savefig(paths.images_path / "34_daily_diff.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/34_daily_diff.png
# :name: fig-34-daily-diff
# Daily absolute difference (ninja − SMARD, GWh) for 2019. Positive values
# (coloured) indicate over-estimation; negative values (red) indicate
# under-estimation. Dashed line shows the annual median difference per technology.
# ```

# %% [markdown]
# ### Combined renewables: daily comparison (absolute, absolute diff, relative diff)
#
# Aggregates all three technologies (solar PV + wind onshore + wind offshore) into
# a single daily total. Days where the SMARD total is below 200 GWh are excluded
# from the relative deviation panel and marked with grey ticks.

# %%
MIN_TOTAL_GWH = 200   # threshold below which relative deviation is suppressed

d_ninja_total = (ninja_solar + ninja_wind_on + ninja_wind_off).resample("D").sum() / 1e3
d_smard_total = (smard_solar + smard_wind_on + smard_wind_off).resample("D").sum() / 1e3
common_d = d_smard_total.index.intersection(d_ninja_total.index)

d_ninja_total = d_ninja_total.loc[common_d]
d_smard_total = d_smard_total.loc[common_d]
d_abs_diff    = d_ninja_total - d_smard_total
d_rel_diff    = (d_abs_diff / d_smard_total.where(d_smard_total >= MIN_TOTAL_GWH)) * 100

excluded_days = common_d[d_smard_total < MIN_TOTAL_GWH]

fig5t, (ax_t1, ax_t2, ax_t3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

# — panel 1: absolute levels —
ax_t1.plot(common_d, d_smard_total, color="0.4", linewidth=1.0, label="SMARD (actual)")
ax_t1.plot(common_d, d_ninja_total, color="#4a90d9", linewidth=1.0, alpha=0.85,
           label="renewables.ninja")
ax_t1.set_ylabel("GWh / day")
ax_t1.set_title(
    f"Combined renewables (PV + onshore + offshore wind) — daily generation {YEAR}",
    fontsize=10,
)
ax_t1.legend(fontsize=9)
ax_t1.yaxis.grid(True, linewidth=0.4, alpha=0.5)
ax_t1.set_axisbelow(True)

# — panel 2: absolute difference —
ax_t2.axhline(0, color="black", linewidth=0.8)
ax_t2.fill_between(common_d, d_abs_diff, 0,
                   where=d_abs_diff >= 0, color="#4a90d9", alpha=0.5, label="Over-estimate")
ax_t2.fill_between(common_d, d_abs_diff, 0,
                   where=d_abs_diff < 0, color="tomato", alpha=0.5, label="Under-estimate")
med_abs = d_abs_diff.median()
ax_t2.axhline(med_abs, color="black", linewidth=1.0, linestyle="--",
              label=f"median {med_abs:+.0f} GWh")
ax_t2.set_ylabel("GWh / day")
ax_t2.set_title("Absolute difference (ninja − SMARD)", fontsize=10)
ax_t2.legend(fontsize=9)
ax_t2.yaxis.grid(True, linewidth=0.4, alpha=0.5)
ax_t2.set_axisbelow(True)

# — panel 3: relative difference —
ax_t3.axhline(0, color="black", linewidth=0.8)
ax_t3.fill_between(common_d, d_rel_diff.clip(-100, 100), 0,
                   where=d_rel_diff.fillna(0) >= 0, color="#4a90d9", alpha=0.5,
                   label="Over-estimate")
ax_t3.fill_between(common_d, d_rel_diff.clip(-100, 100), 0,
                   where=d_rel_diff.fillna(0) < 0, color="tomato", alpha=0.5,
                   label="Under-estimate")
med_rel = d_rel_diff.dropna().median()
ax_t3.axhline(med_rel, color="black", linewidth=1.0, linestyle="--",
              label=f"median {med_rel:+.0f}%")
if len(excluded_days):
    ax_t3.plot(excluded_days, np.zeros(len(excluded_days)),
               marker="|", color="0.5", markersize=8, linewidth=0, alpha=0.7,
               label=f"excluded ({len(excluded_days)} days, SMARD < {MIN_TOTAL_GWH} GWh)")
ax_t3.set_ylabel("% deviation")
ax_t3.set_title("Relative difference (ninja − SMARD) / SMARD", fontsize=10)
ax_t3.legend(fontsize=9)
ax_t3.yaxis.grid(True, linewidth=0.4, alpha=0.5)
ax_t3.set_axisbelow(True)
ax_t3.set_xlabel("Date")

fig5t.tight_layout()
fig5t.savefig(paths.images_path / "34_total_daily_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/34_total_daily_comparison.png
# :name: fig-34-total-daily-comparison
# Daily combined renewables generation (solar PV + onshore + offshore wind) for
# Germany in 2019: absolute levels (top), absolute difference ninja − SMARD (middle),
# and relative difference (bottom). Grey ticks on the zero line mark days excluded
# from the relative panel because SMARD total was below 200 GWh.
# ```

# %% [markdown]
# ### Hourly generation: synthetic vs SMARD — sample weeks
#
# Winter week (Jan 14–20) and summer week (Jul 8–14).

# %%
WEEK_WINTER = ("2019-01-14", "2019-01-20")
WEEK_SUMMER = ("2019-07-08", "2019-07-14")

_weeks = [
    (*WEEK_WINTER, "Winter"),
    (*WEEK_SUMMER, "Summer"),
]
for week_start, week_end, season in _weeks:
    slc = slice(week_start, week_end)
    fig5, axes5 = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, (ninja, smard, label, color) in zip(axes5, _specs_lines):
        common = smard.loc[slc].index.intersection(ninja.loc[slc].index)
        ax.plot(common, smard.loc[common] / 1e3, color="0.4", linewidth=1.2,
                label="SMARD (actual)")
        ax.plot(common, ninja.loc[common] / 1e3, color=color, linewidth=1.2, alpha=0.85,
                label="renewables.ninja")
        ax.set_ylabel("GW")
        ax.set_title(
            f"{label}: hourly generation — {season} week ({week_start} to {week_end})",
            fontsize=10,
        )
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
    axes5[-1].set_xlabel("Hour (Europe/Berlin)")
    fig5.tight_layout()
    fname = f"34_hourly_lines_{season.lower()}.png"
    fig5.savefig(paths.images_path / fname, dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ```{figure} ../../output/images/34_hourly_lines_winter.png
# :name: fig-34-hourly-lines-winter
# Hourly electricity generation (GW) from renewables.ninja (coloured) vs SMARD
# actuals (grey) for a winter week (Jan 14–20, 2019).
# ```
#
# ```{figure} ../../output/images/34_hourly_lines_summer.png
# :name: fig-34-hourly-lines-summer
# Hourly electricity generation (GW) from renewables.ninja (coloured) vs SMARD
# actuals (grey) for a summer week (Jul 8–14, 2019). The midday solar peak is
# clearly visible, with the model consistently exceeding measured output.
# ```

# %% [markdown]
# ### Hourly absolute differences — sample weeks

# %%
for week_start, week_end, season in _weeks:
    slc = slice(week_start, week_end)
    fig6, axes6 = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, (ninja, smard, label, color) in zip(axes6, _specs_lines):
        common = smard.loc[slc].index.intersection(ninja.loc[slc].index)
        diff = (ninja.loc[common] - smard.loc[common]) / 1e3   # GW
        ax.axhline(0, color="black", linewidth=0.8)
        ax.fill_between(diff.index, diff, 0,
                        where=diff >= 0, color=color, alpha=0.5, label="Over-estimate")
        ax.fill_between(diff.index, diff, 0,
                        where=diff < 0, color="tomato", alpha=0.5, label="Under-estimate")
        ax.set_ylabel("GW")
        ax.set_title(
            f"{label}: hourly difference — {season} week ({week_start} to {week_end})",
            fontsize=10,
        )
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
    axes6[-1].set_xlabel("Hour (Europe/Berlin)")
    fig6.tight_layout()
    fname = f"34_hourly_diff_{season.lower()}.png"
    fig6.savefig(paths.images_path / fname, dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ```{figure} ../../output/images/34_hourly_diff_winter.png
# :name: fig-34-hourly-diff-winter
# Hourly absolute difference (ninja − SMARD, GW) for a winter week
# (Jan 14–20, 2019). Positive values indicate over-estimation; negative values
# indicate under-estimation.
# ```
#
# ```{figure} ../../output/images/34_hourly_diff_summer.png
# :name: fig-34-hourly-diff-summer
# Hourly absolute difference (ninja − SMARD, GW) for a summer week
# (Jul 8–14, 2019). The midday solar over-estimation is clearly structured
# around the daily irradiance cycle.
# ```

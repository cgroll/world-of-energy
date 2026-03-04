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
# # PECD Regional Capacity Factors
#
# Analyses hourly capacity factors and power generation from the
# Pan-European Climate Database (PECD) ERA5 reanalysis, covering European
# countries from 1979 onwards at NUTS 0 resolution.
#
# **Inputs**
# - `data/processed/pecd/pecd_regions.parquet`: wide-format parquet with
#   MultiIndex columns `(variable, product_type, country)`

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

from woe.paths import ProjPaths

paths = ProjPaths()

# %%
df = pd.read_parquet(paths.pecd_processed_file)

variables = df.columns.get_level_values("variable").unique().tolist()
product_types = df.columns.get_level_values("product_type").unique().tolist()

print(f"Shape:        {df.shape}")
print(f"Time range:   {df.index[0]} → {df.index[-1]}")
print(f"Variables:    {variables}")
print(f"Product types:{product_types}")

# %%
# Extract capacity factor slices for the three technologies
cf_solar    = df["solar_photovoltaic_power_generation"]["capacity_factor_ratio"]
cf_onshore  = df["wind_power_generation_onshore"]["capacity_factor_ratio"]
cf_offshore = df["wind_power_generation_offshore"]["capacity_factor_ratio"]

# Drop countries with no data at all
cf_solar    = cf_solar.dropna(axis=1, how="all")
cf_onshore  = cf_onshore.dropna(axis=1, how="all")
cf_offshore = cf_offshore.dropna(axis=1, how="all")

print(f"Solar    CF: {len(cf_solar.columns)} countries")
print(f"Onshore  CF: {len(cf_onshore.columns)} countries")
print(f"Offshore CF: {len(cf_offshore.columns)} countries")

# %% [markdown]
# ## Long-run mean capacity factors by country

# %%
mean_solar    = cf_solar.mean().sort_values()
mean_onshore  = cf_onshore.mean().sort_values()
mean_offshore = cf_offshore.mean().sort_values()

fig, axes = plt.subplots(1, 3, figsize=(18, 10))

for ax, series, color, label in [
    (axes[0], mean_solar,    "#f4b942", "Solar PV"),
    (axes[1], mean_onshore,  "#4a90d9", "Wind onshore"),
    (axes[2], mean_offshore, "#1a5fa8", "Wind offshore"),
]:
    ax.barh(series.index, series.values, color=color, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Long-run mean capacity factor")
    ax.set_title(f"{label}\n(ERA5 1979–2026)", fontsize=10)
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)

fig.suptitle("PECD ERA5 — Long-run mean capacity factors by country", fontsize=12)
fig.tight_layout()
fig.savefig(paths.images_path / "37_mean_cf_by_country.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/37_mean_cf_by_country.png
# :name: fig-37-mean-cf-by-country
# Long-run mean capacity factors (1979–2026) by country for solar PV (left),
# onshore wind (middle), and offshore wind (right). Southern European countries
# (MT, CY, ES, PT) lead for solar; northern and coastal countries
# (IS, IE, UK, NO) lead for wind.
# ```

# %% [markdown]
# ## Seasonal profile for Germany

# %%
DE_SOLAR    = cf_solar["DE"]
DE_ONSHORE  = cf_onshore["DE"]
DE_OFFSHORE = cf_offshore.get("DE", pd.Series(dtype=float))

monthly_solar    = DE_SOLAR.groupby(DE_SOLAR.index.month).mean()
monthly_onshore  = DE_ONSHORE.groupby(DE_ONSHORE.index.month).mean()

month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(month_labels, monthly_solar.values,
         color="#f4b942", linewidth=2, marker="o", markersize=5, label="Solar PV")
ax2.plot(month_labels, monthly_onshore.values,
         color="#4a90d9", linewidth=2, marker="o", markersize=5, label="Wind onshore")

if not DE_OFFSHORE.empty:
    monthly_offshore = DE_OFFSHORE.groupby(DE_OFFSHORE.index.month).mean()
    ax2.plot(month_labels, monthly_offshore.values,
             color="#1a5fa8", linewidth=2, marker="o", markersize=5, label="Wind offshore")

ax2.set_ylabel("Mean capacity factor")
ax2.set_title("Germany — monthly mean capacity factors (ERA5 1979–2026)", fontsize=11)
ax2.legend(fontsize=10)
ax2.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax2.set_axisbelow(True)
fig2.tight_layout()
fig2.savefig(paths.images_path / "37_seasonal_profile_de.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/37_seasonal_profile_de.png
# :name: fig-37-seasonal-profile-de
# Monthly mean capacity factors for Germany (ERA5 1979–2026 climatology).
# Solar PV peaks sharply in June–July; wind onshore peaks in winter months,
# creating a natural seasonal complementarity.
# ```

# %% [markdown]
# ## Monthly capacity factor distributions across Europe (jitter plots)

# %%
# Year-month averages for Germany: groupby (year, month) → mean → shape (12, n_years)
monthly_dist_solar = (
    cf_solar["DE"]
    .groupby([cf_solar.index.year, cf_solar.index.month]).mean()
    .unstack(level=0)
)
monthly_dist_onshore = (
    cf_onshore["DE"]
    .groupby([cf_onshore.index.year, cf_onshore.index.month]).mean()
    .unstack(level=0)
)
_de_offshore = cf_offshore.get("DE", pd.Series(dtype=float))
if not _de_offshore.empty:
    monthly_dist_offshore = (
        _de_offshore
        .groupby([_de_offshore.index.year, _de_offshore.index.month]).mean()
        .unstack(level=0)
    )
else:
    monthly_dist_offshore = pd.DataFrame()

rng = np.random.default_rng(42)


def _jitter_plot(ax, monthly_dist, color, title):
    months = range(1, 13)
    for m in months:
        vals = monthly_dist.loc[m].dropna().values
        jitter = rng.uniform(-0.2, 0.2, size=len(vals))
        ax.scatter(m + jitter, vals, color=color, s=40, alpha=0.85,
                   edgecolors="black", linewidths=0.8)

    ax.set_xticks(list(months))
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("Mean capacity factor")
    ax.set_title(title, fontsize=11)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)


# %%
fig_bp1, ax_bp1 = plt.subplots(figsize=(13, 5))
_jitter_plot(ax_bp1, monthly_dist_solar, "#f4b942",
                "Solar PV — monthly CF for Germany, one point per year (ERA5 1979–2026)")
fig_bp1.tight_layout()
fig_bp1.savefig(paths.images_path / "37_monthly_boxplot_solar.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/37_monthly_boxplot_solar.png
# :name: fig-37-monthly-boxplot-solar
# Monthly capacity factors for solar PV in Germany (ERA5 1979–2026). Each jittered
# dot represents the mean capacity factor for one calendar year in that month,
# showing interannual variability. Generation is strongly concentrated in summer
# months, with very low values in winter.
# ```

# %%
fig_bp2, ax_bp2 = plt.subplots(figsize=(13, 5))
_jitter_plot(ax_bp2, monthly_dist_onshore, "#4a90d9",
                "Wind onshore — monthly CF for Germany, one point per year (ERA5 1979–2026)")
fig_bp2.tight_layout()
fig_bp2.savefig(paths.images_path / "37_monthly_boxplot_onshore.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/37_monthly_boxplot_onshore.png
# :name: fig-37-monthly-boxplot-onshore
# Monthly capacity factors for onshore wind in Germany (ERA5 1979–2026). Each
# jittered dot represents the mean capacity factor for one calendar year in that
# month. Wind resources peak in winter and show stronger year-to-year variability
# than solar.
# ```

# %%
fig_bp3, ax_bp3 = plt.subplots(figsize=(13, 5))
_jitter_plot(ax_bp3, monthly_dist_offshore, "#1a5fa8",
                "Wind offshore — monthly CF for Germany, one point per year (ERA5 1979–2026)")
fig_bp3.tight_layout()
fig_bp3.savefig(paths.images_path / "37_monthly_boxplot_offshore.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/37_monthly_boxplot_offshore.png
# :name: fig-37-monthly-boxplot-offshore
# Monthly capacity factors for offshore wind in Germany (ERA5 1979–2026). Each
# jittered dot represents the mean capacity factor for one calendar year in that
# month. Offshore resources are generally higher and more consistent than onshore,
# with a similar winter peak.
# ```

# %% [markdown]
# ## Interannual variability — Germany annual capacity factors

# %%
# Restrict to complete years (1979–2025)
idx_full = (df.index.year >= 1979) & (df.index.year <= 2025)
annual_solar   = cf_solar.loc[idx_full, "DE"].resample("YE").mean()
annual_onshore = cf_onshore.loc[idx_full, "DE"].resample("YE").mean()

fig3, ax3 = plt.subplots(figsize=(14, 5))
years = annual_solar.index.year

ax3.plot(years, annual_solar.values,
         color="#f4b942", linewidth=1.5, marker="o", markersize=3, label="Solar PV")
ax3.plot(years, annual_onshore.values,
         color="#4a90d9", linewidth=1.5, marker="o", markersize=3, label="Wind onshore")
ax3.axhline(annual_solar.mean(), color="#f4b942", linewidth=1.0, linestyle="--", alpha=0.6)
ax3.axhline(annual_onshore.mean(), color="#4a90d9", linewidth=1.0, linestyle="--", alpha=0.6)

ax3.set_xlabel("Year")
ax3.set_ylabel("Annual mean capacity factor")
ax3.set_title("Germany — annual capacity factors, 1979–2025 (ERA5)", fontsize=11)
ax3.legend(fontsize=10)
ax3.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax3.set_axisbelow(True)
fig3.tight_layout()
fig3.savefig(paths.images_path / "37_annual_cf_de.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/37_annual_cf_de.png
# :name: fig-37-annual-cf-de
# Annual mean capacity factors for Germany (ERA5, 1979–2025). Dashed lines show
# the long-run mean. Both technologies exhibit substantial interannual variability;
# wind shows stronger year-to-year swings than solar.
# ```

# %% [markdown]
# ## Solar vs wind complementarity across Europe

# %%
# Scatter: mean solar CF vs mean onshore wind CF per country
shared = sorted(set(mean_solar.index) & set(mean_onshore.index))
x = np.array([mean_solar[c] for c in shared])
y = np.array([mean_onshore[c] for c in shared])

fig4, ax4 = plt.subplots(figsize=(10, 7))
ax4.scatter(x, y, color="#4a90d9", s=60, edgecolors="white", linewidths=0.5, zorder=3)

for country, xi, yi in zip(shared, x, y):
    ax4.annotate(country, (xi, yi), fontsize=7.5,
                 xytext=(3, 3), textcoords="offset points")

ax4.set_xlabel("Mean solar PV capacity factor")
ax4.set_ylabel("Mean onshore wind capacity factor")
ax4.set_title(
    "Solar–wind resource complementarity across Europe\n(ERA5 1979–2026, NUTS 0)",
    fontsize=11,
)
ax4.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax4.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax4.set_axisbelow(True)
fig4.tight_layout()
fig4.savefig(paths.images_path / "37_solar_wind_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/37_solar_wind_scatter.png
# :name: fig-37-solar-wind-scatter
# Mean solar PV vs onshore wind capacity factors across European countries
# (ERA5 1979–2026). There is a broad negative correlation: countries with
# strong solar resources (southern Europe) tend to have weaker onshore wind
# resources, and vice versa — suggesting natural complementarity at the
# continental scale.
# ```

# %% [markdown]
# ## Capacity factor choropleths

# %%
# Map PECD NUTS-0 codes → Natural Earth ADM0_A3 three-letter codes
NUTS_TO_ADM0 = {
    "AL": "ALB", "AT": "AUT", "BA": "BIH", "BE": "BEL", "BG": "BGR",
    "CH": "CHE", "CY": "CYP", "CZ": "CZE", "DE": "DEU", "DK": "DNK",
    "EE": "EST", "EL": "GRC", "ES": "ESP", "FI": "FIN", "FR": "FRA",
    "HR": "HRV", "HU": "HUN", "IE": "IRL", "IS": "ISL", "IT": "ITA",
    "LI": "LIE", "LT": "LTU", "LU": "LUX", "LV": "LVA", "ME": "MNE",
    "MK": "MKD", "MT": "MLT", "NL": "NLD", "NO": "NOR", "PL": "POL",
    "PT": "PRT", "RO": "ROU", "RS": "SRB", "SE": "SWE", "SI": "SVN",
    "SK": "SVK", "TR": "TUR", "UK": "GBR",
}

_shpfile = shpreader.natural_earth(
    resolution="10m", category="cultural", name="admin_0_countries"
)
_country_geoms = {
    r.attributes["ADM0_A3"]: r.geometry
    for r in shpreader.Reader(_shpfile).records()
}

PROJ = ccrs.LambertConformal(central_longitude=10, central_latitude=50)
EUROPE_EXTENT = [-25, 47, 27, 72]


def _fill_choropleth(ax, values: pd.Series, cmap_name: str, title: str) -> None:
    cmap = plt.colormaps[cmap_name]
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

    ax.set_extent(EUROPE_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="#c6def1", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#e0e0e0", zorder=0)

    for nuts_code, val in values.items():
        adm0 = NUTS_TO_ADM0.get(nuts_code)
        geom = _country_geoms.get(adm0)
        if geom is not None:
            ax.add_geometries(
                [geom], ccrs.PlateCarree(),
                facecolor=cmap(norm(val)), edgecolor="black", linewidth=0.25, zorder=1,
            )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.25, linestyle=":", zorder=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation="vertical",
                 fraction=0.03, pad=0.04, label="Mean capacity factor")
    ax.set_title(title, fontsize=10, pad=6)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.4, zorder=3)


# %%
fig5, ax5 = plt.subplots(subplot_kw={"projection": PROJ}, figsize=(10, 7))
_fill_choropleth(ax5, mean_solar, "YlOrRd",
                 "Solar PV — long-run mean CF (ERA5 1979–2026)")
fig5.tight_layout()
fig5.savefig(paths.images_path / "37_cf_choropleth_solar.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/37_cf_choropleth_solar.png
# :name: fig-37-cf-choropleth-solar
# Long-run mean solar PV capacity factors (ERA5 1979–2026) by country.
# The strongest solar resources are in the south (MT, CY, ES, PT, GR);
# northern and central European countries have substantially lower yields.
# Grey countries have no PECD data.
# ```

# %%
fig6, ax6 = plt.subplots(subplot_kw={"projection": PROJ}, figsize=(10, 7))
_fill_choropleth(ax6, mean_onshore, "Blues",
                 "Wind onshore — long-run mean CF (ERA5 1979–2026)")
fig6.tight_layout()
fig6.savefig(paths.images_path / "37_cf_choropleth_onshore.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/37_cf_choropleth_onshore.png
# :name: fig-37-cf-choropleth-onshore
# Long-run mean onshore wind capacity factors (ERA5 1979–2026) by country.
# Iceland, Ireland, the UK, and Norway show the highest capacity factors,
# reflecting the strong and consistent Atlantic westerlies at northern latitudes.
# ```

# %%
fig7, ax7 = plt.subplots(subplot_kw={"projection": PROJ}, figsize=(10, 7))
_fill_choropleth(ax7, mean_offshore, "GnBu",
                 "Wind offshore — long-run mean CF (ERA5 1979–2026)")
fig7.tight_layout()
fig7.savefig(paths.images_path / "37_cf_choropleth_offshore.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/37_cf_choropleth_offshore.png
# :name: fig-37-cf-choropleth-offshore
# Long-run mean offshore wind capacity factors (ERA5 1979–2026) by country.
# Only coastal and island countries are included. The North Sea and Atlantic
# nations (UK, IE, NO, DK) have the highest capacity factors; Mediterranean
# countries have markedly weaker offshore wind resources.
# ```

# %%

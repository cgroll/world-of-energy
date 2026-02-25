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
# # Weather Variability in Germany — Beyond the Seasonal Cycle
#
# Monthly ERA5 climatology tells us what is *typical* for each calendar month,
# but the actual value in any given year can differ substantially from that
# mean.  This notebook shows the full spread of monthly values (1940–2024)
# for temperature, wind speed, and solar radiation, and then analyses
# *Dunkelflaute* — months when both wind power and solar power potential
# are simultaneously very low.
#
# **Data source:** ERA5 monthly Germany spatial means
# (`16_compute_germany_time_series.py`)
#
# | Variable | ERA5 field | Unit |
# |---|---|---|
# | 2 m temperature | `t2m` | °C |
# | 100 m wind speed | `wind_speed_100m` | m/s |
# | Solar radiation | `ssrd` | kWh/m²/day |

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from woe.paths import ProjPaths

paths = ProjPaths()

# %%
# Load Germany monthly time series produced by 16_compute_germany_time_series.py
df = pd.read_parquet(paths.era5_germany_monthly_ts_file)

# Derived columns
df["t2m_c"]    = df["t2m"] - 273.15      # K → °C
df["ssrd_kwh"] = df["ssrd"] / 3.6e6     # J/m² (daily mean) → kWh/m²/day
df["month"]    = df.index.month
df["year"]     = df.index.year

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

print(f"Loaded {len(df)} monthly records, "
      f"{df['year'].min()}–{df['year'].max()}")
print(f"Columns: {list(df.columns)}")

# %%
def jitter_plot(ax, df, var, ylabel, title,
                cmap="RdBu_r", vmin=None, vmax=None,
                hline=None, hline_label=None):
    """Jitter plot: one dot per year for each calendar month.

    Dots are coloured by the variable value.  The black horizontal bar
    marks the calendar-month median.
    """
    rng = np.random.default_rng(42)
    vals = df[var].values
    if vmin is None:
        vmin = np.percentile(vals, 2)
    if vmax is None:
        vmax = np.percentile(vals, 98)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    for m in range(1, 13):
        data = df.loc[df["month"] == m, var].values
        x = m + rng.uniform(-0.28, 0.28, len(data))
        ax.scatter(x, data, c=cmap_obj(norm(data)),
                   s=16, alpha=0.80, zorder=3, linewidths=0)
        med = np.median(data)
        ax.plot([m - 0.32, m + 0.32], [med, med],
                color="black", linewidth=1.8, zorder=4)

    if hline is not None:
        ax.axhline(hline, color="#3a8abf", linewidth=1.0,
                   linestyle="--", alpha=0.7, label=hline_label)
        if hline_label:
            ax.legend(fontsize=8, loc="upper right")

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.set_xlim(0.5, 12.5)


# %% [markdown]
# ## 2 m temperature by calendar month

# %%
fig, ax = plt.subplots(figsize=(13, 5))
jitter_plot(
    ax, df, "t2m_c",
    ylabel="2 m temperature (°C)",
    title="ERA5 monthly 2 m temperature by month — Germany (one dot per year)",
    cmap="RdYlBu_r",
    hline=0, hline_label="0 °C",
)
fig.tight_layout()
fig.savefig(paths.images_path / "17_germany_t2m_jitter.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/17_germany_t2m_jitter.png
# :name: fig-17-germany-t2m-jitter
# ERA5 monthly mean 2 m temperature for Germany (spatial mean), 1940–2024.
# Each dot is one year; the black bar marks the calendar-month median.
# The spread within each month shows the substantial year-to-year variability
# driven by atmospheric circulation patterns such as the NAO and blocking events.
# ```

# %% [markdown]
# ## 100 m wind speed by calendar month
#
# 100 m wind speed is the most direct ERA5 proxy for wind-turbine output.
# Wind is strongest in winter but shows large year-to-year scatter in every
# season, driven by variability in the jet stream and Atlantic storm tracks.

# %%
fig, ax = plt.subplots(figsize=(13, 5))
jitter_plot(
    ax, df, "wind_speed_100m",
    ylabel="100 m wind speed (m/s)",
    title="ERA5 monthly 100 m wind speed by month — Germany (one dot per year)",
    cmap="viridis",
)
fig.tight_layout()
fig.savefig(paths.images_path / "17_germany_wind_jitter.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/17_germany_wind_jitter.png
# :name: fig-17-germany-wind-jitter
# ERA5 monthly mean 100 m wind speed for Germany (spatial mean), 1940–2024.
# Winter months (Nov–Feb) are on average windiest but also show the largest
# absolute spread.  Low-wind winter months are potential Dunkelflaute candidates.
# ```

# %% [markdown]
# ## Surface solar radiation by calendar month
#
# Monthly mean surface solar radiation (`ssrd`), converted to kWh/m²/day,
# is a proxy for photovoltaic output potential.  The near-zero winter values
# reflect Germany's high latitude (~47–55°N) and the short winter day length.
# Summer months show remarkably little year-to-year spread compared with wind.

# %%
fig, ax = plt.subplots(figsize=(13, 5))
jitter_plot(
    ax, df, "ssrd_kwh",
    ylabel="Surface solar radiation (kWh/m²/day)",
    title="ERA5 monthly surface solar radiation by month — Germany (one dot per year)",
    cmap="plasma",
)
fig.tight_layout()
fig.savefig(paths.images_path / "17_germany_ssrd_jitter.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/17_germany_ssrd_jitter.png
# :name: fig-17-germany-ssrd-jitter
# ERA5 monthly mean surface solar radiation downwards for Germany, 1940–2024.
# The seasonal cycle dominates: values approach zero in December and peak in
# June–July.  Year-to-year variability is moderate in summer (cloud-cover
# differences) but negligible in winter where solar geometry constrains output.
# ```

# %% [markdown]
# ## Dunkelflaute analysis
#
# *Dunkelflaute* (lit. "dark doldrums") describes periods when both wind and
# solar renewable generation are simultaneously very low — a critical stress
# scenario for a power system with high renewable penetration.
#
# **Definition used here:** a month is classified as Dunkelflaute when
# Germany-mean 100 m wind speed **and** surface solar radiation are both
# below their respective 25th percentiles computed over all 1020 months
# (1940–2024).  This captures the joint bottom-quartile tail of both
# resource distributions.

# %%
# --- Dunkelflaute thresholds ---
p25_wind  = df["wind_speed_100m"].quantile(0.25)
p25_solar = df["ssrd_kwh"].quantile(0.25)

df["dunkelflaute"] = (
    (df["wind_speed_100m"] < p25_wind) &
    (df["ssrd_kwh"]        < p25_solar)
)

n_df  = df["dunkelflaute"].sum()
frac  = n_df / len(df)
print(f"Thresholds — wind: {p25_wind:.2f} m/s,  solar: {p25_solar:.3f} kWh/m²/day")
print(f"Dunkelflaute months: {n_df} / {len(df)}  ({100*frac:.1f} %)")
print("\nCount per calendar month:")
print(df.groupby("month")["dunkelflaute"].sum().to_string())

# %%
# 12 perceptually distinct colours for the months (cyclic HSV)
MONTH_COLORS = plt.get_cmap("hsv")(np.linspace(0, 1, 13))[:12]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left: wind vs solar scatter, coloured by calendar month ---
ax = axes[0]
for m in range(1, 13):
    sel = df[df["month"] == m]
    ax.scatter(
        sel["wind_speed_100m"], sel["ssrd_kwh"],
        color=MONTH_COLORS[m - 1], s=14, alpha=0.75,
        label=MONTH_LABELS[m - 1], zorder=3, linewidths=0,
    )

# Dunkelflaute quadrant shading
ax.axvline(p25_wind,  color="gray", linewidth=0.9, linestyle="--", alpha=0.6)
ax.axhline(p25_solar, color="gray", linewidth=0.9, linestyle="--", alpha=0.6)
xmax = df["wind_speed_100m"].max() * 1.05
ymax = df["ssrd_kwh"].max() * 1.05
ax.fill_betweenx(
    [0, p25_solar], 0, p25_wind,
    color="steelblue", alpha=0.15, zorder=1,
)
ax.text(
    p25_wind * 0.48, p25_solar * 0.50,
    "Dunkelflaute\nzone",
    ha="center", va="center",
    fontsize=9, color="steelblue", fontweight="bold",
)
ax.set_xlabel("100 m wind speed (m/s)")
ax.set_ylabel("Surface solar radiation (kWh/m²/day)")
ax.set_title(
    "Wind vs solar — Germany monthly means\n(coloured by calendar month)",
    fontsize=10, fontweight="bold",
)
ax.legend(title="Month", ncol=2, fontsize=7, title_fontsize=8,
          loc="upper right", framealpha=0.9)
ax.set_xlim(0, xmax)
ax.set_ylim(0, ymax)
ax.grid(linewidth=0.4, alpha=0.4)

# --- Right: Dunkelflaute frequency by calendar month ---
ax2 = axes[1]
freq = df.groupby("month")["dunkelflaute"].mean() * 100
bar_colors = [MONTH_COLORS[m - 1] for m in freq.index]
ax2.bar(freq.index, freq.values, color=bar_colors, width=0.7, alpha=0.85, zorder=3)
ax2.axhline(frac * 100, color="black", linewidth=1.0, linestyle="--", alpha=0.6,
            label=f"Overall rate ({frac*100:.1f}%)")
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(MONTH_LABELS)
ax2.set_ylabel("Frequency of Dunkelflaute years (%)")
ax2.set_title(
    "Dunkelflaute frequency by calendar month\n"
    "(wind < p25 AND solar < p25, ERA5 1940–2024)",
    fontsize=10, fontweight="bold",
)
ax2.grid(axis="y", linewidth=0.4, alpha=0.4)
ax2.legend(fontsize=8)

fig.suptitle("Dunkelflaute analysis — Germany ERA5 1940–2024",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(paths.images_path / "17_germany_dunkelflaute.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/17_germany_dunkelflaute.png
# :name: fig-17-germany-dunkelflaute
# **Left:** scatter of Germany monthly 100 m wind speed vs surface solar
# radiation, 1940–2024, coloured by calendar month.  The shaded blue quadrant
# marks the Dunkelflaute zone (both resources below their 25th percentile).
# **Right:** fraction of years in each calendar month that fell into the
# Dunkelflaute zone.  Dunkelflaute events are overwhelmingly concentrated in
# November–January, when solar output is structurally near zero and wind can
# also be suppressed by blocking anticyclones.
# ```

# %% [markdown]
# ## Case studies: cold vs warm January
#
# To show how the large-scale circulation drives extreme temperature anomalies
# we compare January 1963 (coldest: Germany mean −7.3 °C, the *Big Freeze*,
# caused by a blocking anticyclone funnelling Arctic/Siberian air into central
# Europe) against January 2007 (warmest: Germany mean +4.9 °C, associated with
# a strongly positive NAO and persistent westerly flow from the Atlantic).
#
# Both months are shown with identical colour scales so differences are
# immediately comparable.

# %%
COLD_YEAR = 1963
WARM_YEAR = 2007
G = 9.80665  # standard gravity (m/s2)

t2m_cold_de = float(df.loc[f"{COLD_YEAR}-01-01", "t2m_c"])
t2m_warm_de = float(df.loc[f"{WARM_YEAR}-01-01", "t2m_c"])
print(f"Cold case: January {COLD_YEAR}  Germany mean = {t2m_cold_de:.1f} °C")
print(f"Warm case: January {WARM_YEAR}  Germany mean = {t2m_warm_de:.1f} °C")

# %%
# Load spatial ERA5 fields for both case months
ds = xr.open_zarr(paths.era5_monthly_zarr_path)

def _load_fields(year):
    date = f"{year}-01"
    t2m  = (ds["t2m"].sel(time=date).squeeze() - 273.15).compute()
    z500 = (ds["z"].sel(time=date, pressure_level=500).squeeze() / G).compute()
    u250 = ds["u"].sel(time=date, pressure_level=250).squeeze().compute()
    v250 = ds["v"].sel(time=date, pressure_level=250).squeeze().compute()
    wspd = np.sqrt(u250**2 + v250**2)
    return t2m, z500, u250, v250, wspd

t2m_cold, z500_cold, u250_cold, v250_cold, wspd250_cold = _load_fields(COLD_YEAR)
t2m_warm, z500_warm, u250_warm, v250_warm, wspd250_warm = _load_fields(WARM_YEAR)

# Shared colour and contour limits (computed across both months)
T2M_VMIN, T2M_VMAX = -40, 20
WIND_VMAX = max(float(wspd250_cold.max()), float(wspd250_warm.max()))

_z_all_min = min(float(z500_cold.min()), float(z500_warm.min()))
_z_all_max = max(float(z500_cold.max()), float(z500_warm.max()))
Z500_LEVELS = np.arange(
    np.floor(_z_all_min / 80) * 80,
    np.ceil(_z_all_max  / 80) * 80 + 1,
    80,
)

print(f"Shared wind vmax: {WIND_VMAX:.1f} m/s")
print(f"Z500 levels: {Z500_LEVELS[0]:.0f} … {Z500_LEVELS[-1]:.0f} m "
      f"({len(Z500_LEVELS)} contours)")

# %%
# Shared map settings
PROJ   = ccrs.Orthographic(central_longitude=-10, central_latitude=55)
EXTENT = [-90, 40, 20, 80]   # W, E, S, N  (PlateCarree)
N = 5  # quiver thinning factor

lons_q = u250_cold.longitude.values[::N]
lats_q = u250_cold.latitude.values[::N]

# %% [markdown]
# ### 2 m temperature — cold vs warm January

# %%
fig, axes = plt.subplots(
    1, 2, figsize=(18, 8),
    subplot_kw={"projection": PROJ},
)

cases_t2m = [
    (axes[0], t2m_cold, z500_cold, COLD_YEAR, t2m_cold_de),
    (axes[1], t2m_warm, z500_warm, WARM_YEAR, t2m_warm_de),
]

for ax, t2m, z500, year, t2m_de in cases_t2m:
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    im = ax.pcolormesh(
        t2m.longitude, t2m.latitude, t2m.values,
        transform=ccrs.PlateCarree(),
        cmap="RdYlBu_r", vmin=T2M_VMIN, vmax=T2M_VMAX,
    )
    cs = ax.contour(
        z500.longitude, z500.latitude, z500.values,
        levels=Z500_LEVELS,
        colors="black", linewidths=0.6, alpha=0.55,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(cs, fmt="%d", fontsize=7, inline=True)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
    ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)
    ax.plot(10.5, 51.2, transform=ccrs.PlateCarree(),
            marker="*", markersize=11, color="gold",
            markeredgecolor="black", markeredgewidth=0.6, zorder=5)
    ax.text(12.5, 51.2, "Germany", transform=ccrs.PlateCarree(),
            fontsize=9, fontweight="bold", color="black", va="center", zorder=5)
    ax.set_title(
        f"January {year}  (Germany mean: {t2m_de:+.1f} °C)\nZ500 contours every 80 m",
        fontsize=11, fontweight="bold",
    )

fig.colorbar(
    im, ax=axes, orientation="horizontal", pad=0.04, fraction=0.03,
    label="2 m temperature (°C)",
)
fig.suptitle(
    "ERA5 2 m temperature — cold vs warm January",
    fontsize=13, fontweight="bold",
)
fig.tight_layout()
fig.savefig(paths.images_path / "17_january_t2m_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/17_january_t2m_comparison.png
# :name: fig-17-january-t2m-comparison
# ERA5 2 m temperature for January 1963 (left, cold) and January 2007 (right,
# warm) on an orthographic projection.  Both panels share the same colour scale
# and Z500 contour levels (every 80 m, black lines) to make comparison direct.
# In 1963 a blocking ridge over Scandinavia/Russia channels Arctic air into
# central Europe (deep blue over Germany); in 2007 a strongly positive NAO
# drives mild Atlantic westerlies across the continent (orange-yellow over
# western Europe).
# ```

# %% [markdown]
# ### 250 hPa wind speed and Z500 — cold vs warm January

# %%
fig, axes = plt.subplots(
    1, 2, figsize=(18, 8),
    subplot_kw={"projection": PROJ},
)

cases_wind = [
    (axes[0], wspd250_cold, z500_cold, u250_cold, v250_cold, COLD_YEAR),
    (axes[1], wspd250_warm, z500_warm, u250_warm, v250_warm, WARM_YEAR),
]

for ax, wspd, z500, u250, v250, year in cases_wind:
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    im = ax.pcolormesh(
        wspd.longitude, wspd.latitude, wspd.values,
        transform=ccrs.PlateCarree(),
        cmap="plasma", vmin=0, vmax=WIND_VMAX,
    )
    cs = ax.contour(
        z500.longitude, z500.latitude, z500.values,
        levels=Z500_LEVELS,
        colors="white", linewidths=0.8, alpha=0.65,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(cs, fmt="%d", fontsize=7, inline=True, colors="white")
    ax.quiver(
        lons_q, lats_q,
        u250.values[::N, ::N], v250.values[::N, ::N],
        transform=ccrs.PlateCarree(),
        scale=2500, width=0.0012, color="white", alpha=0.65,
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
    ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)
    ax.set_title(
        f"January {year}\nZ500 contours every 80 m  |  arrows = wind direction",
        fontsize=11, fontweight="bold",
    )

fig.colorbar(
    im, ax=axes, orientation="horizontal", pad=0.04, fraction=0.03,
    label="250 hPa wind speed (m/s)",
)
fig.suptitle(
    "250 hPa wind speed and Z500 — cold vs warm January",
    fontsize=13, fontweight="bold",
)
fig.tight_layout()
fig.savefig(paths.images_path / "17_january_250hpa_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/17_january_250hpa_comparison.png
# :name: fig-17-january-250hpa-comparison
# 250 hPa wind speed (colour; shared plasma scale) and Z500 contours (white,
# every 80 m) for January 1963 (left) and January 2007 (right).  White arrows
# show wind direction.  In 1963 the jet is displaced southward and weakened
# over central Europe due to the blocking ridge; in 2007 the jet is strong and
# directed straight at Europe, advecting mild air from the Atlantic.
# ```

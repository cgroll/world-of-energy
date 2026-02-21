"""Compute ERA5 monthly spatial aggregates over Germany.

Loads the monthly ERA5 Zarr store produced by 14_nc_to_zarr.py, masks to
Germany's land geometry (Natural Earth 10 m), and computes spatial mean,
median, min, max and std for each monthly time step.

Variables processed:
  - t2m             : 2 m temperature (K)
  - wind_speed_100m : 100 m wind speed (m/s), derived from u100/v100

  Pending download (accumulated flux variables not yet in zarr store):
  - sf              : snowfall (m of water equivalent)
  - tp              : total precipitation (m)
  - ssrd            : surface solar radiation downwards (J/m²)

Output:
  Parquet file with a (time, agg) MultiIndex and one column per variable,
  saved to data/processed/era5_germany_monthly.parquet.
"""

# %%
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom

from woe.paths import ProjPaths

paths = ProjPaths()

# %%
# Load zarr store
print("Loading ERA5 monthly Zarr store...")
ds = xr.open_zarr(paths.era5_monthly_zarr_path)
print(f"  Variables:  {list(ds.data_vars)}")
print(f"  Dimensions: {dict(ds.dims)}")
print(f"  Time range: {ds.time.values[0]} → {ds.time.values[-1]}")

# %%
# Build Germany land mask on the ERA5 grid using Natural Earth 10 m boundaries
print("Building Germany land mask...")
countries_shp = shpreader.natural_earth(
    resolution="10m", category="cultural", name="admin_0_countries"
)
germany_geom = next(
    rec.geometry
    for rec in shpreader.Reader(countries_shp).records()
    if rec.attributes["NAME"] == "Germany"
)

lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
mask = np.vectorize(lambda lon, lat: germany_geom.contains(sgeom.Point(lon, lat)))(
    lons, lats
)

mask_da = xr.DataArray(
    mask,
    coords={"latitude": ds.latitude, "longitude": ds.longitude},
    dims=["latitude", "longitude"],
)
print(f"  Grid cells inside Germany: {int(mask.sum())}")

# %%
# Apply mask and derive 100 m wind speed from U/V components
ds_de = ds.where(mask_da)
ds_de["wind_speed_100m"] = np.sqrt(ds_de["u100"] ** 2 + ds_de["v100"] ** 2)

VARIABLES = ["t2m", "wind_speed_100m"]  # sf, tp, ssrd pending download
AGG_NAMES = ["mean", "median", "min", "max", "std"]

# %%
# Stack spatial dims and drop non-Germany cells (NaN throughout all time steps),
# then apply each aggregation over the cell dimension for all months at once.
print("Computing spatial aggregations...")
times = pd.DatetimeIndex(ds_de.time.values)
per_agg: dict[str, pd.DataFrame] = {agg: pd.DataFrame(index=times) for agg in AGG_NAMES}

for var in VARIABLES:
    print(f"  {var}...")
    da_stacked = (
        ds_de[var]
        .stack(cell=("latitude", "longitude"))
        .dropna("cell", how="all")  # removes non-Germany cells masked to NaN
    )
    per_agg["mean"][var]   = da_stacked.mean("cell").compute().values
    per_agg["median"][var] = da_stacked.median("cell").compute().values
    per_agg["min"][var]    = da_stacked.min("cell").compute().values
    per_agg["max"][var]    = da_stacked.max("cell").compute().values
    per_agg["std"][var]    = da_stacked.std("cell").compute().values

# %%
# Assemble final DataFrame with (time, agg) MultiIndex
frames = []
for agg_name, agg_df in per_agg.items():
    agg_df = agg_df.copy()
    agg_df.index.name = "time"
    agg_df["agg"] = agg_name
    frames.append(agg_df.reset_index().set_index(["time", "agg"]))

df = pd.concat(frames).sort_index()
print(f"Shape: {df.shape}  (time steps × agg types = {len(times)} × {len(AGG_NAMES)})")
print(df.head(10))

# %%
# Save to parquet
output = paths.era5_germany_monthly_file
output.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(output)
print(f"Saved to {output}")

# %%
# Plot mean temperature over time
t2m_mean = df.xs("mean", level="agg")["t2m"] - 273.15  # K → °C

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(t2m_mean.index, t2m_mean.values, color="#e05c2e", linewidth=1.2, label="Monthly mean")

# 12-month rolling average
rolling = t2m_mean.rolling(12, center=True).mean()
ax.plot(rolling.index, rolling.values, color="#333333", linewidth=2.0, label="12-month rolling mean")

ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_minor_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_ylabel("2 m temperature (°C)")
ax.set_title("ERA5 monthly mean 2 m temperature — Germany (spatial mean)")
ax.legend()
ax.grid(axis="y", linewidth=0.4, alpha=0.6)
fig.tight_layout()

fig_path = paths.images_path / "16_germany_t2m_mean.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to {fig_path}")

# %%
# Jitter (strip) plot — one dot per year, grouped by calendar month
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

rng = np.random.default_rng(42)
jitter_width = 0.3

cmap = LinearSegmentedColormap.from_list("cold_hot", ["#313695", "#4575b4", "#d73027", "#a50026"])
t2m_min, t2m_max = t2m_mean.min(), t2m_mean.max()

fig, ax = plt.subplots(figsize=(12, 5))

for m in range(1, 13):
    vals = t2m_mean[t2m_mean.index.month == m].values
    xs = m + rng.uniform(-jitter_width, jitter_width, size=len(vals))
    colors = cmap((vals - t2m_min) / (t2m_max - t2m_min))
    ax.scatter(xs, vals, c=colors, s=18, alpha=0.7, linewidths=0)

    # Overlay median bar
    ax.hlines(np.median(vals), m - jitter_width, m + jitter_width,
              colors="#333333", linewidth=1.8)

ax.axhline(0, color="steelblue", linewidth=0.8, linestyle="--", alpha=0.6)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS)
ax.set_ylabel("2 m temperature (°C)")
ax.set_title("ERA5 monthly 2 m temperature by month — Germany (one dot per year)")
ax.grid(axis="y", linewidth=0.4, alpha=0.5)
fig.tight_layout()

fig_path = paths.images_path / "16_germany_t2m_jitter.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to {fig_path}")

# %%
# Jitter plot — 100 m wind speed by calendar month
# Precipitation (tp) is not yet available in the Zarr store (pending download).
wind_mean = df.xs("mean", level="agg")["wind_speed_100m"]

wind_cmap = LinearSegmentedColormap.from_list("wind", ["#c6dbef", "#6baed6", "#08306b"])
w_min, w_max = wind_mean.min(), wind_mean.max()

fig, ax = plt.subplots(figsize=(12, 5))

for m in range(1, 13):
    vals = wind_mean[wind_mean.index.month == m].values
    xs = m + rng.uniform(-jitter_width, jitter_width, size=len(vals))
    colors = wind_cmap((vals - w_min) / (w_max - w_min))
    ax.scatter(xs, vals, c=colors, s=18, alpha=0.8, linewidths=0)
    ax.hlines(np.median(vals), m - jitter_width, m + jitter_width,
              colors="#333333", linewidth=1.8)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS)
ax.set_ylabel("100 m wind speed (m/s)")
ax.set_title("ERA5 monthly mean 100 m wind speed by month — Germany (one dot per year)")
ax.grid(axis="y", linewidth=0.4, alpha=0.5)
fig.tight_layout()

fig_path = paths.images_path / "16_germany_wind_jitter.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to {fig_path}")

# %%
# 2-D scatter: mean temperature vs mean wind speed, coloured by month
mean_df = df.xs("mean", level="agg").copy()
mean_df["t2m_c"] = mean_df["t2m"] - 273.15

MONTH_COLORS = [plt.get_cmap("hsv")(m / 12) for m in range(12)]

fig, ax = plt.subplots(figsize=(8, 6))

for m in range(1, 13):
    sel = mean_df[mean_df.index.month == m]
    ax.scatter(
        sel["t2m_c"], sel["wind_speed_100m"],
        color=MONTH_COLORS[m - 1], s=22, alpha=0.75, linewidths=0,
        label=MONTH_LABELS[m - 1],
    )

ax.set_xlabel("Mean 2 m temperature (°C)")
ax.set_ylabel("Mean 100 m wind speed (m/s)")
ax.set_title("Germany monthly mean: temperature vs wind speed")
ax.legend(loc="upper right", ncol=2, fontsize=8, markerscale=1.4,
          framealpha=0.8, title="Month")
ax.grid(linewidth=0.4, alpha=0.5)
fig.tight_layout()

fig_path = paths.images_path / "16_germany_t2m_vs_wind.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to {fig_path}")

# %%
# Correlation matrices — month-to-month Pearson correlations.
# First chart: calendar year (Jan–Dec). Second chart: Jul–Jun year.
SHIFTED_LABELS = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                  "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
MONTH_ORDER = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]

def month_corr(series: pd.Series) -> pd.DataFrame:
    """Calendar-year grouping: Jan(1) … Dec(12)."""
    wide = series.to_frame("v")
    wide["year"]  = series.index.year
    wide["month"] = series.index.month
    pivoted = wide.pivot(index="year", columns="month", values="v")
    pivoted.columns = MONTH_LABELS          # Jan … Dec
    return pivoted.corr()

corr_t2m  = month_corr(t2m_mean)
corr_wind = month_corr(wind_mean)

def plot_corr(corr: pd.DataFrame, title: str, ax: plt.Axes) -> None:
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_xticklabels(corr.columns, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(12):
        for j in range(12):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=5.5,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "#333333")
    ax.set_title(title, fontsize=10)
    return im

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im = plot_corr(corr_t2m,  "Temperature (°C)",      axes[0])
plot_corr(corr_wind, "100 m wind speed (m/s)", axes[1])
fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.03, pad=0.04,
             label="Pearson r")
fig.suptitle("Month-to-month correlations (Jan–Dec year) — Germany", fontsize=12)
fig.tight_layout()

fig_path = paths.images_path / "16_germany_month_corr.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to {fig_path}")

# %%
# Correlation matrix with a Jul–Jun year cutoff:
# Jan–Jun are grouped with the preceding Jul–Dec (i.e. assigned to year - 1).
def month_corr_shifted(series: pd.Series) -> pd.DataFrame:
    wide = series.to_frame("v")
    wide["month"] = series.index.month
    wide["group_year"] = np.where(series.index.month <= 6,
                                  series.index.year - 1,
                                  series.index.year)
    pivoted = wide.pivot(index="group_year", columns="month", values="v")
    pivoted = pivoted[MONTH_ORDER]
    pivoted.columns = SHIFTED_LABELS
    return pivoted.corr()

corr_t2m_sh  = month_corr_shifted(t2m_mean)
corr_wind_sh = month_corr_shifted(wind_mean)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im = plot_corr(corr_t2m_sh,  "Temperature (°C)",      axes[0])
plot_corr(corr_wind_sh, "100 m wind speed (m/s)", axes[1])
fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.03, pad=0.04,
             label="Pearson r")
fig.suptitle("Month-to-month correlations (Jul–Jun year) — Germany", fontsize=12)
fig.tight_layout()

fig_path = paths.images_path / "16_germany_month_corr_shifted.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to {fig_path}")

# %%


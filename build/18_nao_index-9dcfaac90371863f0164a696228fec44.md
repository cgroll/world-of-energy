---
jupytext:
  text_representation:
    format_name: percent
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Monthly ERA5 Reanalysis

This notebook loads the monthly ERA5 reanalysis data downloaded from the
Copernicus Climate Data Store and stored as a Zarr archive.  It visualises
2 m temperature on a Robinson projection.

```{code-cell} python
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from woe.paths import ProjPaths

paths = ProjPaths()
```

## Load Zarr store

```{code-cell} python
ds = xr.open_zarr(paths.era5_monthly_zarr_path)
print(f"Variables:   {list(ds.data_vars)}")
print(f"Dimensions:  {dict(ds.dims)}")
print(f"Time range:  {ds.time.values[0]} → {ds.time.values[-1]}")
```

## 2 m temperature — January 2025

```{code-cell} python
DATE = "2024-01"
t2m = ds["t2m"].sel(time=DATE).squeeze() - 273.15  # K → °C

fig, ax = plt.subplots(
    figsize=(14, 7),
    subplot_kw={"projection": ccrs.Robinson()},
)

im = ax.pcolormesh(
    t2m.longitude,
    t2m.latitude,
    t2m.values,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    vmin=-40,
    vmax=40,
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.04)
cbar.set_label("2 m temperature (°C)")

ax.set_title(f"ERA5 monthly mean 2 m temperature — {DATE}", fontsize=13)

fig.tight_layout()
fig.savefig(paths.images_path / "15_era5_t2m_2025_01.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/15_era5_t2m_2025_01.png
:name: fig-15-era5-t2m-2025-01
ERA5 monthly mean 2 m temperature for January 2025 on a Robinson projection.
```

+++

## 2 m temperature — January 2025 (Europe, Lambert Conformal)

```{code-cell} python
proj = ccrs.LambertConformal(central_longitude=10, central_latitude=50)

fig, ax = plt.subplots(figsize=(10, 9), subplot_kw={"projection": proj})

ax.set_extent([-25, 45, 30, 72], crs=ccrs.PlateCarree())

im = ax.pcolormesh(
    t2m.longitude,
    t2m.latitude,
    t2m.values,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    vmin=-25,
    vmax=25,
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
ax.add_feature(cfeature.LAND, facecolor="none", edgecolor="none")
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
gl.top_labels = False
gl.right_labels = False

cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.04, fraction=0.03)
cbar.set_label("2 m temperature (°C)")

ax.set_title(f"ERA5 monthly mean 2 m temperature — {DATE}", fontsize=13)

fig.tight_layout()
fig.savefig(paths.images_path / "15_era5_t2m_europe_2025_01.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/15_era5_t2m_europe_2025_01.png
:name: fig-15-era5-t2m-europe-2025-01
ERA5 monthly mean 2 m temperature for January 2025 over Europe on a Lambert
Conformal projection.
```

+++

## 2 m temperature — January 2025 (full domain, Plate Carrée)

```{code-cell} python
# BOUNDING_BOX = [North=80, West=-90, South=20, East=40]
fig, ax = plt.subplots(
    figsize=(14, 7),
    subplot_kw={"projection": ccrs.PlateCarree()},
)

ax.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())

im = ax.pcolormesh(
    t2m.longitude,
    t2m.latitude,
    t2m.values,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    vmin=-40,
    vmax=40,
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
gl.top_labels = False
gl.right_labels = False

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.04)
cbar.set_label("2 m temperature (°C)")

ax.set_title(f"ERA5 monthly mean 2 m temperature — {DATE}", fontsize=13)

fig.tight_layout()
fig.savefig(paths.images_path / "15_era5_t2m_domain_2025_01.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/15_era5_t2m_domain_2025_01.png
:name: fig-15-era5-t2m-domain-2025-01
ERA5 monthly mean 2 m temperature for January 2025 on a Plate Carrée projection
showing the full downloaded domain.
```

+++

## Winter NAO Index (1940–2024)

The station-based North Atlantic Oscillation (NAO) index is the standardised
mean-sea-level pressure (MSLP) difference between the Azores (Ponta Delgada)
and Iceland (Reykjavik), following the Hurrell (1995) definition.

Positive NAO winters bring stronger-than-normal westerlies across the North
Atlantic, resulting in warmer and wetter conditions over northern Europe.
Negative NAO winters are associated with blocking and cold-air outbreaks.

**Method:**
1. Filter monthly ERA5 MSLP to DJF months only.
2. Resample with `YS-DEC` so that Dec(N), Jan(N+1), Feb(N+1) form one
   season labelled by year N — identical to the Hurrell convention.
3. Drop boundary seasons with fewer than 3 months (first and last groups
   are partial because the dataset starts in January 1940).
4. Snap to the nearest ERA5 grid point for each station.
5. Normalise each series independently (subtract mean, divide by std).
6. NAO index = Azores normalised − Iceland normalised.

```{code-cell} python
# ERA5 short name for mean sea level pressure is 'msl'
msl = ds["msl"]

# --- Step 1: keep only DJF months ---
is_djf = msl["time.month"].isin([12, 1, 2])
winter_mslp = msl.sel(time=is_djf)

# --- Step 2: resample into DJF seasons anchored at December ---
winter_means = winter_mslp.resample(time="YS-DEC").mean()
winter_counts = winter_mslp.resample(time="YS-DEC").count(dim="time")

# --- Step 3: drop incomplete boundary seasons (< 3 months) ---
# Use a single reference point to get a 1-D count series for the time mask.
_count_ref = winter_counts.isel(latitude=0, longitude=0, drop=True)
complete = _count_ref == 3
winter_means = winter_means.sel(time=complete)

# --- Step 4: extract station points (nearest ERA5 grid point) ---
# Ponta Delgada, São Miguel, Azores: 37.74°N, 25.67°W
# Reykjavik, Iceland:               64.13°N, 21.90°W
azores  = winter_means.sel(latitude=37.74, longitude=-25.67, method="nearest")
iceland = winter_means.sel(latitude=64.13, longitude=-21.90, method="nearest")

# --- Step 5: standardise each series over its full period ---
azores_norm  = (azores  - azores.mean("time"))  / azores.std("time")
iceland_norm = (iceland - iceland.mean("time")) / iceland.std("time")

# --- Step 6: NAO index ---
nao = (azores_norm - iceland_norm).compute()

years  = nao.time.dt.year.values
values = nao.values

print(f"NAO index: {len(years)} winters ({years[0]}/{years[0]+1} – {years[-1]}/{years[-1]+1})")
print(f"Mean: {values.mean():.3f}  Std: {values.std():.3f}")
```

### NAO bar chart

```{code-cell} python
# 9-year centred running mean to show decadal variability
nao_smooth = nao.rolling(time=9, center=True, min_periods=5).mean().values

bar_colors = np.where(values >= 0, "crimson", "steelblue")

fig, ax = plt.subplots(figsize=(14, 5))

ax.bar(years, values, color=bar_colors, width=0.85, alpha=0.80,
       label="Annual DJF NAO")
ax.plot(years, nao_smooth, color="black", linewidth=2.0, zorder=3,
        label="9-yr running mean")
ax.axhline(0, color="black", linewidth=0.8, zorder=4)

ax.set_title(
    "Winter (DJF) North Atlantic Oscillation Index — ERA5 1940–2024",
    fontsize=13, fontweight="bold",
)
ax.set_ylabel("Standardised NAO Index")
ax.set_xlabel("Winter Start Year")
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(axis="y", alpha=0.3)

# Annotate the reference stations
station_text = (
    "Azores: Ponta Delgada (37.74°N, 25.67°W)\n"
    "Iceland: Reykjavik (64.13°N, 21.90°W)"
)
ax.text(0.01, 0.03, station_text, transform=ax.transAxes,
        fontsize=8, color="gray", va="bottom")

fig.tight_layout()
fig.savefig(paths.images_path / "15_nao_index.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/15_nao_index.png
:name: fig-15-nao-index
Winter (DJF) NAO index derived from ERA5 MSLP, 1940–2024. Red bars indicate
positive-phase winters (stronger westerlies, milder northern Europe); blue bars
indicate negative-phase winters (weaker westerlies, cold-air outbreaks). The
black line is a 9-year centred running mean highlighting decadal variability.
```

+++

## Spatial correlation: NAO vs winter 2 m temperature

Each grid point shows the Pearson *r* between the DJF NAO index and the
local DJF-mean 2 m temperature over 1940–2024.  Red (positive correlation)
means the location warms during NAO+ winters; blue means it cools.

The canonical NAO fingerprint is visible: warming over northern Europe and
the North Atlantic, and cooling over the Mediterranean and parts of
north-eastern North America.

Reuses the `complete` season mask from the NAO section to guarantee
identical time axes before calling `xr.corr`.

```{code-cell} python
# Build DJF mean temperature with the same resampling and completeness filter
winter_t2m_means = (
    ds["t2m"]
    .sel(time=ds.time.dt.month.isin([12, 1, 2]))
    .resample(time="YS-DEC")
    .mean()
    .sel(time=complete)  # same incomplete-season mask as the NAO index
)

# Pearson r at every grid point along the shared time dimension
print("Computing spatial correlation …")
corr_map = xr.corr(nao, winter_t2m_means, dim="time").compute()
print("Done.")
```

```{code-cell} python
proj = ccrs.Orthographic(central_longitude=-10, central_latitude=55)
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": proj})

ax.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())

im = ax.pcolormesh(
    corr_map.longitude,
    corr_map.latitude,
    corr_map.values,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    vmin=-0.8,
    vmax=0.8,
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)

cbar = fig.colorbar(
    im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.04, shrink=0.85
)
cbar.set_label("Pearson r — NAO index vs DJF 2 m temperature")

ax.set_title(
    "NAO impact on winter (DJF) 2 m temperature — ERA5 1940–2024",
    fontsize=13,
    fontweight="bold",
)

fig.tight_layout()
fig.savefig(
    paths.images_path / "15_nao_t2m_correlation.png", dpi=150, bbox_inches="tight"
)
plt.show()
```

```{figure} ../../output/images/15_nao_t2m_correlation.png
:name: fig-15-nao-t2m-correlation
Pearson correlation between the DJF NAO index and ERA5 DJF 2 m temperature,
1940–2024.  Red shading indicates regions that are warmer than average during
NAO+ winters; blue shading indicates cooler regions.
```

+++

## Germany DJF temperature vs NAO index

Load the Germany spatial-mean 2 m temperature produced by
`16_dev_germany_weather.py`, apply the identical YS-DEC resampling used for
the NAO index, and scatter the two series against each other.

```{code-cell} python
# Load Germany monthly spatial-mean temperature (mean aggregation, K → °C)
df_de = pd.read_parquet(paths.era5_germany_monthly_ts_file)
t2m_de_monthly = df_de["t2m"] - 273.15  # K → °C

# Filter to DJF and resample with YS-DEC (same convention as NAO index)
is_djf_de = t2m_de_monthly.index.month.isin([12, 1, 2])
t2m_de_djf_monthly = t2m_de_monthly[is_djf_de]
t2m_de_djf = t2m_de_djf_monthly.resample("YS-DEC").mean()
counts_de   = t2m_de_djf_monthly.resample("YS-DEC").count()
t2m_de_djf  = t2m_de_djf[counts_de == 3]

# NAO as a pandas Series indexed by the season's December date (YS-DEC labels)
nao_series = pd.Series(values, index=pd.DatetimeIndex(nao.time.values))

# Align on the shared index (inner join → only seasons present in both)
combined = pd.DataFrame({"nao": nao_series, "t2m_de": t2m_de_djf}).dropna()
print(f"Aligned seasons: {len(combined)}  "
      f"({combined.index.year[0]}/{combined.index.year[0]+1} – "
      f"{combined.index.year[-1]}/{combined.index.year[-1]+1})")

# Pearson correlation and OLS trend line
pearson_r = combined["nao"].corr(combined["t2m_de"])
m_slope, m_intercept = np.polyfit(combined["nao"], combined["t2m_de"], 1)
print(f"Pearson r (NAO vs Germany DJF T2m): {pearson_r:.3f}")
print(f"OLS slope:                           {m_slope:+.2f} °C per σ of NAO")

x_fit = np.linspace(combined["nao"].min(), combined["nao"].max(), 100)

fig, ax = plt.subplots(figsize=(7, 6))

sc = ax.scatter(
    combined["nao"], combined["t2m_de"],
    c=combined.index.year, cmap="plasma",
    s=45, alpha=0.85, zorder=3,
)
ax.plot(x_fit, m_slope * x_fit + m_intercept,
        color="crimson", linewidth=1.8, zorder=4,
        label=f"OLS: {m_slope:+.2f} °C / σ  (r = {pearson_r:.2f})")
ax.axhline(combined["t2m_de"].mean(), color="gray", linewidth=0.8,
           linestyle="--", alpha=0.6)
ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

cbar = fig.colorbar(sc, ax=ax, label="Winter start year")
ax.set_xlabel("DJF NAO Index (standardised)")
ax.set_ylabel("Germany DJF mean 2 m temperature (°C)")
ax.set_title(
    "Germany winter temperature vs NAO index\n"
    "ERA5 DJF seasons, 1940–2024",
    fontsize=12,
)
ax.legend(loc="upper left")
ax.grid(linewidth=0.4, alpha=0.5)
fig.tight_layout()

fig.savefig(paths.images_path / "15_nao_germany_t2m.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/15_nao_germany_t2m.png
:name: fig-15-nao-germany-t2m
Scatter plot of Germany DJF mean 2 m temperature against the ERA5-derived
NAO index, 1940–2024.  Each point is one winter season, coloured by year.
The crimson line is an OLS regression; the slope (°C per σ of NAO) quantifies
how much Germany warms for each standard-deviation increase in the NAO index.
```

+++

## DJF temperature maps — extreme NAO winters

Full spatial field of ERA5 DJF mean 2 m temperature for the winters with the
highest and lowest NAO index in the record (1940–2024).  Uses the same
`winter_t2m_means` DataArray (already filtered to complete DJF seasons) that
was used for the NAO computation.

```{code-cell} python
idx_max = int(np.argmax(values))
idx_min = int(np.argmin(values))
year_max, nao_max = years[idx_max], values[idx_max]
year_min, nao_min = years[idx_min], values[idx_min]

print(f"Max NAO winter: {year_max}/{year_max+1}  NAO = {nao_max:+.2f} σ")
print(f"Min NAO winter: {year_min}/{year_min+1}  NAO = {nao_min:+.2f} σ")

t2m_max_field = (winter_t2m_means.sel(time=nao.time[idx_max]) - 273.15).compute()
t2m_min_field = (winter_t2m_means.sel(time=nao.time[idx_min]) - 273.15).compute()

fig, axes = plt.subplots(
    1, 2, figsize=(18, 6),
    subplot_kw={"projection": ccrs.Robinson()},
)

for ax, t2m, year, nao_val, label in [
    (axes[0], t2m_max_field, year_max, nao_max, "Maximum NAO"),
    (axes[1], t2m_min_field, year_min, nao_min, "Minimum NAO"),
]:
    im = ax.pcolormesh(
        t2m.longitude, t2m.latitude, t2m.values,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=-40, vmax=40,
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)
    ax.set_title(
        f"{label}: {year}/{year+1}  (NAO = {nao_val:+.2f} σ)",
        fontsize=12, fontweight="bold",
    )

fig.colorbar(
    im, ax=axes, orientation="horizontal", pad=0.04, fraction=0.03,
    label="DJF mean 2 m temperature (°C)",
)
fig.suptitle(
    "ERA5 DJF mean 2 m temperature — extreme NAO winters",
    fontsize=13, fontweight="bold",
)
fig.tight_layout()
fig.savefig(paths.images_path / "15_nao_extreme_winters.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/15_nao_extreme_winters.png
:name: fig-15-nao-extreme-winters
ERA5 DJF mean 2 m temperature for the winter with the highest NAO index
(left) and lowest NAO index (right) in the 1940–2024 record.  The contrast
between the two panels illustrates the NAO's influence on winter temperatures
across the North Atlantic–European sector.
```

+++

## Temperature difference: max-NAO minus min-NAO winter

```{code-cell} python
t2m_diff = t2m_max_field - t2m_min_field  # signed: positive = warmer in NAO+ winter

vabs = float(np.abs(t2m_diff.values).max())

fig, ax = plt.subplots(
    figsize=(14, 7),
    subplot_kw={"projection": ccrs.Robinson()},
)

im = ax.pcolormesh(
    t2m_diff.longitude, t2m_diff.latitude, t2m_diff.values,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r", vmin=-vabs, vmax=vabs,
)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.04)
cbar.set_label("Temperature difference (°C)  [max-NAO minus min-NAO]")

ax.set_title(
    f"DJF temperature difference: {year_max}/{year_max+1} (NAO={nao_max:+.2f}σ)"
    f"  minus  {year_min}/{year_min+1} (NAO={nao_min:+.2f}σ)",
    fontsize=12, fontweight="bold",
)
fig.tight_layout()
fig.savefig(paths.images_path / "15_nao_extreme_diff.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/15_nao_extreme_diff.png
:name: fig-15-nao-extreme-diff
Signed temperature difference between the maximum-NAO and minimum-NAO DJF
seasons in the ERA5 record.  Red indicates regions that were warmer during
the NAO+ winter; blue indicates regions that were colder.
```

+++

## 250 hPa wind speed — extreme NAO winters

DJF mean 250 hPa wind speed (jet stream level) for the max-NAO and min-NAO
winters.  U and V components are resampled with the same YS-DEC convention
and completeness mask before computing speed = √(u² + v²).  Wind direction
is overlaid as thinned quiver arrows.

```{code-cell} python
# Build DJF mean U and V at 250 hPa with the same season mask as the NAO index
def _djf_season_mean(da):
    return (
        da.sel(time=ds.time.dt.month.isin([12, 1, 2]))
        .resample(time="YS-DEC")
        .mean()
        .sel(time=complete)
    )

u250 = _djf_season_mean(ds["u"].sel(pressure_level=250))
v250 = _djf_season_mean(ds["v"].sel(pressure_level=250))

# Select the two extreme seasons and compute wind speed
u_max = u250.sel(time=nao.time[idx_max]).compute()
v_max = v250.sel(time=nao.time[idx_max]).compute()
u_min = u250.sel(time=nao.time[idx_min]).compute()
v_min = v250.sel(time=nao.time[idx_min]).compute()

wspd_max = np.sqrt(u_max**2 + v_max**2)
wspd_min = np.sqrt(u_min**2 + v_min**2)

vmax_wspd = float(max(wspd_max.values.max(), wspd_min.values.max()))

# Thin the quiver grid to avoid overplotting (every Nth point)
N = 6
lons_q = wspd_max.longitude.values[::N]
lats_q = wspd_max.latitude.values[::N]

fig, axes = plt.subplots(
    1, 2, figsize=(18, 6),
    subplot_kw={"projection": ccrs.Robinson()},
)

for ax, wspd, u_q, v_q, year, nao_val, label in [
    (axes[0], wspd_max, u_max.values[::N, ::N], v_max.values[::N, ::N],
     year_max, nao_max, "Maximum NAO"),
    (axes[1], wspd_min, u_min.values[::N, ::N], v_min.values[::N, ::N],
     year_min, nao_min, "Minimum NAO"),
]:
    im = ax.pcolormesh(
        wspd.longitude, wspd.latitude, wspd.values,
        transform=ccrs.PlateCarree(),
        cmap="YlOrRd", vmin=0, vmax=vmax_wspd,
    )
    ax.quiver(
        lons_q, lats_q, u_q, v_q,
        transform=ccrs.PlateCarree(),
        scale=2000, width=0.001, color="black", alpha=0.55,
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)
    ax.set_title(
        f"{label}: {year}/{year+1}  (NAO = {nao_val:+.2f} σ)",
        fontsize=12, fontweight="bold",
    )

fig.colorbar(
    im, ax=axes, orientation="horizontal", pad=0.04, fraction=0.03,
    label="250 hPa wind speed (m/s)",
)
fig.suptitle(
    "ERA5 DJF mean 250 hPa wind speed — extreme NAO winters",
    fontsize=13, fontweight="bold",
)
fig.tight_layout()
fig.savefig(paths.images_path / "15_nao_extreme_250hpa_wind.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/15_nao_extreme_250hpa_wind.png
:name: fig-15-nao-extreme-250hpa-wind
DJF mean 250 hPa wind speed for the maximum-NAO (left) and minimum-NAO
(right) winters in the ERA5 record.  Colour shows wind speed; arrows show
wind direction (thinned for clarity).  The NAO+ winter typically shows a
stronger, more zonally oriented jet stream over the North Atlantic.
```

+++

## 500 hPa geopotential height — extreme NAO winters

DJF mean 500 hPa geopotential height (Z500) for the max-NAO and min-NAO
winters.  Geopotential (m²/s²) is converted to geopotential height (m) by
dividing by g = 9.80665 m/s².  Filled contours show height; contour lines
highlight the main ridges and troughs.

```{code-cell} python
G = 9.80665  # standard gravity m/s²

z500_max = (_djf_season_mean(ds["z"].sel(pressure_level=500))
            .sel(time=nao.time[idx_max]).compute() / G)
z500_min = (_djf_season_mean(ds["z"].sel(pressure_level=500))
            .sel(time=nao.time[idx_min]).compute() / G)

# Shared colour scale anchored to the combined range
vmin_z = float(min(z500_max.values.min(), z500_min.values.min()))
vmax_z = float(max(z500_max.values.max(), z500_min.values.max()))

fig, axes = plt.subplots(
    1, 2, figsize=(18, 6),
    subplot_kw={"projection": ccrs.Robinson()},
)

for ax, z500, year, nao_val, label in [
    (axes[0], z500_max, year_max, nao_max, "Maximum NAO"),
    (axes[1], z500_min, year_min, nao_min, "Minimum NAO"),
]:
    lons = z500.longitude.values
    lats = z500.latitude.values

    im = ax.pcolormesh(
        lons, lats, z500.values,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=vmin_z, vmax=vmax_z,
    )
    # Contour lines every 80 m
    cs = ax.contour(
        lons, lats, z500.values,
        levels=np.arange(np.floor(vmin_z / 80) * 80,
                         np.ceil(vmax_z / 80) * 80 + 1, 80),
        colors="black", linewidths=0.6, alpha=0.6,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(cs, fmt="%d", fontsize=7, inline=True)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)

    # NAO station markers
    for (lon, lat, name) in [
        (-25.67, 37.74, "Azores"),
        (-21.90, 64.13, "Iceland"),
    ]:
        ax.plot(lon, lat, transform=ccrs.PlateCarree(),
                marker="*", markersize=10, color="gold",
                markeredgecolor="black", markeredgewidth=0.6, zorder=5)
        ax.text(lon + 1.5, lat, name, transform=ccrs.PlateCarree(),
                fontsize=8, fontweight="bold", color="black",
                va="center", zorder=5)

    ax.set_title(
        f"{label}: {year}/{year+1}  (NAO = {nao_val:+.2f} σ)",
        fontsize=12, fontweight="bold",
    )

fig.colorbar(
    im, ax=axes, orientation="horizontal", pad=0.04, fraction=0.03,
    label="500 hPa geopotential height (m)",
)
fig.suptitle(
    "ERA5 DJF mean 500 hPa geopotential height — extreme NAO winters",
    fontsize=13, fontweight="bold",
)
fig.tight_layout()
fig.savefig(paths.images_path / "15_nao_extreme_z500.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/15_nao_extreme_z500.png
:name: fig-15-nao-extreme-z500
DJF mean 500 hPa geopotential height for the maximum-NAO (left) and
minimum-NAO (right) winters.  Contour lines are drawn every 80 m.  The
NAO+ pattern shows higher geopotential over the Azores and lower over
Iceland, steering mild Atlantic air into Europe; NAO− reverses this,
favouring blocking and cold-air advection.
```

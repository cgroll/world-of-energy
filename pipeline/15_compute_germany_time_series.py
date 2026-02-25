"""Compute ERA5 monthly Germany spatial-mean time series.

Loads the monthly ERA5 Zarr store (produced by 14_nc_to_zarr.py), masks to
Germany's land geometry (Natural Earth 10 m), and computes the spatial mean
for each monthly time step over Germany grid cells.

Variables computed (all as Germany spatial means):
  Single-level:
    msl              : mean sea level pressure (Pa)
    t2m              : 2 m temperature (K)
    tp               : total precipitation (m)
    sf               : snowfall (m of water equivalent)
    ssrd             : surface solar radiation downwards (J/m²)
    wind_speed_100m  : 100 m wind speed (m/s), derived from u100 / v100

  Pressure-level (250 hPa and 500 hPa):
    z_250hpa         : geopotential at 250 hPa (J/kg)
    z_500hpa         : geopotential at 500 hPa (J/kg)
    wind_speed_250hpa: wind speed at 250 hPa (m/s), derived from u / v
    wind_speed_500hpa: wind speed at 500 hPa (m/s), derived from u / v

Output:
  data/processed/era5/time_series/germany_monthly.parquet
  DatetimeIndex (monthly), one column per variable.
"""

# %%
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom

from woe.paths import ProjPaths

paths = ProjPaths()

# %%
# Load Zarr store lazily
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
def germany_mean(da: xr.DataArray) -> np.ndarray:
    """Return (n_time,) array of Germany spatial means, computing eagerly."""
    return (
        da.where(mask_da)
        .stack(cell=("latitude", "longitude"))
        .dropna("cell", how="all")
        .mean("cell")
        .compute()
        .values
    )


times = pd.DatetimeIndex(ds.time.values)

# %%
print("Computing Germany spatial means...")
result: dict[str, np.ndarray] = {}

# --- Single-level scalar variables ---
for var in ["msl", "t2m", "tp", "sf", "ssrd"]:
    if var in ds:
        print(f"  {var}...")
        result[var] = germany_mean(ds[var])
    else:
        print(f"  {var}: not in Zarr store, skipping")

# --- 100 m wind speed ---
if "u100" in ds and "v100" in ds:
    print("  wind_speed_100m...")
    result["wind_speed_100m"] = germany_mean(np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2))

# --- Pressure-level variables at 250 and 500 hPa ---
for level in [250, 500]:
    # Geopotential
    if "z" in ds:
        print(f"  z_{level}hpa...")
        result[f"z_{level}hpa"] = germany_mean(ds["z"].sel(pressure_level=level))

    # Wind speed from U/V components
    if "u" in ds and "v" in ds:
        print(f"  wind_speed_{level}hpa...")
        u = ds["u"].sel(pressure_level=level)
        v = ds["v"].sel(pressure_level=level)
        result[f"wind_speed_{level}hpa"] = germany_mean(np.sqrt(u**2 + v**2))

# %%
df = pd.DataFrame(result, index=times)
df.index.name = "time"
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())

# %%
output = paths.era5_germany_monthly_ts_file
output.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(output)
print(f"\nSaved to {output}")

"""Convert downloaded ERA5 monthly NetCDF files to a single Zarr store.

Reads the NetCDF files produced by 14_download_monthly_era5.py,
merges all variables into one dataset, rechunks along the time axis, and
writes to data/downloads/era5/monthly_aggregates/zarr/era5_monthly.zarr.
"""

# %%
import pandas as pd
import xarray as xr

from woe.paths import ProjPaths

paths = ProjPaths()

# %%
# Load all NetCDF files lazily (no data read yet)
print("Loading datasets...")
ds_climate = xr.open_dataset(paths.era5_sl_climate_file)
ds_wind_solar = xr.open_dataset(paths.era5_sl_wind_solar_file)
ds_accumulated = xr.open_dataset(paths.era5_sl_accumulated_file)
ds_pressure = xr.open_dataset(paths.era5_monthly_pressure_levels_file)

# New CDS API uses valid_time instead of time — normalise to time
_all = [
    ("climate", ds_climate),
    ("wind_solar", ds_wind_solar),
    ("accumulated", ds_accumulated),
    ("pressure", ds_pressure),
]
for ds_name, ds in _all:
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
        if ds_name == "climate":
            ds_climate = ds
        elif ds_name == "wind_solar":
            ds_wind_solar = ds
        elif ds_name == "accumulated":
            ds_accumulated = ds
        else:
            ds_pressure = ds

print(f"  Climate variables:     {list(ds_climate.data_vars)}")
print(f"  Wind variables:        {list(ds_wind_solar.data_vars)}")
print(f"  Accumulated variables: {list(ds_accumulated.data_vars)}")
print(f"  Pressure-level variables: {list(ds_pressure.data_vars)}")

# %%
# Align all datasets to a common time axis before merging.
# Datasets from different CDS requests can have subtly different time
# encodings (e.g. int64 vs datetime64, or ns vs us precision), which causes
# xr.merge to expand to an outer-join time axis and run out of memory.
# Normalise every time coordinate to datetime64[ns] first, then merge with
# join='inner' so only months present in all four files are kept.
def _normalise_time(ds):
    t = pd.DatetimeIndex(ds.time.values).normalize()   # drop sub-day offset
    return ds.assign_coords(time=t)

ds_climate    = _normalise_time(ds_climate)
ds_wind_solar = _normalise_time(ds_wind_solar)
ds_accumulated = _normalise_time(ds_accumulated)
ds_pressure   = _normalise_time(ds_pressure)

# Diagnose time coordinate alignment across datasets (after rename+normalise)
for name, ds in [("climate", ds_climate), ("wind_solar", ds_wind_solar),
                 ("accumulated", ds_accumulated), ("pressure", ds_pressure)]:
    t = ds.time
    print(f"  {name:12s}  time: {t.values[0]} → {t.values[-1]}  n={len(t)}  dtype={t.dtype}")

print("Merging datasets...")
ds_merged = xr.merge(
    [ds_climate, ds_wind_solar, ds_accumulated, ds_pressure],
    join="inner",
)

print(f"  Combined variables: {list(ds_merged.data_vars)}")
print(f"  Time range: {ds_merged.time.values[0]} → {ds_merged.time.values[-1]}")
print(f"  Shape: {dict(ds_merged.dims)}")

# %%
# Chunk by 1 year of time (12 months) and the full spatial grid.
# For a European bounding box this produces ~5–10 MB chunks per variable —
# a good balance between read performance and memory pressure.
chunks = {
    "time": 12,
    "latitude": len(ds_merged.latitude),
    "longitude": len(ds_merged.longitude),
}
ds_chunked = ds_merged.chunk(chunks)
print(f"Chunking: {chunks}")

# %%
# Strip original NetCDF chunk encodings — otherwise Zarr ignores our chunks
# and falls back to ECMWF defaults.
for var in ds_chunked.variables:
    ds_chunked[var].encoding.pop("chunks", None)

# %%
# Write to Zarr store
output = paths.era5_monthly_zarr_path
output.parent.mkdir(parents=True, exist_ok=True)

print(f"Writing to {output} ...")
ds_chunked.to_zarr(output, mode="w")
print("Done.")

# %%

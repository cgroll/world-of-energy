"""Download ERA5 z500 (geopotential height at 500 hPa) from ARCO-ERA5.

Source: ARCO-ERA5 public dataset on Google Cloud Storage.
  gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3

3-hourly intervals (00, 03, 06, 09, 12, 15, 18, 21 UTC).
Region: 30°N–80°N, 80°W–40°E.
One NetCDF file per day in data/downloads/z500/daily/ (8 timesteps each).
Existing files are skipped — restartable after interruption.

Variable: geopotential at 500 hPa converted to geopotential height [m] (= Φ/g).

Usage:
  python 60_z500_download.py                          # defaults: 2025-01-01 to 2025-12-31
  python 60_z500_download.py --start 2020-01-01 --end 2024-12-31

Download strategy:
  - zarr-python + fsspec directly (no xarray/dask) to avoid dask task-graph
    accumulation and RSS growth over hundreds of iterations.
  - gcsfs cache_type="none" prevents the block cache from growing unbounded.
  - One day at a time; each day file holds 8 timesteps stacked in memory then
    written in a single to_netcdf call.
  - ARCO chunks are (time=1, level=37, lat=721, lon=1440). Selecting one level
    always downloads all 37 and slices in memory — expected, cannot be avoided.
"""

# %%
import argparse
import asyncio
import gc
import time

import fsspec
import numpy as np
import pandas as pd
import psutil
import xarray as xr
import zarr

from woe.paths import ProjPaths

paths = ProjPaths()

_proc = psutil.Process()


def log(msg: str) -> None:
    rss = _proc.memory_info().rss / 1e6
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts} | {rss:5.0f} MB RSS] {msg}", flush=True)


# %%
parser = argparse.ArgumentParser(description="Download ERA5 z500 for a date range.")
parser.add_argument("--start", default="2025-01-01", help="First date to download (YYYY-MM-DD)")
parser.add_argument("--end",   default="2025-12-31", help="Last date to download (YYYY-MM-DD)")
args = parser.parse_args()

START_DATE = args.start
END_DATE   = args.end

ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# ERA5 time axis: 1940-01-01T00:00, hourly
ERA5_START = pd.Timestamp("1940-01-01T00:00:00")

GRAVITY = 9.80665  # m/s² — convert geopotential (m²/s²) to height (m)

# Spatial domain
LAT_SOUTH = 30
LAT_NORTH = 80
# ARCO uses 0–360° longitude; 80°W = 280°, 40°E = 40°
LON_WEST_0360 = 280
LON_EAST_0360 = 40

HOURS_UTC = [0, 3, 6, 9, 12, 15, 18, 21]

out_daily_dir = paths.z500_downloads_path / "daily"
out_daily_dir.mkdir(parents=True, exist_ok=True)


# %%
log("Opening ARCO-ERA5 zarr store (no dask)...")
mapper = fsspec.get_mapper(ZARR_URL, token="anon", cache_type="none")
root = zarr.open_group(mapper, mode="r")
log("Store opened.")

log("Loading coordinate arrays...")
era5_levels = root["level"][:]      # (37,)
era5_lats   = root["latitude"][:]   # (721,) descending 90 → -90
era5_lons   = root["longitude"][:]  # (1440,) 0 → 359.75

# --- Spatial slices (computed once, reused every iteration) ---
lat_mask  = (era5_lats >= LAT_SOUTH) & (era5_lats <= LAT_NORTH)
lat_idxs  = np.where(lat_mask)[0]
lat_slice = slice(int(lat_idxs[0]), int(lat_idxs[-1]) + 1)
out_lats  = era5_lats[lat_slice]  # descending: 80 … 30°N

# Western longitudes: 280°–359.75° → after -360: -80 … -0.25°
lon_west_idxs  = np.where(era5_lons >= LON_WEST_0360)[0]
lon_west_slice = slice(int(lon_west_idxs[0]), int(lon_west_idxs[-1]) + 1)

# Eastern longitudes: 0°–40°
lon_east_idxs  = np.where(era5_lons <= LON_EAST_0360)[0]
lon_east_slice = slice(int(lon_east_idxs[0]), int(lon_east_idxs[-1]) + 1)

out_lons = np.concatenate([
    era5_lons[lon_west_slice] - 360,  # 280–359.75° → -80 … -0.25°
    era5_lons[lon_east_slice],         # 0–40°
])

level_idx_map = {int(lvl): i for i, lvl in enumerate(era5_levels)}
lev_idx_500   = level_idx_map[500]

log(f"Spatial grid: {len(out_lats)} lats × {len(out_lons)} lons")
log(f"  lat  {out_lats[-1]:.2f}°N – {out_lats[0]:.2f}°N")
log(f"  lon  {out_lons[0]:.2f}° – {out_lons[-1]:.2f}°")


# %%
def fetch_z500(t_idx: int) -> np.ndarray:
    """Fetch geopotential at 500 hPa for one timestep, return as z [m]."""
    data_global = root["geopotential"][t_idx, lev_idx_500, lat_slice, :]
    data_west   = data_global[:, lon_west_slice]
    data_east   = data_global[:, lon_east_slice]
    data        = np.concatenate([data_west, data_east], axis=-1)
    del data_global, data_west, data_east
    return data / GRAVITY


# %%
dates  = pd.date_range(START_DATE, END_DATE, freq="D")
n_days = len(dates)

log(f"Downloading z500 {START_DATE} – {END_DATE}: {n_days} days × {len(HOURS_UTC)} timesteps")

for i, date in enumerate(dates):
    date_str = date.strftime("%Y-%m-%d")
    day_file = out_daily_dir / f"z500_{date_str}.nc"

    if day_file.exists():
        log(f"  [{i+1}/{n_days}] {date_str} — already on disk, skipping")
        continue

    log(f"  [{i+1}/{n_days}] {date_str} — downloading {len(HOURS_UTC)} timesteps...")
    timestamps = [date + pd.Timedelta(hours=h) for h in HOURS_UTC]
    t_idxs     = [int((ts - ERA5_START) / pd.Timedelta("1h")) for ts in timestamps]

    arrays = []
    for h, t_idx in enumerate(t_idxs):
        arrays.append(fetch_z500(t_idx))
        if (h + 1) % 4 == 0:
            gc.collect()

    data_stack = np.stack(arrays, axis=0)  # (8, n_lat, n_lon)
    del arrays
    gc.collect()

    da = xr.DataArray(
        data_stack,
        dims=["time", "latitude", "longitude"],
        coords={
            "time":      timestamps,
            "latitude":  out_lats,
            "longitude": out_lons,
        },
        name="z",
        attrs={
            "long_name": "geopotential height",
            "units":     "m",
            "level":     "500 hPa",
            "source":    "ARCO-ERA5 (gs://gcp-public-data-arco-era5)",
        },
    )
    da.to_netcdf(day_file)
    del da, data_stack
    gc.collect()

    log(f"    saved → {day_file.name}")


# %%
# Proactively close the gcsfs aiohttp session to avoid a noisy RuntimeError
# during interpreter shutdown when the weakref finalizer fires on a dead loop.
del root

try:
    _fs   = mapper.fs
    _loop = _fs.loop
    if _loop is not None and not _loop.is_closed():
        async def _close_session():
            s = getattr(_fs, '_session', None)
            if s is not None and not s.closed:
                await s.close()
        asyncio.run_coroutine_threadsafe(_close_session(), _loop).result(timeout=3)
except Exception:
    pass

mapper.fs.clear_instance_cache()
del mapper
gc.collect()

log("Done.")

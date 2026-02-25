"""Download ERA5 NAO/jet-stream variables from Google Cloud (ARCO-ERA5).

Downloads daily 12:00 UTC data for two winter periods:
  - 2014-12-01 to 2015-02-28
  - 2009-12-01 to 2010-02-28

Variables:
  Pressure-level:
  - Geopotential at 500 hPa (z500) — ridge/trough patterns
  - U-component of wind at 250 hPa — jet stream zonal flow
  - V-component of wind at 250 hPa — jet stream meridional flow
  Single-level:
  - 2m temperature — surface air temperature for meteorological context

Spatial domain: Atlantic and Europe [80°N–20°N, 90°W–40°E],
matching the bounding box used in pipeline/14_download_monthly_era5.py.

ARCO-ERA5 uses 0–360° longitude. The requested domain straddles the
wrap-around point, so western Atlantic (270°–360°) and Europe (0°–40°)
are selected separately, concatenated, and longitudes converted to the
–180°..180° convention. The resulting longitude axis is monotonically
increasing (–90 … –0.25, 0 … 40°E).

Download strategy:
  - zarr-python is used directly (no xarray, no dask) for data reads.
    This avoids dask task-graph and tokenisation-cache accumulation that
    causes RSS to grow to OOM over hundreds of compute() calls.
  - gcsfs cache_type="none" prevents the block cache from retaining
    downloaded chunks between iterations.
  - One day at a time in a loop for fine-grained progress and
    restartability.  Each day is saved as a separate NetCDF checkpoint;
    skipped on re-run so the job can be resumed after interruption.
  - Per-variable files merged from daily checkpoints, then merged into
    the final output.
  - Wall-clock timestamps and RSS logged on every line.

Folder layout (data/downloads/era5/nao_jetstream/):
  daily/{period}/{var}_{level}hpa/{date}.nc   ← one file per day (pressure-level vars)
  daily/{period}/{var}_sfc/{date}.nc          ← one file per day (single-level vars)
  {period}_{var}_{level}hpa.nc                ← merged per-variable checkpoint
  {period}_{var}_sfc.nc                       ← merged single-level checkpoint
  era5_nao_jetstream_{period}.nc              ← final merged output
"""

# %%
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
ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# ERA5 time axis starts at 1940-01-01T00:00:00 and is exactly hourly.
# Chunk structure: (time=1, level=37, lat=721, lon=1440) = ~154 MB per chunk.
# All 37 levels are bundled together — selecting one level always downloads
# all 37 and slices in memory. This is expected and cannot be avoided.
ERA5_START = pd.Timestamp("1940-01-01T00:00:00")

# Bounding box — matches pipeline/14_download_monthly_era5.py
LAT_NORTH = 80
LAT_SOUTH = 20
LON_WEST_0360 = 270   # -90°W in 0–360° convention
LON_EAST_0360 = 40

# Variables: (ARCO dataset name, pressure level in hPa, or None for single-level)
VARIABLES = [
    ("geopotential", 500),
    ("u_component_of_wind", 250),
    ("v_component_of_wind", 250),
    ("2m_temperature", None),    # single-level: no pressure dimension in ARCO
]

# Two winter periods: daily at 12:00 UTC
PERIODS = {
    "2014-2015": ("2014-12-01", "2015-02-28"),
    "2009-2010": ("2009-12-01", "2010-02-28"),
}

nao_path = paths.downloads_path / "era5" / "nao_jetstream"
nao_path.mkdir(parents=True, exist_ok=True)


# %%
# --- Open zarr store directly (no xarray, no dask) ---
log("Opening ARCO-ERA5 zarr store via zarr-python (no dask)...")
mapper = fsspec.get_mapper(ZARR_URL, token="anon", cache_type="none")
root = zarr.open_group(mapper, mode="r")
log("Store opened.")

# Load small coordinate arrays once — these fit easily in memory.
log("Loading coordinate arrays...")
era5_levels = root["level"][:]           # (37,) int
era5_lats   = root["latitude"][:]        # (721,) float64, descending 90→-90
era5_lons   = root["longitude"][:]       # (1440,) float64, 0→359.75

log(f"ERA5 time axis starts {ERA5_START}, hourly steps.")

# Pre-compute spatial integer slices — done once, reused every iteration.
lat_mask       = (era5_lats >= LAT_SOUTH) & (era5_lats <= LAT_NORTH)
lat_idxs       = np.where(lat_mask)[0]
lat_slice      = slice(int(lat_idxs[0]), int(lat_idxs[-1]) + 1)
out_lats       = era5_lats[lat_slice]

lon_west_idxs  = np.where(era5_lons >= LON_WEST_0360)[0]
lon_east_idxs  = np.where(era5_lons <= LON_EAST_0360)[0]
lon_west_slice = slice(int(lon_west_idxs[0]), int(lon_west_idxs[-1]) + 1)
lon_east_slice = slice(int(lon_east_idxs[0]), int(lon_east_idxs[-1]) + 1)
out_lons       = np.concatenate([
    era5_lons[lon_west_slice] - 360,   # 270–359.75° → -90 … -0.25°
    era5_lons[lon_east_slice],          # 0–40°
])

# Level index map
level_idx_map = {int(lvl): i for i, lvl in enumerate(era5_levels)}

log(f"Spatial grid: {len(out_lats)} lats × {len(out_lons)} lons")


# %%
def make_timestamps(start: str, end: str) -> list[str]:
    """Return daily 12:00 UTC ISO timestamps from start to end (inclusive)."""
    dates = pd.date_range(start, end, freq="D")
    return [f"{d.strftime('%Y-%m-%d')}T12:00:00" for d in dates]


def download_day(var: str, level: int | None, ts: str) -> xr.DataArray:
    """Download one variable/level/timestamp as a numpy-backed DataArray.

    Uses zarr-python integer indexing directly — no dask graphs created.
    Reads one zarr chunk (full global lat×lon for the given time+level),
    then slices the Atlantic–Europe region in memory.

    level=None selects a single-level variable (shape time×lat×lon in ARCO).
    """
    t_idx = int((pd.Timestamp(ts) - ERA5_START) / pd.Timedelta("1h"))

    # Read full lat band for the selected time (one zarr chunk fetch).
    # Shape before slicing: (n_lats_global, n_lons_global)
    if level is not None:
        lev_idx     = level_idx_map[level]
        data_global = root[var][t_idx, lev_idx, lat_slice, :]  # (241, 1440)
    else:
        # Single-level variable: ARCO array is (time, lat, lon) — no level dim.
        data_global = root[var][t_idx, lat_slice, :]            # (241, 1440)

    data_west = data_global[:, lon_west_slice]                  # (241, 360)
    data_east = data_global[:, lon_east_slice]                  # (241, 161)
    data      = np.concatenate([data_west, data_east], axis=-1) # (241, 521)
    del data_global, data_west, data_east

    return xr.DataArray(
        data[np.newaxis],   # add time dim back → (1, 241, 521)
        dims=["time", "latitude", "longitude"],
        coords={
            "time":      [pd.Timestamp(ts)],
            "latitude":  out_lats,
            "longitude": out_lons,
        },
        name=var,
    )


# %%
for period_name, (start, end) in PERIODS.items():
    out_file = nao_path / f"era5_nao_jetstream_{period_name}.nc"

    if out_file.exists():
        with xr.open_dataset(out_file) as _ds:
            expected_vars = {v for v, _ in VARIABLES}
            if expected_vars.issubset(set(_ds.data_vars)):
                log(f"[{period_name}] Final file already on disk with all variables, skipping.")
                continue
        log(f"[{period_name}] Final file exists but is missing variables — will re-merge.")

    log(f"[{period_name}] Downloading {start} – {end} at 12:00 UTC")
    timestamps = make_timestamps(start, end)
    n = len(timestamps)
    log(f"  {n} daily timesteps")

    var_files = []
    for var, level in VARIABLES:
        var_tag  = f"{var}_{level}hpa" if level is not None else f"{var}_sfc"
        var_label = f"{var} @ {level} hPa" if level is not None else f"{var} (sfc)"
        var_file = nao_path / f"{period_name}_{var_tag}.nc"
        var_files.append(var_file)

        if var_file.exists():
            log(f"  [{var_label}] merged file already on disk, skipping.")
            continue

        day_dir = nao_path / "daily" / period_name / var_tag
        day_dir.mkdir(parents=True, exist_ok=True)

        day_files = []
        for i, ts in enumerate(timestamps):
            date_str = ts[:10]
            day_file = day_dir / f"{date_str}.nc"
            day_files.append(day_file)

            if day_file.exists():
                log(f"  [{var_label}] day {i+1}/{n} {date_str} — already on disk")
                continue

            log(f"  [{var_label}] day {i+1}/{n} {date_str} — downloading...")
            da = download_day(var, level, ts)
            da.to_netcdf(day_file)
            del da
            gc.collect()

        log(f"  [{var_label}] merging {n} daily files → {var_file.name}...")
        # chunks={"time": 1} ensures streaming writes (one timestep at a time).
        with xr.open_mfdataset(day_files, combine="by_coords", chunks={"time": 1}) as ds_var:
            ds_var.to_netcdf(var_file)
        log(f"  [{var_label}] saved: {var_file.name}")

    log(f"[{period_name}] Merging variables → {out_file.name}...")
    with xr.open_mfdataset(var_files, combine="by_coords", compat="override", chunks={"time": 1}) as ds_merged:
        log(f"  shape: {dict(ds_merged.sizes)}")
        ds_merged.to_netcdf(out_file)
    log(f"[{period_name}] Done: {out_file.name}")

# %%
# Explicitly close the zarr/gcsfs objects before Python's shutdown sequence.
# gcsfs registers a weakref finalizer that calls close_session(loop, session)
# when the GCSFileSystem is GC'd.  If that happens during interpreter shutdown
# the asyncio state is inconsistent and the call raises a noisy RuntimeError.
# Fix: proactively close the aiohttp session on the fsspec background loop via
# run_coroutine_threadsafe (which submits to the running daemon-thread loop
# and blocks until done), so the finalizer is a no-op by the time GC runs.
del root

try:
    _fs   = mapper.fs
    _loop = _fs.loop          # fsspec background event loop (daemon thread)
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

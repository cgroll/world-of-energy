"""Download ERA5 reanalysis data from Google Cloud (ARCO-ERA5).

Downloads a single time snapshot (13:00 UTC on 2026-01-20) of surface and
pressure-level variables from the ARCO-ERA5 dataset on Google Cloud Storage.

Surface variables:
- 2m temperature
- 10m u/v wind components
- Mean sea level pressure
- Total precipitation
- Snowfall
- Total cloud cover

Pressure-level variables (400, 350, 300, 250, 225, 200, 175, 150, 125, 100 hPa):
- Geopotential
- Temperature
- U/V wind components

Download strategy:
  Mirrors the approach in 11_dev_pressure_levels.py which is fast and
  memory-efficient:
  - chunks={"time": 48}: lets dask align reads with the actual zarr objects
  - scalar time selection: xarray/zarr fetches only the one relevant chunk
  - plain .compute(): default threaded scheduler, no custom configuration
  - one variable at a time: ~4-40 MB per compute(), well within RAM limits

  Each variable is saved to an individual checkpoint file immediately after
  downloading. The script skips any file that already exists, making it fully
  restartable after interruption.

  Every log line includes a wall-clock timestamp and the process RSS so that
  memory growth and OOM kills are easy to spot.
"""

# %%
import gc
import sys
import time
import traceback

import psutil
import xarray as xr

from woe.paths import ProjPaths

paths = ProjPaths()

_proc = psutil.Process()


def log(msg: str) -> None:
    rss = _proc.memory_info().rss / 1e6
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts} | {rss:5.0f} MB RSS] {msg}", flush=True)


# %%
log("Opening ARCO-ERA5 zarr dataset (lazy)...")
ds = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    chunks={"time": 48},
    consolidated=True,
    storage_options={"token": "anon"},
)
log(f"Opened. dims={dict(ds.sizes)}")

# %%
# --- Configuration ---
DATE = "2026-01-20"
TIME = f"{DATE}T13:00:00"       # scalar: time dim is dropped after selection
PRESSURE_LEVELS = [400, 350, 300, 250, 225, 200, 175, 150, 125, 100]

SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation",
    "snowfall",
    "total_cloud_cover",
]

PRESSURE_LEVEL_VARS = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]

era5_path = paths.downloads_path / "era5"
era5_path.mkdir(parents=True, exist_ok=True)


def download_var(da_lazy, ind_file):
    """Download one lazy DataArray, save to disk, free memory."""
    log("  computing...")
    try:
        da = da_lazy.compute()
    except MemoryError:
        log("  MemoryError — RSS at time of failure shown above")
        sys.exit(1)
    except Exception:
        log("  unexpected error:")
        traceback.print_exc()
        sys.exit(1)
    log(f"  {da.nbytes / 1e6:.1f} MB in RAM — saving to {ind_file.name}...")
    da.to_netcdf(ind_file)
    del da
    gc.collect()
    log(f"  saved. RSS after gc: {_proc.memory_info().rss / 1e6:.0f} MB")


# %%
# --- Download surface variables ---
surface_file = era5_path / f"era5_surface_{DATE}_13utc.nc"

if surface_file.exists():
    log(f"Surface file exists, skipping: {surface_file.name}")
else:
    log(f"--- Surface variables ({TIME}) ---")
    var_files = []

    for var in SURFACE_VARS:
        var_file = era5_path / f"era5_{DATE}_13utc_surface_{var}.nc"
        var_files.append(var_file)
        if var_file.exists():
            log(f"  [{var}] already on disk — skipping")
            continue
        log(f"  [{var}] downloading (~4 MB)...")
        download_var(ds[var].sel(time=TIME), var_file)

    log(f"Merging {len(var_files)} surface files → {surface_file.name}...")
    with xr.open_mfdataset(var_files, combine="by_coords",
                           compat="override") as ds_surface:
        log(f"  shape: {dict(ds_surface.sizes)}")
        ds_surface.to_netcdf(surface_file)
    log(f"Surface file saved: {surface_file.name}")

# %%
# --- Download pressure-level variables ---
#
# All 10 levels are selected per variable in a single .compute() call.
# Since all levels share one zarr chunk (time=1, level=37, lat, lon ≈ 150 MB),
# selecting them together downloads that chunk only once.
# Each variable produces a ~40 MB array (10/37 levels retained).

pressure_file = era5_path / f"era5_pressure_levels_{DATE}_13utc.nc"

if pressure_file.exists():
    log(f"Pressure-level file exists, skipping: {pressure_file.name}")
else:
    log(f"--- Pressure-level variables ({TIME}, {PRESSURE_LEVELS}) ---")
    ind_files = []

    for var in PRESSURE_LEVEL_VARS:
        ind_file = era5_path / f"era5_{DATE}_13utc_{var}.nc"
        ind_files.append(ind_file)
        if ind_file.exists():
            log(f"  [{var}] already on disk — skipping")
            continue
        log(f"  [{var}] downloading 10 levels (~40 MB)...")
        download_var(ds[var].sel(time=TIME, level=PRESSURE_LEVELS), ind_file)

    log(f"Merging {len(ind_files)} pressure-level files → {pressure_file.name}...")
    with xr.open_mfdataset(ind_files, combine="by_coords",
                           compat="override") as ds_pressure:
        log(f"  shape: {dict(ds_pressure.sizes)}")
        ds_pressure.to_netcdf(pressure_file)
    log(f"Pressure-level file saved: {pressure_file.name}")

log("Done.")

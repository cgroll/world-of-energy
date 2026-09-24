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
# # Download ERA5 Snapshots for Offshore Wind Maximum Drawdown
#
# Downloads 10 evenly-spaced daily 12:00 UTC ERA5 snapshots from the public
# ARCO-ERA5 Zarr store on Google Cloud Storage, covering the maximum
# offshore-wind drawdown period identified in `45_RE_drawdowns.py`, plus the
# same calendar window one year earlier as a "normal wind" comparison.
#
# **Download strategy** (mirrors `19_download_daily_nao_jetstream.py`):
# zarr-python direct reads (no dask), one timestamp at a time, per-day
# NetCDF checkpoints for restartability.
#
# **Inputs**
# - `data/processed/re_drawdowns.parquet` — drawdown periods from script 45
#
# **Outputs**
# - `data/downloads/era5/nao_jetstream/daily/offshore_drawdown/` — drawdown period
# - `data/downloads/era5/nao_jetstream/daily/offshore_comparison/` — comparison period

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


# %% [markdown]
# ## Load drawdown periods and select snapshot dates

# %%
drawdowns = pd.read_parquet(paths.re_drawdowns_file)
offshore = drawdowns[drawdowns["source"] == "Wind offshore"].iloc[0]
peak_time = pd.Timestamp(offshore["peak_time"])
trough_time = pd.Timestamp(offshore["trough_time"])

dur_days = (trough_time - peak_time) / pd.Timedelta("1d")
print(f"Offshore wind max drawdown: {peak_time.date()} -> {trough_time.date()}"
      f"  ({dur_days:.0f} days)")

N_FRAMES = 10
all_days = pd.date_range(peak_time.normalize(), trough_time.normalize(), freq="D")
indices = np.linspace(0, len(all_days) - 1, N_FRAMES, dtype=int)
snapshot_dates = all_days[indices]

print(f"\nSnapshot dates ({N_FRAMES} frames):")
for d in snapshot_dates:
    print(f"  {d.date()}")

# %% [markdown]
# ## Download from ARCO-ERA5
#
# Variables (all at 12:00 UTC):
#
# | Variable | Level | Purpose |
# |---|---|---|
# | `geopotential` | 500 hPa | Z500 ridge/trough contours |
# | `u_component_of_wind` | 250 hPa | Jet stream zonal flow |
# | `v_component_of_wind` | 250 hPa | Jet stream meridional flow |
# | `100m_u_component_of_wind` | surface | 100 m zonal wind (turbine hub height proxy) |
# | `100m_v_component_of_wind` | surface | 100 m meridional wind |

# %%
ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ERA5_START = pd.Timestamp("1940-01-01T00:00:00")

# Bounding box — Atlantic + Europe (matches scripts 14 & 19)
LAT_NORTH, LAT_SOUTH = 80, 20
LON_WEST_0360, LON_EAST_0360 = 270, 40

VARIABLES = [
    ("geopotential", 500),
    ("100m_u_component_of_wind", None),
    ("100m_v_component_of_wind", None),
]

out_dir = paths.era5_nao_jetstream_path / "daily" / "offshore_drawdown"
out_dir.mkdir(parents=True, exist_ok=True)

# Comparison period: same calendar window, one year earlier
COMPARISON_OFFSET = pd.DateOffset(years=-1)
comparison_dates = snapshot_dates + COMPARISON_OFFSET
comp_dir = paths.era5_nao_jetstream_path / "daily" / "offshore_comparison"
comp_dir.mkdir(parents=True, exist_ok=True)

print(f"\nComparison dates (offset {COMPARISON_OFFSET}):")
for d in comparison_dates:
    print(f"  {d.date()}")

# %%
log("Opening ARCO-ERA5 zarr store...")
mapper = fsspec.get_mapper(ZARR_URL, token="anon", cache_type="none")
root = zarr.open_group(mapper, mode="r")

era5_levels = root["level"][:]
era5_lats = root["latitude"][:]
era5_lons = root["longitude"][:]

lat_mask = (era5_lats >= LAT_SOUTH) & (era5_lats <= LAT_NORTH)
lat_idxs = np.where(lat_mask)[0]
lat_slice = slice(int(lat_idxs[0]), int(lat_idxs[-1]) + 1)
out_lats = era5_lats[lat_slice]

lon_west_idxs = np.where(era5_lons >= LON_WEST_0360)[0]
lon_east_idxs = np.where(era5_lons <= LON_EAST_0360)[0]
lon_west_slice = slice(int(lon_west_idxs[0]), int(lon_west_idxs[-1]) + 1)
lon_east_slice = slice(int(lon_east_idxs[0]), int(lon_east_idxs[-1]) + 1)
out_lons = np.concatenate([
    era5_lons[lon_west_slice] - 360,
    era5_lons[lon_east_slice],
])

level_idx_map = {int(lvl): i for i, lvl in enumerate(era5_levels)}
log(f"Spatial grid: {len(out_lats)} lats x {len(out_lons)} lons")


# %%
def download_day(var: str, level: int | None, ts: str) -> xr.DataArray:
    """Download one variable/level/timestamp from ARCO-ERA5 via zarr-python."""
    t_idx = int((pd.Timestamp(ts) - ERA5_START) / pd.Timedelta("1h"))

    if level is not None:
        lev_idx = level_idx_map[level]
        data_global = root[var][t_idx, lev_idx, lat_slice, :]
    else:
        data_global = root[var][t_idx, lat_slice, :]

    data_west = data_global[:, lon_west_slice]
    data_east = data_global[:, lon_east_slice]
    data = np.concatenate([data_west, data_east], axis=-1)
    del data_global, data_west, data_east

    return xr.DataArray(
        data[np.newaxis],
        dims=["time", "latitude", "longitude"],
        coords={
            "time": [pd.Timestamp(ts)],
            "latitude": out_lats,
            "longitude": out_lons,
        },
        name=var,
    )


# %%
for label, dates, target_dir in [
    ("drawdown", snapshot_dates, out_dir),
    ("comparison", comparison_dates, comp_dir),
]:
    log(f"--- Downloading {label} snapshots ---")
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        ts = f"{date_str}T12:00:00"

        for var, level in VARIABLES:
            var_tag = f"{var}_{level}hpa" if level is not None else f"{var}_sfc"
            day_dir = target_dir / var_tag
            day_dir.mkdir(parents=True, exist_ok=True)
            day_file = day_dir / f"{date_str}.nc"

            if day_file.exists():
                log(f"  {var_tag} {date_str} — cached")
                continue

            log(f"  {var_tag} {date_str} — downloading...")
            da = download_day(var, level, ts)
            da.to_netcdf(day_file)
            del da
            gc.collect()

log("All snapshots downloaded.")

# %% — Clean up zarr/gcsfs connections
del root
try:
    _fs = mapper.fs
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

# %%

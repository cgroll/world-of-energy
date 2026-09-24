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
# # Jet Stream Animation — Maximum Offshore Wind Drawdown vs Normal Year
#
# Renders 10 daily ERA5 snapshots (downloaded by
# `47_download_offshore_ddown_era5.py`) as 100 m wind speed (colour) +
# Z500 geopotential-height contours and assembles them into animated GIFs.
#
# Two GIFs are produced:
# 1. **Drawdown period** — the worst sustained offshore-wind drought
# 2. **Comparison period** — same calendar window one year earlier (normal wind)
#
# Both share identical colour scales for direct visual comparison.
#
# **Inputs**
# - `data/processed/re_drawdowns.parquet` — drawdown periods from script 45
# - `data/downloads/era5/nao_jetstream/daily/offshore_drawdown/` — drawdown NetCDFs
# - `data/downloads/era5/nao_jetstream/daily/offshore_comparison/` — comparison NetCDFs

# %%
import shutil

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

from woe.paths import ProjPaths


def show():
    """plt.show() wrapper: no-op when matplotlib uses a non-interactive backend."""
    try:
        plt.show()
    except Exception:
        pass


paths = ProjPaths()

# %% [markdown]
# ## Load drawdown period and snapshot dates

# %%
drawdowns = pd.read_parquet(paths.re_drawdowns_file)
offshore = drawdowns[drawdowns["source"] == "Wind offshore"].iloc[0]
peak_time = pd.Timestamp(offshore["peak_time"])
trough_time = pd.Timestamp(offshore["trough_time"])

N_FRAMES = 10
all_days = pd.date_range(peak_time.normalize(), trough_time.normalize(), freq="D")
indices = np.linspace(0, len(all_days) - 1, N_FRAMES, dtype=int)
snapshot_dates = all_days[indices]

COMPARISON_OFFSET = pd.DateOffset(years=-1)
comparison_dates = snapshot_dates + COMPARISON_OFFSET

print(f"Offshore wind max drawdown: {peak_time.date()} -> {trough_time.date()}")
print(f"{N_FRAMES} snapshot dates loaded.")
print(f"Comparison dates: {comparison_dates[0].date()} -> {comparison_dates[-1].date()}")

# %% [markdown]
# ## Load downloaded ERA5 fields

# %%
G = 9.80665

VARIABLES = [
    ("geopotential", 500),
    ("100m_u_component_of_wind", None),
    ("100m_v_component_of_wind", None),
]


def load_snapshots(data_dir, dates):
    """Load ERA5 snapshots from per-day NetCDF files."""
    datasets = []
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        arrays = {}
        for var, level in VARIABLES:
            var_tag = f"{var}_{level}hpa" if level is not None else f"{var}_sfc"
            f = data_dir / var_tag / f"{date_str}.nc"
            arrays[var_tag] = xr.open_dataarray(f)
        datasets.append(arrays)
    return datasets


drawdown_dir = paths.era5_nao_jetstream_path / "daily" / "offshore_drawdown"
comparison_dir = paths.era5_nao_jetstream_path / "daily" / "offshore_comparison"

drawdown_data = load_snapshots(drawdown_dir, snapshot_dates)
comparison_data = load_snapshots(comparison_dir, comparison_dates)

print(f"Loaded {len(drawdown_data)} drawdown + {len(comparison_data)} comparison snapshots.")

# %% [markdown]
# ## Pre-compute fields and shared limits across both periods

# %%
PROJ = ccrs.Orthographic(central_longitude=-10, central_latitude=55)
EXTENT = [-90, 40, 20, 80]
N_THIN = 5


def precompute_fields(datasets):
    """Extract wind speed, Z500 height, and wind components from loaded data."""
    result = []
    for arrays in datasets:
        u100 = arrays["100m_u_component_of_wind_sfc"].squeeze().values
        v100 = arrays["100m_v_component_of_wind_sfc"].squeeze().values
        wspd100 = np.sqrt(u100**2 + v100**2)
        z500 = arrays["geopotential_500hpa"].squeeze().values / G
        lats = arrays["100m_u_component_of_wind_sfc"].latitude.values
        lons = arrays["100m_u_component_of_wind_sfc"].longitude.values
        result.append((wspd100, z500, u100, v100, lats, lons))
    return result


drawdown_fields = precompute_fields(drawdown_data)
comparison_fields = precompute_fields(comparison_data)

# Shared limits across both periods for consistent colour scales
all_fields = drawdown_fields + comparison_fields
z_all_min = min(float(f[1].min()) for f in all_fields)
z_all_max = max(float(f[1].max()) for f in all_fields)
wspd100_max = max(float(f[0].max()) for f in all_fields)

Z500_LEVELS = np.arange(
    np.floor(z_all_min / 80) * 80,
    np.ceil(z_all_max / 80) * 80 + 1,
    80,
)

print(f"Shared 100 m wind speed vmax: {wspd100_max:.1f} m/s")
print(f"Z500 levels: {Z500_LEVELS[0]:.0f} – {Z500_LEVELS[-1]:.0f} m "
      f"({len(Z500_LEVELS)} contours)")

# %% [markdown]
# ## Render frames


# %%
def render_frames(dates, fields, prefix, suptitle):
    """Render map frames and return list of saved file paths."""
    frame_paths = []
    for i, (date, (wspd100, z500, u100, v100, lats, lons)) in enumerate(
        zip(dates, fields)
    ):
        fig, ax = plt.subplots(
            figsize=(10, 8),
            subplot_kw={"projection": PROJ},
        )
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

        im = ax.pcolormesh(
            lons, lats, wspd100,
            transform=ccrs.PlateCarree(),
            cmap="YlGnBu", vmin=0, vmax=wspd100_max,
        )
        cs = ax.contour(
            lons, lats, z500,
            levels=Z500_LEVELS,
            colors="black", linewidths=0.6, alpha=0.55,
            transform=ccrs.PlateCarree(),
        )
        ax.clabel(cs, fmt="%d", fontsize=7, inline=True)

        lons_q = lons[::N_THIN]
        lats_q = lats[::N_THIN]
        ax.quiver(
            lons_q, lats_q,
            u100[::N_THIN, ::N_THIN],
            v100[::N_THIN, ::N_THIN],
            transform=ccrs.PlateCarree(),
            scale=1500, width=0.0012, color="k", alpha=0.5,
        )

        ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
        ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)
        ax.plot(
            10.5, 51.2, transform=ccrs.PlateCarree(),
            marker="*", markersize=11, color="red",
            markeredgecolor="black", markeredgewidth=0.6, zorder=5,
        )
        ax.text(
            12.5, 51.2, "Germany", transform=ccrs.PlateCarree(),
            fontsize=9, fontweight="bold", color="black", va="center", zorder=5,
        )

        fig.colorbar(
            im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.04,
            label="100 m wind speed (m/s)",
        )

        ax.set_title(
            f"{date.strftime('%d %B %Y')}  (12:00 UTC)\n"
            f"Z500 contours every 80 m  |  100 m wind arrows",
            fontsize=11, fontweight="bold",
        )
        fig.suptitle(
            f"{suptitle}  —  frame {i + 1}/{N_FRAMES}",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()

        frame_file = paths.images_path / f"{prefix}_frame_{i:02d}.png"
        fig.savefig(frame_file, dpi=120, bbox_inches="tight")
        frame_paths.append(frame_file)
        plt.close(fig)

    print(f"Saved {len(frame_paths)} {prefix} frames.")
    return frame_paths


# %%
drawdown_frames = render_frames(
    snapshot_dates, drawdown_fields,
    prefix="48_offshore_jetstream",
    suptitle=(f"Wind offshore max drawdown  "
              f"({peak_time.strftime('%b %Y')} – {trough_time.strftime('%b %Y')})"),
)

comparison_frames = render_frames(
    comparison_dates, comparison_fields,
    prefix="48_offshore_comparison",
    suptitle=(f"Wind offshore — normal year comparison  "
              f"({comparison_dates[0].strftime('%b %Y')} – "
              f"{comparison_dates[-1].strftime('%b %Y')})"),
)

# %% [markdown]
# ## Assemble GIFs


# %%
def assemble_gif(frame_paths, gif_path, start_path=None, end_path=None):
    """Assemble frames into an animated GIF and optionally save start/end PNGs."""
    frames = [Image.open(p) for p in frame_paths]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=1200,  # ms per frame
        loop=0,         # infinite loop
    )
    print(f"GIF saved: {gif_path}")
    if start_path:
        shutil.copy(frame_paths[0], start_path)
    if end_path:
        shutil.copy(frame_paths[-1], end_path)


# %%
assemble_gif(
    drawdown_frames,
    paths.images_path / "48_offshore_jetstream.gif",
    start_path=paths.images_path / "48_offshore_jetstream_start.png",
    end_path=paths.images_path / "48_offshore_jetstream_end.png",
)

assemble_gif(
    comparison_frames,
    paths.images_path / "48_offshore_comparison.gif",
    start_path=paths.images_path / "48_offshore_comparison_start.png",
    end_path=paths.images_path / "48_offshore_comparison_end.png",
)
show()

# %% [markdown]
# ```{figure} ../../output/images/48_offshore_jetstream.gif
# :name: fig-48-offshore-jetstream
# Animated sequence of 10 daily ERA5 snapshots spanning the maximum
# cumulative-balance drawdown of offshore wind capacity factors in Germany.
# Colour shows 100 m wind speed (a proxy for turbine-hub-height wind), black
# contours are Z500 geopotential height (every 80 m), and arrows indicate
# 100 m wind direction.  The sequence reveals the persistent large-scale
# circulation pattern — typically blocking or a displaced jet stream —
# responsible for the worst sustained offshore wind drought in the 1980–2019
# PECD record.
# ```

# %% [markdown]
# ```{figure} ../../output/images/48_offshore_comparison.gif
# :name: fig-48-offshore-comparison
# Same calendar window as {numref}`fig-48-offshore-jetstream` but one year
# earlier, showing typical wind conditions for comparison.  The stronger
# 100 m wind speeds and more zonal jet stream flow contrast sharply with the
# blocked pattern during the drawdown period.
# ```

# %%

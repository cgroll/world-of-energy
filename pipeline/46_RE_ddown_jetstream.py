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
# # Jet Stream During Maximum Renewable Energy Drawdowns — Germany
#
# For each renewable source (solar PV, wind onshore, wind offshore) we
# identify the maximum cumulative-balance drawdown period from the PECD ERA5
# capacity factor record (see `45_RE_drawdowns.py`) and then show the
# corresponding monthly 250 hPa wind speed and Z500 geopotential height
# fields from ERA5.  This reveals how persistent large-scale circulation
# anomalies (blocking, jet-stream displacement) drive the worst sustained
# energy shortfalls.
#
# **Inputs**
# - `data/processed/pecd/pecd_regions.parquet` — PECD ERA5 capacity factors
# - `data/downloads/era5/monthly_aggregates/zarr/era5_monthly.zarr` — ERA5 monthly fields

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from woe.paths import ProjPaths


def show():
    """plt.show() wrapper: no-op when matplotlib uses a non-interactive backend."""
    try:
        plt.show()
    except Exception:
        pass


paths = ProjPaths()

# %% [markdown]
# ## Reproduce maximum drawdown periods from script 45

# %%
df = pd.read_parquet(paths.pecd_processed_file)

cf_solar = df["solar_photovoltaic_power_generation"]["capacity_factor_ratio"]
cf_onshore = df["wind_power_generation_onshore"]["capacity_factor_ratio"]
cf_offshore = df["wind_power_generation_offshore"]["capacity_factor_ratio"]

cf_solar = cf_solar.dropna(axis=1, how="all")
cf_onshore = cf_onshore.dropna(axis=1, how="all")
cf_offshore = cf_offshore.dropna(axis=1, how="all")

COUNTRY = "DE"
QUANTILE = 0.20

sources = {}
if COUNTRY in cf_solar.columns:
    sources["Solar PV"] = (cf_solar[COUNTRY], "#f4b942")
if COUNTRY in cf_onshore.columns:
    sources["Wind onshore"] = (cf_onshore[COUNTRY], "#4a90d9")
if COUNTRY in cf_offshore.columns:
    sources["Wind offshore"] = (cf_offshore[COUNTRY], "#1a5fa8")


def cumulative_balance(series: pd.Series) -> pd.Series:
    """Cumulative sum of (CF - p20), y-axis in equivalent full-load hours."""
    deviation = series - series.quantile(QUANTILE)
    return deviation.cumsum()


def max_drawdown_info(balance: pd.Series) -> dict:
    """Return start, trough, and magnitude of the maximum drawdown."""
    running_max = balance.cummax()
    drawdown = balance - running_max
    trough_time = drawdown.idxmin()
    trough_val = balance[trough_time]
    peak_val = running_max[trough_time]
    peak_time = balance[:trough_time].idxmax()
    magnitude = peak_val - trough_val
    return {
        "peak_time": peak_time,
        "trough_time": trough_time,
        "peak_val": peak_val,
        "trough_val": trough_val,
        "magnitude": magnitude,
    }


balances = {name: cumulative_balance(s) for name, (s, _) in sources.items()}
drawdown_info = {name: max_drawdown_info(bal) for name, bal in balances.items()}

for name, info in drawdown_info.items():
    dur_days = (info["trough_time"] - info["peak_time"]) / pd.Timedelta("1d")
    print(
        f"  {name}: peak {info['peak_time'].date()} -> trough {info['trough_time'].date()}"
        f"  ({dur_days:.0f} days)"
    )

# %% [markdown]
# ## Load ERA5 monthly fields

# %%
G = 9.80665  # standard gravity (m/s2)
ds = xr.open_zarr(paths.era5_monthly_zarr_path)

# Map settings (same projection as script 17)
PROJ = ccrs.Orthographic(central_longitude=-10, central_latitude=55)
EXTENT = [-90, 40, 20, 80]  # W, E, S, N (PlateCarree)
N_THIN = 5  # quiver thinning factor

# %% [markdown]
# ## Plot consecutive monthly jet-stream maps for each drawdown

# %%
MAX_MONTHS = 12  # cap the number of panels to keep figures readable


def months_in_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Return a monthly DatetimeIndex covering start to end (inclusive)."""
    first = start.to_period("M").to_timestamp()
    last = end.to_period("M").to_timestamp()
    return pd.date_range(first, last, freq="MS")


for name, info in drawdown_info.items():
    months = months_in_range(info["peak_time"], info["trough_time"])
    if len(months) > MAX_MONTHS:
        months = months[:MAX_MONTHS]
        truncated = True
    else:
        truncated = False

    n_months = len(months)
    ncols = min(n_months, 4)
    nrows = int(np.ceil(n_months / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 5.5 * nrows),
        subplot_kw={"projection": PROJ},
    )
    axes_flat = np.atleast_1d(axes).ravel()

    # Pre-compute shared colour limits across all months for this drawdown
    wspd_max = 0.0
    z_all_min, z_all_max = 1e9, 0.0
    fields = []
    for month in months:
        date_str = month.strftime("%Y-%m")
        z500 = (ds["z"].sel(time=date_str, pressure_level=500).squeeze() / G).compute()
        u250 = ds["u"].sel(time=date_str, pressure_level=250).squeeze().compute()
        v250 = ds["v"].sel(time=date_str, pressure_level=250).squeeze().compute()
        wspd = np.sqrt(u250**2 + v250**2)
        fields.append((z500, u250, v250, wspd))
        wspd_max = max(wspd_max, float(wspd.max()))
        z_all_min = min(z_all_min, float(z500.min()))
        z_all_max = max(z_all_max, float(z500.max()))

    z500_levels = np.arange(
        np.floor(z_all_min / 80) * 80,
        np.ceil(z_all_max / 80) * 80 + 1,
        80,
    )

    lons_q = fields[0][1].longitude.values[::N_THIN]
    lats_q = fields[0][1].latitude.values[::N_THIN]

    im = None
    for i, (month, (z500, u250, v250, wspd)) in enumerate(zip(months, fields)):
        ax = axes_flat[i]
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

        im = ax.pcolormesh(
            wspd.longitude, wspd.latitude, wspd.values,
            transform=ccrs.PlateCarree(),
            cmap="plasma", vmin=0, vmax=wspd_max,
        )
        cs = ax.contour(
            z500.longitude, z500.latitude, z500.values,
            levels=z500_levels,
            colors="white", linewidths=0.8, alpha=0.65,
            transform=ccrs.PlateCarree(),
        )
        ax.clabel(cs, fmt="%d", fontsize=6, inline=True, colors="white")
        ax.quiver(
            lons_q, lats_q,
            u250.values[::N_THIN, ::N_THIN],
            v250.values[::N_THIN, ::N_THIN],
            transform=ccrs.PlateCarree(),
            scale=2500, width=0.0012, color="white", alpha=0.65,
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
        ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)

        # Mark Germany
        ax.plot(
            10.5, 51.2, transform=ccrs.PlateCarree(),
            marker="*", markersize=9, color="gold",
            markeredgecolor="black", markeredgewidth=0.5, zorder=5,
        )

        ax.set_title(month.strftime("%B %Y"), fontsize=10, fontweight="bold")

    # Hide unused axes
    for j in range(n_months, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Colour bar
    fig.colorbar(
        im, ax=axes_flat[:n_months].tolist(),
        orientation="horizontal", pad=0.04, fraction=0.03,
        label="250 hPa wind speed (m/s)",
    )

    trunc_note = f" (first {MAX_MONTHS} months shown)" if truncated else ""
    fig.suptitle(
        f"{name} — jet stream during maximum drawdown{trunc_note}\n"
        f"{info['peak_time'].strftime('%b %Y')} → {info['trough_time'].strftime('%b %Y')}"
        f"  |  250 hPa wind & Z500 (ERA5)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    slug = name.lower().replace(" ", "_")
    fig.savefig(
        paths.images_path / f"46_{slug}_jetstream.png",
        dpi=150, bbox_inches="tight",
    )
    show()

# %% [markdown]
# ```{figure} ../../output/images/46_solar_pv_jetstream.png
# :name: fig-46-solar-pv-jetstream
# 250 hPa wind speed (colour) and Z500 geopotential height contours (white,
# every 80 m) for the months spanning the maximum cumulative-balance drawdown
# of solar PV capacity factors in Germany.  White arrows indicate wind
# direction at 250 hPa.
# ```
#
# ```{figure} ../../output/images/46_wind_onshore_jetstream.png
# :name: fig-46-wind-onshore-jetstream
# As above, for the maximum drawdown of onshore wind capacity factors.
# ```
#
# ```{figure} ../../output/images/46_wind_offshore_jetstream.png
# :name: fig-46-wind-offshore-jetstream
# As above, for the maximum drawdown of offshore wind capacity factors.
# ```

# %%

# %%
import io

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image

from woe.paths import ProjPaths

paths = ProjPaths()
paths.images_path.mkdir(parents=True, exist_ok=True)

# %%
G = 9.80665            # standard gravity (m/s²)
JET_THRESHOLD = 30     # m/s — standard meteorological jet-stream criterion
GIF_DURATION_MS = 300  # milliseconds per frame

# North Atlantic / Europe domain
EXTENT = [-90, 40, 20, 80]   # [lon_min, lon_max, lat_min, lat_max] in PlateCarree

PROJ = ccrs.LambertConformal(
    central_longitude=-20,
    central_latitude=55,
    standard_parallels=(35, 65),
)

PERIODS = ["2014-2015", "2009-2010"]


# %%
def render_frame_t2m_z500(
    lats: np.ndarray,
    lons: np.ndarray,
    t2m_celsius: np.ndarray,
    gph500: np.ndarray,
    date_str: str,
    period_name: str,
) -> Image.Image:
    """Render a 2m-temperature + Z500-contour frame and return a PIL Image."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=PROJ)

    im = ax.pcolormesh(
        lons, lats, t2m_celsius,
        cmap="RdBu_r",
        vmin=-30, vmax=20,
        transform=ccrs.PlateCarree(),
        shading="auto",
        zorder=1,
    )

    z500_levels = np.arange(4800, 6001, 60)
    cs = ax.contour(
        lons, lats, gph500,
        levels=z500_levels,
        colors="black",
        linewidths=0.6,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    ax.clabel(cs, fmt="%d m", fontsize=6, inline=True, inline_spacing=3)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, color="0.15", zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle="--",
                   color="0.35", zorder=4)

    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.3,
        color="gray",
        alpha=0.5,
        linestyle=":",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02,
                        fraction=0.03, aspect=30, shrink=0.85)
    cbar.set_label("2m temperature (°C)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        f"2m temperature (colour) + Z500 geopotential height (contours, m)\n"
        f"{date_str} 12:00 UTC  |  ERA5  |  Winter {period_name}",
        fontsize=11, pad=8,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# %%
for period_name in PERIODS:
    data_file = paths.era5_nao_jetstream_path / f"era5_nao_jetstream_{period_name}.nc"
    gif_file = paths.images_path / f"18_t2m_z500_{period_name}.gif"

    print(f"[{period_name}] Loading {data_file.name} ...")
    ds = xr.open_dataset(data_file)
    lats = ds.latitude.values
    lons = ds.longitude.values
    n = len(ds.time)

    frames = []
    for i, t in enumerate(ds.time.values):
        date_str = str(t)[:10]
        gph500 = ds["geopotential"].sel(time=t).values / G
        t2m_celsius = ds["2m_temperature"].sel(time=t).values - 273.15

        frame = render_frame_t2m_z500(lats, lons, t2m_celsius, gph500,
                                      date_str, period_name)
        frames.append(frame)
        print(f"  frame {i + 1:3d}/{n}  {date_str}", end="\r")

    frames[0].save(
        gif_file,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_DURATION_MS,
        loop=0,
    )
    print(f"\n[{period_name}] Saved: {gif_file.name}  ({n} frames)")
    ds.close()


# %%
def render_frame(
    lats: np.ndarray,
    lons: np.ndarray,
    gph500: np.ndarray,
    wspd250: np.ndarray,
    date_str: str,
    period_name: str,
) -> Image.Image:
    """Render a single Z500 + jet-stream synoptic map frame and return a PIL Image."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=PROJ)

    ax.add_feature(cfeature.OCEAN, facecolor="#d0e8f5", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#f0ede8", zorder=0)

    z500_levels = np.arange(4800, 6001, 60)
    cs = ax.contour(
        lons, lats, gph500,
        levels=z500_levels,
        colors="black",
        linewidths=0.6,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    ax.clabel(cs, fmt="%d m", fontsize=6, inline=True, inline_spacing=3)

    jet_speed_masked = np.where(wspd250 >= JET_THRESHOLD, wspd250, np.nan)
    ax.contourf(
        lons, lats, jet_speed_masked,
        levels=[JET_THRESHOLD, 300],
        colors=["mediumpurple"],
        alpha=0.75,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )
    ax.contour(
        lons, lats, wspd250,
        levels=[JET_THRESHOLD],
        colors=["rebeccapurple"],
        linewidths=1.0,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, color="black", zorder=6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle="--",
                   color="0.35", zorder=6)

    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.3,
        color="gray",
        alpha=0.5,
        linestyle=":",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    legend_elements = [
        mpatches.Patch(
            facecolor="mediumpurple", edgecolor="rebeccapurple",
            alpha=0.75, label=f"Jet stream ≥ {JET_THRESHOLD} m/s  (250 hPa)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9,
              framealpha=0.85)

    ax.set_title(
        f"Z500 geopotential height (contours, m) + Jet stream (purple)\n"
        f"{date_str} 12:00 UTC  |  ERA5  |  Winter {period_name}",
        fontsize=11, pad=8,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# %%
for period_name in PERIODS:
    data_file = paths.era5_nao_jetstream_path / f"era5_nao_jetstream_{period_name}.nc"
    gif_file = paths.images_path / f"18_nao_jet_stream_{period_name}.gif"

    print(f"[{period_name}] Loading {data_file.name} ...")
    ds = xr.open_dataset(data_file)
    lats = ds.latitude.values
    lons = ds.longitude.values
    n = len(ds.time)
    print(f"  {n} timesteps, lat {lats[0]:.1f}–{lats[-1]:.1f}, "
          f"lon {lons[0]:.1f}–{lons[-1]:.1f}")

    frames = []
    for i, t in enumerate(ds.time.values):
        date_str = str(t)[:10]

        gph500 = ds["geopotential"].sel(time=t).values / G

        u250 = ds["u_component_of_wind"].sel(time=t).values
        v250 = ds["v_component_of_wind"].sel(time=t).values
        wspd250 = np.sqrt(u250 ** 2 + v250 ** 2)

        frame = render_frame(lats, lons, gph500, wspd250, date_str, period_name)
        frames.append(frame)
        print(f"  frame {i + 1:3d}/{n}  {date_str}", end="\r")

    frames[0].save(
        gif_file,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_DURATION_MS,
        loop=0,
    )
    print(f"\n[{period_name}] Saved: {gif_file.name}  ({n} frames)")
    ds.close()


# %%
def render_frame_combined(
    lats: np.ndarray,
    lons: np.ndarray,
    gph500: np.ndarray,
    wspd250: np.ndarray,
    t2m_celsius: np.ndarray,
    date_str: str,
    period_name: str,
) -> Image.Image:
    """Two-panel frame: left Z500+jet, right Z500+T2m. Returns a PIL Image."""
    fig = plt.figure(figsize=(22, 9))
    ax_jet = fig.add_subplot(1, 2, 1, projection=PROJ)
    ax_t2m = fig.add_subplot(1, 2, 2, projection=PROJ)

    z500_levels = np.arange(4800, 6001, 60)

    # ---- Left panel: Z500 + jet stream ----
    ax_jet.add_feature(cfeature.OCEAN, facecolor="#d0e8f5", zorder=0)
    ax_jet.add_feature(cfeature.LAND, facecolor="#f0ede8", zorder=0)

    cs_jet = ax_jet.contour(
        lons, lats, gph500,
        levels=z500_levels, colors="black", linewidths=0.6,
        transform=ccrs.PlateCarree(), zorder=3,
    )
    ax_jet.clabel(cs_jet, fmt="%d m", fontsize=5, inline=True, inline_spacing=2)

    jet_masked = np.where(wspd250 >= JET_THRESHOLD, wspd250, np.nan)
    ax_jet.contourf(
        lons, lats, jet_masked,
        levels=[JET_THRESHOLD, 300], colors=["mediumpurple"], alpha=0.75,
        transform=ccrs.PlateCarree(), zorder=4,
    )
    ax_jet.contour(
        lons, lats, wspd250,
        levels=[JET_THRESHOLD], colors=["rebeccapurple"], linewidths=1.0,
        transform=ccrs.PlateCarree(), zorder=5,
    )

    ax_jet.add_feature(cfeature.COASTLINE, linewidth=0.6, color="black", zorder=6)
    ax_jet.add_feature(cfeature.BORDERS, linewidth=0.35, linestyle="--",
                       color="0.35", zorder=6)
    ax_jet.set_extent(EXTENT, crs=ccrs.PlateCarree())

    gl_jet = ax_jet.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                               linewidth=0.3, color="gray", alpha=0.5, linestyle=":")
    gl_jet.top_labels = False
    gl_jet.right_labels = False
    gl_jet.xlabel_style = {"size": 7}
    gl_jet.ylabel_style = {"size": 7}

    ax_jet.legend(
        handles=[mpatches.Patch(facecolor="mediumpurple", edgecolor="rebeccapurple",
                                alpha=0.75, label=f"Jet ≥ {JET_THRESHOLD} m/s (250 hPa)")],
        loc="lower left", fontsize=8, framealpha=0.85,
    )
    ax_jet.set_title("Z500 (contours) + Jet stream (purple)", fontsize=10, pad=6)

    # ---- Right panel: Z500 + 2m temperature ----
    im = ax_t2m.pcolormesh(
        lons, lats, t2m_celsius,
        cmap="RdBu_r", vmin=-30, vmax=20,
        transform=ccrs.PlateCarree(), shading="auto", zorder=1,
    )

    cs_t2m = ax_t2m.contour(
        lons, lats, gph500,
        levels=z500_levels, colors="black", linewidths=0.6,
        transform=ccrs.PlateCarree(), zorder=3,
    )
    ax_t2m.clabel(cs_t2m, fmt="%d m", fontsize=5, inline=True, inline_spacing=2)

    ax_t2m.add_feature(cfeature.COASTLINE, linewidth=0.6, color="0.15", zorder=4)
    ax_t2m.add_feature(cfeature.BORDERS, linewidth=0.35, linestyle="--",
                       color="0.35", zorder=4)
    ax_t2m.set_extent(EXTENT, crs=ccrs.PlateCarree())

    gl_t2m = ax_t2m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                               linewidth=0.3, color="gray", alpha=0.5, linestyle=":")
    gl_t2m.top_labels = False
    gl_t2m.left_labels = False
    gl_t2m.xlabel_style = {"size": 7}
    gl_t2m.ylabel_style = {"size": 7}

    cbar = fig.colorbar(im, ax=ax_t2m, orientation="vertical",
                        pad=0.04, fraction=0.025, aspect=35, shrink=0.9)
    cbar.set_label("2m temperature (°C)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax_t2m.set_title("Z500 (contours) + 2m temperature", fontsize=10, pad=6)

    fig.suptitle(
        f"{date_str} 12:00 UTC  |  ERA5 reanalysis  |  Winter {period_name}",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# %%
for period_name in PERIODS:
    data_file = paths.era5_nao_jetstream_path / f"era5_nao_jetstream_{period_name}.nc"
    gif_file = paths.images_path / f"18_combined_{period_name}.gif"

    print(f"[{period_name}] Loading {data_file.name} ...")
    ds = xr.open_dataset(data_file)
    lats = ds.latitude.values
    lons = ds.longitude.values
    n = len(ds.time)

    frames = []
    for i, t in enumerate(ds.time.values):
        date_str = str(t)[:10]

        gph500 = ds["geopotential"].sel(time=t).values / G
        u250 = ds["u_component_of_wind"].sel(time=t).values
        v250 = ds["v_component_of_wind"].sel(time=t).values
        wspd250 = np.sqrt(u250 ** 2 + v250 ** 2)
        t2m_celsius = ds["2m_temperature"].sel(time=t).values - 273.15

        frame = render_frame_combined(lats, lons, gph500, wspd250, t2m_celsius,
                                      date_str, period_name)
        frames.append(frame)
        print(f"  frame {i + 1:3d}/{n}  {date_str}", end="\r")

    frames[0].save(
        gif_file,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_DURATION_MS,
        loop=0,
    )
    print(f"\n[{period_name}] Saved: {gif_file.name}  ({n} frames)")
    ds.close()

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
# # Jet Stream — Wind at 250 hPa
#
# The **jet stream** is a narrow band of fast-moving air in the upper
# troposphere, typically at around 250 hPa (~10 km altitude).  It plays a
# central role in steering weather systems across the mid-latitudes and is
# closely connected to energy systems:
#
# - **Wind energy**: Surface wind patterns are strongly influenced by the
#   position and intensity of the jet stream.  A southward-displaced (or
#   "blocked") jet can bring persistent calm, anticyclonic conditions that
#   depress wind output.
# - **Temperature**: Jet stream excursions poleward or equatorward modulate
#   cold-air outbreaks, affecting heating demand.
# - **Solar**: Ridging associated with a northward jet displacement often
#   brings clear skies and elevated solar output.
#
# This notebook loads ERA5 reanalysis wind data at 250 hPa downloaded from
# Google Cloud (ARCO-ERA5) and visualises the global jet stream on a Robinson
# projection.

# %%
import io

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from PIL import Image

from woe.paths import ProjPaths

paths = ProjPaths()

# %% [markdown]
# ## Load ERA5 pressure-level data
#
# The ERA5 pressure-level file was downloaded by `11_dev_download_era5.py` and
# stored as a NetCDF file in `data/downloads/era5/`.  It contains geopotential,
# temperature, and u/v wind components on selected pressure levels for a single
# time snapshot (13:00 UTC).

# %%
DATE = "2026-01-20"
era5_path = paths.downloads_path / "era5"
pressure_file = era5_path / f"era5_pressure_levels_{DATE}_13utc.nc"

ds = xr.open_dataset(pressure_file)
print(f"Dataset dimensions: {dict(ds.dims)}")
print(f"Variables:          {list(ds.data_vars)}")
print(f"Pressure levels:    {ds.level.values.tolist()}")
print(f"Time:               {ds.time.values}")

# %% [markdown]
# ## Plot wind speed and direction at every pressure level
#
# The file contains wind data on 10 levels spanning the upper troposphere and
# lower stratosphere (400–100 hPa).  We iterate over all levels and produce one
# Robinson-projection map per level with a **fixed colour scale** (0–80 m/s) so
# that images are directly comparable.
#
# | Level (hPa) | Approx. altitude | Layer |
# |-------------|-----------------|-------|
# | 400 | ~7 km | mid-troposphere |
# | 350 | ~8 km | mid-troposphere |
# | 300 | ~9 km | upper troposphere |
# | 250 | ~10 km | jet-stream core |
# | 225 | ~11 km | jet-stream core |
# | 200 | ~12 km | jet / tropopause |
# | 175 | ~13 km | tropopause |
# | 150 | ~14 km | lower stratosphere |
# | 125 | ~15 km | lower stratosphere |
# | 100 | ~16 km | lower stratosphere |

# %%
lats = ds.latitude.values   # descending: 90 → -90
lons = ds.longitude.values  # ascending:  0 → 359.75

levels_cf = np.arange(0, 85, 5)  # fixed 0–80 m/s scale for comparability
stride = 20                       # 0.25° × 20 = 5° subsampling for arrows

for level in ds.level.values.tolist():
    ds_lev = ds.sel(level=level)

    u = ds_lev["u_component_of_wind"]
    v = ds_lev["v_component_of_wind"]
    wspd = np.sqrt(u.values ** 2 + v.values ** 2)  # m/s

    print(f"{level:3d} hPa  wind speed range: {wspd.min():.1f} – {wspd.max():.1f} m/s")

    # Add cyclic point to close the gap at the prime meridian
    wspd_c, lons_c = add_cyclic_point(wspd, coord=lons)
    u_c, _ = add_cyclic_point(u.values, coord=lons)
    v_c, _ = add_cyclic_point(v.values, coord=lons)

    # Subsample and normalise direction arrows
    lons_sub = lons_c[::stride]
    lats_sub = lats[::stride]
    u_sub = u_c[::stride, ::stride]
    v_sub = v_c[::stride, ::stride]
    wspd_sub = np.sqrt(u_sub ** 2 + v_sub ** 2)
    wspd_sub = np.where(wspd_sub == 0, 1, wspd_sub)
    u_norm = u_sub / wspd_sub
    v_norm = v_sub / wspd_sub

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=0))
    ax.set_global()

    cf = ax.contourf(
        lons_c, lats, wspd_c,
        levels=levels_cf,
        cmap="YlOrRd",
        transform=ccrs.PlateCarree(),
        extend="max",
    )

    ax.quiver(
        lons_sub, lats_sub, u_norm, v_norm,
        transform=ccrs.PlateCarree(),
        color="steelblue",
        alpha=0.7,
        scale=80,
        width=0.002,
        headwidth=3,
        headlength=3,
        headaxislength=3.75,
        regrid_shape=None,
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color="black", alpha=0.7)
    ax.add_feature(cfeature.OCEAN, facecolor="aliceblue", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.4,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator(range(-90, 91, 30))

    cbar = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.03,
                        fraction=0.04, aspect=40)
    cbar.set_label(f"Wind speed at {level} hPa (m/s)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        f"Upper-level wind — {level} hPa speed and direction\n"
        f"{DATE} 13:00 UTC  |  ERA5 reanalysis",
        fontsize=13, pad=10,
    )

    fig.tight_layout()
    out_path = paths.images_path / f"12_jet_stream_{level}hpa.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path.name}")

# %% [markdown]
# ```{figure} ../../output/images/12_jet_stream_250hpa.png
# :name: fig-12-jet-stream-250hpa
# Global jet stream on 2026-01-20 at 13:00 UTC.  Filled colours show wind speed
# at 250 hPa (m/s); blue arrows indicate wind direction.  The subtropical and
# polar jet streams appear as elongated bands of fast-moving air (orange/red)
# encircling the mid-latitudes.
# ```

# %% [markdown]
# ## Key features to look for
#
# - **Subtropical jet** (~25–35°N/S): driven by the Hadley cell overturning,
#   typically the faster of the two jets in winter.
# - **Polar front jet** (~50–65°N): marks the boundary between cold polar air
#   and warmer mid-latitude air.  Its north–south meanders are the Rossby waves
#   that steer surface weather systems.
# - **Jet breaks and split flow**: where the jet bifurcates or weakens, blocking
#   patterns can develop, associated with persistent weather extremes —
#   droughts, cold snaps, or heat waves.
# - **Vertical structure**: comparing levels from 400 hPa up to 100 hPa reveals
#   how the jet intensifies with altitude through the troposphere and then weakens
#   in the lower stratosphere above the tropopause (~150–200 hPa in winter).
#
# For energy system analysis, the January snapshot captures the Northern
# Hemisphere winter jet stream at its seasonal peak strength.

# %% [markdown]
# ## Column-maximum wind speed
#
# Taking the **maximum wind speed across all pressure levels** at each grid
# point gives a single map that captures the jet cores regardless of the exact
# altitude at which they peak.  Grid points where the jet tilts with height or
# splits into multiple cores are still highlighted.

# %%
# Stack all levels: shape (n_levels, lat, lon)
u_all = ds["u_component_of_wind"].values   # (level, lat, lon)
v_all = ds["v_component_of_wind"].values
wspd_all = np.sqrt(u_all ** 2 + v_all ** 2)  # (level, lat, lon)

# Maximum over the level axis → (lat, lon)
wspd_max = wspd_all.max(axis=0)

print(f"Column-max wind speed range: {wspd_max.min():.1f} – {wspd_max.max():.1f} m/s")

# %%
wspd_max_c, lons_c = add_cyclic_point(wspd_max, coord=lons)

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=0))
ax.set_global()

levels_cf = np.arange(0, 85, 5)
cf = ax.contourf(
    lons_c, lats, wspd_max_c,
    levels=levels_cf,
    cmap="YlOrRd",
    transform=ccrs.PlateCarree(),
    extend="max",
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="black")
ax.add_feature(cfeature.BORDERS, linewidth=0.3, color="black", alpha=0.7)
ax.add_feature(cfeature.OCEAN, facecolor="aliceblue", zorder=0)
ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)

gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=False,
    linewidth=0.4,
    color="gray",
    alpha=0.5,
    linestyle="--",
)
gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
gl.ylocator = mticker.FixedLocator(range(-90, 91, 30))

cbar = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.03,
                    fraction=0.04, aspect=40)
cbar.set_label("Column-maximum wind speed (m/s)", fontsize=11)
cbar.ax.tick_params(labelsize=9)

ax.set_title(
    f"Column-maximum wind speed (400–100 hPa)\n{DATE} 13:00 UTC  |  ERA5 reanalysis",
    fontsize=13, pad=10,
)

fig.tight_layout()
fig.savefig(paths.images_path / "12_jet_stream_colmax_wspd.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved → 12_jet_stream_colmax_wspd.png")

# %% [markdown]
# ## Jet-stream mask
#
# A simple but effective way to delineate where the jet stream is present: mark
# every grid point where the column-maximum wind speed exceeds a threshold.
# A common operational choice is **30 m/s** (~60 kt), which isolates the
# fast-moving cores while excluding the broad, slower background flow.
#
# The mask is overlaid in **purple** on top of a light background so the
# jet-stream footprint stands out clearly.

# %%
JET_THRESHOLD = 30  # m/s  — standard meteorological jet-stream criterion

jet_mask = wspd_max >= JET_THRESHOLD   # boolean (lat, lon)
jet_mask_float = jet_mask.astype(float)
jet_mask_float[~jet_mask] = np.nan     # NaN outside jet → transparent in imshow

jet_mask_c, _ = add_cyclic_point(jet_mask_float, coord=lons)

print(f"Jet-stream threshold: {JET_THRESHOLD} m/s")
print(f"Fraction of globe covered: {jet_mask.mean()*100:.1f}%")

# %%
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=0))
ax.set_global()

# Light grey background fill for wind speed context
cf_bg = ax.contourf(
    lons_c, lats, wspd_max_c,
    levels=np.arange(0, 85, 5),
    cmap="Greys",
    transform=ccrs.PlateCarree(),
    extend="max",
    alpha=0.4,
)

# Jet-stream mask in purple
ax.contourf(
    lons_c, lats, jet_mask_c,
    levels=[0.5, 1.5],
    colors=["mediumpurple"],
    transform=ccrs.PlateCarree(),
    alpha=0.85,
)
# Crisp outline of the jet boundary
ax.contour(
    lons_c, lats, jet_mask_c,
    levels=[0.5],
    colors=["purple"],
    linewidths=0.8,
    transform=ccrs.PlateCarree(),
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="black")
ax.add_feature(cfeature.BORDERS, linewidth=0.3, color="black", alpha=0.7)
ax.add_feature(cfeature.OCEAN, facecolor="aliceblue", zorder=0)
ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)

gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=False,
    linewidth=0.4,
    color="gray",
    alpha=0.5,
    linestyle="--",
)
gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
gl.ylocator = mticker.FixedLocator(range(-90, 91, 30))

# Legend proxy
legend_elements = [mpatches.Patch(facecolor="mediumpurple", edgecolor="purple",
                         alpha=0.85, label=f"Jet stream (max wind ≥ {JET_THRESHOLD} m/s)")]
ax.legend(handles=legend_elements, loc="lower left", fontsize=10,
          framealpha=0.8)

ax.set_title(
    f"Jet-stream footprint — column-max wind ≥ {JET_THRESHOLD} m/s\n"
    f"{DATE} 13:00 UTC  |  ERA5 reanalysis",
    fontsize=13, pad=10,
)

fig.tight_layout()
fig.savefig(paths.images_path / "12_jet_stream_mask.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved → 12_jet_stream_mask.png")

# %% [markdown]
# ## Surface wind (10 m)
#
# ERA5 also provides 10 m wind components (`10m_u_component_of_wind`,
# `10m_v_component_of_wind`) in the surface file.  Surface winds are directly
# relevant to wind-energy generation and heating demand, and can be compared
# visually with the upper-level jet to see how much momentum couples downward.

# %%
surface_file = era5_path / f"era5_surface_{DATE}_13utc.nc"
ds_sfc = xr.open_dataset(surface_file)

u10 = ds_sfc["10m_u_component_of_wind"].values.squeeze()   # (lat, lon)
v10 = ds_sfc["10m_v_component_of_wind"].values.squeeze()
wspd10 = np.sqrt(u10 ** 2 + v10 ** 2)

lats_sfc = ds_sfc.latitude.values
lons_sfc = ds_sfc.longitude.values

print(f"10 m wind speed range: {wspd10.min():.1f} – {wspd10.max():.1f} m/s")

# %%
wspd10_c, lons10_c = add_cyclic_point(wspd10, coord=lons_sfc)
u10_c, _ = add_cyclic_point(u10, coord=lons_sfc)
v10_c, _ = add_cyclic_point(v10, coord=lons_sfc)

stride = 20   # 0.25° × 20 = 5° subsampling for arrows
lons10_sub = lons10_c[::stride]
lats10_sub = lats_sfc[::stride]
u10_sub = u10_c[::stride, ::stride]
v10_sub = v10_c[::stride, ::stride]

wspd10_sub = np.sqrt(u10_sub ** 2 + v10_sub ** 2)
wspd10_sub = np.where(wspd10_sub == 0, 1, wspd10_sub)
u10_norm = u10_sub / wspd10_sub
v10_norm = v10_sub / wspd10_sub

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=0))
ax.set_global()

# Surface winds are much weaker than jet-level winds; use a tighter scale
levels_sfc = np.arange(0, 25, 2)   # 0–24 m/s in 2 m/s steps
cf = ax.contourf(
    lons10_c, lats_sfc, wspd10_c,
    levels=levels_sfc,
    cmap="YlOrRd",
    transform=ccrs.PlateCarree(),
    extend="max",
)

ax.quiver(
    lons10_sub, lats10_sub, u10_norm, v10_norm,
    transform=ccrs.PlateCarree(),
    color="steelblue",
    alpha=0.7,
    scale=80,
    width=0.002,
    headwidth=3,
    headlength=3,
    headaxislength=3.75,
    regrid_shape=None,
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="black")
ax.add_feature(cfeature.BORDERS, linewidth=0.3, color="black", alpha=0.7)
ax.add_feature(cfeature.OCEAN, facecolor="aliceblue", zorder=0)
ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)

gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=False,
    linewidth=0.4,
    color="gray",
    alpha=0.5,
    linestyle="--",
)
gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
gl.ylocator = mticker.FixedLocator(range(-90, 91, 30))

cbar = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.03,
                    fraction=0.04, aspect=40)
cbar.set_label("10 m wind speed (m/s)", fontsize=11)
cbar.ax.tick_params(labelsize=9)

ax.set_title(
    f"Surface wind — 10 m speed and direction\n{DATE} 13:00 UTC  |  ERA5 reanalysis",
    fontsize=13, pad=10,
)

fig.tight_layout()
fig.savefig(paths.images_path / "12_surface_wind.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved → 12_surface_wind.png")

# %% [markdown]
# ## Vertical cross-section: wind speed vs latitude and altitude
#
# A cut at a fixed longitude through the jet stream reveals its full
# **vertical structure**: how wind speed varies simultaneously with latitude
# (x-axis) and altitude (y-axis).  The y-axis uses **geopotential height**
# (km) so altitude increases upward naturally; secondary labels on the right
# show the corresponding pressure level.
#
# `LON_CUT` selects the meridian; 300 °E (60 °W) typically bisects the
# North Atlantic jet in winter.

# %%
G = 9.80665   # standard gravity m/s²

# Six meridians spaced 60° apart, covering the globe
LON_CUTS = [0, 60, 120, 180, 240, 300]


def _fmt_cross_section_axes(ax):
    """Apply common axis style to a cross-section panel."""
    ax.set_xlabel("Latitude (°)", fontsize=11)
    ax.set_ylabel("Geopotential height (km)", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d°"))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(15))
    ax.grid(axis="x", linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
    ax.grid(axis="y", linewidth=0.4, color="gray", alpha=0.3, linestyle=":")


def plot_cross_sections(ds, lon_cut, date, paths):
    ds_cut = ds.sel(longitude=lon_cut, method="nearest")
    lon_actual = float(ds_cut.longitude)
    lon_plot   = lon_actual if lon_actual <= 180 else lon_actual - 360
    lon_tag    = f"{int(round(lon_actual)):03d}E"

    # Extract fields and sort levels bottom → top (ascending geopotential height)
    u_cut    = ds_cut["u_component_of_wind"].values   # (level, lat)
    v_cut    = ds_cut["v_component_of_wind"].values
    z_cut    = ds_cut["geopotential"].values
    t_cut    = ds_cut["temperature"].values
    lats_cut = ds_cut.latitude.values                 # descending 90 → -90

    z_km        = z_cut / G / 1000
    level_order = np.argsort(z_km.mean(axis=1))
    levels_s    = ds.level.values[level_order]
    z_km_s      = z_km[level_order][:, ::-1]          # south→north
    wspd_s      = np.sqrt(u_cut**2 + v_cut**2)[level_order][:, ::-1]
    t_s         = t_cut[level_order][:, ::-1] - 273.15
    press_2d    = np.broadcast_to(levels_s[:, np.newaxis], z_km_s.shape).copy()

    lat_asc  = lats_cut[::-1]
    lat_2d   = np.broadcast_to(lat_asc, z_km_s.shape).copy()

    print(f"\n=== {lon_actual:.1f}°E ({lon_plot:+.0f}°) ===")

    # 1. Location map
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="aliceblue")
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, color="black", alpha=0.7)
    ax.plot([lon_plot, lon_plot], [-90, 90], color="crimson", linewidth=2,
            transform=ccrs.PlateCarree(), zorder=5)
    ax.text(lon_plot + 2, 60, f"{lon_actual:.0f}°E\n({lon_plot:+.0f}°)",
            transform=ccrs.PlateCarree(), fontsize=10, color="crimson",
            va="center", fontweight="bold")
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4,
                      color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator(range(-90, 91, 30))
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}
    ax.set_title(f"Cross-section location — {lon_actual:.1f}°E ({lon_plot:+.0f}°)\n"
                 f"{date} 13:00 UTC  |  ERA5 reanalysis", fontsize=13, pad=10)
    fig.tight_layout()
    out = paths.images_path / f"12_{lon_tag}_jet_stream_cross_section_location.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.name}")

    # 2. Wind speed
    fig, ax = plt.subplots(figsize=(14, 7))
    cf = ax.contourf(lat_2d, z_km_s, wspd_s, levels=np.arange(0, 85, 5),
                     cmap="YlOrRd", extend="max")
    cs = ax.contour(lat_2d, z_km_s, wspd_s, levels=[30, 50, 70],
                    colors="black", linewidths=0.8, alpha=0.6)
    ax.clabel(cs, fmt="%d m/s", fontsize=8)
    fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.12,
                 fraction=0.03, aspect=30).set_label("Wind speed (m/s)", fontsize=11)
    _fmt_cross_section_axes(ax)
    ax.set_title(f"Vertical cross-section of wind speed at {lon_actual:.1f}°E "
                 f"({lon_plot:+.0f}°)\n{date} 13:00 UTC  |  ERA5 reanalysis",
                 fontsize=13, pad=10)
    fig.tight_layout()
    out = paths.images_path / f"12_{lon_tag}_jet_stream_cross_section.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.name}")

    # 3. Temperature
    fig, ax = plt.subplots(figsize=(14, 7))
    cf = ax.contourf(lat_2d, z_km_s, t_s, levels=np.arange(-80, 25, 5),
                     cmap="RdBu_r", extend="both")
    cs = ax.contour(lat_2d, z_km_s, t_s, levels=np.arange(-80, 25, 10),
                    colors="black", linewidths=0.6, alpha=0.5)
    ax.clabel(cs, fmt="%d°C", fontsize=8)
    fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.12,
                 fraction=0.03, aspect=30).set_label("Temperature (°C)", fontsize=11)
    _fmt_cross_section_axes(ax)
    ax.set_title(f"Vertical cross-section of temperature at {lon_actual:.1f}°E "
                 f"({lon_plot:+.0f}°)\n{date} 13:00 UTC  |  ERA5 reanalysis",
                 fontsize=13, pad=10)
    fig.tight_layout()
    out = paths.images_path / f"12_{lon_tag}_jet_stream_cross_section_temperature.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.name}")

    # 4. Pressure
    fig, ax = plt.subplots(figsize=(14, 7))
    cf = ax.contourf(lat_2d, z_km_s, press_2d,
                     levels=np.linspace(levels_s.min(), levels_s.max(), 40),
                     cmap="viridis_r", extend="neither")
    cs = ax.contour(lat_2d, z_km_s, press_2d, levels=np.sort(levels_s),
                    colors="black", linewidths=0.7, alpha=0.6)
    ax.clabel(cs, fmt="%d hPa", fontsize=8)
    fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.12,
                 fraction=0.03, aspect=30).set_label("Pressure (hPa)", fontsize=11)
    _fmt_cross_section_axes(ax)
    ax.set_title(f"Vertical cross-section of pressure at {lon_actual:.1f}°E "
                 f"({lon_plot:+.0f}°)\n{date} 13:00 UTC  |  ERA5 reanalysis",
                 fontsize=13, pad=10)
    fig.tight_layout()
    out = paths.images_path / f"12_{lon_tag}_jet_stream_cross_section_pressure.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.name}")


# %%
for lon_cut in LON_CUTS:
    plot_cross_sections(ds, lon_cut, DATE, paths)

# %% [markdown]
# ## Combined cross-section: wind speed and temperature at 60°E
#
# Side-by-side view of the two thermodynamic cross-sections at the same
# meridian, making it easy to compare the jet-core position (left) with
# the meridional temperature gradient that drives it (right).

# %%
LON_COMBINED = 60

ds_c    = ds.sel(longitude=LON_COMBINED, method="nearest")
lon_c   = float(ds_c.longitude)
lon_c_p = lon_c if lon_c <= 180 else lon_c - 360

u_c  = ds_c["u_component_of_wind"].values
v_c  = ds_c["v_component_of_wind"].values
z_c  = ds_c["geopotential"].values
t_c  = ds_c["temperature"].values
lats_c = ds_c.latitude.values

z_km_c      = z_c / G / 1000
level_ord_c = np.argsort(z_km_c.mean(axis=1))
z_km_c      = z_km_c[level_ord_c][:, ::-1]
wspd_c      = np.sqrt(u_c**2 + v_c**2)[level_ord_c][:, ::-1]
t_c         = t_c[level_ord_c][:, ::-1] - 273.15

lat_asc_c = lats_c[::-1]
lat_2d_c  = np.broadcast_to(lat_asc_c, z_km_c.shape).copy()

# %%
fig, (ax_w, ax_t) = plt.subplots(1, 2, figsize=(22, 7), sharey=True)

# --- Left: wind speed ---
cf_w = ax_w.contourf(lat_2d_c, z_km_c, wspd_c, levels=np.arange(0, 85, 5),
                     cmap="YlOrRd", extend="max")
cs_w = ax_w.contour(lat_2d_c, z_km_c, wspd_c, levels=[30, 50, 70],
                    colors="black", linewidths=0.8, alpha=0.6)
ax_w.clabel(cs_w, fmt="%d m/s", fontsize=8)
fig.colorbar(cf_w, ax=ax_w, orientation="vertical", pad=0.02,
             fraction=0.03, aspect=30).set_label("Wind speed (m/s)", fontsize=11)

ax_w.set_xlabel("Latitude (°)", fontsize=11)
ax_w.set_ylabel("Geopotential height (km)", fontsize=11)
ax_w.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d°"))
ax_w.xaxis.set_major_locator(mticker.MultipleLocator(15))
ax_w.grid(axis="x", linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
ax_w.grid(axis="y", linewidth=0.4, color="gray", alpha=0.3, linestyle=":")

ax_w.set_title(f"Wind speed at {lon_c:.1f}°E ({lon_c_p:+.0f}°)", fontsize=12)

# --- Right: temperature ---
cf_t = ax_t.contourf(lat_2d_c, z_km_c, t_c, levels=np.arange(-80, 25, 5),
                     cmap="RdBu_r", extend="both")
cs_t = ax_t.contour(lat_2d_c, z_km_c, t_c, levels=np.arange(-80, 25, 10),
                    colors="black", linewidths=0.6, alpha=0.5)
ax_t.clabel(cs_t, fmt="%d°C", fontsize=8)
fig.colorbar(cf_t, ax=ax_t, orientation="vertical", pad=0.02,
             fraction=0.03, aspect=30).set_label("Temperature (°C)", fontsize=11)

ax_t.set_xlabel("Latitude (°)", fontsize=11)
ax_t.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d°"))
ax_t.xaxis.set_major_locator(mticker.MultipleLocator(15))
ax_t.grid(axis="x", linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
ax_t.grid(axis="y", linewidth=0.4, color="gray", alpha=0.3, linestyle=":")

ax_t.set_title(f"Temperature at {lon_c:.1f}°E ({lon_c_p:+.0f}°)", fontsize=12)

fig.suptitle(
    f"Vertical cross-sections at {lon_c:.1f}°E ({lon_c_p:+.0f}°)\n"
    f"{DATE} 13:00 UTC  |  ERA5 reanalysis",
    fontsize=13, y=1.02,
)

fig.tight_layout()
out = paths.images_path / "12_060E_jet_stream_cross_section_combined.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved → {out.name}")

# %% [markdown]
# ## Animation: cross-sections at every 5° longitude
#
# A GIF cycling through all 72 meridians (0°–355° in 5° steps), each frame
# showing the wind-speed (left) and temperature (right) cross-sections.

# %%
gif_frames = []

for lon_cut in range(0, 360, 5):
    ds_g   = ds.sel(longitude=lon_cut, method="nearest")
    lon_g  = float(ds_g.longitude)
    lon_gp = lon_g if lon_g <= 180 else lon_g - 360

    u_g    = ds_g["u_component_of_wind"].values
    v_g    = ds_g["v_component_of_wind"].values
    z_g    = ds_g["geopotential"].values
    t_g    = ds_g["temperature"].values
    lats_g = ds_g.latitude.values

    z_km_g   = z_g / G / 1000
    ord_g    = np.argsort(z_km_g.mean(axis=1))
    z_km_g   = z_km_g[ord_g][:, ::-1]
    wspd_g   = np.sqrt(u_g**2 + v_g**2)[ord_g][:, ::-1]
    t_g      = t_g[ord_g][:, ::-1] - 273.15
    lat_2d_g = np.broadcast_to(lats_g[::-1], z_km_g.shape).copy()

    fig, (ax_w, ax_t) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    cf_w = ax_w.contourf(lat_2d_g, z_km_g, wspd_g, levels=np.arange(0, 85, 5),
                         cmap="YlOrRd", extend="max")
    ax_w.contour(lat_2d_g, z_km_g, wspd_g, levels=[30, 50, 70],
                 colors="black", linewidths=0.8, alpha=0.6)
    fig.colorbar(cf_w, ax=ax_w, fraction=0.04, pad=0.02,
                 aspect=25).set_label("Wind speed (m/s)", fontsize=9)
    _fmt_cross_section_axes(ax_w)
    ax_w.set_title("Wind speed", fontsize=10)

    cf_t = ax_t.contourf(lat_2d_g, z_km_g, t_g, levels=np.arange(-80, 25, 5),
                         cmap="RdBu_r", extend="both")
    ax_t.contour(lat_2d_g, z_km_g, t_g, levels=np.arange(-80, 25, 10),
                 colors="black", linewidths=0.6, alpha=0.5)
    fig.colorbar(cf_t, ax=ax_t, fraction=0.04, pad=0.02,
                 aspect=25).set_label("Temperature (°C)", fontsize=9)
    ax_t.set_xlabel("Latitude (°)", fontsize=10)
    ax_t.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d°"))
    ax_t.xaxis.set_major_locator(mticker.MultipleLocator(15))
    ax_t.grid(axis="x", linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
    ax_t.grid(axis="y", linewidth=0.4, color="gray", alpha=0.3, linestyle=":")
    ax_t.set_title("Temperature", fontsize=10)

    fig.suptitle(
        f"Vertical cross-section at {lon_g:.1f}°E ({lon_gp:+.0f}°)  —  "
        f"{DATE} 13:00 UTC  |  ERA5 reanalysis",
        fontsize=11,
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    gif_frames.append(Image.open(buf).copy())
    print(f"  frame {lon_g:5.1f}°E done", end="\r")

out_gif = paths.images_path / "12_jet_stream_cross_section_animation.gif"
gif_frames[0].save(
    out_gif,
    save_all=True,
    append_images=gif_frames[1:],
    duration=200,   # ms per frame
    loop=0,         # loop forever
)
print(f"\nsaved → {out_gif.name}  ({len(gif_frames)} frames)")

# %%

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
# # ERA5 Weather Data Visualization (Dev)
#
# Download ERA5 2m temperature data for a single day from Google Cloud
# (ARCO-ERA5 analysis-ready dataset) and visualize it on a map with country borders.

# %%
from datetime import datetime

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature.nightshade import Nightshade
from matplotlib.colors import LinearSegmentedColormap

from woe.paths import ProjPaths

paths = ProjPaths()

# %% [markdown]
# ## Download ERA5 data from Google Cloud
#
# The ARCO-ERA5 analysis-ready dataset is publicly available on Google Cloud
# Storage. We use `xr.open_zarr` to lazily open the dataset and only download
# the time slice we need.

# %%
ds = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    chunks={"time": 48},
    consolidated=True,
    storage_options={"token": "anon"},
)
print(f"Dataset dimensions: {dict(ds.dims)}")
print(f"Data variables:     {len(ds.data_vars)}")
print(f"Time range:         {ds.time.values[0]} to {ds.time.values[-1]}")

# %%
# Select 2m temperature at noon on 2025-06-03 and convert K -> °C
t2m = ds["2m_temperature"].sel(time="2025-06-03T12:00:00").compute()
t2m_celsius = t2m - 273.15

print(f"Downloaded dimensions: {dict(t2m.sizes)}")
print(f"Downloaded size:      {t2m.nbytes / 1e6:.1f} MB")
print(f"Longitude range:      {float(t2m_celsius.longitude.min())} to {float(t2m_celsius.longitude.max())}")
print(f"Temperature range:    {float(t2m_celsius.min()):.1f}°C to {float(t2m_celsius.max()):.1f}°C")

# %% [markdown]
# ## Global map — comparing projections
#
# | Projection | Preserves | Best use case |
# |---|---|---|
# | Plate Carree | Grid indices | Internal data checks / quick plots |
# | Equal Earth | Area | Global distribution of energy or rain |
# | Mercator | Direction/Angles | Navigation (avoid for global climate data) |
# | Robinson | Compromise | Balanced, pretty global overview |
# | Orthographic | Perspective | Viewing the Earth as a sphere (3D feel) |

# %%
projections = [
    ("Plate Carrée", ccrs.PlateCarree()),
    ("Equal Earth", ccrs.EqualEarth()),
    ("Robinson", ccrs.Robinson()),
    ("Orthographic", ccrs.Orthographic(central_longitude=10, central_latitude=50)),
    ("Mercator", ccrs.Mercator()),
]

fig, axes = plt.subplots(
    3, 2, figsize=(18, 20),
    subplot_kw={"projection": ccrs.Robinson()},  # placeholder, overridden below
)
axes = axes.flat

for i, (name, proj) in enumerate(projections):
    ax = fig.add_subplot(3, 2, i + 1, projection=proj)
    axes[i].set_visible(False)  # hide the placeholder

    im = ax.pcolormesh(
        t2m_celsius.longitude,
        t2m_celsius.latitude,
        t2m_celsius.values,
        transform=ccrs.PlateCarree(),
        cmap="RdYlBu_r",
        vmin=-40,
        vmax=45,
        shading="auto",
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")
    ax.set_global()
    ax.set_title(name, fontsize=13)

# Hide the unused 6th subplot
axes[5].set_visible(False)

fig.suptitle("ERA5 2m Temperature — 2025-06-03 12:00 UTC", fontsize=15, y=0.98)
fig.subplots_adjust(bottom=0.06)
cbar = fig.colorbar(
    im, ax=fig.get_axes(), orientation="horizontal",
    pad=0.04, shrink=0.5, aspect=40,
)
cbar.set_label("Temperature (°C)")

fig.savefig(paths.images_path / "08_era5_t2m_projections.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/08_era5_t2m_projections.png
# :name: fig-08-era5-t2m-projections
# ERA5 2m temperature on 2025-06-03 at 12:00 UTC shown in five common map projections.
# ```

# %% [markdown]
# ## Europe close-up

# %%
fig, ax = plt.subplots(
    figsize=(12, 10),
    subplot_kw={"projection": ccrs.LambertConformal(central_longitude=10, central_latitude=50)},
)

# Use the full global data — set_extent clips the view.
# This avoids issues with the 0..360 longitude convention when slicing
# across the prime meridian.
im = ax.pcolormesh(
    t2m_celsius.longitude,
    t2m_celsius.latitude,
    t2m_celsius.values,
    transform=ccrs.PlateCarree(),
    cmap="RdYlBu_r",
    vmin=-5,
    vmax=35,
    shading="auto",
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="--")
ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.5)
ax.set_extent([-15, 35, 34, 72], crs=ccrs.PlateCarree())
ax.set_title("ERA5 2m Temperature — Europe — 2025-06-03 12:00 UTC", fontsize=14)

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
cbar.set_label("Temperature (°C)")

fig.tight_layout()
fig.savefig(paths.images_path / "08_era5_t2m_europe.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/08_era5_t2m_europe.png
# :name: fig-08-era5-t2m-europe
# ERA5 2m temperature over Europe on 2025-06-03 at 12:00 UTC.
# ```

# %% [markdown]
# ## Additional variables — Robinson projection

# %%
# Download precipitation, cloud cover, and wind components for the same time step
tp = ds["total_precipitation"].sel(time="2025-06-03T12:00:00").compute()
tcc = ds["total_cloud_cover"].sel(time="2025-06-03T12:00:00").compute()
u10 = ds["10m_u_component_of_wind"].sel(time="2025-06-03T12:00:00").compute()
v10 = ds["10m_v_component_of_wind"].sel(time="2025-06-03T12:00:00").compute()
wind_speed = np.sqrt(u10**2 + v10**2)

print(f"Precipitation size: {tp.nbytes / 1e6:.1f} MB")
print(f"Cloud cover size:   {tcc.nbytes / 1e6:.1f} MB")
print(f"Wind (u+v) size:    {(u10.nbytes + v10.nbytes) / 1e6:.1f} MB")

# %% [markdown]
# ### Precipitation

# %%
fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={"projection": ccrs.Robinson()})

im = ax.pcolormesh(
    tp.longitude, tp.latitude, tp.values * 1000,
    transform=ccrs.PlateCarree(),
    cmap="YlGnBu", vmin=0, vmax=5, shading="auto",
)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")
ax.set_global()
ax.set_title("ERA5 Total Precipitation — 2025-06-03 12:00 UTC", fontsize=14)

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
cbar.set_label("Precipitation (mm)")

fig.tight_layout()
fig.savefig(paths.images_path / "08_era5_precipitation.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/08_era5_precipitation.png
# :name: fig-08-era5-precipitation
# ERA5 total precipitation on 2025-06-03 at 12:00 UTC.
# ```

# %% [markdown]
# ### Cloud cover

# %%
fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={"projection": ccrs.Robinson()})

im = ax.pcolormesh(
    tcc.longitude, tcc.latitude, tcc.values * 100,
    transform=ccrs.PlateCarree(),
    cmap="gray_r", vmin=0, vmax=100, shading="auto",
)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="cyan")
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--", color="cyan")
ax.set_global()
ax.set_title("ERA5 Total Cloud Cover — 2025-06-03 12:00 UTC", fontsize=14)

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
cbar.set_label("Cloud cover (%)")

fig.tight_layout()
fig.savefig(paths.images_path / "08_era5_cloud_cover.png", dpi=150, bbox_inches="tight")
plt.show()



# %% [markdown]
# ```{figure} ../../output/images/08_era5_cloud_cover.png
# :name: fig-08-era5-cloud-cover
# ERA5 total cloud cover on 2025-06-03 at 12:00 UTC.
# ```

# %% [markdown]
# ### Cloud cover over satellite imagery
#
# Overlay ERA5 cloud cover on the Natural Earth Blue Marble background
# shipped with cartopy (`stock_img`). Cloud-free areas reveal the land/ocean
# surface beneath.

# %%
# White-only colormap: transparent where clear, opaque white where cloudy
cloud_cmap = LinearSegmentedColormap.from_list("clouds", [
    (1, 1, 1, 0),   # fully transparent at 0 % cloud
    (1, 1, 1, 1),   # fully opaque white at 100 % cloud
])

fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={"projection": ccrs.Robinson()})
ax.stock_img()

ax.pcolormesh(
    tcc.longitude, tcc.latitude, tcc.values,
    transform=ccrs.PlateCarree(),
    cmap=cloud_cmap, vmin=0, vmax=1, shading="auto",
)
ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="0.3")
ax.add_feature(cfeature.BORDERS, linewidth=0.2, linestyle="--", color="0.4")
ax.set_global()
ax.set_title("ERA5 Cloud Cover over Blue Marble — 2025-06-03 12:00 UTC", fontsize=14)

fig.tight_layout()
fig.savefig(
    paths.images_path / "08_era5_cloud_cover_satellite.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/08_era5_cloud_cover_satellite.png
# :name: fig-08-era5-cloud-cover-satellite
# ERA5 cloud cover overlaid on Blue Marble satellite imagery, 2025-06-03 12:00 UTC.
# ```

# %% [markdown]
# ### Wind speed

# %%
fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={"projection": ccrs.Robinson()})

im = ax.pcolormesh(
    wind_speed.longitude, wind_speed.latitude, wind_speed.values,
    transform=ccrs.PlateCarree(),
    cmap="plasma", vmin=0, vmax=20, shading="auto",
)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")
ax.set_global()
ax.set_title("ERA5 10m Wind Speed — 2025-06-03 12:00 UTC", fontsize=14)

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
cbar.set_label("Wind speed (m/s)")

fig.tight_layout()
fig.savefig(paths.images_path / "08_era5_wind_speed.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/08_era5_wind_speed.png
# :name: fig-08-era5-wind-speed
# ERA5 10m wind speed on 2025-06-03 at 12:00 UTC.
# ```

# %% [markdown]
# ## Jet stream — 250 hPa winds
#
# The jet stream lives near the tropopause, around 250 hPa (~10 km altitude).
# We visualize wind speed at that pressure level and overlay streamlines
# to show the flow direction.

# %%
u250 = ds["u_component_of_wind"].sel(time="2025-06-03T12:00:00", level=250).compute()
v250 = ds["v_component_of_wind"].sel(time="2025-06-03T12:00:00", level=250).compute()
jet_speed = np.sqrt(u250**2 + v250**2)

print(f"Jet stream data dimensions: {dict(jet_speed.sizes)}")
print(f"Jet stream data size:       {(u250.nbytes + v250.nbytes) / 1e6:.1f} MB")
print(f"Max wind speed:             {float(jet_speed.max()):.1f} m/s ({float(jet_speed.max()) * 3.6:.0f} km/h)")

# %% [markdown]
# ### Global jet stream

# %%
fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={"projection": ccrs.Robinson()})

im = ax.pcolormesh(
    jet_speed.longitude, jet_speed.latitude, jet_speed.values,
    transform=ccrs.PlateCarree(),
    cmap="hot_r", vmin=0, vmax=80, shading="auto",
)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")
ax.set_global()
ax.set_title("ERA5 Jet Stream (250 hPa Wind Speed) — 2025-06-03 12:00 UTC", fontsize=14)

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
cbar.set_label("Wind speed (m/s)")

fig.tight_layout()
fig.savefig(paths.images_path / "08_era5_jet_stream_global.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/08_era5_jet_stream_global.png
# :name: fig-08-era5-jet-stream-global
# ERA5 jet stream (250 hPa wind speed) on 2025-06-03 at 12:00 UTC.
# ```

# %% [markdown]
# ### Northern Hemisphere — polar view with streamlines

# %%
fig, ax = plt.subplots(
    figsize=(12, 12),
    subplot_kw={"projection": ccrs.NorthPolarStereo()},
)

im = ax.pcolormesh(
    jet_speed.longitude, jet_speed.latitude, jet_speed.values,
    transform=ccrs.PlateCarree(),
    cmap="hot_r", vmin=0, vmax=80, shading="auto",
)

# Subsample for streamlines (full resolution is too dense)
step = 8
lons = u250.longitude.values[::step]
lats = u250.latitude.values[::step]
u_sub = u250.values[::step, ::step]
v_sub = v250.values[::step, ::step]

ax.streamplot(
    lons, lats, u_sub, v_sub,
    transform=ccrs.PlateCarree(),
    color="white", linewidth=0.6, density=2, arrowsize=1,
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")
ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())
ax.set_title("Northern Hemisphere Jet Stream — 250 hPa — 2025-06-03 12:00 UTC", fontsize=13)

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
cbar.set_label("Wind speed (m/s)")

fig.tight_layout()
fig.savefig(paths.images_path / "08_era5_jet_stream_nh.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/08_era5_jet_stream_nh.png
# :name: fig-08-era5-jet-stream-nh
# Northern Hemisphere jet stream with streamlines at 250 hPa on 2025-06-03.
# ```

# %% [markdown]
# ### Europe — jet stream with streamlines

# %%
fig, ax = plt.subplots(
    figsize=(14, 10),
    subplot_kw={"projection": ccrs.LambertConformal(central_longitude=10, central_latitude=50)},
)

im = ax.pcolormesh(
    jet_speed.longitude, jet_speed.latitude, jet_speed.values,
    transform=ccrs.PlateCarree(),
    cmap="hot_r", vmin=0, vmax=80, shading="auto",
)

step = 4
lons = u250.longitude.values[::step]
lats = u250.latitude.values[::step]
u_sub = u250.values[::step, ::step]
v_sub = v250.values[::step, ::step]

ax.streamplot(
    lons, lats, u_sub, v_sub,
    transform=ccrs.PlateCarree(),
    color="white", linewidth=0.7, density=3, arrowsize=1.2,
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="--")
ax.set_extent([-30, 45, 30, 72], crs=ccrs.PlateCarree())
ax.set_title("Jet Stream over Europe — 250 hPa — 2025-06-03 12:00 UTC", fontsize=14)

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
cbar.set_label("Wind speed (m/s)")

fig.tight_layout()
fig.savefig(paths.images_path / "08_era5_jet_stream_europe.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/08_era5_jet_stream_europe.png
# :name: fig-08-era5-jet-stream-europe
# Jet stream over Europe with streamlines at 250 hPa on 2025-06-03.
# ```

# %% [markdown]
# ### Jet stream as temperature boundary
#
# The jet stream marks the boundary between polar and subtropical air masses.
# Here we show 2m temperature everywhere and overlay the jet stream core
# (wind speed > 30 m/s) on top, so the temperature contrast across the jet
# becomes visible.

# %%
# Mask jet speed: keep only the strong core, set the rest to NaN
jet_core = jet_speed.where(jet_speed > 30)

fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={"projection": ccrs.Robinson()})

# Background: 2m temperature
im_temp = ax.pcolormesh(
    t2m_celsius.longitude, t2m_celsius.latitude, t2m_celsius.values,
    transform=ccrs.PlateCarree(),
    cmap="RdYlBu_r", vmin=-40, vmax=45, shading="auto",
)

# Overlay: jet stream core only (NaN regions stay transparent)
jet_cmap = LinearSegmentedColormap.from_list("jet_core", [
    (0.2, 0.2, 0.2, 0.6),  # dark gray, semi-transparent at 30 m/s
    (1.0, 1.0, 1.0, 1.0),  # bright white at 80 m/s
])
ax.pcolormesh(
    jet_core.longitude, jet_core.latitude, jet_core.values,
    transform=ccrs.PlateCarree(),
    cmap=jet_cmap, vmin=30, vmax=80, shading="auto",
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")
ax.set_global()
ax.set_title(
    "2m Temperature with Jet Stream Core (> 30 m/s) — 2025-06-03 12:00 UTC",
    fontsize=13,
)

cbar = fig.colorbar(im_temp, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
cbar.set_label("Temperature (°C)")

fig.tight_layout()
fig.savefig(
    paths.images_path / "08_era5_jet_stream_temperature.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/08_era5_jet_stream_temperature.png
# :name: fig-08-era5-jet-stream-temperature
# The jet stream core (250 hPa wind > 30 m/s, white band) divides warm subtropical
# air from colder polar air, visible in the 2m temperature field beneath.
# ```

# %% [markdown]
# ## Jet stream and pressure systems
#
# The jet stream is intimately linked to the large-scale pressure pattern.
# At 500 hPa, troughs (low geopotential height) correspond to surface lows
# and ridges (high geopotential height) to surface highs.  The jet flows
# along the steepest height gradient — between troughs and ridges.

# %%
# Download 500 hPa geopotential, MSLP, and 850 hPa temperature
geopot_500 = ds["geopotential"].sel(time="2025-06-03T12:00:00", level=500).compute()
gph_500 = geopot_500 / 9.80665  # geopotential height in metres

mslp = ds["mean_sea_level_pressure"].sel(time="2025-06-03T12:00:00").compute()
mslp_hpa = mslp / 100  # Pa -> hPa

t850 = ds["temperature"].sel(time="2025-06-03T12:00:00", level=850).compute()
t850_celsius = t850 - 273.15

print(f"500 hPa geopotential size: {geopot_500.nbytes / 1e6:.1f} MB")
print(f"MSLP size:                 {mslp.nbytes / 1e6:.1f} MB")
print(f"850 hPa temperature size:  {t850.nbytes / 1e6:.1f} MB")

# %% [markdown]
# ### 500 hPa geopotential height + jet stream overlay
#
# The classic synoptic meteorology chart.  Geopotential height is shown as
# filled colour, with the jet stream core overlaid as a grey cloud.
# Troughs (blue) correspond to surface lows, ridges (red) to surface highs.

# %%
jet_core = jet_speed.where(jet_speed > 30)
jet_cmap = LinearSegmentedColormap.from_list("jet_overlay", [
    (0.3, 0.3, 0.3, 0.4),  # semi-transparent dark grey at threshold
    (0.9, 0.9, 0.9, 0.85), # near-opaque light grey at strongest
])

fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={"projection": ccrs.Robinson()})

im = ax.pcolormesh(
    gph_500.longitude, gph_500.latitude, gph_500.values,
    transform=ccrs.PlateCarree(),
    cmap="RdYlBu_r", vmin=5000, vmax=5900, shading="auto",
)

ax.pcolormesh(
    jet_core.longitude, jet_core.latitude, jet_core.values,
    transform=ccrs.PlateCarree(),
    cmap=jet_cmap, vmin=30, vmax=80, shading="auto",
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")
ax.set_global()
ax.set_title(
    "500 hPa Geopotential Height + Jet Stream (> 30 m/s) — 2025-06-03 12:00 UTC",
    fontsize=13,
)

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
cbar.set_label("500 hPa geopotential height (m)")

fig.tight_layout()
fig.savefig(
    paths.images_path / "08_era5_jet_stream_gph500.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/08_era5_jet_stream_gph500.png
# :name: fig-08-era5-jet-stream-gph500
# 500 hPa geopotential height (colour) with the jet stream core overlaid (grey).
# The jet follows the steepest height gradient between troughs (blue) and ridges (red).
# ```

# %% [markdown]
# ### Mean sea level pressure + jet stream overlay

# %%
fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={"projection": ccrs.Robinson()})

im = ax.pcolormesh(
    mslp_hpa.longitude, mslp_hpa.latitude, mslp_hpa.values,
    transform=ccrs.PlateCarree(),
    cmap="RdYlBu_r", vmin=980, vmax=1040, shading="auto",
)

ax.pcolormesh(
    jet_core.longitude, jet_core.latitude, jet_core.values,
    transform=ccrs.PlateCarree(),
    cmap=jet_cmap, vmin=30, vmax=80, shading="auto",
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")
ax.set_global()
ax.set_title(
    "Mean Sea Level Pressure + Jet Stream (> 30 m/s) — 2025-06-03 12:00 UTC",
    fontsize=13,
)

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
cbar.set_label("MSLP (hPa)")

fig.tight_layout()
fig.savefig(
    paths.images_path / "08_era5_jet_stream_mslp.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/08_era5_jet_stream_mslp.png
# :name: fig-08-era5-jet-stream-mslp
# Mean sea level pressure (colour) with the jet stream core overlaid (grey).
# Surface lows (blue) and highs (red) relate to the jet entrance/exit regions.
# ```

# %% [markdown]
# ### Jet stream vs 850 hPa temperature
#
# The 850 hPa level (~1.5 km) is the standard level for identifying air mass
# boundaries.  Unlike surface temperature it is not affected by terrain or
# the diurnal cycle, making the frontal contrast across the jet even clearer
# than 2m temperature.

# %%
fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={"projection": ccrs.Robinson()})

# Background: 850 hPa temperature
im_temp = ax.pcolormesh(
    t850_celsius.longitude, t850_celsius.latitude, t850_celsius.values,
    transform=ccrs.PlateCarree(),
    cmap="RdYlBu_r", vmin=-30, vmax=30, shading="auto",
)

# Overlay: jet stream core (> 30 m/s) as grey cloud
ax.pcolormesh(
    jet_core.longitude, jet_core.latitude, jet_core.values,
    transform=ccrs.PlateCarree(),
    cmap=jet_cmap, vmin=30, vmax=80, shading="auto",
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")
ax.set_global()
ax.set_title(
    "850 hPa Temperature with Jet Stream Core (> 30 m/s) — 2025-06-03 12:00 UTC",
    fontsize=13,
)

cbar = fig.colorbar(im_temp, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
cbar.set_label("850 hPa Temperature (°C)")

fig.tight_layout()
fig.savefig(
    paths.images_path / "08_era5_jet_stream_t850.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/08_era5_jet_stream_t850.png
# :name: fig-08-era5-jet-stream-t850
# The jet stream core (white band) overlaid on 850 hPa temperature. The 850 hPa
# level removes terrain and diurnal effects, making the air mass boundary across
# the jet sharper than at the surface.
# ```

# %% [markdown]
# ## Composite: satellite + clouds + jet stream
#
# Blue Marble background with white clouds and the jet stream core rendered
# in neon purple.

# %%
# Neon-purple jet colormap: transparent below threshold, vivid purple above
jet_neon_cmap = LinearSegmentedColormap.from_list("jet_neon", [
    (0.6, 0.0, 1.0, 0.0),   # transparent at threshold
    (0.6, 0.0, 1.0, 0.5),   # semi-transparent purple
    (0.85, 0.2, 1.0, 0.9),  # bright neon purple
    (1.0, 0.5, 1.0, 1.0),   # hot pink-purple at strongest
])

fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={"projection": ccrs.Robinson()})

# Layer 1: satellite background
ax.stock_img()

# Layer 2: cloud cover (white)
ax.pcolormesh(
    tcc.longitude, tcc.latitude, tcc.values,
    transform=ccrs.PlateCarree(),
    cmap=cloud_cmap, vmin=0, vmax=1, shading="auto",
)

# Layer 3: jet stream core in neon purple
ax.pcolormesh(
    jet_core.longitude, jet_core.latitude, jet_core.values,
    transform=ccrs.PlateCarree(),
    cmap=jet_neon_cmap, vmin=30, vmax=80, shading="auto",
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color="0.4")
ax.set_global()
ax.set_title(
    "Blue Marble + Clouds + Jet Stream — 2025-06-03 12:00 UTC",
    fontsize=14, color="white", pad=12,
)

fig.patch.set_facecolor("black")
ax.set_facecolor("black")

fig.tight_layout()
fig.savefig(
    paths.images_path / "08_era5_composite_satellite.png",
    dpi=150, bbox_inches="tight", facecolor="black",
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/08_era5_composite_satellite.png
# :name: fig-08-era5-composite-satellite
# Composite view: Blue Marble satellite imagery with ERA5 cloud cover (white) and
# the jet stream core at 250 hPa (neon purple, > 30 m/s).
# ```

# %% [markdown]
# ## Day/night and solar declination
#
# The Earth's axial tilt (23.44°) causes the latitude of maximum direct
# sunlight — the **subsolar point** — to oscillate between the tropics over
# the year.  On 2025-06-03, the subsolar latitude is about 22°N (close to
# the summer solstice).
#
# We use cartopy's built-in `Nightshade` to shade the dark side of the Earth
# and compute the subsolar point from simple astronomical formulae.

# %%
obs_time = datetime(2025, 6, 3, 12, 0, 0)

# Solar declination: latitude where the sun is directly overhead
day_of_year = obs_time.timetuple().tm_yday
declination = 23.44 * np.sin(np.radians(360 / 365 * (day_of_year - 81)))

# Subsolar longitude: the sun is overhead where it's local solar noon.
# At 12:00 UTC, solar noon is at 0° longitude.
hours_utc = obs_time.hour + obs_time.minute / 60
subsolar_lon = 180 - (hours_utc / 24) * 360  # 0° at 12 UTC

print(f"Date/time:           {obs_time} UTC")
print(f"Day of year:         {day_of_year}")
print(f"Solar declination:   {declination:.2f}°N")
print(f"Subsolar point:      {declination:.2f}°N, {subsolar_lon:.2f}°E")

# %% [markdown]
# ### Blue Marble with day/night terminator

# %%
fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={"projection": ccrs.Robinson()})

# Layer 1: satellite background
ax.stock_img()

# Layer 2: clouds
ax.pcolormesh(
    tcc.longitude, tcc.latitude, tcc.values,
    transform=ccrs.PlateCarree(),
    cmap=cloud_cmap, vmin=0, vmax=1, shading="auto",
)

# Layer 3: nightshade
ax.add_feature(Nightshade(obs_time, alpha=0.4))

# Layer 4: subsolar point marker
ax.plot(
    subsolar_lon, declination,
    marker="*", markersize=18, color="yellow", markeredgecolor="orange",
    markeredgewidth=0.8, transform=ccrs.PlateCarree(), zorder=10,
)

# Layer 5: declination latitude line
ax.plot(
    [-180, 180], [declination, declination],
    color="yellow", linewidth=1.2, linestyle="--", alpha=0.7,
    transform=ccrs.PlateCarree(), zorder=9,
)
# Equator for reference
ax.plot(
    [-180, 180], [0, 0],
    color="white", linewidth=0.6, linestyle=":", alpha=0.5,
    transform=ccrs.PlateCarree(), zorder=9,
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="0.3")
ax.set_global()
ax.set_title(
    f"Day/Night + Clouds — 2025-06-03 12:00 UTC — "
    f"Subsolar point {declination:.1f}°N",
    fontsize=13, color="white", pad=12,
)

fig.patch.set_facecolor("black")
ax.set_facecolor("black")

fig.tight_layout()
fig.savefig(
    paths.images_path / "08_era5_daynight.png",
    dpi=150, bbox_inches="tight", facecolor="black",
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/08_era5_daynight.png
# :name: fig-08-era5-daynight
# Blue Marble with ERA5 cloud cover, the day/night terminator (shaded), the
# subsolar point (yellow star), and the solar declination latitude (dashed line).
# On 2025-06-03 the sun is nearly overhead at 22°N, close to the summer solstice.
# ```


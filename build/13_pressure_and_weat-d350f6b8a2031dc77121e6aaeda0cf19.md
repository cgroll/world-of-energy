---
jupytext:
  text_representation:
    format_name: percent
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Pressure Levels and the Jet Stream

The atmosphere is not just a flat layer of air — it has rich three-dimensional
structure.  Weather systems, temperature gradients, and especially the
**jet stream** only make sense once you understand how the atmosphere is
stacked vertically.

This notebook explores that structure using ERA5 reanalysis data for a
single snapshot (2025-06-03 12:00 UTC), combining surface observations with
pressure-level data to build an intuition for the jet stream as a
**three-dimensional tube of fast-moving air** circling the globe at
roughly 10 km altitude.

We start at the surface, then work our way up through the atmosphere,
before looking at vertical cross-sections that reveal the full 3D shape.

```{code-cell} python
import io

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from woe.paths import ProjPaths

paths = ProjPaths()
```

```{code-cell} python
DATE = "2025-06-03"
TIME = "2025-06-03T12:00:00"
G = 9.80665  # standard gravity m/s²

ds_sfc = xr.open_dataset(paths.era5_snapshot_20250603_1200_surface_file)
ds_pl  = xr.open_dataset(paths.era5_snapshot_20250603_1200_pressure_file)

print(f"Surface dimensions:        {dict(ds_sfc.dims)}")
print(f"Surface variables:         {list(ds_sfc.data_vars)}")
print(f"Pressure-level dimensions: {dict(ds_pl.dims)}")
print(f"Pressure-level variables:  {list(ds_pl.data_vars)}")
print(f"Pressure levels (hPa):     {sorted(ds_pl.level.values.tolist(), reverse=True)}")
```

```{code-cell} python
# --- Surface fields ---
t2m      = ds_sfc["2m_temperature"].squeeze()
t2m_c    = t2m - 273.15
u10      = ds_sfc["10m_u_component_of_wind"].squeeze()
v10      = ds_sfc["10m_v_component_of_wind"].squeeze()
wspd10   = np.sqrt(u10**2 + v10**2)
tcc      = ds_sfc["total_cloud_cover"].squeeze()

lats_sfc = ds_sfc.latitude.values
lons_sfc = ds_sfc.longitude.values

# --- Pressure-level fields ---
lats_pl = ds_pl.latitude.values
lons_pl = ds_pl.longitude.values

# Levels sorted from high pressure (low altitude) → low pressure (high altitude)
levels_sorted = sorted(ds_pl.level.values.tolist(), reverse=True)

u250   = ds_pl["u_component_of_wind"].sel(level=250).squeeze()
v250   = ds_pl["v_component_of_wind"].sel(level=250).squeeze()
wspd250 = np.sqrt(u250**2 + v250**2)

gph500  = ds_pl["geopotential"].sel(level=500).squeeze() / G  # metres

print(f"2m temperature:    {float(t2m_c.min()):.1f}°C to {float(t2m_c.max()):.1f}°C")
print(f"10m wind speed:    {float(wspd10.min()):.1f}–{float(wspd10.max()):.1f} m/s")
print(f"250 hPa max wind:  {float(wspd250.max()):.1f} m/s")
print(f"500 hPa GPH range: {float(gph500.min()):.0f}–{float(gph500.max()):.0f} m")
```

## Surface weather

Before climbing into the atmosphere, we look at what is happening at the
Earth's surface on this snapshot.

+++

### 2m Temperature

```{code-cell} python
fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={"projection": ccrs.Robinson()})

im = ax.pcolormesh(
    lons_sfc, lats_sfc, t2m_c.values,
    transform=ccrs.PlateCarree(),
    cmap="RdYlBu_r", vmin=-40, vmax=45, shading="auto",
)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")
ax.set_global()
ax.set_title(f"ERA5 2m Temperature — {DATE} 12:00 UTC", fontsize=14)

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
cbar.set_label("Temperature (°C)")

fig.tight_layout()
fig.savefig(paths.images_path / "13_surface_temperature.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/13_surface_temperature.png
:name: fig-13-surface-temperature
ERA5 2m temperature on 2025-06-03 at 12:00 UTC.  The classic summer contrast
between the warm continents (especially the Middle East and Central Asia) and the
cooler oceans and polar regions is clearly visible.
```

+++

### 10m Wind Speed

```{code-cell} python
wspd10_c, lons10_c = add_cyclic_point(wspd10.values, coord=lons_sfc)
u10_c, _           = add_cyclic_point(u10.values,    coord=lons_sfc)
v10_c, _           = add_cyclic_point(v10.values,    coord=lons_sfc)

stride     = 20  # 0.25° × 20 = 5° subsampling for arrows
lons10_sub = lons10_c[::stride]
lats10_sub = lats_sfc[::stride]
u10_sub    = u10_c[::stride, ::stride]
v10_sub    = v10_c[::stride, ::stride]
wspd10_sub = np.sqrt(u10_sub**2 + v10_sub**2)
wspd10_sub = np.where(wspd10_sub == 0, 1, wspd10_sub)
u10_norm   = u10_sub / wspd10_sub
v10_norm   = v10_sub / wspd10_sub

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax.set_global()

cf = ax.contourf(
    lons10_c, lats_sfc, wspd10_c,
    levels=np.arange(0, 25, 2),
    cmap="YlOrRd", transform=ccrs.PlateCarree(), extend="max",
)
ax.quiver(
    lons10_sub, lats10_sub, u10_norm, v10_norm,
    transform=ccrs.PlateCarree(),
    color="steelblue", alpha=0.7, scale=80, width=0.002,
    headwidth=3, headlength=3, headaxislength=3.75, regrid_shape=None,
)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="black")
ax.add_feature(cfeature.BORDERS,   linewidth=0.3, color="black", alpha=0.7)
ax.add_feature(cfeature.OCEAN,     facecolor="aliceblue", zorder=0)
ax.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
gl.ylocator = mticker.FixedLocator(range(-90, 91,  30))

cbar = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.03, fraction=0.04, aspect=40)
cbar.set_label("10 m wind speed (m/s)", fontsize=11)
ax.set_title(
    f"Surface wind — 10 m speed and direction\n{DATE} 12:00 UTC  |  ERA5 reanalysis",
    fontsize=13, pad=10,
)

fig.tight_layout()
fig.savefig(paths.images_path / "13_surface_wind.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/13_surface_wind.png
:name: fig-13-surface-wind
ERA5 10 m wind speed (filled contours) and direction (blue arrows) on
2025-06-03 at 12:00 UTC.  The strongest surface winds are found over the
storm-track regions of the Southern Ocean and North Atlantic.  Blue arrows
show normalised wind direction; colour shows magnitude.
```

+++

### Cloud cover over satellite imagery

```{code-cell} python
cloud_cmap = LinearSegmentedColormap.from_list("clouds", [
    (1, 1, 1, 0),   # transparent where cloud-free
    (1, 1, 1, 1),   # opaque white where fully overcast
])

fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={"projection": ccrs.Robinson()})
ax.stock_img()

ax.pcolormesh(
    lons_sfc, lats_sfc, tcc.values,
    transform=ccrs.PlateCarree(),
    cmap=cloud_cmap, vmin=0, vmax=1, shading="auto",
)
ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="0.3")
ax.add_feature(cfeature.BORDERS,   linewidth=0.2, linestyle="--", color="0.4")
ax.set_global()
ax.set_title(f"ERA5 Cloud Cover over Blue Marble — {DATE} 12:00 UTC", fontsize=14)

fig.tight_layout()
fig.savefig(paths.images_path / "13_cloud_satellite.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/13_cloud_satellite.png
:name: fig-13-cloud-satellite
ERA5 total cloud cover overlaid on Blue Marble satellite imagery.  Cloud-free
areas reveal the land/ocean surface; white patches mark frontal systems and
convective clouds over the tropics.
```

+++

## What are pressure levels?

The atmosphere is divided into **pressure levels** — horizontal surfaces where
air pressure has the same value everywhere.  Standard levels include
1000 hPa (near sea level), 850 hPa (~1.5 km), 500 hPa (~5.5 km), and
250 hPa (~10 km).

A crucial insight is that these surfaces are **not flat**.  Where the air
column is warm — and therefore expanded — a given pressure level sits
higher; where the air is cold and compressed, it sits lower.  These
undulations (measured as *geopotential height*, in metres) encode the
large-scale temperature and pressure patterns that drive weather systems.

The figure below shows the ERA5 data cross-sectioned along **55°N** — a
latitude that cuts through Iceland, the United Kingdom, central Europe,
and central Russia.  Each coloured band is the atmospheric layer between
two successive pressure levels.  Thicker bands indicate warmer (more
expanded) air; thinner bands indicate colder air.

```{code-cell} python
LAT_TRANSECT = 55.0

# Geopotential height in km at 55°N, all levels, all longitudes
gph_transect = ds_pl["geopotential"].sel(latitude=LAT_TRANSECT, method="nearest").squeeze()
gph_km       = gph_transect / G / 1000  # km

# ERA5 uses 0–360° longitudes; convert to −180–180° and sort west→east
lons_180 = np.where(lons_pl > 180, lons_pl - 360, lons_pl)
sort_idx  = np.argsort(lons_180)
lons_sort = lons_180[sort_idx]
gph_sort  = gph_km.isel(longitude=sort_idx)

# Colour palette: cool (low levels, high pressure) → warm (high levels, low pressure)
band_cmap   = plt.cm.coolwarm
n_bands     = len(levels_sorted) - 1
band_colors = [band_cmap(i / max(n_bands - 1, 1)) for i in range(n_bands)]

fig = plt.figure(figsize=(22, 7))
gs  = fig.add_gridspec(1, 5, wspace=0.45)

# --- Left: world map showing the transect latitude ---
ax_map = fig.add_subplot(gs[0, :2], projection=ccrs.Robinson())
ax_map.set_global()
ax_map.add_feature(cfeature.OCEAN,     facecolor="aliceblue")
ax_map.add_feature(cfeature.LAND,      facecolor="whitesmoke")
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax_map.add_feature(cfeature.BORDERS,   linewidth=0.3, alpha=0.7)
ax_map.plot(
    [-180, 180], [LAT_TRANSECT, LAT_TRANSECT],
    color="crimson", linewidth=2.5, transform=ccrs.PlateCarree(), zorder=5,
)
ax_map.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                 linewidth=0.3, color="gray", alpha=0.4, linestyle="--")
ax_map.set_title(
    f"Cross-section location\n(red line = {LAT_TRANSECT:.0f}°N)",
    fontsize=12, color="crimson",
)

# --- Right: pressure level bands ---
ax_b = fig.add_subplot(gs[0, 2:])

for i in range(n_bands):
    lower_lev = levels_sorted[i]       # higher pressure = lower altitude
    upper_lev = levels_sorted[i + 1]   # lower pressure  = higher altitude
    lower_h   = gph_sort.sel(level=lower_lev).values
    upper_h   = gph_sort.sel(level=upper_lev).values
    ax_b.fill_between(lons_sort, lower_h, upper_h,
                      color=band_colors[i], alpha=0.65)

# Pressure-surface lines and labels
for lev in levels_sorted:
    h = gph_sort.sel(level=lev).values
    ax_b.plot(lons_sort, h, color="black", linewidth=0.8, alpha=0.7)
    ax_b.text(lons_sort[-1] + 2, h[-1], f" {lev} hPa",
              va="center", fontsize=8, fontweight="bold")

# Shade the jet-stream zone (250–200 hPa) with a thin annotation
jet_low  = gph_sort.sel(level=250).values.mean()
jet_high = gph_sort.sel(level=200).values.mean()
ax_b.axhspan(jet_low, jet_high, color="gold", alpha=0.25, zorder=0)
ax_b.text(lons_sort[0] + 2, (jet_low + jet_high) / 2,
          "← jet stream zone", va="center", fontsize=9,
          color="goldenrod", fontweight="bold")

ax_b.set_xlabel("Longitude (°E)", fontsize=12)
ax_b.set_ylabel("Geopotential height (km)", fontsize=12)
ax_b.set_xlim(lons_sort.min(), lons_sort.max() + 14)
ax_b.grid(True, alpha=0.2)
ax_b.set_title(
    f"Atmospheric layers along {LAT_TRANSECT:.0f}°N — {DATE} 12:00 UTC",
    fontsize=13,
)

fig.tight_layout()
fig.savefig(paths.images_path / "13_pressure_bands.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/13_pressure_bands.png
:name: fig-13-pressure-bands
**Left:** world map showing the cross-section location (red line = 55°N).
**Right:** atmospheric layers along 55°N.  Each coloured band represents the
layer between two pressure levels; black lines are the actual geopotential
height of each surface.  Thicker bands indicate warmer (expanded) air.  The
gold shading marks the jet-stream zone (~200–250 hPa, ~10–12 km altitude).
Note how the pressure surfaces undulate by hundreds of metres — these waves
are the Rossby waves that steer surface weather systems.
```

+++

## The jet stream in 2D

The **jet stream** is a narrow corridor of fast-moving air located near the
tropopause, typically at 250 hPa (~10 km altitude).  Two jets exist in each
hemisphere:

- **Polar front jet** (~50–65°): marks the boundary between cold polar air
  and warmer mid-latitude air.  Its meanders steer surface weather systems.
- **Subtropical jet** (~25–35°): driven by the Hadley cell; typically
  stronger in winter, weaker in summer.

On a summer snapshot like ours (June), the polar jet is weaker and displaced
poleward.  We visualise it in two ways:
1. **Global Robinson map** — 250 hPa wind speed in colour, with 500 hPa
   geopotential height as contour lines to show the pressure wave pattern.
2. **Polar (top-down) view** — shows the jet stream as a ring around the
   North Pole.

+++

### Global view: 250 hPa wind + 500 hPa geopotential

```{code-cell} python
wspd250_c, lons250_c = add_cyclic_point(wspd250.values, coord=lons_pl)
gph500_c,  _         = add_cyclic_point(gph500.values,  coord=lons_pl)

# Geopotential-height contour levels in 80 m steps
z500_levels = np.arange(
    np.floor(float(gph500.min()) / 100) * 100,
    np.ceil(float(gph500.max())  / 100) * 100 + 1,
    80,
)

fig = plt.figure(figsize=(16, 8))
ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax.set_global()

cf = ax.contourf(
    lons250_c, lats_pl, wspd250_c,
    levels=np.arange(0, 85, 5),
    cmap="hot_r", transform=ccrs.PlateCarree(), extend="max",
)
cs = ax.contour(
    lons250_c, lats_pl, gph500_c,
    levels=z500_levels,
    colors="steelblue", linewidths=0.8, alpha=0.7,
    transform=ccrs.PlateCarree(),
)
ax.clabel(cs, fmt="%d m", fontsize=7, inline=True)

ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle="--")

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
gl.ylocator = mticker.FixedLocator(range(-90,  91,  30))

cbar = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04, shrink=0.6)
cbar.set_label("250 hPa wind speed (m/s)")
ax.set_title(
    f"250 hPa wind speed + 500 hPa geopotential height — {DATE} 12:00 UTC",
    fontsize=13,
)

fig.tight_layout()
fig.savefig(paths.images_path / "13_z500_wind250.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/13_z500_wind250.png
:name: fig-13-z500-wind250
250 hPa wind speed (filled, hot colourmap) with 500 hPa geopotential height
contours (blue lines, labelled in metres).  The jet stream appears as the orange/red
band encircling the mid-latitudes.  The 500 hPa contours reveal the trough/ridge
wave pattern that the jet stream follows: the jet accelerates through troughs
(low geopotential) and decelerates through ridges (high geopotential).
```

+++

### 500 hPa geopotential + 250 hPa jet stream footprint

The same Z500 wave pattern as above, but now the 250 hPa wind speed is
collapsed to a binary mask: grid points where the jet core exceeds 30 m/s
are filled in purple, while slower regions are transparent.  This style —
borrowed from operational meteorological charts — immediately shows the
jet-stream corridor without the distraction of a continuous colour scale.

```{code-cell} python
JET_THRESHOLD = 30  # m/s — standard meteorological jet-stream criterion

# Reuse cyclic arrays already computed above
z500_levels   = np.arange(
    np.floor(float(gph500.min()) / 60) * 60,
    np.ceil(float(gph500.max())  / 60) * 60 + 1,
    80,
)
jet_masked_c = np.where(wspd250_c >= JET_THRESHOLD, wspd250_c, np.nan)

fig = plt.figure(figsize=(16, 8))
ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax.set_global()

ax.add_feature(cfeature.OCEAN, facecolor="#d0e8f5", zorder=0)
ax.add_feature(cfeature.LAND,  facecolor="#f0ede8", zorder=0)

cs = ax.contour(
    lons250_c, lats_pl, gph500_c,
    levels=z500_levels, colors="black", linewidths=0.6,
    transform=ccrs.PlateCarree(), zorder=3,
)
ax.clabel(cs, fmt="%d m", fontsize=7, inline=True, inline_spacing=3)

ax.contourf(
    lons250_c, lats_pl, jet_masked_c,
    levels=[JET_THRESHOLD, 300], colors=["mediumpurple"], alpha=0.75,
    transform=ccrs.PlateCarree(), zorder=4,
)
ax.contour(
    lons250_c, lats_pl, wspd250_c,
    levels=[JET_THRESHOLD], colors=["rebeccapurple"], linewidths=1.0,
    transform=ccrs.PlateCarree(), zorder=5,
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="black", zorder=6)
ax.add_feature(cfeature.BORDERS,   linewidth=0.3, color="black", alpha=0.7, zorder=6)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
gl.ylocator = mticker.FixedLocator(range(-90,  91,  30))

ax.legend(
    handles=[mpatches.Patch(facecolor="mediumpurple", edgecolor="rebeccapurple",
                            alpha=0.75, label=f"Jet stream ≥ {JET_THRESHOLD} m/s  (250 hPa)")],
    loc="lower left", fontsize=10, framealpha=0.85,
)
ax.set_title(
    f"500 hPa geopotential height (contours) + jet stream footprint — {DATE} 12:00 UTC",
    fontsize=13,
)

fig.tight_layout()
fig.savefig(paths.images_path / "13_z500_jet_purple.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/13_z500_jet_purple.png
:name: fig-13-z500-jet-purple
500 hPa geopotential height (black contours, labelled in metres) with the
250 hPa jet stream footprint (purple, ≥ 30 m/s).  The jet hugs the steepest
geopotential gradient, flowing fastest where contours are most tightly packed.
Troughs (contours bending equatorward) and ridges (bending poleward) mark
the Rossby-wave pattern that steers the jet.
```

+++

### 2m temperature + 500 hPa geopotential contours

The Z500 wave pattern also reflects the surface temperature distribution —
warm air expands the lower troposphere and pushes pressure surfaces higher,
while cold air compresses them.  Overlaying Z500 contours on 2m temperature
makes this connection explicit: ridges (high Z500) sit over warm air masses,
troughs (low Z500) over cold ones.

```{code-cell} python
t2m_c_cyc, lons_t2m_c = add_cyclic_point(t2m_c.values, coord=lons_sfc)

fig = plt.figure(figsize=(16, 8))
ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax.set_global()

im = ax.pcolormesh(
    lons_t2m_c, lats_sfc, t2m_c_cyc,
    cmap="RdBu_r", vmin=-40, vmax=45,
    transform=ccrs.PlateCarree(), shading="auto",
)
cs = ax.contour(
    lons250_c, lats_pl, gph500_c,
    levels=z500_levels, colors="black", linewidths=0.6,
    transform=ccrs.PlateCarree(), zorder=3,
)
ax.clabel(cs, fmt="%d m", fontsize=7, inline=True, inline_spacing=3)

ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="0.15", zorder=4)
ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle="--", color="0.35", zorder=4)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
gl.ylocator = mticker.FixedLocator(range(-90,  91,  30))

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
cbar.set_label("2m temperature (°C)")
ax.set_title(
    f"500 hPa geopotential height (contours) + 2m temperature — {DATE} 12:00 UTC",
    fontsize=13,
)

fig.tight_layout()
fig.savefig(paths.images_path / "13_t2m_z500.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/13_t2m_z500.png
:name: fig-13-t2m-z500
2m temperature (filled, red–blue scale) with 500 hPa geopotential height
contours (black).  Ridges (contours arching poleward) align with warm air
masses; troughs (contours dipping equatorward) sit over cold air.  The
tight coupling between the upper-level wave pattern and the surface
temperature field is a key driver of both weather and energy demand.
```

+++

### 2m temperature with 250 hPa jet stream core overlay

The jet stream is the boundary between cold polar air and warm subtropical
air.  Overlaying the jet core (> 30 m/s, shown here as a white band that
brightens with wind speed) directly on the 2m temperature field makes this
air-mass boundary visible: north of the jet lies colder, denser polar air;
south of it lies warmer subtropical air.

```{code-cell} python
jet_core_c  = np.where(wspd250_c > JET_THRESHOLD, wspd250_c, np.nan)
jet_core_cmap = LinearSegmentedColormap.from_list("jet_core", [
    (0.2, 0.2, 0.2, 0.6),  # dark grey, semi-transparent at threshold
    (1.0, 1.0, 1.0, 1.0),  # bright white at 80 m/s
])

fig = plt.figure(figsize=(16, 9))
ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax.set_global()

im_temp = ax.pcolormesh(
    lons_t2m_c, lats_sfc, t2m_c_cyc,
    cmap="RdYlBu_r", vmin=-40, vmax=45,
    transform=ccrs.PlateCarree(), shading="auto",
)
ax.pcolormesh(
    lons250_c, lats_pl, jet_core_c,
    cmap=jet_core_cmap, vmin=JET_THRESHOLD, vmax=80,
    transform=ccrs.PlateCarree(), shading="auto",
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle="--")

cbar = fig.colorbar(im_temp, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
cbar.set_label("Temperature (°C)")
ax.set_title(
    f"2m temperature with jet stream core (> {JET_THRESHOLD} m/s at 250 hPa) — {DATE} 12:00 UTC",
    fontsize=13,
)

fig.tight_layout()
fig.savefig(paths.images_path / "13_t2m_jet_core.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/13_t2m_jet_core.png
:name: fig-13-t2m-jet-core
2m temperature (filled, RdYlBu_r) with the 250 hPa jet stream core overlaid
as a white band (brightening with wind speed above 30 m/s).  The jet marks
the sharp boundary between cold polar air to the north and warm subtropical
air to the south — a boundary that in winter can be associated with
temperature differences of 20–30 °C over just a few degrees of latitude.
```

+++

### Polar view: the jet stream as a ring around the North Pole

```{code-cell} python
# Subsample for streamlines (dense grids overwhelm cartopy's streamplot)
step  = 8
lons_sub = u250.longitude.values[::step]
lats_sub = lats_pl[::step]
u_sub    = u250.values[::step, ::step]
v_sub    = v250.values[::step, ::step]

fig, ax = plt.subplots(
    figsize=(10, 10),
    subplot_kw={"projection": ccrs.NorthPolarStereo()},
)

im = ax.pcolormesh(
    lons_pl, lats_pl, wspd250.values,
    transform=ccrs.PlateCarree(),
    cmap="hot_r", vmin=0, vmax=80, shading="auto",
)
ax.streamplot(
    lons_sub, lats_sub, u_sub, v_sub,
    transform=ccrs.PlateCarree(),
    color="white", linewidth=0.5, density=1.5, arrowsize=0.8,
)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle="--")
ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
cbar.set_label("250 hPa wind speed (m/s)")
ax.set_title(
    f"Jet stream viewed from above — 250 hPa\n{DATE} 12:00 UTC  |  ERA5",
    fontsize=13,
)

fig.tight_layout()
fig.savefig(paths.images_path / "13_jet_stream_polar.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/13_jet_stream_polar.png
:name: fig-13-jet-stream-polar
The Northern Hemisphere jet stream viewed from directly above the North Pole
(North Polar Stereographic projection).  Wind speed at 250 hPa is shown in
colour; white streamlines indicate wind direction.  The jet stream traces an
irregular ring around the pole, with Rossby-wave meanders visible as north–south
excursions.  This "ring" perspective makes it clear why the jet stream is
sometimes described as a three-dimensional tube encircling the planet.
```

+++

## Jet stream across altitude levels

The figures above show the jet stream at a single pressure level (250 hPa).
But the jet has **vertical extent** — it intensifies from the mid-troposphere
upward through the tropopause, then weakens in the stratosphere above.

The animation below steps through all available pressure levels from
850 hPa (~1.5 km, near-surface) to 100 hPa (~16 km, lower stratosphere).
Watch how the fast-moving core (red/orange) emerges at mid-tropospheric
levels, peaks near 200–250 hPa, then fades in the stratosphere above.

| Level (hPa) | Approx. altitude | Layer |
|-------------|-----------------|-------|
| 850 | ~1.5 km | lower troposphere |
| 500 | ~5.5 km | mid-troposphere |
| 400 | ~7 km   | mid-troposphere |
| 350 | ~8 km   | upper troposphere |
| 300 | ~9 km   | upper troposphere |
| 250 | ~10 km  | **jet-stream core** |
| 225 | ~11 km  | jet-stream core |
| 200 | ~12 km  | jet / tropopause |
| 175 | ~13 km  | tropopause |
| 150 | ~14 km  | lower stratosphere |
| 125 | ~15 km  | lower stratosphere |
| 100 | ~16 km  | lower stratosphere |

```{code-cell} python
_ALT_LABELS = {
    850: "~1.5 km (lower troposphere)",
    500: "~5.5 km (mid-troposphere)",
    400: "~7 km",
    350: "~8 km",
    300: "~9 km (upper troposphere)",
    250: "~10 km ★ jet-stream core",
    225: "~11 km ★ jet-stream core",
    200: "~12 km (jet / tropopause)",
    175: "~13 km (tropopause)",
    150: "~14 km (lower stratosphere)",
    125: "~15 km",
    100: "~16 km (lower stratosphere)",
}

gif_frames_pl = []

for level in levels_sorted:
    u_lev   = ds_pl["u_component_of_wind"].sel(level=level).squeeze()
    v_lev   = ds_pl["v_component_of_wind"].sel(level=level).squeeze()
    wspd_lv = np.sqrt(u_lev.values**2 + v_lev.values**2)

    wspd_c, lons_c = add_cyclic_point(wspd_lv,       coord=lons_pl)
    u_c,    _      = add_cyclic_point(u_lev.values,   coord=lons_pl)
    v_c,    _      = add_cyclic_point(v_lev.values,   coord=lons_pl)

    stride   = 20
    lons_sub = lons_c[::stride]
    lats_sub = lats_pl[::stride]
    u_sub    = u_c[::stride, ::stride]
    v_sub    = v_c[::stride, ::stride]
    wspd_sub = np.sqrt(u_sub**2 + v_sub**2)
    wspd_sub = np.where(wspd_sub == 0, 1, wspd_sub)
    u_norm   = u_sub / wspd_sub
    v_norm   = v_sub / wspd_sub

    fig = plt.figure(figsize=(16, 8))
    ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()

    cf = ax.contourf(
        lons_c, lats_pl, wspd_c,
        levels=np.arange(0, 85, 5),
        cmap="YlOrRd", transform=ccrs.PlateCarree(), extend="max",
    )
    ax.quiver(
        lons_sub, lats_sub, u_norm, v_norm,
        transform=ccrs.PlateCarree(),
        color="steelblue", alpha=0.7, scale=80, width=0.002,
        headwidth=3, headlength=3, headaxislength=3.75, regrid_shape=None,
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="black")
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, color="black", alpha=0.7)
    ax.add_feature(cfeature.OCEAN,     facecolor="aliceblue",  zorder=0)
    ax.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator(range(-90,  91,  30))

    cbar = fig.colorbar(cf, ax=ax, orientation="horizontal",
                        pad=0.03, fraction=0.04, aspect=40)
    cbar.set_label(f"Wind speed at {level} hPa (m/s)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        f"Upper-level wind — {level} hPa  ({_ALT_LABELS.get(level, '')})\n"
        f"{DATE} 12:00 UTC  |  ERA5 reanalysis",
        fontsize=13, pad=10,
    )

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    gif_frames_pl.append(Image.open(buf).copy())
    print(f"  pressure level frame: {level} hPa")

out_gif_pl = paths.images_path / "13_pressure_levels_animation.gif"
gif_frames_pl[0].save(
    out_gif_pl,
    save_all=True,
    append_images=gif_frames_pl[1:],
    duration=700,   # ms per frame — slow enough to read the title
    loop=0,
)
print(f"saved → {out_gif_pl.name}  ({len(gif_frames_pl)} frames)")
```

```{figure} ../../output/images/13_pressure_levels_animation.gif
:name: fig-13-pressure-levels-animation
Animation stepping from 850 hPa (near-surface) up to 100 hPa (lower stratosphere).
The jet stream emerges as the fast-moving core (orange/red) intensifies through the
mid-troposphere, peaks at 200–250 hPa, then weakens in the stratosphere above.
Blue arrows show wind direction.  This animation reveals the jet stream as a
vertically coherent structure rather than just a surface phenomenon.
```

+++

## Vertical cross-sections

A **vertical cross-section** cuts through the atmosphere at a fixed longitude
and shows how wind speed and temperature vary simultaneously with latitude
(x-axis) and altitude (y-axis).  This is the most direct way to see the jet
stream's three-dimensional shape.

The y-axis uses **geopotential height** (km) so altitude increases naturally
upward.  Each cross-section is paired with a world map highlighting the
meridian it was cut along.

+++

### Cross-section helper

```{code-cell} python
def _fmt_cross_axes(ax):
    ax.set_xlabel("Latitude (°)", fontsize=10)
    ax.set_ylabel("Geopotential height (km)", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d°"))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(15))
    ax.grid(axis="x", linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
    ax.grid(axis="y", linewidth=0.4, color="gray", alpha=0.3, linestyle=":")


def _cross_section_data(ds_pl, lon_cut):
    """Extract and sort cross-section arrays for a given longitude."""
    ds_cut    = ds_pl.sel(longitude=lon_cut, method="nearest")
    lon_act   = float(ds_cut.longitude)
    lon_plot  = lon_act if lon_act <= 180 else lon_act - 360

    u_cut  = ds_cut["u_component_of_wind"].squeeze().values  # (level, lat)
    v_cut  = ds_cut["v_component_of_wind"].squeeze().values
    z_cut  = ds_cut["geopotential"].squeeze().values
    t_cut  = ds_cut["temperature"].squeeze().values
    lats_c = ds_cut.latitude.values  # descending 90→−90

    z_km       = z_cut / G / 1000
    level_ord  = np.argsort(z_km.mean(axis=1))  # bottom → top

    z_km_s  = z_km[level_ord][:, ::-1]                                # south→north
    wspd_s  = np.sqrt(u_cut**2 + v_cut**2)[level_ord][:, ::-1]
    t_s     = t_cut[level_ord][:, ::-1] - 273.15
    lat_2d  = np.broadcast_to(lats_c[::-1], z_km_s.shape).copy()

    return lon_act, lon_plot, lat_2d, z_km_s, wspd_s, t_s
```

### Static cross-section at 0 °E (prime meridian) with location map

```{code-cell} python
LON_STATIC = 0
lon_act, lon_plot, lat_2d, z_km_s, wspd_s, t_s = _cross_section_data(ds_pl, LON_STATIC)

fig = plt.figure(figsize=(22, 6))
gs  = fig.add_gridspec(1, 10, wspace=0.5)

# --- Location map ---
ax_map = fig.add_subplot(gs[0, :2], projection=ccrs.PlateCarree())
ax_map.set_global()
ax_map.add_feature(cfeature.OCEAN,     facecolor="aliceblue")
ax_map.add_feature(cfeature.LAND,      facecolor="whitesmoke")
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax_map.add_feature(cfeature.BORDERS,   linewidth=0.3, alpha=0.7)
ax_map.plot([lon_plot, lon_plot], [-90, 90],
            color="crimson", linewidth=2.5, transform=ccrs.PlateCarree(), zorder=5)
gl_m = ax_map.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
gl_m.top_labels   = False
gl_m.right_labels = False
gl_m.xlocator     = mticker.FixedLocator(range(-180, 181, 60))
gl_m.ylocator     = mticker.FixedLocator(range(-90,  91,  30))
gl_m.xlabel_style = {"size": 8}
gl_m.ylabel_style = {"size": 8}
ax_map.set_title(
    f"Cross-section at\n{lon_act:.0f}°E ({lon_plot:+.0f}°)",
    fontsize=10, color="crimson",
)

# --- Wind speed ---
ax_w = fig.add_subplot(gs[0, 2:6])
cf_w = ax_w.contourf(lat_2d, z_km_s, wspd_s,
                     levels=np.arange(0, 85, 5), cmap="YlOrRd", extend="max")
cs_w = ax_w.contour(lat_2d, z_km_s, wspd_s, levels=[30, 50, 70],
                    colors="black", linewidths=0.8, alpha=0.6)
ax_w.clabel(cs_w, fmt="%d m/s", fontsize=8)
fig.colorbar(cf_w, ax=ax_w, orientation="vertical", pad=0.02,
             fraction=0.03, aspect=30).set_label("Wind speed (m/s)", fontsize=10)
_fmt_cross_axes(ax_w)
ax_w.set_title(f"Wind speed — {lon_act:.0f}°E ({lon_plot:+.0f}°)", fontsize=11)

# --- Temperature ---
ax_t = fig.add_subplot(gs[0, 6:10], sharey=ax_w)
cf_t = ax_t.contourf(lat_2d, z_km_s, t_s,
                     levels=np.arange(-80, 25, 5), cmap="RdBu_r", extend="both")
cs_t = ax_t.contour(lat_2d, z_km_s, t_s, levels=np.arange(-80, 25, 10),
                    colors="black", linewidths=0.6, alpha=0.5)
ax_t.clabel(cs_t, fmt="%d°C", fontsize=8)
fig.colorbar(cf_t, ax=ax_t, orientation="vertical", pad=0.02,
             fraction=0.03, aspect=30).set_label("Temperature (°C)", fontsize=10)
ax_t.set_xlabel("Latitude (°)", fontsize=10)
ax_t.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d°"))
ax_t.xaxis.set_major_locator(mticker.MultipleLocator(15))
ax_t.grid(axis="x", linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
ax_t.grid(axis="y", linewidth=0.4, color="gray", alpha=0.3, linestyle=":")
ax_t.tick_params(labelleft=False)
ax_t.set_title(f"Temperature — {lon_act:.0f}°E ({lon_plot:+.0f}°)", fontsize=11)

fig.suptitle(
    f"Vertical cross-section at {lon_act:.0f}°E ({lon_plot:+.0f}°) — "
    f"{DATE} 12:00 UTC  |  ERA5 reanalysis",
    fontsize=13, y=1.02,
)

fig.tight_layout()
fig.savefig(paths.images_path / "13_cross_section_0E.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/13_cross_section_0E.png
:name: fig-13-cross-section-0E
Vertical cross-section through the atmosphere at 0°E (prime meridian).
**Left:** location map with the section meridian (red).
**Centre:** wind speed — the jet stream appears as the orange/red core at
40–60°N between 8 and 14 km altitude.
**Right:** temperature — the stratospheric warm layer above ~12 km and the
sharp temperature gradient across the jet are clearly visible.
```

+++

### Animation: cross-sections at every 5° longitude

The GIF below steps through all meridians in 5° increments, each frame
showing the wind-speed and temperature cross-section alongside a world map
that highlights the current longitude.

Watch the jet stream core (orange/red, 8–14 km) shift northward and
southward as the section rotates around the globe, tracing the Rossby-wave
meanders of the jet.

```{code-cell} python
gif_frames_cs = []

for lon_cut in range(0, 360, 5):
    lon_act, lon_plot, lat_2d, z_km_s, wspd_s, t_s = _cross_section_data(ds_pl, lon_cut)

    fig = plt.figure(figsize=(18, 5))
    gs  = fig.add_gridspec(1, 10, wspace=0.5)

    # Location map
    ax_map = fig.add_subplot(gs[0, :2], projection=ccrs.Robinson())
    ax_map.set_global()
    ax_map.add_feature(cfeature.OCEAN,     facecolor="aliceblue")
    ax_map.add_feature(cfeature.LAND,      facecolor="whitesmoke")
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax_map.plot([lon_plot, lon_plot], [-90, 90],
                color="crimson", linewidth=2.5,
                transform=ccrs.PlateCarree(), zorder=5)
    ax_map.set_title(
        f"{lon_act:.0f}°E ({lon_plot:+.0f}°)",
        fontsize=9, color="crimson",
    )

    # Wind speed
    ax_w = fig.add_subplot(gs[0, 2:6])
    cf_w = ax_w.contourf(lat_2d, z_km_s, wspd_s,
                         levels=np.arange(0, 85, 5), cmap="YlOrRd", extend="max")
    ax_w.contour(lat_2d, z_km_s, wspd_s, levels=[30, 50, 70],
                 colors="black", linewidths=0.8, alpha=0.6)
    fig.colorbar(cf_w, ax=ax_w, fraction=0.04, pad=0.02,
                 aspect=25).set_label("Wind speed (m/s)", fontsize=9)
    _fmt_cross_axes(ax_w)
    ax_w.set_title("Wind speed", fontsize=10)

    # Temperature
    ax_t = fig.add_subplot(gs[0, 6:10], sharey=ax_w)
    cf_t = ax_t.contourf(lat_2d, z_km_s, t_s,
                         levels=np.arange(-80, 25, 5), cmap="RdBu_r", extend="both")
    ax_t.contour(lat_2d, z_km_s, t_s, levels=np.arange(-80, 25, 10),
                 colors="black", linewidths=0.6, alpha=0.5)
    fig.colorbar(cf_t, ax=ax_t, fraction=0.04, pad=0.02,
                 aspect=25).set_label("Temperature (°C)", fontsize=9)
    ax_t.set_xlabel("Latitude (°)", fontsize=9)
    ax_t.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d°"))
    ax_t.xaxis.set_major_locator(mticker.MultipleLocator(15))
    ax_t.grid(axis="x", linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
    ax_t.grid(axis="y", linewidth=0.4, color="gray", alpha=0.3, linestyle=":")
    ax_t.tick_params(labelleft=False)
    ax_t.set_title("Temperature", fontsize=10)

    fig.suptitle(
        f"Vertical cross-section at {lon_act:.0f}°E ({lon_plot:+.0f}°)  —  "
        f"{DATE} 12:00 UTC  |  ERA5 reanalysis",
        fontsize=11,
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    gif_frames_cs.append(Image.open(buf).copy())
    print(f"  cross-section frame: {lon_act:.0f}°E", end="\r")

out_gif_cs = paths.images_path / "13_cross_section_animation.gif"
gif_frames_cs[0].save(
    out_gif_cs,
    save_all=True,
    append_images=gif_frames_cs[1:],
    duration=200,   # ms per frame
    loop=0,
)
print(f"\nsaved → {out_gif_cs.name}  ({len(gif_frames_cs)} frames)")
```

```{figure} ../../output/images/13_cross_section_animation.gif
:name: fig-13-cross-section-animation
Vertical cross-sections rotating around the globe in 5° steps.  Each frame
shows wind speed (centre) and temperature (right) vs. latitude and geopotential
height, paired with a world map highlighting the current meridian (red line).
The jet-stream core (orange/red at 8–14 km, 40–65°N) shifts north and south
with the Rossby-wave pattern as the section rotates — demonstrating the
jet's three-dimensional tube-like shape encircling the Northern Hemisphere.
```

+++

## Column-maximum wind speed

By taking the **maximum wind speed across all pressure levels** at each
horizontal grid point, we collapse the full vertical structure into a single
map that highlights every location where the jet core passes — regardless
of its exact altitude.

Grid points where the jet tilts vertically or splits into multiple cores
are still captured, making this a robust indicator of the jet's
horizontal footprint.

```{code-cell} python
u_all    = ds_pl["u_component_of_wind"].squeeze()  # (level, lat, lon)
v_all    = ds_pl["v_component_of_wind"].squeeze()
wspd_all = np.sqrt(u_all**2 + v_all**2)
wspd_max = wspd_all.max(dim="level")

print(f"Column-max wind range: {float(wspd_max.min()):.1f}–{float(wspd_max.max()):.1f} m/s")
```

```{code-cell} python
wspd_max_c, lons_max_c = add_cyclic_point(wspd_max.values, coord=lons_pl)

fig = plt.figure(figsize=(16, 8))
ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax.set_global()

cf = ax.contourf(
    lons_max_c, lats_pl, wspd_max_c,
    levels=np.arange(0, 85, 5),
    cmap="YlOrRd", transform=ccrs.PlateCarree(), extend="max",
)

# Jet-stream footprint: contour line at 30 m/s threshold
ax.contour(
    lons_max_c, lats_pl, wspd_max_c,
    levels=[30],
    colors=["mediumpurple"], linewidths=1.5,
    transform=ccrs.PlateCarree(),
)
legend_el = [mpatches.Patch(facecolor="mediumpurple", edgecolor="mediumpurple",
                             label="Jet-stream threshold (≥ 30 m/s)")]
ax.legend(handles=legend_el, loc="lower left", fontsize=10, framealpha=0.85)

ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="black")
ax.add_feature(cfeature.BORDERS,   linewidth=0.3, color="black", alpha=0.7)
ax.add_feature(cfeature.OCEAN,     facecolor="aliceblue",  zorder=0)
ax.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
gl.ylocator = mticker.FixedLocator(range(-90,  91,  30))

cbar = fig.colorbar(cf, ax=ax, orientation="horizontal",
                    pad=0.03, fraction=0.04, aspect=40)
cbar.set_label("Column-maximum wind speed (m/s)", fontsize=11)
cbar.ax.tick_params(labelsize=9)

ax.set_title(
    f"Column-maximum wind speed (850–100 hPa)\n{DATE} 12:00 UTC  |  ERA5 reanalysis",
    fontsize=13, pad=10,
)

fig.tight_layout()
fig.savefig(paths.images_path / "13_jet_colmax.png", dpi=150, bbox_inches="tight")
plt.show()
```

```{figure} ../../output/images/13_jet_colmax.png
:name: fig-13-jet-colmax
Column-maximum wind speed from 850 to 100 hPa.  By collapsing all altitude
levels into a single value per grid point, this map shows the **horizontal
footprint** of the jet stream regardless of where it sits vertically.  The
purple contour marks the 30 m/s threshold commonly used to define the
jet-stream boundary.  In this summer snapshot, the jet is weaker than in
winter and displaced poleward, leaving a narrower and more fragmented footprint.
```

+++

## Summary

Combining the visualisations above builds a coherent picture of the jet
stream as a **three-dimensional structure**:

- At the **surface** (10 m wind, cloud cover, 2m temperature), the jet's
  influence is indirect — it steers cyclones and anticyclones that modulate
  surface conditions.
- In the **mid-troposphere** (500 hPa geopotential), the Rossby-wave pattern
  shows the large-scale trough/ridge structure that the jet follows.
- At **jet-stream altitude** (200–250 hPa), the fast core is concentrated
  in a band 20–30° latitude wide, reaching 60–80 m/s in winter (weaker in
  this June snapshot).
- **Vertical cross-sections** reveal the core sitting between ~8 and 14 km,
  with the strongest winds near the tropopause (~200 hPa).
- The **polar view** and **column-maximum map** show the jet's horizontal
  footprint as a ring around the Northern Hemisphere, punctuated by the
  Rossby-wave meanders that connect upper-level dynamics to surface weather.

For energy systems, the key takeaway is that wind and solar availability
across Europe depend critically on the current state of this three-dimensional
jet — its latitude, its strength, and whether it is blocked or meandering.

```{code-cell} python

```

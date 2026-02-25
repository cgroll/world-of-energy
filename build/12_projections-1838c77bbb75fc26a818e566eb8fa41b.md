---
jupytext:
  text_representation:
    format_name: percent
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Map Projections — ERA5 2 m Temperature

Loads the ERA5 2 m temperature snapshot for 2025-06-03 12:00 UTC from the
locally cached surface file (downloaded by `11_download_era5_single_timestamp.py`)
and visualises it in a variety of map projections.

**Overview panel** — four common global projections side by side:
Plate Carrée, Equal Earth, Robinson, Mercator.

**Individual plots** — four projections that emphasise different subsets of
the globe or introduce a 3-D perspective:
- Orthographic (centred on Europe)
- Robinson — global (same as the overview but full-size)
- Plate Carrée — North Atlantic / European domain
- Orthographic centred on the North Atlantic (NAO-style)

```{code-cell} python
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from woe.paths import ProjPaths

paths = ProjPaths()
```

## Load data

```{code-cell} python
ds = xr.open_dataset(paths.era5_snapshot_20250603_1200_surface_file)
t2m = ds["2m_temperature"].squeeze() - 273.15  # K → °C

print(f"Dimensions:        {dict(t2m.sizes)}")
print(f"Temperature range: {float(t2m.min()):.1f} °C to {float(t2m.max()):.1f} °C")
```

## Overview — four global projections

| Projection | Preserves | Best use case |
|---|---|---|
| Plate Carrée | Grid indices | Quick plots / data checks |
| Equal Earth | Area | Global distribution comparisons |
| Robinson | Compromise | Balanced, appealing global overview |
| Mercator | Direction/Angles | Navigation (distorts polar regions) |

```{code-cell} python
projections = [
    ("Plate Carrée", ccrs.PlateCarree()),
    ("Equal Earth",  ccrs.EqualEarth()),
    ("Robinson",     ccrs.Robinson()),
    ("Mercator",     ccrs.Mercator()),
]

fig = plt.figure(figsize=(18, 13))
fig.subplots_adjust(top=0.93, bottom=0.09, left=0.02, right=0.98,
                    hspace=0.08, wspace=0.04)

for i, (name, proj) in enumerate(projections):
    ax = fig.add_subplot(2, 2, i + 1, projection=proj)

    im = ax.pcolormesh(
        t2m.longitude,
        t2m.latitude,
        t2m.values,
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

fig.suptitle("ERA5 2 m Temperature — 2025-06-03 12:00 UTC", fontsize=15)

# Explicit colorbar axes pinned to the reserved bottom strip
cax = fig.add_axes([0.25, 0.03, 0.50, 0.022])
cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
cbar.set_label("Temperature (°C)")

fig.savefig(
    paths.images_path / "12_t2m_projections_overview.png",
    dpi=150, bbox_inches="tight",
)
plt.show()
```

```{figure} ../../output/images/12_t2m_projections_overview.png
:name: fig-12-t2m-projections-overview
ERA5 2 m temperature on 2025-06-03 at 12:00 UTC in four common global map
projections.  Plate Carrée preserves grid indices; Equal Earth preserves
area; Robinson is a balanced compromise; Mercator preserves direction but
strongly distorts polar regions.
```

+++

## Orthographic — Europe-centred perspective

```{code-cell} python
fig, ax = plt.subplots(
    figsize=(10, 10),
    subplot_kw={"projection": ccrs.Orthographic(central_longitude=10, central_latitude=50)},
)

im = ax.pcolormesh(
    t2m.longitude,
    t2m.latitude,
    t2m.values,
    transform=ccrs.PlateCarree(),
    cmap="RdYlBu_r",
    vmin=-40,
    vmax=45,
    shading="auto",
)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")
ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)
ax.set_global()
ax.set_title("ERA5 2 m Temperature — Orthographic (Europe) — 2025-06-03 12:00 UTC", fontsize=12)

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.04, shrink=0.8)
cbar.set_label("Temperature (°C)")

fig.tight_layout()
fig.savefig(
    paths.images_path / "12_t2m_orthographic.png",
    dpi=150, bbox_inches="tight",
)
plt.show()
```

```{figure} ../../output/images/12_t2m_orthographic.png
:name: fig-12-t2m-orthographic
ERA5 2 m temperature on 2025-06-03 at 12:00 UTC on an Orthographic projection
centred over Europe.  This perspective gives a sense of how the temperature
field looks from space, with the curvature of the Earth visible at the edges.
```

+++

## Robinson — global full-size

The Robinson projection is a compromise that minimises distortion of both
shape and area, making it the standard choice for presenting global climate
fields.

```{code-cell} python
fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={"projection": ccrs.Robinson()})

im = ax.pcolormesh(
    t2m.longitude,
    t2m.latitude,
    t2m.values,
    transform=ccrs.PlateCarree(),
    cmap="RdYlBu_r",
    vmin=-40,
    vmax=45,
)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)
ax.set_global()

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.04)
cbar.set_label("2 m temperature (°C)")

ax.set_title("ERA5 2 m Temperature — Robinson — 2025-06-03 12:00 UTC", fontsize=13)

fig.tight_layout()
fig.savefig(
    paths.images_path / "12_t2m_robinson.png",
    dpi=150, bbox_inches="tight",
)
plt.show()
```

```{figure} ../../output/images/12_t2m_robinson.png
:name: fig-12-t2m-robinson
ERA5 2 m temperature on 2025-06-03 at 12:00 UTC on a Robinson projection.
The warm summer temperatures across mid-latitude land masses (>30 °C) contrast
with cold air over Antarctica and the high Arctic.
```

+++

## Plate Carrée — North Atlantic / European domain

A rectangular view centred on the North Atlantic and Europe — the same
domain used throughout the NAO analysis (90°W–40°E, 20°N–80°N).

```{code-cell} python
fig, ax = plt.subplots(
    figsize=(14, 7),
    subplot_kw={"projection": ccrs.PlateCarree()},
)

ax.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())

im = ax.pcolormesh(
    t2m.longitude,
    t2m.latitude,
    t2m.values,
    transform=ccrs.PlateCarree(),
    cmap="RdYlBu_r",
    vmin=-40,
    vmax=45,
)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
gl.top_labels = False
gl.right_labels = False

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.04)
cbar.set_label("2 m temperature (°C)")

ax.set_title("ERA5 2 m Temperature — North Atlantic domain — 2025-06-03 12:00 UTC", fontsize=13)

fig.tight_layout()
fig.savefig(
    paths.images_path / "12_t2m_domain.png",
    dpi=150, bbox_inches="tight",
)
plt.show()
```

```{figure} ../../output/images/12_t2m_domain.png
:name: fig-12-t2m-domain
ERA5 2 m temperature on 2025-06-03 at 12:00 UTC on a Plate Carrée projection
showing the North Atlantic / European domain (90°W–40°E, 20°N–80°N).  The
rectangular grid makes it easy to read off latitudes and longitudes directly.
```

+++

## Orthographic — North Atlantic (NAO-style)

An Orthographic projection centred over the North Atlantic (10°W, 55°N) —
the same viewpoint used for NAO correlation maps.  This perspective places
Iceland and the Azores, the two NAO reference stations, near the centre of
the frame.

```{code-cell} python
fig, ax = plt.subplots(
    figsize=(10, 8),
    subplot_kw={"projection": ccrs.Orthographic(central_longitude=-10, central_latitude=55)},
)

ax.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())

im = ax.pcolormesh(
    t2m.longitude,
    t2m.latitude,
    t2m.values,
    transform=ccrs.PlateCarree(),
    cmap="RdYlBu_r",
    vmin=-40,
    vmax=45,
)
ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)

cbar = fig.colorbar(
    im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.04, shrink=0.85
)
cbar.set_label("2 m temperature (°C)")

ax.set_title(
    "ERA5 2 m Temperature — Orthographic (North Atlantic) — 2025-06-03 12:00 UTC",
    fontsize=12,
)

fig.tight_layout()
fig.savefig(
    paths.images_path / "12_t2m_orthographic_nao.png",
    dpi=150, bbox_inches="tight",
)
plt.show()
```

```{figure} ../../output/images/12_t2m_orthographic_nao.png
:name: fig-12-t2m-orthographic-nao
ERA5 2 m temperature on 2025-06-03 at 12:00 UTC on an Orthographic projection
centred over the North Atlantic (10°W, 55°N).  This is the same viewpoint
used for NAO correlation maps; the spherical perspective highlights the
curvature of the temperature gradient across the Atlantic storm track.
```

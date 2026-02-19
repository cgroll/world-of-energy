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
# # Pressure Levels and Geopotential Height
#
# In weather and climate science, the atmosphere is sliced into **pressure
# levels** — imaginary surfaces where air pressure has the same value
# everywhere on that surface.  Standard levels include 1000, 925, 850, 700,
# 500, 300, and 250 hPa.
#
# A key insight is that these pressure surfaces are **not flat**.  Where the
# air column is warm (and therefore expanded), a given pressure level sits
# higher up; where the air is cold (compressed), it sits lower.  The height
# of a pressure surface is called its **geopotential height** and is measured
# in metres.
#
# This notebook downloads ERA5 geopotential data from Google Cloud
# (ARCO-ERA5) and visualises the undulations of pressure surfaces along
# a west–east transect at 55°N — across the North Sea, northern Germany,
# Poland, and the Baltic.

# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from woe.paths import ProjPaths

paths = ProjPaths()

# %% [markdown]
# ## Download ERA5 geopotential from Google Cloud
#
# The ARCO-ERA5 dataset on Google Cloud Storage contains geopotential on all
# 37 pressure levels at 0.25° resolution.  We open the dataset lazily and
# download only the slice we need: a single time step, selected pressure
# levels, along a narrow latitude band.

# %%
ds = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    chunks={"time": 48},
    consolidated=True,
    storage_options={"token": "anon"},
)
print(f"Dataset dimensions: {dict(ds.dims)}")
print(f"Available levels:   {ds.level.values.tolist()}")

# %%
# Select geopotential at the tropospheric levels we care about
levels = [1000, 925, 850, 700, 500, 300, 250]
time_sel = "2025-06-03T12:00:00"

geopot = (
    ds["geopotential"]
    .sel(time=time_sel, level=levels)
    .compute()
)

# Convert geopotential (m²/s²) to geopotential height (m)
gph = geopot / 9.80665

print(f"Downloaded shape: {dict(gph.sizes)}")
print(f"Downloaded size:  {gph.nbytes / 1e6:.1f} MB")

# %% [markdown]
# ## Extract the 55°N transect
#
# We select the latitude band closest to 55°N and extract a west–east
# cross-section from −10°E (Atlantic) to 40°E (western Russia).  This
# crosses the North Sea, Denmark/northern Germany, Poland, the Baltic
# states, and Belarus.

# %%
# ERA5 longitudes are 0..360; convert our desired range
lon_min, lon_max = -10, 40
lat_sel = 55.0

# Select nearest latitude
transect = gph.sel(latitude=lat_sel, method="nearest")

# Handle longitude wrapping: ERA5 uses 0..360
lon_values = transect.longitude.values
# Convert to -180..180 for selection
lon_180 = np.where(lon_values > 180, lon_values - 360, lon_values)

# Create a mask for our longitude range
lon_mask = (lon_180 >= lon_min) & (lon_180 <= lon_max)
transect = transect.isel(longitude=lon_mask)

# Build a clean longitude array in degrees east for plotting
plot_lons = np.where(
    transect.longitude.values > 180,
    transect.longitude.values - 360,
    transect.longitude.values,
)

# Sort by longitude so the transect runs west→east (the mask may have
# selected 350..360 and 0..40 as two disjoint chunks in the original
# 0..360 coordinate, leaving the negative lons at the end).
sort_idx = np.argsort(plot_lons)
plot_lons = plot_lons[sort_idx]
transect = transect.isel(longitude=sort_idx)

print(f"Transect shape:     {dict(transect.sizes)}")
print(f"Longitude range:    {plot_lons.min():.2f}°E to {plot_lons.max():.2f}°E")
print(f"Actual latitude:    {float(transect.latitude):.2f}°N")

# %% [markdown]
# ## What do the pressure surfaces look like?
#
# Each pressure level is a line that undulates up and down.  The variations
# are small compared to the absolute height (e.g. the 500 hPa surface
# averages ~5500 m), but they are meteorologically significant: a 100 m dip
# in the 500 hPa surface signals a trough (low-pressure system), while a
# rise signals a ridge (high-pressure system).
#
# Let's first plot each pressure level as a simple line.

# %%
fig, ax = plt.subplots(figsize=(14, 8))

cmap = plt.cm.viridis_r
colors = [cmap(i / (len(levels) - 1)) for i in range(len(levels))]

for i, level in enumerate(levels):
    height = transect.sel(level=level).values
    ax.plot(plot_lons, height, color=colors[i], linewidth=2,
            label=f"{level} hPa ({np.mean(height):.0f} m avg)")

ax.set_xlabel("Longitude (°E)", fontsize=12)
ax.set_ylabel("Geopotential height (m)", fontsize=12)
ax.set_title(
    f"Pressure surfaces along {float(transect.latitude):.0f}°N — "
    f"2025-06-03 12:00 UTC",
    fontsize=14,
)
ax.legend(loc="center left", fontsize=10)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(
    paths.images_path / "11_pressure_level_lines.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/11_pressure_level_lines.png
# :name: fig-11-pressure-level-lines
# Geopotential height of seven pressure surfaces along 55°N.  Higher pressure
# levels (1000 hPa) are near the surface; lower pressures (250 hPa) are in the
# upper troposphere.
# ```

# %% [markdown]
# The plot above shows the full vertical extent, but the undulations within
# each level are hard to see because the spacing between levels dominates.
# Next we zoom into individual levels to reveal the structure.

# %%
fig, axes = plt.subplots(len(levels), 1, figsize=(14, 2.5 * len(levels)),
                         sharex=True)

for i, level in enumerate(levels):
    ax = axes[i]
    height = transect.sel(level=level).values
    ax.fill_between(plot_lons, height, alpha=0.3, color=colors[i])
    ax.plot(plot_lons, height, color=colors[i], linewidth=1.5)
    ax.set_ylabel("Height (m)", fontsize=9)
    ax.set_title(f"{level} hPa", fontsize=11, loc="left", fontweight="bold")
    ax.grid(True, alpha=0.3)
    # Let matplotlib auto-scale to show the undulations
    margin = (height.max() - height.min()) * 0.15
    if margin < 5:
        margin = 20
    ax.set_ylim(height.min() - margin, height.max() + margin)

axes[-1].set_xlabel("Longitude (°E)", fontsize=12)
fig.suptitle(
    f"Pressure surface undulations along {float(transect.latitude):.0f}°N — "
    f"2025-06-03 12:00 UTC",
    fontsize=14, y=1.01,
)

fig.tight_layout()
fig.savefig(
    paths.images_path / "11_pressure_level_panels.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/11_pressure_level_panels.png
# :name: fig-11-pressure-level-panels
# Each panel shows the geopotential height variation of one pressure level
# along 55°N, with y-axis zoomed to reveal the undulations.  Dips correspond
# to troughs (colder air), rises to ridges (warmer air).
# ```

# %% [markdown]
# ## Stacked area chart — the atmosphere as coloured bands
#
# The most intuitive way to see pressure levels is as **layers of the
# atmosphere**.  We fill the area between consecutive pressure surfaces,
# creating a cross-sectional view of the troposphere.  The y-axis is true
# altitude in metres, and the bands reveal how the layer thicknesses change
# along the transect.
#
# **Thicker layers mean warmer air** (warm air expands, pushing the upper
# pressure surface higher), while **thinner layers mean colder air**.

# %%
fig, ax = plt.subplots(figsize=(14, 8))

# Levels are ordered from highest pressure (lowest altitude) to lowest pressure
# (highest altitude).  We fill between consecutive levels.
band_cmap = plt.cm.coolwarm
n_bands = len(levels) - 1
band_colors = [band_cmap(i / (n_bands - 1)) for i in range(n_bands)]

for i in range(n_bands):
    lower_level = levels[i]       # higher pressure = lower altitude
    upper_level = levels[i + 1]   # lower pressure = higher altitude
    lower_height = transect.sel(level=lower_level).values
    upper_height = transect.sel(level=upper_level).values

    color = to_rgba(band_colors[i], alpha=0.6)
    ax.fill_between(
        plot_lons, lower_height, upper_height,
        color=color,
        label=f"{lower_level}–{upper_level} hPa",
    )

# Draw the pressure surface lines on top
for i, level in enumerate(levels):
    height = transect.sel(level=level).values
    ax.plot(plot_lons, height, color="black", linewidth=0.8, alpha=0.7)
    # Label on the right edge
    ax.text(
        plot_lons[-1] + 0.5, height[-1],
        f" {level} hPa",
        va="center", fontsize=9, fontweight="bold",
    )

ax.set_xlabel("Longitude (°E)", fontsize=12)
ax.set_ylabel("Geopotential height (m)", fontsize=12)
ax.set_title(
    f"Atmospheric layers along {float(transect.latitude):.0f}°N — "
    f"2025-06-03 12:00 UTC",
    fontsize=14,
)
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax.set_xlim(plot_lons.min(), plot_lons.max() + 4)  # extra space for labels
ax.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig(
    paths.images_path / "11_pressure_level_bands.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/11_pressure_level_bands.png
# :name: fig-11-pressure-level-bands
# Cross-section of the troposphere along 55°N.  Each coloured band represents
# the layer between two pressure surfaces.  Thicker bands indicate warmer
# (expanded) air; thinner bands indicate colder (compressed) air.  The black
# lines are the actual geopotential height of each pressure level.
# ```

# %% [markdown]
# ## Layer thickness and temperature
#
# The relationship between layer thickness and temperature is not a
# coincidence — it follows directly from the **hypsometric equation**:
#
# $$
# \Delta z = \frac{R_d \, \bar{T}}{g} \ln\frac{p_{\text{lower}}}{p_{\text{upper}}}
# $$
#
# where $R_d$ is the gas constant for dry air (287 J/kg/K), $\bar{T}$ is
# the mean virtual temperature of the layer, $g$ is gravitational
# acceleration, and $p$ is pressure.  A warmer layer is thicker.
#
# Let's verify this by downloading the temperature field and comparing
# layer-mean temperature with layer thickness along our transect.

# %%
temp = (
    ds["temperature"]
    .sel(time=time_sel, level=levels)
    .sel(latitude=lat_sel, method="nearest")
    .isel(longitude=lon_mask)
    .compute()
)

print(f"Temperature shape: {dict(temp.sizes)}")

# %%
# Pick the 1000–500 hPa layer (a classic thickness measure)
z_1000 = transect.sel(level=1000).values
z_500 = transect.sel(level=500).values
thickness_1000_500 = z_500 - z_1000

# Mean temperature in K across levels between 1000 and 500 hPa
mid_levels = [l for l in levels if 500 <= l <= 1000]
t_mean = temp.sel(level=mid_levels).mean(dim="level").values

fig, ax1 = plt.subplots(figsize=(14, 5))

color_thick = "tab:blue"
color_temp = "tab:red"

ax1.plot(plot_lons, thickness_1000_500, color=color_thick, linewidth=2,
         label="1000–500 hPa thickness")
ax1.set_xlabel("Longitude (°E)", fontsize=12)
ax1.set_ylabel("Thickness (m)", fontsize=12, color=color_thick)
ax1.tick_params(axis="y", labelcolor=color_thick)

ax2 = ax1.twinx()
ax2.plot(plot_lons, t_mean - 273.15, color=color_temp, linewidth=2,
         linestyle="--", label="Mean temperature")
ax2.set_ylabel("Mean temperature (°C)", fontsize=12, color=color_temp)
ax2.tick_params(axis="y", labelcolor=color_temp)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=10)

ax1.set_title(
    f"1000–500 hPa thickness vs. mean temperature along "
    f"{float(transect.latitude):.0f}°N — 2025-06-03 12:00 UTC",
    fontsize=13,
)
ax1.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(
    paths.images_path / "11_thickness_vs_temperature.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/11_thickness_vs_temperature.png
# :name: fig-11-thickness-vs-temperature
# The 1000–500 hPa thickness (blue) closely tracks the mean layer temperature
# (red dashed), confirming the hypsometric equation: warmer air columns are
# thicker.
# ```

# %% [markdown]
# ## Why this matters for energy
#
# Pressure levels and geopotential heights are not just abstract meteorology —
# they directly affect energy systems:
#
# - **Wind energy**: The jet stream (250–300 hPa) steers surface weather
#   systems.  Large geopotential height gradients at 500 hPa indicate strong
#   pressure differences at the surface, which drive wind.
# - **Solar energy**: Troughs in the geopotential field are associated with
#   cloud cover and precipitation, reducing solar irradiance.  Ridges bring
#   clear skies.
# - **Temperature-driven demand**: The thickness of the lower atmosphere
#   (1000–850 hPa) is a proxy for surface temperature, which drives heating
#   and cooling demand.
# - **Forecasting**: Numerical weather prediction models work on pressure
#   levels.  Understanding the vertical structure of the atmosphere is
#   essential for interpreting weather forecasts that drive energy trading.

# %%

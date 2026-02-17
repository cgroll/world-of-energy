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
# # Sunlight on Earth — Animated Day Cycle
#
# This script creates a GIF animation showing where sunlight hits the Earth
# over 24 hours on a day in June (near summer solstice). Each frame is one
# hour. The animation uses a satellite background image (Blue Marble) with
# cartopy's `Nightshade` to shade the dark side.

# %%
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature.nightshade import Nightshade
from matplotlib.animation import FuncAnimation, PillowWriter

from woe.paths import ProjPaths

paths = ProjPaths()

# %% [markdown]
# ## Configuration
#
# We create two animations — summer solstice (2025-06-21) and winter solstice
# (2025-12-21) — to contrast the extremes of Northern Hemisphere daylight.

# %%
dates = [datetime(2025, 6, 21), datetime(2025, 12, 21)]
n_frames = 64
frame_hours = np.linspace(0, 24, n_frames, endpoint=False)

# %% [markdown]
# ## Build the animations
#
# For each date and each frame we:
# 1. Draw the Blue Marble satellite background
# 2. Overlay the Nightshade (dark side of Earth)
# 3. Mark the subsolar point and declination latitude


# %%
def create_sunlight_animation(anim_date, output_path):
    fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={"projection": ccrs.Robinson()})
    fig.patch.set_facecolor("black")

    def draw_frame(fractional_hour):
        ax.clear()
        h = int(fractional_hour)
        m = int((fractional_hour - h) * 60)
        obs_time = datetime(anim_date.year, anim_date.month, anim_date.day, h, m, 0)

        # Solar geometry
        day_of_year = obs_time.timetuple().tm_yday
        declination = 23.44 * np.sin(np.radians(360 / 365 * (day_of_year - 81)))
        hours_utc = obs_time.hour + obs_time.minute / 60
        subsolar_lon = 180 - (hours_utc / 24) * 360

        # Layer 1: satellite background
        ax.stock_img()

        # Layer 2: nightshade
        ax.add_feature(Nightshade(obs_time, alpha=0.4))

        # Layer 3: subsolar point marker
        ax.plot(
            subsolar_lon, declination,
            marker="*", markersize=18, color="yellow", markeredgecolor="orange",
            markeredgewidth=0.8, transform=ccrs.PlateCarree(), zorder=10,
        )

        # Layer 4: declination latitude line
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
        ax.set_facecolor("black")
        ax.set_title(
            f"Sunlight on Earth — {obs_time:%Y-%m-%d %H:%M} UTC — "
            f"Subsolar point {declination:.1f}°N, {subsolar_lon:.1f}°E",
            fontsize=13, color="white", pad=12,
        )

    anim = FuncAnimation(fig, draw_frame, frames=list(frame_hours), interval=1000 / 15)
    anim.save(str(output_path), writer=PillowWriter(fps=15), dpi=120, savefig_kwargs={"facecolor": "black"})
    plt.close(fig)
    print(f"Saved animation to {output_path}")


for d in dates:
    suffix = f"{d:%Y_%m_%d}"
    create_sunlight_animation(d, paths.images_path / f"09_sunlight_animation_{suffix}.gif")

# %% [markdown]
# ```{figure} ../../output/images/09_sunlight_animation_2025_06_21.gif
# :name: fig-09-sunlight-summer
# Sunlight on Earth over 24 hours on 2025-06-21 (summer solstice). The yellow
# star marks the subsolar point; the dashed line shows the solar declination
# latitude (~23.4°N).
# ```
#
# ```{figure} ../../output/images/09_sunlight_animation_2025_12_21.gif
# :name: fig-09-sunlight-winter
# Sunlight on Earth over 24 hours on 2025-12-21 (winter solstice). The subsolar
# point is now at ~23.4°S, showing the opposite extreme of daylight distribution.
# ```

# %% [markdown]
# ## Solar elevation heatmaps
#
# The **solar elevation** (altitude) is the angle of the sun above the horizon
# (0° at sunrise/sunset, 90° when directly overhead). We compute it for every
# day of the year and every minute of the day using standard solar position
# equations, comparing Munich (48.14°N) and Sydney (33.87°S).

# %%
locations = [
    {"name": "Munich", "lat": 48.14, "lon": 11.58, "utc_offset": 1, "tz_label": "CET"},
    {"name": "Sydney", "lat": -33.87, "lon": 151.21, "utc_offset": 11, "tz_label": "AEDT"},
]

days = np.arange(1, 366)
hours_of_day = np.arange(0, 24, 1 / 60)  # one-minute resolution
day_grid, hour_grid = np.meshgrid(days, hours_of_day)

# Solar declination (same for all locations)
declination_rad = np.radians(
    23.44 * np.sin(np.radians(360 / 365 * (day_grid - 81)))
)

month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

for loc in locations:
    lat_rad = np.radians(loc["lat"])

    # Y-axis in local time; compute solar position in UTC
    local_hours = hours_of_day
    utc_hours = local_hours - loc["utc_offset"]
    day_grid_loc, utc_hour_grid = np.meshgrid(days, utc_hours)
    _, local_hour_grid = np.meshgrid(days, local_hours)

    # Hour angle: sun's angular displacement from solar noon
    solar_hour = utc_hour_grid + loc["lon"] / 15
    hour_angle_rad = np.radians(15 * (solar_hour - 12))

    # Solar elevation (altitude)
    sin_elev = (
        np.sin(lat_rad) * np.sin(declination_rad)
        + np.cos(lat_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad)
    )
    elevation_deg = np.degrees(np.arcsin(np.clip(sin_elev, -1, 1)))
    elevation_deg = np.where(elevation_deg < 0, np.nan, elevation_deg)

    fig, ax = plt.subplots(figsize=(14, 6))

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("0.15")

    im = ax.pcolormesh(
        day_grid_loc, local_hour_grid, elevation_deg,
        cmap=cmap, vmin=0, vmax=70, shading="auto",
    )

    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Solar elevation (°)", fontsize=11)

    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels)
    ax.set_yticks(range(0, 25, 3))
    ax.set_ylabel(f"Hour of day ({loc['tz_label']}, UTC{loc['utc_offset']:+d})", fontsize=11)
    ax.set_xlabel("Day of year", fontsize=11)
    ax.set_title(
        f"Solar Elevation — {loc['name']} ({abs(loc['lat']):.1f}°{'N' if loc['lat'] >= 0 else 'S'}, "
        f"{abs(loc['lon']):.1f}°{'E' if loc['lon'] >= 0 else 'W'})",
        fontsize=13,
    )

    fig.tight_layout()
    filename = f"09_elevation_heatmap_{loc['name'].lower()}.png"
    fig.savefig(paths.images_path / filename, dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ```{figure} ../../output/images/09_elevation_heatmap_munich.png
# :name: fig-09-elevation-munich
# Solar elevation at Munich (48.1°N, 11.6°E) for every day and minute of the year.
# The elevation is the angle of the sun above the horizon (0° at sunrise/sunset,
# up to ~65° at summer solstice noon). Grey areas indicate nighttime.
# ```
#
# ```{figure} ../../output/images/09_elevation_heatmap_sydney.png
# :name: fig-09-elevation-sydney
# Solar elevation at Sydney (33.9°S, 151.2°E). Being in the Southern Hemisphere,
# Sydney's seasons are inverted — longest days around December, shortest in June.
# The higher maximum elevation (~80°) reflects its lower latitude compared to Munich.
# ```

# %% [markdown]
# ## Subsolar point migration over the year
#
# The Earth's axial tilt causes the subsolar latitude to oscillate between
# ~23.4°N (June solstice) and ~23.4°S (December solstice). This animation
# shows one frame per week, tracing the subsolar declination line across the
# year at solar noon (12:00 UTC).

# %%
weeks = np.arange(0, 52)
week_dates = [datetime(2025, 1, 1) + timedelta(weeks=int(w)) for w in weeks]

fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={"projection": ccrs.Robinson()})
fig.patch.set_facecolor("black")


def draw_subsolar_frame(i):
    ax.clear()
    d = week_dates[i]
    obs_time = datetime(d.year, d.month, d.day, 12, 0, 0)

    day_of_year = obs_time.timetuple().tm_yday
    declination = 23.44 * np.sin(np.radians(360 / 365 * (day_of_year - 81)))

    # Subsolar longitude at 12:00 UTC = 0°E
    subsolar_lon = 0.0

    ax.stock_img()
    ax.add_feature(Nightshade(obs_time, alpha=0.3))

    # Declination latitude line
    ax.plot(
        [-180, 180], [declination, declination],
        color="yellow", linewidth=2, linestyle="--", alpha=0.9,
        transform=ccrs.PlateCarree(), zorder=9,
    )

    # Subsolar point
    ax.plot(
        subsolar_lon, declination,
        marker="*", markersize=20, color="yellow", markeredgecolor="orange",
        markeredgewidth=0.8, transform=ccrs.PlateCarree(), zorder=10,
    )

    # Tropic lines for reference
    for tropic_lat, label in [(23.44, "Tropic of Cancer"), (-23.44, "Tropic of Capricorn")]:
        ax.plot(
            [-180, 180], [tropic_lat, tropic_lat],
            color="white", linewidth=0.5, linestyle=":", alpha=0.4,
            transform=ccrs.PlateCarree(), zorder=8,
        )

    # Equator
    ax.plot(
        [-180, 180], [0, 0],
        color="white", linewidth=0.6, linestyle=":", alpha=0.5,
        transform=ccrs.PlateCarree(), zorder=8,
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="0.3")
    ax.set_global()
    ax.set_facecolor("black")
    ax.set_title(
        f"Subsolar Latitude — {obs_time:%Y-%m-%d} (week {i + 1}/52) — "
        f"Declination {declination:.1f}°{'N' if declination >= 0 else 'S'}",
        fontsize=13, color="white", pad=12,
    )


anim = FuncAnimation(fig, draw_subsolar_frame, frames=len(week_dates), interval=1000 / 15)

output_file = paths.images_path / "09_subsolar_migration.gif"
anim.save(str(output_file), writer=PillowWriter(fps=15), dpi=120, savefig_kwargs={"facecolor": "black"})
plt.close(fig)
print(f"Saved animation to {output_file}")

# %% [markdown]
# ```{figure} ../../output/images/09_subsolar_migration.gif
# :name: fig-09-subsolar-migration
# Migration of the subsolar latitude over the year (one frame per week, 15 fps).
# The yellow dashed line shows where the sun is directly overhead at noon UTC.
# It oscillates between the Tropic of Cancer (~23.4°N in June) and the Tropic
# of Capricorn (~23.4°S in December).
# ```

---
jupytext:
  text_representation:
    format_name: percent
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Earth's Seasons — From Sunlight to Vegetation

Why do we have seasons? The answer lies in a single geometric fact: the
Earth's rotation axis is tilted by about **23.4°** relative to the plane of
its orbit around the Sun. As the Earth travels along its annual orbit, this
tilt causes different hemispheres to lean toward or away from the Sun,
changing both the **angle** at which sunlight strikes the surface and the
**number of hours** of daylight each location receives.

This chapter traces the chain of consequences:

1. **Sunlight geometry** — how the Sun illuminates different parts of the
   Earth during the day, and how this pattern shifts between summer and winter.
2. **The subsolar point** — the latitude where the Sun is directly overhead,
   and how it migrates between the Tropics over the year.
3. **Solar elevation** — the angle of the Sun above the horizon for any
   location and time, which determines how concentrated the incoming energy is.
4. **Temperature** — how the uneven distribution of sunlight drives the
   global temperature pattern, both over the course of a day and across seasons.
5. **Vegetation** — how the resulting temperature and moisture patterns
   control where and when vegetation grows, visible from space as a seasonal
   "green wave".

```{code-cell} python
import calendar
import math
import os
from datetime import datetime, date, timedelta

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature.nightshade import Nightshade
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image, ImageDraw

from woe.paths import ProjPaths

paths = ProjPaths()
```

## Sunlight on Earth — the day cycle

At any given moment, exactly half of the Earth is illuminated by the Sun.
But the angle at which sunlight arrives varies enormously by latitude and
time of day. Near the equator, sunlight can arrive nearly **vertically**,
concentrating its energy on a small area. Near the poles, it arrives at a
**shallow angle**, spreading the same energy over a much larger area — which
is why polar regions are cold even when they receive 24 hours of daylight in
summer.

The two animations below show the illumination pattern over 24 hours on
the **summer solstice** (June 21) and the **winter solstice** (December 21).
The contrast is striking: in June, the North Pole enjoys continuous daylight
while the South Pole is in complete darkness. In December, the situation is
reversed.

```{code-cell} python
dates = [datetime(2025, 6, 21), datetime(2025, 12, 21)]
n_frames = 64
frame_hours = np.linspace(0, 24, n_frames, endpoint=False)


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

        ax.stock_img()
        ax.add_feature(Nightshade(obs_time, alpha=0.4))

        # Subsolar point marker
        ax.plot(
            subsolar_lon, declination,
            marker="*", markersize=18, color="yellow", markeredgecolor="orange",
            markeredgewidth=0.8, transform=ccrs.PlateCarree(), zorder=10,
        )

        # Declination latitude line
        ax.plot(
            [-180, 180], [declination, declination],
            color="yellow", linewidth=1.2, linestyle="--", alpha=0.7,
            transform=ccrs.PlateCarree(), zorder=9,
        )
        # Equator
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
    create_sunlight_animation(d, paths.images_path / f"10_sunlight_animation_{suffix}.gif")
```

```{figure} ../../output/images/10_sunlight_animation_2025_06_21.gif
:name: fig-10-sunlight-summer
Sunlight on Earth over 24 hours on 2025-06-21 (summer solstice). The yellow
star marks the subsolar point; the dashed line shows the solar declination
latitude (~23.4°N). Notice that the Arctic never enters darkness.
```

```{figure} ../../output/images/10_sunlight_animation_2025_12_21.gif
:name: fig-10-sunlight-winter
Winter solstice (2025-12-21). The subsolar point has moved to ~23.4°S.
Now it is the Antarctic that enjoys continuous daylight, while the Arctic
remains in polar night.
```

+++

## The subsolar point and its annual migration

The **subsolar point** is the location on Earth where the Sun is directly
overhead — i.e. the solar elevation is exactly 90°. Its latitude equals the
**solar declination**, which can be approximated as:

$$
\delta = 23.44° \times \sin\!\left(\frac{360°}{365} \times (d - 81)\right)
$$

where $d$ is the day of the year (so $d = 81$ corresponds to the spring
equinox around March 22, when $\delta = 0$).

Over the course of a year, the subsolar latitude oscillates between the
**Tropic of Cancer** (~23.4°N, around June 21) and the **Tropic of
Capricorn** (~23.4°S, around December 21). The animation below shows one
frame per week, tracing this migration at solar noon (12:00 UTC).

```{code-cell} python
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
    subsolar_lon = 0.0  # at 12:00 UTC

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

    # Tropic lines
    for tropic_lat in [23.44, -23.44]:
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
output_file = paths.images_path / "10_subsolar_migration.gif"
anim.save(str(output_file), writer=PillowWriter(fps=15), dpi=120, savefig_kwargs={"facecolor": "black"})
plt.close(fig)
print(f"Saved animation to {output_file}")
```

```{figure} ../../output/images/10_subsolar_migration.gif
:name: fig-10-subsolar-migration
Migration of the subsolar latitude over the year (one frame per week). The
yellow dashed line shows where the Sun is directly overhead at noon UTC. It
oscillates between the Tropic of Cancer (~23.4°N in June) and the Tropic of
Capricorn (~23.4°S in December).
```

+++

## Solar elevation — the Sun's angle above the horizon

The **solar elevation** (or altitude angle) is the angle of the Sun above
the horizon, measured from 0° (sunrise/sunset) to 90° (directly overhead).
It determines how much energy a square metre of surface receives: a high
Sun concentrates light into a small area, while a low Sun spreads it thin.

For any location (latitude $\varphi$) and time, the elevation $\alpha$ is:

$$
\sin(\alpha) = \sin(\varphi)\,\sin(\delta) + \cos(\varphi)\,\cos(\delta)\,\cos(h)
$$

where $\delta$ is the solar declination (computed above) and $h$ is the
**hour angle** — the Sun's angular displacement from local solar noon
($h = 0$ at noon, 15° per hour).

The heatmaps below show solar elevation for every day of the year and every
minute of the day, for two cities at very different latitudes: **Munich**
(48.1°N) and **Sydney** (33.9°S). Grey areas indicate nighttime (elevation
below 0°).

```{code-cell} python
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

    local_hours = hours_of_day
    utc_hours = local_hours - loc["utc_offset"]
    day_grid_loc, utc_hour_grid = np.meshgrid(days, utc_hours)
    _, local_hour_grid = np.meshgrid(days, local_hours)

    # Hour angle: sun's angular displacement from solar noon
    solar_hour = utc_hour_grid + loc["lon"] / 15
    hour_angle_rad = np.radians(15 * (solar_hour - 12))

    # Solar elevation
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
    filename = f"10_elevation_heatmap_{loc['name'].lower()}.png"
    fig.savefig(paths.images_path / filename, dpi=150, bbox_inches="tight")
    plt.show()
```

```{figure} ../../output/images/10_elevation_heatmap_munich.png
:name: fig-10-elevation-munich
Solar elevation at Munich (48.1°N, 11.6°E) for every day and minute of
the year. In summer, the Sun reaches ~65° and daylight lasts over 16 hours.
In winter, the maximum elevation drops to ~18° with less than 8 hours of
daylight. Grey areas indicate nighttime.
```

```{figure} ../../output/images/10_elevation_heatmap_sydney.png
:name: fig-10-elevation-sydney
Solar elevation at Sydney (33.9°S, 151.2°E). Being in the Southern
Hemisphere, Sydney's seasons are inverted — longest days around December,
shortest in June. The higher maximum elevation (~80°) reflects its lower
latitude compared to Munich, and the seasonal daylight variation is less
extreme.
```

+++

## Global temperatures — the diurnal cycle

The varying sunlight intensity directly drives temperature patterns across
the globe. To observe this, we use **ERA5 reanalysis data** — a global
gridded dataset produced by the European Centre for Medium-Range Weather
Forecasts (ECMWF) that combines weather observations with numerical models.
The variable shown is **2-metre temperature** (the air temperature at 2 m
above the surface), which is the standard measure of surface air temperature.

A key feature visible in the diurnal animation is the difference between
**land and ocean**: land heats and cools much faster than water due to its
lower heat capacity. This is why continents show a large day-night temperature
swing while ocean temperatures barely change over 24 hours. This also
explains why coastal cities have milder climates than continental interiors.

```{code-cell} python
import ee
import geemap
from dotenv import load_dotenv

load_dotenv()
ee.Initialize(project=os.environ["EE_GCP_PROJECT_ID"])
```

```{code-cell} python
era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").select("temperature_2m")

start = "2024-01-01"
end = "2025-01-01"
era5_year = era5.filterDate(start, end)

# Build a 24-image collection: one mean image per UTC hour
hourly_images = []
for hour in range(24):
    hourly_mean = (
        era5_year
        .filter(ee.Filter.calendarRange(hour, hour, "hour"))
        .mean()
        .subtract(273.15)  # K -> °C
    )
    hourly_mean = hourly_mean.set("hour", hour).set("label", f"{hour:02d}:00 UTC")
    hourly_images.append(hourly_mean)

col = ee.ImageCollection(hourly_images)
print(f"Collection size: {col.size().getInfo()} images")
```

```{code-cell} python
vis_params = {
    "min": -40,
    "max": 40,
    "palette": [
        "#313695", "#4575b4", "#74add1", "#abd9e9",
        "#e0f3f8", "#ffffbf", "#fee090", "#fdae61",
        "#f46d43", "#d73027", "#a50026",
    ],
    "dimensions": 800,
    "framesPerSecond": 3,
}

gif_path = str(paths.images_path / "10_ee_temperature_by_hour.gif")

geemap.download_ee_video(col, vis_params, gif_path)
print(f"GIF saved to {gif_path}")
```

```{code-cell} python
labels = [f"{h:02d}:00 UTC" for h in range(24)]

geemap.add_text_to_gif(
    gif_path,
    gif_path,
    xy=("5%", "90%"),
    text_sequence=labels,
    font_size=20,
    font_color="white",
    add_progress_bar=True,
    progress_bar_color="cyan",
    progress_bar_height=5,
)
print(f"Annotated GIF saved to {gif_path}")
```

```{figure} ../../output/images/10_ee_temperature_by_hour.gif
:name: fig-10-temperature-diurnal
Average 2-metre temperature by hour of day (UTC), computed from a full year
of ERA5 reanalysis data. The "heat wave" sweeps westward as the Sun moves,
and the land-sea contrast is clearly visible: continents show a large
day-night swing, while oceans remain nearly constant.
```

+++

## Global temperatures — the seasonal cycle

Over the course of months, the sustained difference in sunlight between
hemispheres creates the familiar **seasons**. Note that temperature extremes
typically lag the solstices by about 4–6 weeks — this **thermal lag** occurs
because the Earth's surface and atmosphere take time to heat up and cool
down, similar to how the hottest part of the day is usually mid-afternoon
rather than noon.

The ocean again plays a moderating role: maritime and coastal climates have
milder seasonal swings, while continental interiors (like Siberia or central
Canada) experience the most extreme temperature differences between summer
and winter.

```{code-cell} python
monthly_images = []
monthly_labels = []
for month in range(1, 13):
    month_slices = []
    for year in range(2022, 2025):
        days_in_month = calendar.monthrange(year, month)[1]
        m_start = f"{year}-{month:02d}-01"
        m_end = f"{year}-{month:02d}-{days_in_month:02d}T23:59"
        month_slices.append(
            era5.filterDate(m_start, m_end)
            .filter(ee.Filter.calendarRange(12, 12, "hour"))
        )
    merged = month_slices[0]
    for s in month_slices[1:]:
        merged = merged.merge(s)
    monthly_mean = merged.mean().subtract(273.15)
    label = calendar.month_abbr[month]
    monthly_mean = monthly_mean.set("month", month).set("label", label)
    monthly_images.append(monthly_mean)
    monthly_labels.append(label)

col_monthly = ee.ImageCollection(monthly_images)
print(f"Monthly collection size: {col_monthly.size().getInfo()} images")
```

```{code-cell} python
vis_params_monthly = {
    "min": -40,
    "max": 40,
    "palette": [
        "#313695", "#4575b4", "#74add1", "#abd9e9",
        "#e0f3f8", "#ffffbf", "#fee090", "#fdae61",
        "#f46d43", "#d73027", "#a50026",
    ],
    "dimensions": 600,
    "framesPerSecond": 1,
}

gif_monthly_path = str(paths.images_path / "10_ee_temperature_by_month.gif")

geemap.download_ee_video(col_monthly, vis_params_monthly, gif_monthly_path)
print(f"Monthly GIF saved to {gif_monthly_path}")
```

```{code-cell} python
geemap.add_text_to_gif(
    gif_monthly_path,
    gif_monthly_path,
    xy=("5%", "90%"),
    text_sequence=monthly_labels,
    font_size=20,
    font_color="white",
    add_progress_bar=True,
    progress_bar_color="cyan",
    progress_bar_height=5,
)
print(f"Annotated monthly GIF saved to {gif_monthly_path}")
```

```{figure} ../../output/images/10_ee_temperature_by_month.gif
:name: fig-10-temperature-seasonal
Average 2-metre temperature by month, computed from ERA5 noon snapshots
across 2022–2024. The hemispheric temperature contrast shifts as the
subsolar latitude migrates. Continental interiors show the most extreme
seasonal swings, while oceans remain comparatively stable.
```

+++

## Vegetation — the seasonal green wave

The temperature and moisture patterns driven by the Sun's position have a
direct and visible impact on vegetation. We can observe this from space
using the **Normalized Difference Vegetation Index (NDVI)**, computed from
satellite imagery as:

$$
\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}
$$

where NIR is near-infrared reflectance and Red is visible red reflectance.
Healthy green vegetation **strongly reflects** near-infrared light (which
it doesn't need) and **absorbs** red light (which it uses for
photosynthesis). This makes NDVI a reliable indicator of vegetation health:

- **NDVI > 0.6** — dense, healthy vegetation (tropical forests, croplands
  in growing season)
- **NDVI 0.2–0.6** — moderate vegetation (grasslands, sparse forests)
- **NDVI < 0.1** — bare soil, water, snow, or desert

We use **MODIS Terra** 16-day NDVI composites (MOD13A2, 1 km resolution)
to build a multi-year median for each 16-day period, producing ~23 frames
that reveal the seasonal "green wave". The wave tracks the subsolar latitude
— as warmth and moisture arrive, vegetation pulses to life. We overlay the
subsolar declination line (yellow dashed) to make this connection visible.

```{code-cell} python
ndvi_col = ee.ImageCollection("MODIS/061/MOD13A2").select("NDVI")

ndvi_col = ndvi_col.map(
    lambda img: img.set(
        "doy", ee.Date(img.get("system:time_start")).getRelative("day", "year")
    )
)

# Use one year as the DOY template (23 composites, every 16 days)
distinct_doy = ndvi_col.filterDate("2020-01-01", "2021-01-01")

# Join all years by matching DOY, then take the median
join_filter = ee.Filter.equals(leftField="doy", rightField="doy")
join = ee.Join.saveAll("doy_matches")
joined = ee.ImageCollection(join.apply(distinct_doy, ndvi_col, join_filter))

ndvi_composites = joined.map(
    lambda img: (
        ee.ImageCollection.fromImages(img.get("doy_matches"))
        .reduce(ee.Reducer.median())
        .rename("NDVI")
        .copyProperties(img, ["doy", "system:time_start"])
    )
)

print(f"NDVI composites: {ndvi_composites.size().getInfo()} frames")
```

```{code-cell} python
ndvi_vis = {
    "min": 0,
    "max": 9000,
    "palette": [
        "FFFFFF", "CE7E45", "DF923D", "F1B555", "FCD163", "99B718", "74A901",
        "66A000", "529400", "3E8601", "207401", "056201", "004C00", "023B01",
        "012E01", "011D01", "011301",
    ],
}

ndvi_rgb = ndvi_composites.map(lambda img: img.visualize(**ndvi_vis))

gif_ndvi_params = {
    "dimensions": 600,
    "framesPerSecond": 4,
}

gif_ndvi_path = str(paths.images_path / "10_ee_ndvi_by_doy.gif")

geemap.download_ee_video(ndvi_rgb, gif_ndvi_params, gif_ndvi_path)
print(f"NDVI GIF saved to {gif_ndvi_path}")
```

```{code-cell} python
doy_list = list(range(1, 366, 16))  # 1, 17, 33, ...
ndvi_labels = [
    (date(2024, 1, 1) + timedelta(days=d - 1)).strftime("%b %d")
    for d in doy_list
]

geemap.add_text_to_gif(
    gif_ndvi_path,
    gif_ndvi_path,
    xy=("5%", "90%"),
    text_sequence=ndvi_labels,
    font_size=20,
    font_color="white",
    add_progress_bar=True,
    progress_bar_color="green",
    progress_bar_height=5,
)
print(f"Annotated NDVI GIF saved to {gif_ndvi_path}")
```

### Subsolar latitude overlay

To highlight the connection between the Sun's position and vegetation, we
draw the subsolar declination latitude as a yellow dashed line on each frame.

```{code-cell} python
global_lat_min, global_lat_max = -90, 90


def solar_declination(doy: int) -> float:
    """Solar declination in degrees for a given day-of-year."""
    return 23.44 * math.sin(math.radians(360 / 365 * (doy - 81)))


def add_subsolar_line(gif_in_path, lat_min, lat_max):
    """Add a dashed yellow subsolar-latitude line to each frame of a GIF."""
    gif = Image.open(gif_in_path)
    frames = []
    durations = []
    frame_idx = 0

    try:
        while True:
            frame = gif.copy().convert("RGBA")
            draw = ImageDraw.Draw(frame)
            w, h = frame.size

            doy = doy_list[frame_idx % len(doy_list)]
            decl = solar_declination(doy)
            y = int((lat_max - decl) / (lat_max - lat_min) * h)

            dash_len, gap_len = 12, 6
            x = 0
            while x < w:
                draw.line([(x, y), (min(x + dash_len, w), y)], fill="yellow", width=2)
                x += dash_len + gap_len

            frames.append(frame)
            durations.append(gif.info.get("duration", 250))
            frame_idx += 1
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass

    frames[0].save(
        gif_in_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )


add_subsolar_line(gif_ndvi_path, global_lat_min, global_lat_max)
print(f"Subsolar line added to {gif_ndvi_path}")
```

```{figure} ../../output/images/10_ee_ndvi_by_doy.gif
:name: fig-10-ndvi-global
Global NDVI seasonal cycle (multi-year median of MODIS Terra 16-day
composites). The yellow dashed line marks the subsolar latitude. Vegetation
greens up in each hemisphere's spring/summer, closely tracking the Sun's
position. The boreal forest "green wave" sweeping northward in May–June is
one of the most visible seasonal changes on Earth.
```

+++

## Vegetation — Africa close-up

Africa straddles both tropics, making it an ideal region to observe the
seasonal vegetation pulse. The most dramatic change occurs in the **Sahel**
— the semi-arid transition zone between the Sahara and the tropical savannas.
During the West African monsoon (approximately June–September), moisture
from the Gulf of Guinea pushes northward, triggering a rapid greening of the
Sahel. This vegetation pulse closely follows the northward migration of the
subsolar latitude and retreats as the Sun moves south again.

```{code-cell} python
africa = ee.Geometry.Rectangle([-20, -35, 55, 38])

ndvi_rgb_africa = ndvi_composites.map(
    lambda img: img.visualize(**ndvi_vis).clip(africa)
)

gif_ndvi_africa_params = {
    "dimensions": 800,
    "framesPerSecond": 4,
    "region": africa,
}

gif_ndvi_africa_path = str(paths.images_path / "10_ee_ndvi_africa_by_doy.gif")

geemap.download_ee_video(ndvi_rgb_africa, gif_ndvi_africa_params, gif_ndvi_africa_path)
print(f"NDVI Africa GIF saved to {gif_ndvi_africa_path}")
```

```{code-cell} python
geemap.add_text_to_gif(
    gif_ndvi_africa_path,
    gif_ndvi_africa_path,
    xy=("5%", "90%"),
    text_sequence=ndvi_labels,
    font_size=20,
    font_color="white",
    add_progress_bar=True,
    progress_bar_color="green",
    progress_bar_height=5,
)
print(f"Annotated NDVI Africa GIF saved to {gif_ndvi_africa_path}")
```

```{code-cell} python
add_subsolar_line(gif_ndvi_africa_path, lat_min=-35, lat_max=38)
print(f"Subsolar line added to {gif_ndvi_africa_path}")
```

```{figure} ../../output/images/10_ee_ndvi_africa_by_doy.gif
:name: fig-10-ndvi-africa
NDVI seasonal cycle over Africa. The Sahel's rapid greening during the
monsoon season (June–September) is clearly visible as a band of green
pushing northward, closely following the subsolar latitude (yellow dashed
line). Southern Africa shows the opposite pattern, greening during the
austral summer (November–March).
```

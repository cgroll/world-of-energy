---
jupytext:
  text_representation:
    format_name: percent
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Earth Engine — Average Temperature by Hour of Day

Compute the mean 2m temperature for each UTC hour (0–23) from a full year
of ERA5 reanalysis data on Google Earth Engine, then export an animated GIF
showing the diurnal temperature cycle across the globe.

```{code-cell} python
import os

import ee
import geemap
from dotenv import load_dotenv
from woe.paths import ProjPaths

load_dotenv()
ee.Initialize(project=os.environ["EE_GCP_PROJECT_ID"])
paths = ProjPaths()
```

## Build hourly climatology

For each UTC hour (0–23), average all ERA5 snapshots from a full year.
This produces 24 images showing the mean diurnal temperature pattern.

```{code-cell} python
era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").select("temperature_2m")

# Use a full year of data for a robust average
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
    # Tag the image with the hour for labelling
    hourly_mean = hourly_mean.set("hour", hour).set("label", f"{hour:02d}:00 UTC")
    hourly_images.append(hourly_mean)

col = ee.ImageCollection(hourly_images)
print(f"Collection size: {col.size().getInfo()} images")
```

## Export animated GIF

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

## Add hour-of-day annotation

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

## Seasonal cycle — average temperature per month

Average 3 years of ERA5 daily noon snapshots by calendar month (1–12).
Using fewer years and filtering by date range per month (instead of
scanning the full decade) keeps computation within GEE memory limits.

```{code-cell} python
import calendar

# Build a 12-image collection: one mean image per month
# Filter per-month date ranges to avoid scanning a huge base collection
monthly_images = []
month_labels = []
for month in range(1, 13):
    # Collect this calendar month across 3 years
    month_slices = []
    for year in range(2022, 2025):
        days_in_month = calendar.monthrange(year, month)[1]
        start = f"{year}-{month:02d}-01"
        end = f"{year}-{month:02d}-{days_in_month:02d}T23:59"
        month_slices.append(
            era5.filterDate(start, end)
            .filter(ee.Filter.calendarRange(12, 12, "hour"))
        )
    merged = month_slices[0]
    for s in month_slices[1:]:
        merged = merged.merge(s)
    monthly_mean = merged.mean().subtract(273.15)  # K -> °C
    label = calendar.month_abbr[month]
    monthly_mean = monthly_mean.set("month", month).set("label", label)
    monthly_images.append(monthly_mean)
    month_labels.append(label)

col_monthly = ee.ImageCollection(monthly_images)
print(f"Monthly collection size: {col_monthly.size().getInfo()} images")
```

## Export seasonal GIF

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

## Add month annotation

```{code-cell} python
geemap.add_text_to_gif(
    gif_monthly_path,
    gif_monthly_path,
    xy=("5%", "90%"),
    text_sequence=month_labels,
    font_size=20,
    font_color="white",
    add_progress_bar=True,
    progress_bar_color="cyan",
    progress_bar_height=5,
)
print(f"Annotated monthly GIF saved to {gif_monthly_path}")
```

## NDVI seasonal cycle

Use MODIS Terra 16-day NDVI composites (MOD13A2, 1 km) to show
vegetation greenness throughout the year. For each 16-day period we
compute the multi-year median across 2015–2024, producing ~23 frames
that reveal the seasonal "green wave" migrating between hemispheres.

```{code-cell} python
ndvi_col = ee.ImageCollection("MODIS/061/MOD13A2").select("NDVI")

# Tag each image with its day-of-year
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

## Export NDVI GIF

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

# Visualize each frame as RGB
ndvi_rgb = ndvi_composites.map(lambda img: img.visualize(**ndvi_vis))

gif_ndvi_params = {
    "dimensions": 600,
    "framesPerSecond": 4,
}

gif_ndvi_path = str(paths.images_path / "10_ee_ndvi_by_doy.gif")

geemap.download_ee_video(ndvi_rgb, gif_ndvi_params, gif_ndvi_path)
print(f"NDVI GIF saved to {gif_ndvi_path}")
```

## Add DOY annotation

```{code-cell} python
# Build labels from the DOY values (every 16 days starting at day 1)
from datetime import date, timedelta

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

## Add subsolar latitude line — global NDVI

Draw the subsolar latitude as a yellow dashed line on each frame of the
global NDVI GIF.

```{code-cell} python
import math
from PIL import Image, ImageDraw

global_lat_min, global_lat_max = -90, 90

def solar_declination(doy: int) -> float:
    """Solar declination in degrees for a given day-of-year."""
    return 23.44 * math.sin(math.radians(360 / 365 * (doy - 81)))

def add_subsolar_line(gif_path, lat_min, lat_max):
    """Add a dashed yellow subsolar-latitude line to each frame of a GIF."""
    gif = Image.open(gif_path)
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
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )

add_subsolar_line(gif_ndvi_path, global_lat_min, global_lat_max)
print(f"Subsolar line added to {gif_ndvi_path}")
```

## NDVI seasonal cycle — Africa

Same multi-year median NDVI composites, but clipped to Africa at higher
resolution.  The Sahel greening pulse during the monsoon season is
particularly visible.

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

## Add DOY annotation — Africa

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

## Add subsolar latitude line — Africa

```{code-cell} python
add_subsolar_line(gif_ndvi_africa_path, lat_min=-35, lat_max=38)
print(f"Subsolar line added to {gif_ndvi_africa_path}")
```

```{code-cell} python

```

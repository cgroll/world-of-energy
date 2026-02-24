---
jupytext:
  text_representation:
    format_name: percent
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Seasonality Heatmaps

Heatmaps of hour of day vs day of year reveal the combined diurnal and seasonal
patterns in generation, load, and prices across Germany's electricity system.

```{code-cell} python
import pandas as pd
import matplotlib.pyplot as plt

from woe.paths import ProjPaths
```

```{code-cell} python
# Load all data files
paths = ProjPaths()

solar = pd.read_parquet(paths.smard_solar_file)
wind_onshore = pd.read_parquet(paths.smard_wind_onshore_file)
wind_offshore = pd.read_parquet(paths.smard_wind_offshore_file)
total_load = pd.read_parquet(paths.smard_total_load_file)
prices = pd.read_parquet(paths.smard_prices_file)
```

```{code-cell} python
# Combine into single DataFrame
df = pd.concat([
    solar.rename(columns={solar.columns[0]: "solar"}),
    wind_onshore.rename(columns={wind_onshore.columns[0]: "wind_onshore"}),
    wind_offshore.rename(columns={wind_offshore.columns[0]: "wind_offshore"}),
    total_load.rename(columns={total_load.columns[0]: "total_load"}),
    prices.rename(columns={prices.columns[0]: "price"}),
], axis=1)

df = df.dropna()

# Derived columns
df["renewables"] = df["solar"] + df["wind_onshore"] + df["wind_offshore"]
df["residual_load"] = df["total_load"] - df["renewables"]
df["wind"] = df["wind_onshore"] + df["wind_offshore"]
df["hour"] = df.index.hour
df["dayofyear"] = df.index.dayofyear

print(f"Combined data: {len(df)} records")
print(f"Date range: {df.index.min()} to {df.index.max()}")
```

```{code-cell} python
# Heatmap helper
month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def plot_heatmap(df, column, title, cmap, unit, diverging=False, save_name=None):
    """Plot a heatmap of hour of day vs day of year."""
    pivot = df.groupby(["dayofyear", "hour"])[column].mean().unstack(level=0)

    fig, ax = plt.subplots(figsize=(14, 6))

    vmin, vmax = pivot.min().min(), pivot.max().max()
    if diverging:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    im = ax.pcolormesh(
        pivot.columns,
        pivot.index,
        pivot.values,
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(unit)

    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of Day")
    ax.set_title(title)
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels)
    ax.set_yticks(range(0, 24, 3))
    ax.invert_yaxis()

    fig.tight_layout()
    if save_name:
        fig.savefig(paths.images_path / save_name, dpi=150, bbox_inches="tight")
    plt.show()
```

## Solar Generation

Solar output peaks around midday in summer months. The heatmap clearly shows
zero generation at night and the seasonal shift in daylight hours.

```{code-cell} python
plot_heatmap(df, "solar", "Average Solar Generation (Hour vs Day of Year)", "YlOrRd", "MW",
             save_name="05_seasonality_solar.png")
```

```{figure} ../../output/images/05_seasonality_solar.png
:name: fig-05-seasonality-solar
Average solar generation by hour of day and day of year
```

+++

## Wind Generation

Wind generation (onshore + offshore) tends to be higher in winter and shows
less pronounced diurnal patterns compared to solar.

```{code-cell} python
plot_heatmap(df, "wind", "Average Wind Generation (Hour vs Day of Year)", "YlGnBu", "MW",
             save_name="05_seasonality_wind.png")
```

```{figure} ../../output/images/05_seasonality_wind.png
:name: fig-05-seasonality-wind
Average wind generation by hour of day and day of year
```

+++

## Aggregate Renewable Generation

The combined solar and wind heatmap reveals the complementary nature of these
sources: solar dominates summer midday, wind dominates winter.

```{code-cell} python
plot_heatmap(df, "renewables", "Average Renewable Generation (Hour vs Day of Year)", "YlGn", "MW",
             save_name="05_seasonality_renewables.png")
```

```{figure} ../../output/images/05_seasonality_renewables.png
:name: fig-05-seasonality-renewables
Average aggregate renewable generation by hour of day and day of year
```

+++

## Electricity Load

Load follows a clear double-peak pattern (morning and evening) with lower
demand on summer middays and overnight. Winter months show higher overall load.

```{code-cell} python
plot_heatmap(df, "total_load", "Average Electricity Load (Hour vs Day of Year)", "inferno", "MW",
             save_name="05_seasonality_load.png")
```

```{figure} ../../output/images/05_seasonality_load.png
:name: fig-05-seasonality-load
Average electricity load by hour of day and day of year
```

+++

## Residual Load

The residual load heatmap highlights when conventional generation is most needed
(red) and when renewable surpluses occur (blue). Negative values in summer
midday reflect solar overproduction.

```{code-cell} python
plot_heatmap(df, "residual_load", "Average Residual Load (Hour vs Day of Year)", "RdBu_r", "MW",
             diverging=True, save_name="05_seasonality_residual_load.png")
```

```{figure} ../../output/images/05_seasonality_residual_load.png
:name: fig-05-seasonality-residual-load
Average residual load by hour of day and day of year
```

+++

## Electricity Price

Price patterns mirror residual load closely. Low or negative prices appear
during summer midday (solar surplus), while winter evening peaks drive the
highest prices.

```{code-cell} python
plot_heatmap(df, "price", "Average Electricity Price (Hour vs Day of Year)", "RdBu_r", "EUR/MWh",
             diverging=True, save_name="05_seasonality_price.png")
```

```{figure} ../../output/images/05_seasonality_price.png
:name: fig-05-seasonality-price
Average electricity price by hour of day and day of year
```

```{code-cell} python
:tags: [remove-input]

from datetime import datetime
from IPython.display import Markdown

Markdown(f"Last run: {datetime.now().strftime('%Y-%m-%d')}")
```

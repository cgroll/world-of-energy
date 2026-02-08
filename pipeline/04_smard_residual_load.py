# %% [markdown]
# # Residual Load Analysis
#
# This notebook analyzes the relationship between renewable generation, total
# electricity consumption, and residual load in Germany.
#
# ## What is Residual Load?
#
# **Residual load** is the electricity demand that must be met by dispatchable
# (controllable) power plants after subtracting variable renewable generation:
#
# $$\text{Residual Load} = \text{Total Load} - \text{Solar} - \text{Wind Onshore} - \text{Wind Offshore}$$
#
# This metric is crucial for understanding:
#
# - **Grid balancing needs:** How much conventional capacity is required
# - **Storage requirements:** When excess renewable energy could be stored
# - **Price formation:** Residual load strongly correlates with electricity prices
#
# ### Interpreting Residual Load
#
# - **High residual load:** Demand exceeds renewable supply → conventional plants
#   must run → higher prices
# - **Low residual load:** Renewables cover most demand → fewer conventional
#   plants needed → lower prices
# - **Negative residual load:** Renewable generation exceeds total demand →
#   curtailment, exports, or storage needed → often negative prices

# %%
import pandas as pd
import matplotlib.pyplot as plt

from woe.paths import ProjPaths

# %%
# Load all data files
paths = ProjPaths()

print("Loading data files...")
solar = pd.read_parquet(paths.smard_solar_file)
wind_onshore = pd.read_parquet(paths.smard_wind_onshore_file)
wind_offshore = pd.read_parquet(paths.smard_wind_offshore_file)
total_load = pd.read_parquet(paths.smard_total_load_file)
prices = pd.read_parquet(paths.smard_prices_file)

print(f"Solar: {len(solar)} records")
print(f"Wind Onshore: {len(wind_onshore)} records")
print(f"Wind Offshore: {len(wind_offshore)} records")
print(f"Total Load: {len(total_load)} records")
print(f"Prices: {len(prices)} records")

# %%
# Combine into single DataFrame
df = pd.concat([
    solar.rename(columns={solar.columns[0]: "solar"}),
    wind_onshore.rename(columns={wind_onshore.columns[0]: "wind_onshore"}),
    wind_offshore.rename(columns={wind_offshore.columns[0]: "wind_offshore"}),
    total_load.rename(columns={total_load.columns[0]: "total_load"}),
    prices.rename(columns={prices.columns[0]: "price"}),
], axis=1)

# Drop rows with missing values
df = df.dropna()

print(f"Combined data: {len(df)} records")
print(f"Date range: {df.index.min()} to {df.index.max()}")
df.head()

# %% [markdown]
# ## Renewable Generation vs Total Load
#
# The stacked area chart shows solar, wind onshore, and wind offshore generation,
# with total load overlaid as a line. The gap between the renewables stack and
# the load line represents the residual load.

# %%
# Plot renewables stacked with total load as line
fig, ax = plt.subplots(figsize=(14, 6))

# Resample to daily averages for cleaner visualization
daily = df.resample("D").mean()

# Stacked area plot for renewables
ax.stackplot(
    daily.index,
    daily["solar"],
    daily["wind_onshore"],
    daily["wind_offshore"],
    labels=["Solar", "Wind Onshore", "Wind Offshore"],
    alpha=0.7,
)

# Total load as line
ax.plot(daily.index, daily["total_load"], color="black", linewidth=1, label="Total Load")

ax.set_xlabel("Date")
ax.set_ylabel("Power (MW)")
ax.set_title("Renewable Generation vs Total Load (Daily Averages)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Residual Load Calculation
#
# We compute residual load by subtracting all renewable generation from total load.

# %%
# Compute residual load
df["renewables"] = df["solar"] + df["wind_onshore"] + df["wind_offshore"]
df["residual_load"] = df["total_load"] - df["renewables"]

print(f"Residual load statistics:")
print(df["residual_load"].describe())

# Count negative residual load hours
negative_hours = (df["residual_load"] < 0).sum()
print(f"\nHours with negative residual load: {negative_hours} ({100*negative_hours/len(df):.1f}%)")

# %%
# Plot residual load time series
fig, ax = plt.subplots(figsize=(14, 6))

daily = df.resample("D").mean()

ax.plot(daily.index, daily["residual_load"], linewidth=0.5, alpha=0.8)
ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)
ax.fill_between(
    daily.index,
    daily["residual_load"],
    0,
    where=daily["residual_load"] < 0,
    color="green",
    alpha=0.3,
    label="Negative (renewable surplus)",
)
ax.fill_between(
    daily.index,
    daily["residual_load"],
    0,
    where=daily["residual_load"] >= 0,
    color="gray",
    alpha=0.3,
    label="Positive (conventional needed)",
)

ax.set_xlabel("Date")
ax.set_ylabel("Residual Load (MW)")
ax.set_title("Residual Load Over Time (Daily Averages)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Residual Load vs Electricity Prices
#
# There is typically a strong correlation between residual load and electricity
# prices. When residual load is high, expensive conventional plants must run,
# pushing prices up. When residual load is low or negative, prices tend to drop.

# %%
# Plot residual load vs price as time series
fig, ax1 = plt.subplots(figsize=(14, 6))

daily = df.resample("D").mean()

# Residual load on left axis
color1 = "tab:blue"
ax1.set_xlabel("Date")
ax1.set_ylabel("Residual Load (MW)", color=color1)
ax1.plot(daily.index, daily["residual_load"], color=color1, linewidth=0.5, alpha=0.7)
ax1.tick_params(axis="y", labelcolor=color1)

# Price on right axis
ax2 = ax1.twinx()
color2 = "tab:orange"
ax2.set_ylabel("Price (EUR/MWh)", color=color2)
ax2.plot(daily.index, daily["price"], color=color2, linewidth=0.5, alpha=0.7)
ax2.tick_params(axis="y", labelcolor=color2)

ax1.set_title("Residual Load and Electricity Price (Daily Averages)")
ax1.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %%
# Scatter plot: residual load vs price
fig, ax = plt.subplots(figsize=(10, 8))

# Use hourly data but sample for performance
sample = df.sample(min(50000, len(df)), random_state=42)

scatter = ax.scatter(
    sample["residual_load"],
    sample["price"],
    alpha=0.1,
    s=1,
)

ax.set_xlabel("Residual Load (MW)")
ax.set_ylabel("Price (EUR/MWh)")
ax.set_title("Residual Load vs Electricity Price")
ax.grid(True, alpha=0.3)

# Add correlation coefficient
corr = df["residual_load"].corr(df["price"])
ax.text(
    0.05, 0.95,
    f"Correlation: {corr:.3f}",
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Summer 2025 Detail View (June - October)
#
# A closer look at the summer period shows the daily and weekly patterns more
# clearly, including the strong solar contribution during midday hours.

# %%
# Filter to summer 2025 period
df_summer = df.loc["2025-06-01":"2025-10-01"].copy()
print(f"Summer 2025 subset: {len(df_summer)} records")
print(f"Date range: {df_summer.index.min()} to {df_summer.index.max()}")

# %%
# Renewables vs total load - summer 2025
fig, ax = plt.subplots(figsize=(14, 6))

daily_summer = df_summer.resample("D").mean()

ax.stackplot(
    daily_summer.index,
    daily_summer["solar"],
    daily_summer["wind_onshore"],
    daily_summer["wind_offshore"],
    labels=["Solar", "Wind Onshore", "Wind Offshore"],
    alpha=0.7,
)
ax.plot(daily_summer.index, daily_summer["total_load"], color="black", linewidth=1, label="Total Load")

ax.set_xlabel("Date")
ax.set_ylabel("Power (MW)")
ax.set_title("Renewable Generation vs Total Load - Summer 2025 (Daily Averages)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %%
# Residual load - summer 2025
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(daily_summer.index, daily_summer["residual_load"], linewidth=1, alpha=0.8)
ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)
ax.fill_between(
    daily_summer.index,
    daily_summer["residual_load"],
    0,
    where=daily_summer["residual_load"] < 0,
    color="green",
    alpha=0.3,
    label="Negative (renewable surplus)",
)
ax.fill_between(
    daily_summer.index,
    daily_summer["residual_load"],
    0,
    where=daily_summer["residual_load"] >= 0,
    color="gray",
    alpha=0.3,
    label="Positive (conventional needed)",
)

ax.set_xlabel("Date")
ax.set_ylabel("Residual Load (MW)")
ax.set_title("Residual Load - Summer 2025 (Daily Averages)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %%
# Residual load vs price - summer 2025
fig, ax1 = plt.subplots(figsize=(14, 6))

color1 = "tab:blue"
ax1.set_xlabel("Date")
ax1.set_ylabel("Residual Load (MW)", color=color1)
ax1.plot(daily_summer.index, daily_summer["residual_load"], color=color1, linewidth=1, alpha=0.7)
ax1.tick_params(axis="y", labelcolor=color1)

ax2 = ax1.twinx()
color2 = "tab:orange"
ax2.set_ylabel("Price (EUR/MWh)", color=color2)
ax2.plot(daily_summer.index, daily_summer["price"], color=color2, linewidth=1, alpha=0.7)
ax2.tick_params(axis="y", labelcolor=color2)

ax1.set_title("Residual Load and Electricity Price - Summer 2025 (Daily Averages)")
ax1.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %%
# Scatter plot - summer 2025
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(
    df_summer["residual_load"],
    df_summer["price"],
    alpha=0.3,
    s=5,
)

ax.set_xlabel("Residual Load (MW)")
ax.set_ylabel("Price (EUR/MWh)")
ax.set_title("Residual Load vs Electricity Price - Summer 2025")
ax.grid(True, alpha=0.3)

corr_summer = df_summer["residual_load"].corr(df_summer["price"])
ax.text(
    0.05, 0.95,
    f"Correlation: {corr_summer:.3f}",
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Seasonality Analysis
#
# Boxplots reveal the typical patterns in generation and consumption across
# different time scales: hourly, daily, and monthly.

# %%
# Add time components for grouping
df["hour"] = df.index.hour
df["month"] = df.index.month
df["dayofweek"] = df.index.dayofweek  # 0=Monday, 6=Sunday
df["wind"] = df["wind_onshore"] + df["wind_offshore"]

# %% [markdown]
# ### Solar Generation Patterns
#
# Solar generation follows a clear diurnal pattern with peak output around midday,
# and shows strong seasonal variation with higher output in summer months.

# %%
# Solar generation by hour of day
fig, ax = plt.subplots(figsize=(14, 5))

df.boxplot(column="solar", by="hour", ax=ax, grid=False)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Solar Generation (MW)")
ax.set_title("Solar Generation by Hour of Day")
plt.suptitle("")  # Remove automatic title
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.show()

# %%
# Solar generation by month
fig, ax = plt.subplots(figsize=(12, 5))

df.boxplot(column="solar", by="month", ax=ax, grid=False)
ax.set_xlabel("Month")
ax.set_ylabel("Solar Generation (MW)")
ax.set_title("Solar Generation by Month")
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.suptitle("")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.show()

# %% [markdown]
# ### Wind Generation Patterns
#
# Wind generation (onshore + offshore combined) shows less pronounced hourly
# patterns but significant seasonal variation, typically higher in winter months.

# %%
# Wind generation by hour of day
fig, ax = plt.subplots(figsize=(14, 5))

df.boxplot(column="wind", by="hour", ax=ax, grid=False)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Wind Generation (MW)")
ax.set_title("Wind Generation by Hour of Day (Onshore + Offshore)")
plt.suptitle("")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.show()

# %%
# Wind generation by month
fig, ax = plt.subplots(figsize=(12, 5))

df.boxplot(column="wind", by="month", ax=ax, grid=False)
ax.set_xlabel("Month")
ax.set_ylabel("Wind Generation (MW)")
ax.set_title("Wind Generation by Month (Onshore + Offshore)")
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.suptitle("")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.show()

# %% [markdown]
# ### Aggregate Renewable Generation Patterns
#
# Combined solar and wind generation shows how the complementary nature of these
# sources affects total renewable output throughout the day and year.

# %%
# Aggregate renewables by hour of day
fig, ax = plt.subplots(figsize=(14, 5))

df.boxplot(column="renewables", by="hour", ax=ax, grid=False)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Renewable Generation (MW)")
ax.set_title("Aggregate Renewable Generation by Hour of Day (Solar + Wind)")
plt.suptitle("")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.show()

# %%
# Aggregate renewables by month
fig, ax = plt.subplots(figsize=(12, 5))

df.boxplot(column="renewables", by="month", ax=ax, grid=False)
ax.set_xlabel("Month")
ax.set_ylabel("Renewable Generation (MW)")
ax.set_title("Aggregate Renewable Generation by Month (Solar + Wind)")
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.suptitle("")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.show()

# %% [markdown]
# ### Load Patterns
#
# Electricity demand shows clear patterns across all time scales:
# - **Hourly:** Morning and evening peaks with overnight lows
# - **Weekly:** Lower demand on weekends
# - **Monthly:** Higher demand in winter (heating) and summer (cooling)

# %%
# Load by hour of day
fig, ax = plt.subplots(figsize=(14, 5))

df.boxplot(column="total_load", by="hour", ax=ax, grid=False)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Total Load (MW)")
ax.set_title("Electricity Load by Hour of Day")
plt.suptitle("")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.show()

# %%
# Load by day of week
fig, ax = plt.subplots(figsize=(10, 5))

df.boxplot(column="total_load", by="dayofweek", ax=ax, grid=False)
ax.set_xlabel("Day of Week")
ax.set_ylabel("Total Load (MW)")
ax.set_title("Electricity Load by Day of Week")
ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
plt.suptitle("")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.show()

# %%
# Load by month
fig, ax = plt.subplots(figsize=(12, 5))

df.boxplot(column="total_load", by="month", ax=ax, grid=False)
ax.set_xlabel("Month")
ax.set_ylabel("Total Load (MW)")
ax.set_title("Electricity Load by Month")
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.suptitle("")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.show()

# %% [markdown]
# ### Residual Load Patterns
#
# Residual load patterns reveal when conventional generation or storage is most
# needed to balance the grid.

# %%
# Residual load by hour of day
fig, ax = plt.subplots(figsize=(14, 5))

df.boxplot(column="residual_load", by="hour", ax=ax, grid=False)
ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Residual Load (MW)")
ax.set_title("Residual Load by Hour of Day")
plt.suptitle("")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.show()

# %%
# Residual load by day of week
fig, ax = plt.subplots(figsize=(10, 5))

df.boxplot(column="residual_load", by="dayofweek", ax=ax, grid=False)
ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlabel("Day of Week")
ax.set_ylabel("Residual Load (MW)")
ax.set_title("Residual Load by Day of Week")
ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
plt.suptitle("")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.show()

# %%
# Residual load by month
fig, ax = plt.subplots(figsize=(12, 5))

df.boxplot(column="residual_load", by="month", ax=ax, grid=False)
ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlabel("Month")
ax.set_ylabel("Residual Load (MW)")
ax.set_title("Residual Load by Month")
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.suptitle("")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.show()

# %% [markdown]
# ### Residual Load: Summer vs Winter
#
# Comparing residual load patterns between summer (Apr-Sep) and winter (Oct-Mar)
# reveals the seasonal shift in when conventional generation is most needed.

# %%
# Create summer/winter subsets
df_winter = df[df["month"].isin([10, 11, 12, 1, 2, 3])].copy()
df_summer_season = df[df["month"].isin([4, 5, 6, 7, 8, 9])].copy()

print(f"Winter months (Oct-Mar): {len(df_winter)} records")
print(f"Summer months (Apr-Sep): {len(df_summer_season)} records")

# %%
# Residual load by hour - Summer
fig, ax = plt.subplots(figsize=(14, 5))

df_summer_season.boxplot(column="residual_load", by="hour", ax=ax, grid=False)
ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Residual Load (MW)")
ax.set_title("Residual Load by Hour of Day - Summer (Apr-Sep)")
plt.suptitle("")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.show()

# %%
# Residual load by hour - Winter
fig, ax = plt.subplots(figsize=(14, 5))

df_winter.boxplot(column="residual_load", by="hour", ax=ax, grid=False)
ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Residual Load (MW)")
ax.set_title("Residual Load by Hour of Day - Winter (Oct-Mar)")
plt.suptitle("")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.show()

# %% tags=["remove-input"]
from datetime import datetime
from IPython.display import Markdown

Markdown(f"Last run: {datetime.now().strftime('%Y-%m-%d')}")

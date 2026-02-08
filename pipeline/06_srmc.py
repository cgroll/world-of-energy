# %% [markdown]
# # Fossil Fuel Short Run Marginal Costs (SRMC)
#
# This notebook calculates and visualizes the Short Run Marginal Costs (SRMC)
# of generating electricity using hard coal and fossil gas in Europe.

# %% [markdown]
# ## Background
#
# The **Short Run Marginal Cost (SRMC)** represents the variable cost of producing
# one additional MWh of electricity from a power plant. It determines the merit
# order position of generators in wholesale electricity markets.
#
# ### Components of SRMC
#
# SRMC consists of three main components:
#
# 1. **Fuel costs** - The cost of the primary fuel (gas or coal) needed to
#    generate electricity, adjusted for plant efficiency.
#
# 2. **Carbon costs** - The cost of CO2 emissions under the EU Emissions Trading
#    Scheme (EU ETS), based on the carbon intensity of the fuel.
#
# 3. **Variable O&M costs** - Operating and maintenance costs that vary with
#    electricity output.
#
# ### Price References
#
# **Fossil Gas:**
# - Dutch Title Transfer Facility (TTF) is the benchmark for gas traded in Europe
# - Other European hubs (CEGH VTP, THE, PSV, etc.) trade at spreads to TTF
# - Prices sourced from commodity exchanges (e.g., ICE, EEX)
#
# **Hard Coal:**
# - API 2 Rotterdam is the benchmark for coal imported into Northwest Europe
# - Front month settlement prices in USD/metric ton
# - Thermal content approximately 6,000 kcal/kg (NAR)
#
# **Carbon:**
# - EU ETS allowance prices (EUA) for the front December contract
# - Prices in EUR per tonne CO2
#
# ### Calculation Assumptions
#
# | Parameter | Hard Coal | Fossil Gas |
# |-----------|-----------|------------|
# | Efficiency (HHV) | 40% | 50% |
# | Carbon intensity | 0.83 tCO2/MWh | 0.37 tCO2/MWh |
# | Variable O&M | €2/MWh | €2/MWh |
#
# The carbon intensity values represent emissions per MWh of electricity generated,
# accounting for the power plant efficiency.
#
# ### Why SRMC Matters
#
# In liberalized electricity markets, generators bid close to their marginal cost.
# The SRMC comparison between coal and gas determines:
#
# - **Fuel switching** - When gas SRMC falls below coal, utilities switch from
#   coal to gas, reducing emissions
# - **Price formation** - The highest SRMC plant needed to meet demand often
#   sets the wholesale electricity price
# - **Investment signals** - Persistent SRMC relationships influence new build
#   decisions and plant retirements

# %%
import pandas as pd
import matplotlib.pyplot as plt

from woe.paths import ProjPaths

# %% [markdown]
# ## SRMC Calculation Parameters

# %%
# Power plant efficiency rates (Higher Heating Value / Gross Calorific Value)
COAL_EFFICIENCY = 0.40  # 40%
GAS_EFFICIENCY = 0.50  # 50%

# Carbon intensity (tCO2 per MWh of electricity generated)
COAL_CARBON_INTENSITY = 0.83  # tCO2/MWh
GAS_CARBON_INTENSITY = 0.37  # tCO2/MWh

# Variable Operating and Maintenance costs (EUR/MWh)
VOM_COST = 2.0

# Coal thermal content (MWh per metric ton)
# API 2 coal specification: 6,000 kcal/kg NAR
# 6,000 kcal/kg = 6,000,000 kcal/t ÷ 860,421 kcal/MWh ≈ 6.98 MWh/t
COAL_THERMAL_CONTENT = 6.98  # MWh/t

# %% [markdown]
# ## Load Price Data

# %%
paths = ProjPaths()


def load_investing_com_csv(filepath: str) -> pd.Series:
    """Load price data from Investing.com CSV export.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Series with date index and price values.
    """
    df = pd.read_csv(
        filepath,
        encoding="utf-8-sig",  # Handle BOM
        parse_dates=["Date"],
        dayfirst=False,  # MM/DD/YYYY format
    )
    df = df.set_index("Date").sort_index()
    return df["Price"]


# %%
# Load gas prices (TTF, EUR/MWh)
print(f"Loading gas prices from: {paths.ttf_gas_prices_file}")
gas_df = pd.read_parquet(paths.ttf_gas_prices_file)
gas_prices = gas_df["Close"].rename("gas_price")

print(f"Gas price data: {gas_prices.index.min()} to {gas_prices.index.max()}")
print(f"Records: {len(gas_prices)}")

# %%
# Load coal prices (API 2 Rotterdam, USD/mt)
print(f"Loading coal prices from: {paths.rotterdam_coal_prices_file}")
coal_prices = load_investing_com_csv(paths.rotterdam_coal_prices_file).rename(
    "coal_price"
)

print(f"Coal price data: {coal_prices.index.min()} to {coal_prices.index.max()}")
print(f"Records: {len(coal_prices)}")

# %%
# Load carbon prices (EU ETS, EUR/tCO2)
print(f"Loading carbon prices from: {paths.eu_carbon_prices_file}")
carbon_prices = load_investing_com_csv(paths.eu_carbon_prices_file).rename(
    "carbon_price"
)

print(f"Carbon price data: {carbon_prices.index.min()} to {carbon_prices.index.max()}")
print(f"Records: {len(carbon_prices)}")

# %% [markdown]
# ## Combine and Align Data

# %%
# Combine all price series into a single DataFrame
# Use inner join to only keep dates where all prices are available
prices = pd.concat([gas_prices, coal_prices, carbon_prices], axis=1, join="inner")

# Forward fill any gaps (e.g., weekends, holidays)
prices = prices.ffill()

print(f"Combined data range: {prices.index.min()} to {prices.index.max()}")
print(f"Records with all prices: {len(prices)}")
prices.head()

# %% [markdown]
# ## Calculate SRMC
#
# ### Gas SRMC
# $$\text{Gas SRMC} = \frac{\text{Gas Price}}{\text{Efficiency}} + \text{Carbon Intensity} \times \text{Carbon Price} + \text{VOM}$$
#
# ### Coal SRMC
# $$\text{Coal SRMC} = \frac{\text{Coal Price (EUR/MWh)}}{\text{Efficiency}} + \text{Carbon Intensity} \times \text{Carbon Price} + \text{VOM}$$
#
# Note: Coal prices in USD/t need to be converted to EUR/MWh using:
# - Exchange rate (USD/EUR)
# - Thermal content (MWh/t)

# %%
# For simplicity, we assume coal prices are already in EUR/MWh
# If coal prices are in USD/t, additional conversion would be needed:
# coal_eur_mwh = coal_usd_mt / exchange_rate / COAL_THERMAL_CONTENT

# Convert coal from USD/t to EUR/MWh
# Using approximate EUR/USD rate - in production, this should use actual FX data
EURUSD_RATE = 1.08  # EUR per USD (approximate average)
coal_eur_mwh = prices["coal_price"] / EURUSD_RATE / COAL_THERMAL_CONTENT

# Calculate fuel costs (EUR/MWh of electricity output)
gas_fuel_cost = prices["gas_price"] / GAS_EFFICIENCY
coal_fuel_cost = coal_eur_mwh / COAL_EFFICIENCY

# Calculate carbon costs (EUR/MWh of electricity output)
gas_carbon_cost = GAS_CARBON_INTENSITY * prices["carbon_price"]
coal_carbon_cost = COAL_CARBON_INTENSITY * prices["carbon_price"]

# Calculate total SRMC
srmc = pd.DataFrame(index=prices.index)
srmc["gas_srmc"] = gas_fuel_cost + gas_carbon_cost + VOM_COST
srmc["coal_srmc"] = coal_fuel_cost + coal_carbon_cost + VOM_COST

# Also store components for analysis
srmc["gas_fuel_cost"] = gas_fuel_cost
srmc["gas_carbon_cost"] = gas_carbon_cost
srmc["coal_fuel_cost"] = coal_fuel_cost
srmc["coal_carbon_cost"] = coal_carbon_cost

print("SRMC Summary Statistics (EUR/MWh):")
print(srmc[["gas_srmc", "coal_srmc"]].describe().round(2))

# %% [markdown]
# ## SRMC Time Series

# %%
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(srmc.index, srmc["gas_srmc"], label="Gas SRMC", linewidth=1, alpha=0.8)
ax.plot(srmc.index, srmc["coal_srmc"], label="Coal SRMC", linewidth=1, alpha=0.8)

ax.set_xlabel("Date")
ax.set_ylabel("SRMC (EUR/MWh)")
ax.set_title("Short Run Marginal Costs: Coal vs Gas")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Monthly Average SRMC

# %%
monthly_srmc = srmc[["gas_srmc", "coal_srmc"]].resample("ME").mean()

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    monthly_srmc.index,
    monthly_srmc["gas_srmc"],
    label="Gas SRMC",
    marker="o",
    markersize=3,
)
ax.plot(
    monthly_srmc.index,
    monthly_srmc["coal_srmc"],
    label="Coal SRMC",
    marker="o",
    markersize=3,
)

ax.set_xlabel("Date")
ax.set_ylabel("SRMC (EUR/MWh)")
ax.set_title("Monthly Average Short Run Marginal Costs")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Fuel Switching Indicator
#
# When gas SRMC is lower than coal SRMC, it is economically favorable to dispatch
# gas plants over coal plants, leading to lower emissions. This "fuel switching"
# is a key mechanism for carbon price effectiveness.

# %%
# Calculate the spread (positive = gas cheaper than coal)
srmc["spread"] = srmc["coal_srmc"] - srmc["gas_srmc"]

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# SRMC comparison
ax1 = axes[0]
ax1.plot(srmc.index, srmc["gas_srmc"], label="Gas SRMC", linewidth=1, alpha=0.8)
ax1.plot(srmc.index, srmc["coal_srmc"], label="Coal SRMC", linewidth=1, alpha=0.8)
ax1.set_ylabel("SRMC (EUR/MWh)")
ax1.set_title("Short Run Marginal Costs and Fuel Switching")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Spread (coal - gas)
ax2 = axes[1]
ax2.fill_between(
    srmc.index,
    srmc["spread"],
    0,
    where=srmc["spread"] >= 0,
    color="green",
    alpha=0.5,
    label="Gas cheaper (fuel switch)",
)
ax2.fill_between(
    srmc.index,
    srmc["spread"],
    0,
    where=srmc["spread"] < 0,
    color="red",
    alpha=0.5,
    label="Coal cheaper",
)
ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
ax2.set_xlabel("Date")
ax2.set_ylabel("Spread (EUR/MWh)")
ax2.set_title("Coal-Gas Spread (positive = gas cheaper)")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## SRMC Cost Components
#
# Breaking down the SRMC into its fuel and carbon cost components shows the
# relative importance of each driver.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gas SRMC components
ax1 = axes[0]
ax1.stackplot(
    srmc.index,
    srmc["gas_fuel_cost"],
    srmc["gas_carbon_cost"],
    [VOM_COST] * len(srmc),
    labels=["Fuel Cost", "Carbon Cost", "Variable O&M"],
    alpha=0.8,
)
ax1.set_xlabel("Date")
ax1.set_ylabel("Cost (EUR/MWh)")
ax1.set_title("Gas SRMC Components")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# Coal SRMC components
ax2 = axes[1]
ax2.stackplot(
    srmc.index,
    srmc["coal_fuel_cost"],
    srmc["coal_carbon_cost"],
    [VOM_COST] * len(srmc),
    labels=["Fuel Cost", "Carbon Cost", "Variable O&M"],
    alpha=0.8,
)
ax2.set_xlabel("Date")
ax2.set_ylabel("Cost (EUR/MWh)")
ax2.set_title("Coal SRMC Components")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %% tags=["remove-input"]
from datetime import datetime
from IPython.display import Markdown

Markdown(f"Last run: {datetime.now().strftime('%Y-%m-%d')}")

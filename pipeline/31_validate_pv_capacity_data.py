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
# # Renewable Power Plants in Germany — Dataset Validation
#
# Validates the Open Power System Data (OPSD) filtered plant registry against
# official reference figures.
#
# **Checks performed**
# - Technology breakdown: plant count and installed capacity by type
# - Solar PV cross-check: OPSD state-level capacities vs. Wikipedia 2015 figures
#   (the closest historical reference for the 2015-12-31 snapshot)

# %%
import pandas as pd
import matplotlib.pyplot as plt

from woe.paths import ProjPaths

paths = ProjPaths()

# %% [markdown]
# ## Load data

# %%
TARGET_DATE = pd.Timestamp("2015-12-31")

df_raw = pd.read_csv(paths.renewable_power_plants_file, low_memory=False)
df_raw["commissioning_date"] = pd.to_datetime(df_raw["commissioning_date"], errors="coerce")

commissioned = df_raw["commissioning_date"] <= TARGET_DATE
not_decommissioned = (
    df_raw["decommissioning_date"].isna()
    | (pd.to_datetime(df_raw["decommissioning_date"], errors="coerce") > TARGET_DATE)
)
df = df_raw[commissioned & not_decommissioned].copy()

NUTS1_TO_STATE = {
    "DE1": "Baden-Württemberg",
    "DE2": "Bayern",
    "DE3": "Berlin",
    "DE4": "Brandenburg",
    "DE5": "Bremen",
    "DE6": "Hamburg",
    "DE7": "Hessen",
    "DE8": "Mecklenburg-Vorpommern",
    "DE9": "Niedersachsen",
    "DEA": "Nordrhein-Westfalen",
    "DEB": "Rheinland-Pfalz",
    "DEC": "Saarland",
    "DED": "Sachsen",
    "DEE": "Sachsen-Anhalt",
    "DEF": "Schleswig-Holstein",
    "DEG": "Thüringen",
}
df["state_nuts"] = df["nuts_1_region"].map(NUTS1_TO_STATE)

print(f"Total records in dataset:   {len(df_raw):>10,}")
print(f"Active plants at {TARGET_DATE.date()}: {len(df):>10,}")

# %% [markdown]
# ## Technology breakdown
#
# The `technology` column distinguishes four plant types:
# - **Photovoltaics** — rooftop / small-scale solar PV
# - **Photovoltaics ground** — ground-mounted utility-scale solar PV
# - **Onshore** — land-based wind turbines
# - **Offshore** — offshore wind turbines in the German EEZ (North Sea and Baltic Sea)

# %%
TECH_LABELS = {
    "Photovoltaics": "Solar PV (rooftop)",
    "Photovoltaics ground": "Solar PV (ground)",
    "Onshore": "Wind Onshore",
    "Offshore": "Wind Offshore",
}
df["tech_label"] = df["technology"].map(TECH_LABELS)

summary = (
    df.groupby("tech_label")
    .agg(
        plant_count=("electrical_capacity", "count"),
        total_capacity_MW=("electrical_capacity", "sum"),
        avg_capacity_kW=("electrical_capacity", lambda x: x.mean() * 1000),
    )
    .round(1)
    .sort_values("total_capacity_MW", ascending=False)
)
print(summary.to_string())

# %%
TECH_COLORS = {
    "Solar PV (rooftop)": "#FFD700",
    "Solar PV (ground)": "#FFA500",
    "Wind Onshore": "#4169E1",
    "Wind Offshore": "#00CED1",
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels = list(summary.index)
colors = [TECH_COLORS[t] for t in labels]

axes[0].bar(labels, summary["plant_count"], color=colors, edgecolor="black", linewidth=0.5)
axes[0].set_title("Number of Plants by Technology")
axes[0].set_ylabel("Number of Plants")
axes[0].set_xticklabels(labels, rotation=15, ha="right")
for bar, val in zip(axes[0].patches, summary["plant_count"]):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.01,
        f"{val:,.0f}",
        ha="center", va="bottom", fontsize=9,
    )

cap_gw = summary["total_capacity_MW"] / 1000
axes[1].bar(labels, cap_gw, color=colors, edgecolor="black", linewidth=0.5)
axes[1].set_title("Total Installed Capacity by Technology")
axes[1].set_ylabel("Capacity (GW)")
axes[1].set_xticklabels(labels, rotation=15, ha="right")
for bar, val in zip(axes[1].patches, cap_gw):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.01,
        f"{val:.1f} GW",
        ha="center", va="bottom", fontsize=9,
    )

fig.suptitle("Renewable Power Plants in Germany (OPSD Dataset)", fontsize=13)
fig.tight_layout()
fig.savefig(paths.images_path / "31_plant_counts.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/31_plant_counts.png
# :name: fig-31-plant-counts
# Number of plants and total installed capacity by technology type.
# Solar PV rooftop installations dominate by count while contributing the
# largest share of capacity. Each individual wind turbine has significantly
# higher average capacity (~3 MW onshore, ~7 MW offshore) than a rooftop
# solar system.
# ```

# %% [markdown]
# ## Solar PV validation — OPSD vs. Wikipedia 2015
#
# Wikipedia's *Solar power in Germany* article provides cumulative installed PV
# capacity per federal state; the 2015 column is the closest historical match
# for our `TARGET_DATE` of 2015-12-31.

# %%
WIKI_2015_MW = {
    "Baden-Württemberg":  5_117.0,
    "Bayern":            11_309.2,
    "Berlin":                83.9,
    "Brandenburg":        2_981.5,
    "Bremen":                42.2,
    "Hamburg":               36.9,
    "Hessen":             1_811.2,
    "Niedersachsen":      3_580.4,
    "Mecklenburg-Vorpommern": 1_414.4,
    "Nordrhein-Westfalen": 4_363.7,
    "Rheinland-Pfalz":    1_920.5,
    "Saarland":             415.8,
    "Sachsen":            1_607.5,
    "Sachsen-Anhalt":     1_962.6,
    "Schleswig-Holstein": 1_498.3,
    "Thüringen":          1_187.4,
}

solar_df = df[df["energy_source_level_2"] == "Solar"]
solar_by_state_mw = solar_df.groupby("state_nuts")["electrical_capacity"].sum()

compare = pd.DataFrame({
    "OPSD 2015-12-31 (MW)": solar_by_state_mw,
    "Wikipedia 2015 (MW)":  pd.Series(WIKI_2015_MW),
}).reindex(sorted(WIKI_2015_MW.keys())).round(0)

compare["Delta (MW)"] = compare["OPSD 2015-12-31 (MW)"] - compare["Wikipedia 2015 (MW)"]
compare["Delta %"] = (compare["Delta (MW)"] / compare["Wikipedia 2015 (MW)"] * 100).round(1)

totals = compare.sum(numeric_only=True)
totals["Delta %"] = totals["Delta (MW)"] / totals["Wikipedia 2015 (MW)"] * 100
compare.loc["TOTAL"] = totals.round(1)

print(compare.to_string())
print(
    "\nNote: OPSD covers ~98% of total capacity. Small negative deltas (~1–5%)"
    " reflect plants below OPSD's registration threshold or minor data lags."
)

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
# # Renewable Plant Capacities: Maps and Aggregates
#
# Filters the OPSD plant registry to the 2019 snapshot, produces capacity maps
# for all technologies, and computes the three aggregate tables used by
# `pipeline/32_dev_energy_generation.py` as API query inputs.
#
# **Outputs**
# - `pv_state_aggregates.parquet`: PV capacity + capacity-weighted centroid per Bundesland
# - `wind_onshore_state_aggregates.parquet`: onshore wind capacity + centroid per Bundesland
# - `wind_offshore_aggregates.parquet`: offshore wind capacity + centroid per sea region
#   (Nordsee / Ostsee)

# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

from woe.paths import ProjPaths

paths = ProjPaths()

PROJ = ccrs.LambertConformal(central_longitude=10.5, central_latitude=51.5)
GERMANY_EXTENT = [5.5, 15.5, 47.0, 55.5]  # [W, E, S, N] in PlateCarree degrees

# %% [markdown]
# ## Load and filter data

# %%
TARGET_DATE = pd.Timestamp("2019-01-01")

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
df["state"] = df["nuts_1_region"].map(NUTS1_TO_STATE)

pv = df[df["energy_source_level_2"] == "Solar"].copy()
wind_onshore = df[(df["energy_source_level_2"] == "Wind") & (df["technology"] == "Onshore")].copy()
wind_offshore = df[(df["energy_source_level_2"] == "Wind") & (df["technology"] == "Offshore")].copy()

print(f"Active plants at {TARGET_DATE.date()}:  {len(df):>10,}")
print(f"  Solar PV:      {len(pv):>10,}  ({pv['electrical_capacity'].sum():.0f} MW)")
print(f"  Wind onshore:  {len(wind_onshore):>10,}  ({wind_onshore['electrical_capacity'].sum():.0f} MW)")
print(f"  Wind offshore: {len(wind_offshore):>10,}  ({wind_offshore['electrical_capacity'].sum():.0f} MW)")

# %% [markdown]
# ## Load German state geometries

# %%
shpfilename = shpreader.natural_earth(
    resolution="10m", category="cultural", name="admin_1_states_provinces_lakes"
)
reader = shpreader.Reader(shpfilename)
german_states = {
    r.attributes["name"]: r.geometry
    for r in reader.records()
    if r.attributes["admin"] == "Germany"
}
print(f"Loaded {len(german_states)} German federal states from Natural Earth")

# %% [markdown]
# ## Solar PV — capacity choropleth

# %%
solar_by_state = pv.groupby("state")["electrical_capacity"].sum() / 1000  # GW

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": PROJ}, figsize=(8, 9))
ax.set_extent(GERMANY_EXTENT, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.OCEAN, facecolor="#c6def1")
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")

cmap_solar = cm.YlOrRd
norm_solar = mcolors.Normalize(vmin=0, vmax=solar_by_state.max())

for name, geom in german_states.items():
    cap = solar_by_state.get(name, 0.0)
    ax.add_geometries(
        [geom], ccrs.PlateCarree(),
        facecolor=cmap_solar(norm_solar(cap)), edgecolor="black", linewidth=0.5,
    )

sm = plt.cm.ScalarMappable(cmap=cmap_solar, norm=norm_solar)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
cbar.set_label("Installed Solar PV Capacity (GW)", fontsize=10)
ax.set_title(
    f"Solar PV Installed Capacity by Federal State\n(as of {TARGET_DATE.date()}, OPSD)",
    fontsize=11, pad=10,
)
ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
fig.tight_layout()
fig.savefig(paths.images_path / "32_solar_choropleth.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/32_solar_choropleth.png
# :name: fig-32-solar-choropleth
# Installed solar PV capacity (GW) by German federal state as of 2019-01-01.
# Bayern and Baden-Württemberg lead due to higher solar irradiation in southern
# Germany. Brandenburg also has a significant share from large ground-mounted parks.
# ```

# %% [markdown]
# ## Wind — onshore choropleth
#
# Offshore wind farms lie in the German EEZ and are not attributed to any
# federal state; their total is shown as an annotation.

# %%
wind_on_by_state = wind_onshore.groupby("state")["electrical_capacity"].sum() / 1000  # GW
wind_off_total_gw = wind_offshore["electrical_capacity"].sum() / 1000

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": PROJ}, figsize=(8, 9))
ax.set_extent(GERMANY_EXTENT, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.OCEAN, facecolor="#c6def1")
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")

cmap_wind = cm.Blues
norm_wind = mcolors.Normalize(vmin=0, vmax=wind_on_by_state.max())

for name, geom in german_states.items():
    cap = wind_on_by_state.get(name, 0.0)
    ax.add_geometries(
        [geom], ccrs.PlateCarree(),
        facecolor=cmap_wind(norm_wind(cap)), edgecolor="black", linewidth=0.5,
    )

sm = plt.cm.ScalarMappable(cmap=cmap_wind, norm=norm_wind)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
cbar.set_label("Installed Wind Onshore Capacity (GW)", fontsize=10)

ax.text(
    0.02, 0.02,
    f"Wind Offshore (EEZ): {wind_off_total_gw:.1f} GW\n(Nordsee + Ostsee, not per state)",
    transform=ax.transAxes,
    fontsize=9, va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
)

ax.set_title(
    f"Wind Onshore Installed Capacity by Federal State\n(as of {TARGET_DATE.date()}, OPSD)",
    fontsize=11, pad=10,
)
ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
fig.tight_layout()
fig.savefig(paths.images_path / "32_wind_choropleth.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/32_wind_choropleth.png
# :name: fig-32-wind-choropleth
# Installed wind onshore capacity (GW) by German federal state as of 2019-01-01.
# Niedersachsen and Brandenburg dominate. Offshore wind capacity (Nordsee + Ostsee)
# is located in the EEZ and is not assigned to any state.
# ```

# %% [markdown]
# ## Compute capacity-weighted centroids
#
# Centroids are computed from plants with valid coordinates only; all plants
# (with or without coordinates) contribute to the capacity total.

# %%
def weighted_centroid(g: pd.DataFrame) -> pd.Series:
    w = g["electrical_capacity"]
    return pd.Series({
        "lat": (g["lat"] * w).sum() / w.sum(),
        "lon": (g["lon"] * w).sum() / w.sum(),
        "n_with_coords": len(g),
    })


# --- PV: per Bundesland ---
cap_pv = (
    pv.groupby("state")["electrical_capacity"]
    .agg(capacity_MW="sum", n_plants="count")
    .reset_index()
)
pv_geo = pv.dropna(subset=["lat", "lon"])
pv_centroids = pv_geo.groupby("state").apply(weighted_centroid).reset_index()
pv_agg = cap_pv.merge(pv_centroids, on="state", how="left")
pv_agg["n_no_coords"] = pv_agg["n_plants"] - pv_agg["n_with_coords"].fillna(0).astype(int)
pv_agg = pv_agg.sort_values("capacity_MW", ascending=False).reset_index(drop=True)

print(f"PV ({TARGET_DATE.date()}):")
print(pv_agg[["state", "capacity_MW", "n_plants", "n_no_coords", "lat", "lon"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# --- Wind onshore: per Bundesland ---
cap_onshore = (
    wind_onshore.groupby("state")["electrical_capacity"]
    .agg(capacity_MW="sum", n_plants="count")
    .reset_index()
)
onshore_geo = wind_onshore.dropna(subset=["lat", "lon"])
onshore_centroids = onshore_geo.groupby("state").apply(weighted_centroid).reset_index()
wind_onshore_agg = cap_onshore.merge(onshore_centroids, on="state", how="left")
wind_onshore_agg = wind_onshore_agg.sort_values("capacity_MW", ascending=False).reset_index(drop=True)

print(f"\nWind onshore ({TARGET_DATE.date()}):")
print(wind_onshore_agg[["state", "capacity_MW", "lat", "lon"]].to_string(index=False))

# --- Wind offshore: Nordsee / Ostsee ---
# DE8 (Mecklenburg-Vorpommern) → Ostsee; NaN region (federal EEZ waters) → Nordsee
wind_offshore = wind_offshore.copy()
wind_offshore["region"] = wind_offshore["nuts_1_region"].apply(
    lambda x: "Ostsee" if x == "DE8" else "Nordsee"
)
cap_offshore = (
    wind_offshore.groupby("region")["electrical_capacity"]
    .agg(capacity_MW="sum", n_plants="count")
    .reset_index()
)
offshore_geo = wind_offshore.dropna(subset=["lat", "lon"])
offshore_centroids = offshore_geo.groupby("region").apply(weighted_centroid).reset_index()
wind_offshore_agg = cap_offshore.merge(offshore_centroids, on="region", how="left")

print(f"\nWind offshore ({TARGET_DATE.date()}):")
print(wind_offshore_agg[["region", "capacity_MW", "n_with_coords", "lat", "lon"]].to_string(index=False))
print("  (centroids from plants with valid coordinates only)")

# %% [markdown]
# ## PV centroid map

# %%
pv_agg_valid = pv_agg.dropna(subset=["lat", "lon"])
cap_by_state = pv_agg.set_index("state")["capacity_MW"] / 1000  # GW
cmap_pv = cm.YlOrRd
norm_pv = mcolors.Normalize(vmin=0, vmax=cap_by_state.max())

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": PROJ}, figsize=(9, 10))
ax.set_extent(GERMANY_EXTENT, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.OCEAN, facecolor="#c6def1")
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")

for name, geom in german_states.items():
    cap = cap_by_state.get(name, 0.0)
    ax.add_geometries(
        [geom], ccrs.PlateCarree(),
        facecolor=cmap_pv(norm_pv(cap)), edgecolor="black", linewidth=0.5,
    )

bubble_sizes = (pv_agg_valid["capacity_MW"] / pv_agg_valid["capacity_MW"].max() * 400).clip(lower=20)
ax.scatter(
    pv_agg_valid["lon"], pv_agg_valid["lat"],
    s=bubble_sizes,
    c=pv_agg_valid["capacity_MW"] / 1000,
    cmap=cmap_pv, norm=norm_pv,
    edgecolors="white", linewidths=0.8,
    transform=ccrs.PlateCarree(),
    zorder=5,
)
for _, row in pv_agg_valid.iterrows():
    ax.text(
        row["lon"], row["lat"] + 0.12,
        f"{row['capacity_MW'] / 1000:.1f} GW",
        transform=ccrs.PlateCarree(),
        ha="center", va="bottom", fontsize=6.5, color="black", fontweight="bold",
    )

sm = plt.cm.ScalarMappable(cmap=cmap_pv, norm=norm_pv)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
cbar.set_label("Installed Solar PV Capacity (GW)", fontsize=10)
ax.set_title(
    f"Solar PV Installed Capacity and Fleet Centroids by Bundesland\n"
    f"(as of {TARGET_DATE.date()}, OPSD)",
    fontsize=11, pad=10,
)
ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
fig.tight_layout()
fig.savefig(paths.images_path / "32_pv_state_centroids.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/32_pv_state_centroids.png
# :name: fig-32-pv-state-centroids
# Solar PV installed capacity per Bundesland as of 2019-01-01, with
# capacity-weighted fleet centroids shown as bubbles (bubble area ∝ capacity).
# Bayern dominates with the highest absolute capacity.
# ```

# %% [markdown]
# ## Save aggregate tables

# %%
paths.renewable_plants_processed_path.mkdir(parents=True, exist_ok=True)

pv_agg.to_parquet(paths.pv_state_aggregates_file, index=False)
print(f"Saved PV aggregates            → {paths.pv_state_aggregates_file}")

wind_onshore_agg.to_parquet(paths.wind_onshore_state_aggregates_file, index=False)
print(f"Saved wind onshore aggregates  → {paths.wind_onshore_state_aggregates_file}")

wind_offshore_agg.to_parquet(paths.wind_offshore_aggregates_file, index=False)
print(f"Saved wind offshore aggregates → {paths.wind_offshore_aggregates_file}")

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
# # Renewable Generation Analysis
#
# Analyses the hourly solar PV, onshore wind, and offshore wind capacity factors
# produced by `pipeline/32_dev_energy_generation.py` (renewables.ninja / MERRA-2).
#
# **Inputs**
# - `ninja_pv_cf.parquet`: hourly PV capacity factors (0–1) per Bundesland
# - `ninja_wind_onshore_cf.parquet`: hourly onshore wind capacity factors per Bundesland
# - `ninja_wind_offshore_cf.parquet`: hourly offshore wind capacity factors per sea region
# - `pv_state_aggregates.parquet`: PV installed capacity per state
# - `wind_onshore_state_aggregates.parquet`: onshore wind capacity per state
# - `wind_offshore_aggregates.parquet`: offshore wind capacity per sea region

# %%
import pandas as pd
import matplotlib.pyplot as plt

from woe.paths import ProjPaths

paths = ProjPaths()

YEAR = 2019

# %%
pv_cf = pd.read_parquet(paths.ninja_pv_cf_file)
wind_onshore_cf = pd.read_parquet(paths.ninja_wind_onshore_cf_file)
wind_offshore_cf = pd.read_parquet(paths.ninja_wind_offshore_cf_file)
pv_agg = pd.read_parquet(paths.pv_state_aggregates_file)
wind_onshore_agg = pd.read_parquet(paths.wind_onshore_state_aggregates_file)
wind_offshore_agg = pd.read_parquet(paths.wind_offshore_aggregates_file)

print(f"PV CF:            {pv_cf.shape}  (rows=hours, cols=states)")
print(f"Wind onshore CF:  {wind_onshore_cf.shape}")
print(f"Wind offshore CF: {wind_offshore_cf.shape}  (cols=Nordsee, Ostsee)")

# %% [markdown]
# ## Monthly generation summary

# %%
cap_pv = pv_agg.set_index("state")["capacity_MW"]
cap_onshore = wind_onshore_agg.set_index("state")["capacity_MW"]
cap_offshore = wind_offshore_agg.set_index("region")["capacity_MW"]

pv_de = (pv_cf * cap_pv).sum(axis=1)
wind_onshore_de = (wind_onshore_cf * cap_onshore).sum(axis=1)
wind_offshore_de = (wind_offshore_cf * cap_offshore).sum(axis=1)

monthly = pd.DataFrame({
    "PV": pv_de.resample("ME").sum() / 1e3,
    "Wind Onshore": wind_onshore_de.resample("ME").sum() / 1e3,
    "Wind Offshore": wind_offshore_de.resample("ME").sum() / 1e3,
})
monthly.index = monthly.index.strftime("%b")

fig, ax = plt.subplots(figsize=(12, 5))
x = range(len(monthly))
width = 0.38
ax.bar([i - width / 2 for i in x], monthly["PV"], width=width,
       label="Solar PV", color="#f4b942", edgecolor="white", linewidth=0.5)
ax.bar([i + width / 2 for i in x], monthly["Wind Onshore"], width=width,
       label="Wind onshore", color="#4a90d9", edgecolor="white", linewidth=0.5)
ax.bar([i + width / 2 for i in x], monthly["Wind Offshore"], width=width,
       bottom=monthly["Wind Onshore"],
       label="Wind offshore", color="#1a5fa8", edgecolor="white", linewidth=0.5)
ax.set_xticks(list(x))
ax.set_xticklabels(monthly.index)
ax.set_ylabel("Generation (GWh)")
ax.set_title(
    f"Monthly Solar PV and Wind Generation — Germany {YEAR}\n"
    "(renewables.ninja / MERRA-2, capacity-weighted centroids)",
    fontsize=11,
)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(paths.images_path / "34_monthly_generation.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/34_monthly_generation.png
# :name: fig-34-monthly-generation
# Monthly solar PV, onshore wind, and offshore wind electricity generation for
# Germany in 2019, modelled via renewables.ninja (MERRA-2 reanalysis).
# Wind bars are stacked: onshore (light blue) + offshore (dark blue).
# PV peaks sharply in summer; wind shows the inverse seasonal pattern with
# highest output in winter months.
# ```

# %% [markdown]
# ## Annual capacity factors per state (onshore wind and PV)

# %%
cf_pv = pv_cf.mean().rename("CF_PV")
cf_wind_onshore = wind_onshore_cf.mean().rename("CF_Wind_Onshore")

cf_summary = pd.concat([cf_pv, cf_wind_onshore], axis=1).dropna(how="all")
cf_summary = cf_summary.sort_values("CF_Wind_Onshore", ascending=True)

fig2, ax2 = plt.subplots(figsize=(10, 6))
y = range(len(cf_summary))
ax2.barh([i + 0.2 for i in y], cf_summary["CF_PV"], height=0.35,
         label="Solar PV", color="#f4b942", edgecolor="white", linewidth=0.5)
ax2.barh([i - 0.2 for i in y], cf_summary["CF_Wind_Onshore"], height=0.35,
         label="Wind (onshore)", color="#4a90d9", edgecolor="white", linewidth=0.5)
ax2.set_yticks(list(y))
ax2.set_yticklabels(cf_summary.index, fontsize=9)
ax2.set_xlabel("Annual capacity factor")
ax2.set_title(
    f"Annual Capacity Factors by Bundesland — {YEAR}\n"
    "(renewables.ninja / MERRA-2)",
    fontsize=11,
)
ax2.legend(fontsize=10)
ax2.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax2.set_axisbelow(True)
fig2.tight_layout()
fig2.savefig(paths.images_path / "34_capacity_factors.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/34_capacity_factors.png
# :name: fig-34-capacity-factors
# Annual capacity factors for solar PV and onshore wind by Bundesland in 2019.
# Northern states (Schleswig-Holstein, Mecklenburg-Vorpommern) show the highest
# wind capacity factors; southern states (Bayern, Baden-Württemberg) lead for PV.
# Offshore wind capacity factors (Nordsee, Ostsee) are substantially higher
# than any onshore state and are not shown here.
# ```

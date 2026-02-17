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
# # Forecast vs Actual: Solar and Wind Generation
#
# Compare day-ahead and intraday forecasts against actual generation for solar,
# wind onshore, and wind offshore.

# %%
import pandas as pd
import matplotlib.pyplot as plt

from woe.paths import ProjPaths


# %%
paths = ProjPaths()

# Load individual forecast files
forecast_files = {
    "da_solar": paths.smard_forecast_da_solar_file,
    "da_wind_onshore": paths.smard_forecast_da_wind_onshore_file,
    "da_wind_offshore": paths.smard_forecast_da_wind_offshore_file,
    "id_solar": paths.smard_forecast_id_solar_file,
    "id_wind_onshore": paths.smard_forecast_id_wind_onshore_file,
    "id_wind_offshore": paths.smard_forecast_id_wind_offshore_file,
}

forecasts = pd.DataFrame()
for col, path in forecast_files.items():
    if path.exists():
        tmp = pd.read_parquet(path)
        tmp.columns = [col]
        forecasts = tmp if forecasts.empty else forecasts.join(tmp, how="outer")

# Load actual generation
solar = pd.read_parquet(paths.smard_solar_file)
wind_onshore = pd.read_parquet(paths.smard_wind_onshore_file)
wind_offshore = pd.read_parquet(paths.smard_wind_offshore_file)

# %%
print(f"Forecasts shape: {forecasts.shape}")
print(f"Forecasts columns: {list(forecasts.columns)}")
print(f"Forecasts index range: {forecasts.index.min()} to {forecasts.index.max()}")
print()

for prefix, source in [("da", "Day-ahead"), ("id", "Intraday")]:
    for var in ["solar", "wind_onshore", "wind_offshore"]:
        col = f"{prefix}_{var}"
        exists = col in forecasts.columns
        n = forecasts[col].notna().sum() if exists else 0
        print(f"  {source} {var:15s} ({col}): exists={exists}, non-null={n}")

# %% [markdown]
# ## Align forecasts with actuals
#
# The actual generation data is hourly, while forecasts are quarter-hourly.
# An inner join keeps only the on-the-hour timestamps that both share,
# so no resampling is needed.

# %%
actual = pd.concat([
    solar.rename(columns={solar.columns[0]: "actual_solar"}),
    wind_onshore.rename(columns={wind_onshore.columns[0]: "actual_wind_onshore"}),
    wind_offshore.rename(columns={wind_offshore.columns[0]: "actual_wind_offshore"}),
], axis=1)

df = forecasts.join(actual, how="inner")
print(f"Joined shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"\nNull counts:\n{df.isna().sum()}")

# %% [markdown]
# ## Time series comparison (sample week)

# %%
SOURCES = [
    ("solar", "Solar"),
    ("wind_onshore", "Wind Onshore"),
    ("wind_offshore", "Wind Offshore"),
]

# Pick a recent summer week for solar, or last 90 days as fallback
summer = df.loc["2025-06":"2025-06"]
if len(summer) == 0:
    summer = df.last("90D")
sample = summer.iloc[:7*24]

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for ax, (var, title) in zip(axes, SOURCES):
    ax.plot(sample.index, sample[f"actual_{var}"], label="Actual", color="black", linewidth=1.2)
    if f"da_{var}" in df.columns:
        ax.plot(sample.index, sample[f"da_{var}"], label="Day-ahead", alpha=0.8)
    if f"id_{var}" in df.columns:
        ax.plot(sample.index, sample[f"id_{var}"], label="Intraday", alpha=0.8)
    ax.set_ylabel("Generation [MW]")
    ax.set_title(f"{title}: Forecast vs Actual")
    ax.legend()

fig.tight_layout()
fig.savefig(paths.images_path / "07_forecast_sample_week.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Forecast errors

# %%
for var, title in SOURCES:
    df[f"err_da_{var}"] = df[f"da_{var}"] - df[f"actual_{var}"]
    if f"id_{var}" in df.columns:
        df[f"err_id_{var}"] = df[f"id_{var}"] - df[f"actual_{var}"]

rows = []
for var, title in SOURCES:
    for prefix, label in [("da", "Day-ahead"), ("id", "Intraday")]:
        col = f"err_{prefix}_{var}"
        if col in df.columns:
            rows.append({
                "source": title,
                "forecast": label,
                "MAE [MW]": df[col].abs().mean(),
                "Bias [MW]": df[col].mean(),
                "RMSE [MW]": (df[col] ** 2).mean() ** 0.5,
            })

print(pd.DataFrame(rows).to_string(index=False, float_format="{:,.0f}".format))

# %%
fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharey="row")

for row, (var, title) in enumerate(SOURCES):
    for col_idx, (prefix, label) in enumerate([("da", "Day-ahead"), ("id", "Intraday")]):
        ax = axes[row, col_idx]
        err_col = f"err_{prefix}_{var}"
        if err_col not in df.columns:
            ax.set_visible(False)
            continue
        ax.hist(df[err_col].dropna(), bins=100, alpha=0.7, edgecolor="none")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Forecast error [MW]")
        ax.set_title(f"{title} — {label}")
        mae = df[err_col].abs().mean()
        bias = df[err_col].mean()
        ax.annotate(f"MAE = {mae:,.0f} MW\nBias = {bias:,.0f} MW",
                    xy=(0.97, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    axes[row, 0].set_ylabel("Count")

fig.tight_layout()
fig.savefig(paths.images_path / "07_forecast_errors.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Scatter: forecast vs actual

# %%
fig, axes = plt.subplots(3, 2, figsize=(12, 14))

for row, (var, title) in enumerate(SOURCES):
    for col_idx, (prefix, label) in enumerate([("da", "Day-ahead"), ("id", "Intraday")]):
        ax = axes[row, col_idx]
        fcst_col = f"{prefix}_{var}"
        actual_col = f"actual_{var}"
        if fcst_col not in df.columns:
            ax.set_visible(False)
            continue
        sub = df[[fcst_col, actual_col]].dropna()
        ax.scatter(sub[actual_col], sub[fcst_col], s=1, alpha=0.05)
        lims = [0, max(sub[actual_col].max(), sub[fcst_col].max()) * 1.05]
        ax.plot(lims, lims, "k--", linewidth=0.8, label="perfect forecast")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"Actual {title} [MW]")
        ax.set_ylabel(f"Forecast {title} [MW]")
        ax.set_title(f"{title} — {label}")
        ax.set_aspect("equal")
        ax.legend(loc="upper left")

fig.tight_layout()
fig.savefig(paths.images_path / "07_forecast_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

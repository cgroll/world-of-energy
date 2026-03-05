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
# # Demand Nowcasting with XGBoost and BDEW Profiles
#
# XGBoost model that reconstructs quarter-hourly electricity consumption in
# Germany (2023–2025) from structural and temporal features.
#
# **Features**
# - 11 BDEW standard load profiles (H0, G0–G6, L0–L2) as structural regressors
# - Sin/cos encoding of month-of-year and hour-of-day for smooth seasonality
# - Day-type dummies: Saturday, Sunday, public holiday (DE national)
#
# **Methodology**
# - BDEW profile values are extracted via a seasonal × day-type lookup table
#   (independent of timestamp convention, timezone-agnostic)
# - Train: 2023–2024 · Test: 2025
# - Target: actual quarter-hourly consumption from SMARD (TOTAL_LOAD, DE-LU)

# %%
from datetime import datetime

import holidays as hol
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from demandlib import bdew
from woe.paths import ProjPaths
from woe.smard import Resolution, Region, Variable, download_smard_data

def show():
    """plt.show() wrapper: no-op when matplotlib uses a non-interactive backend."""
    try:
        plt.show()
    except Exception:
        pass


paths = ProjPaths()

YEARS = [2023, 2024, 2025]
PROFILES = ["h0", "g0", "g1", "g2", "g3", "g4", "g5", "g6", "l0", "l1", "l2"]

# %% [markdown]
# ## Quarter-hourly consumption data
#
# The SMARD download script (01) fetches total load at **hourly** resolution.
# This script downloads and caches the same variable at **quarter-hourly**
# resolution for 2023–2025.

# %%
qh_file = paths.smard_total_load_qh_file

if qh_file.exists():
    load_qh = pd.read_parquet(qh_file)
    print(f"Loaded {len(load_qh):,} records from cache ({qh_file.name})")
else:
    print("Downloading quarter-hourly TOTAL_LOAD from SMARD …")
    load_qh = download_smard_data(
        region=Region.DE_LU.value,
        resolution=Resolution.QUARTER_HOUR.value,
        variable=Variable.TOTAL_LOAD.value,
        variable_name="load_mw",
        start_time=datetime(YEARS[0], 1, 1),
    )
    paths.smard_downloads_path.mkdir(parents=True, exist_ok=True)
    load_qh.to_parquet(qh_file)
    print(f"Downloaded {len(load_qh):,} records → saved to {qh_file.name}")

load_qh = load_qh[load_qh.index.year.isin(YEARS)].copy()
print(f"\nTime range:     {load_qh.index[0]} → {load_qh.index[-1]}")
print(f"Records:        {len(load_qh):,}")
print(f"Missing values: {load_qh['load_mw'].isna().sum():,}")

# %% [markdown]
# ## BDEW feature lookup table
#
# Instead of aligning BDEW timestamps directly with SMARD (which may differ
# by one quarter-hour depending on period-start vs period-end convention),
# a lookup table is built from the reference profile:
#
# ```
# (season, day_type, quarter_of_day) → {profile: mean value}
# ```
#
# This mapping is timezone-agnostic and works regardless of the labelling
# convention used in the SMARD download.

# %%
# Generate BDEW profile for a reference non-leap year
ref_slp = bdew.ElecSlp(2023)
ref_df = ref_slp.slp_frame[PROFILES].copy()
ref_seasons = ref_slp._seasons


def classify_season(ts: pd.Timestamp, seasons: dict) -> str:
    """Return BDEW season name for a timestamp."""
    for name, (sm, sd, em, ed) in seasons.items():
        start = pd.Timestamp(ts.year, sm, sd)
        end = pd.Timestamp(ts.year, em, ed)
        if start <= ts.normalize() <= end:
            return name.rstrip("12")  # 'winter', 'transition', 'summer'
    return "unknown"


ref_df["season"] = [classify_season(ts, ref_seasons) for ts in ref_df.index]
ref_df["day_type"] = ref_df.index.map(
    lambda ts: "Workday" if ts.dayofweek < 5 else ("Saturday" if ts.dayofweek == 5 else "Sunday")
)
ref_df["qh"] = ref_df.index.hour * 4 + ref_df.index.minute // 15

# Build lookup: one mean value per (season, day_type, qh) cell
lookup = ref_df.groupby(["season", "day_type", "qh"])[PROFILES].mean()

print(f"Lookup table shape: {lookup.shape}  (3 seasons × 3 day-types × 96 quarter-hours)")
print(f"Seasons in lookup:  {sorted(lookup.index.get_level_values('season').unique())}")

# %% [markdown]
# ## Feature matrix

# %%
df = load_qh.dropna(subset=["load_mw"]).copy()

# German national public holidays
de_holidays = hol.Germany(years=YEARS)
holiday_dates = {d for d in de_holidays.keys()}

# Classify each timestamp
df["season"] = [classify_season(ts, ref_seasons) for ts in df.index]
df["day_type_raw"] = df.index.map(
    lambda ts: "Workday" if ts.dayofweek < 5 else ("Saturday" if ts.dayofweek == 5 else "Sunday")
)
# Holidays shift to "Sunday" category in BDEW convention
df["is_holiday"] = df.index.normalize().map(lambda d: d in holiday_dates).astype(int)
df["day_type"] = df["day_type_raw"].where(df["is_holiday"] == 0, "Sunday")
df["qh"] = df.index.hour * 4 + df.index.minute // 15

# Look up BDEW profile values
bdew_features = df.join(
    lookup, on=["season", "day_type", "qh"], how="left", rsuffix="_bdew"
)
for p in PROFILES:
    df[f"bdew_{p}"] = bdew_features[p].values

# Sin/cos temporal features
df["sin_month"] = np.sin(2 * np.pi * df.index.month / 12)
df["cos_month"] = np.cos(2 * np.pi * df.index.month / 12)
df["sin_hour"] = np.sin(2 * np.pi * df.index.hour / 24)
df["cos_hour"] = np.cos(2 * np.pi * df.index.hour / 24)

# Day-type dummies (Workday is the reference / baseline)
df["is_saturday"] = (df.index.dayofweek == 5).astype(int)
df["is_sunday"] = (df.index.dayofweek == 6).astype(int)
# is_holiday already computed above

# Drop helper columns and any remaining NaNs
df.drop(columns=["season", "day_type_raw", "day_type", "qh"], inplace=True)
df.dropna(inplace=True)

FEATURE_COLS = [c for c in df.columns if c != "load_mw"]
print(f"Feature matrix: {df.shape}")
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Holiday rows:  {df['is_holiday'].sum():,}")

# %% [markdown]
# ## Train / test split
#
# Train on 2023–2024, evaluate on 2025.

# %%
train = df[df.index.year < 2025]
test = df[df.index.year == 2025]

X_train, y_train = train[FEATURE_COLS], train["load_mw"]
X_test, y_test = test[FEATURE_COLS], test["load_mw"]

print(f"Train: {len(X_train):,} rows  {X_train.index[0].date()} → {X_train.index[-1].date()}")
print(f"Test:  {len(X_test):,} rows   {X_test.index[0].date()} → {X_test.index[-1].date()}")

# %% [markdown]
# ## XGBoost model

# %%
model = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# %% [markdown]
# ## Evaluation metrics

# %%
def _metrics(y_true, y_pred, label):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{label:<8} R²={r2:.4f}  MAE={mae:,.0f} MW  RMSE={rmse:,.0f} MW  MAPE={mape:.2f}%")
    return dict(r2=r2, mae=mae, rmse=rmse, mape=mape)

metrics_train = _metrics(y_train, y_pred_train, "Train")
metrics_test = _metrics(y_test, y_pred_test, "Test")

# %% [markdown]
# ## Feature importance

# %%
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()

_colors = []
for f in importances.index:
    if f.startswith("bdew_h"):
        _colors.append("#e6734a")
    elif f.startswith("bdew_g"):
        _colors.append("#4a90d9")
    elif f.startswith("bdew_l"):
        _colors.append("#5ab55e")
    else:
        _colors.append("#9b59b6")

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(importances.index, importances.values, color=_colors, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Feature importance (gain, normalised)")
ax.set_title("XGBoost feature importances — quarter-hourly demand model (2023–2024 training)", fontsize=11)
ax.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
legend_elements = [
    Patch(facecolor="#e6734a", label="BDEW Residential (H0)"),
    Patch(facecolor="#4a90d9", label="BDEW Commercial (G0–G6)"),
    Patch(facecolor="#5ab55e", label="BDEW Agricultural (L0–L2)"),
    Patch(facecolor="#9b59b6", label="Temporal / day-type"),
]
ax.legend(handles=legend_elements, loc="lower right")
fig.tight_layout()
fig.savefig(paths.images_path / "39_feature_importance.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/39_feature_importance.png
# :name: fig-39-feature-importance
# XGBoost feature importances (gain) for the quarter-hourly demand nowcasting
# model. BDEW profiles dominate — H0 (residential) typically ranks highest
# because households make up the largest share of German consumption. The sin/cos
# month features carry the annual heating cycle, while day-type dummies and
# sin/cos hour add intraday level adjustments.
# ```

# %% [markdown]
# ## Actual vs predicted — scatter (test 2025)

# %%
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_pred_test, alpha=0.08, s=2, color="#4a90d9", rasterized=True)
lims = [
    min(float(y_test.min()), float(y_pred_test.min())),
    max(float(y_test.max()), float(y_pred_test.max())),
]
ax.plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
ax.set_xlabel("Actual consumption (MW)")
ax.set_ylabel("Predicted consumption (MW)")
ax.set_title(
    f"Demand nowcast — actual vs predicted, test 2025\n"
    f"R²={metrics_test['r2']:.4f}  MAE={metrics_test['mae']:,.0f} MW  "
    f"RMSE={metrics_test['rmse']:,.0f} MW  MAPE={metrics_test['mape']:.2f}%",
    fontsize=10,
)
ax.legend()
fig.tight_layout()
fig.savefig(paths.images_path / "39_scatter_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/39_scatter_actual_vs_predicted.png
# :name: fig-39-scatter
# Scatter plot of actual vs predicted quarter-hourly consumption for the 2025
# test period. Points cluster tightly along the 1:1 line, confirming that BDEW
# profiles plus temporal sin/cos features capture the dominant variance in
# German electricity demand.
# ```

# %% [markdown]
# ## Sample week time series

# %%
y_pred_test_s = pd.Series(y_pred_test, index=test.index, name="predicted")

for label, start_date in [("winter", "2025-01-13"), ("summer", "2025-07-07")]:
    start = pd.Timestamp(start_date)
    end = start + pd.Timedelta(days=7)
    mask = (test.index >= start) & (test.index < end)
    actual = test.loc[mask, "load_mw"]
    predicted = y_pred_test_s.loc[mask]

    if actual.empty:
        print(f"No data for {label} week — skipping")
        continue

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(actual.index, actual.values, color="#333333", linewidth=1.0, label="Actual")
    ax.plot(predicted.index, predicted.values, color="#e6734a", linewidth=1.0,
            label="Predicted", alpha=0.85)
    ax.set_ylabel("Consumption (MW)")
    ax.set_title(f"Demand nowcast — {label} week 2025 ({start_date})", fontsize=11)
    ax.legend()
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(paths.images_path / f"39_sample_week_{label}.png", dpi=150, bbox_inches="tight")
    show()

# %% [markdown]
# ```{figure} ../../output/images/39_sample_week_winter.png
# :name: fig-39-week-winter
# Actual vs predicted quarter-hourly consumption for a representative winter week
# (January 2025). The model tracks the intraday double-peak (morning/evening)
# and captures the reduced weekend load.
# ```
#
# ```{figure} ../../output/images/39_sample_week_summer.png
# :name: fig-39-week-summer
# Actual vs predicted quarter-hourly consumption for a representative summer week
# (July 2025). The flatter intraday profile and weaker day-type contrast compared
# to winter are reproduced well.
# ```

# %% [markdown]
# ## Residuals by hour of day

# %%
residuals = pd.Series(
    y_test.values - y_pred_test, index=test.index, name="residual"
)
resid_by_hour = [residuals[residuals.index.hour == h].values for h in range(24)]

fig, ax = plt.subplots(figsize=(12, 5))
ax.boxplot(
    resid_by_hour,
    positions=range(24),
    widths=0.6,
    patch_artist=True,
    boxprops=dict(facecolor="#4a90d9", alpha=0.7),
    medianprops=dict(color="white", linewidth=1.5),
    flierprops=dict(marker=".", markersize=2, alpha=0.3),
)
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Residual (MW)")
ax.set_title("Prediction residuals by hour of day — test set 2025", fontsize=11)
ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], fontsize=8)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(paths.images_path / "39_residuals_by_hour.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/39_residuals_by_hour.png
# :name: fig-39-residuals
# Distribution of prediction residuals by hour of day for the 2025 test period.
# Medians near zero indicate unbiased predictions across all hours. Wider
# interquartile ranges during morning ramp-up (07:00–09:00) and evening peak
# (17:00–20:00) reflect higher uncertainty at demand transition points.
# ```

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
# XGBoost model that reconstructs hourly electricity consumption in
# Germany (2023–2025) from structural and temporal features.
#
# **Features**
# - 11 BDEW standard load profiles (H0, G0–G6, L0–L2) as structural regressors
# - Sin/cos encoding of month-of-year and hour-of-day for smooth seasonality
# - Day-type dummies: Saturday, Sunday, public holiday (DE national)
#
# **Methodology**
# - BDEW quarter-hourly profiles are aggregated to hourly means and stored in
#   a seasonal × day-type × hour-of-day lookup table (timezone-agnostic)
# - Train: 2023–2024 · Test: 2025
# - Target: actual hourly consumption from SMARD (TOTAL_LOAD, DE-LU)

# %%
import holidays as hol
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from demandlib import bdew
from woe.paths import ProjPaths

def show():
    """plt.show() wrapper: no-op when matplotlib uses a non-interactive backend."""
    try:
        plt.show()
    except Exception:
        pass


paths = ProjPaths()

YEARS = [2022, 2023, 2024, 2025]
PROFILES = ["h0", "g0", "g1", "g2", "g3", "g4", "g5", "g6", "l0", "l1", "l2"]

# %% [markdown]
# ## Hourly consumption data
#
# The SMARD download script (01) fetches total load at **hourly** resolution.
# We read that file directly — no additional download required.

# %%
load_h = pd.read_parquet(paths.smard_total_load_file)
load_h = load_h.rename(columns={"TOTAL_LOAD": "load_mw"})
load_h = load_h[load_h.index.year.isin(YEARS)].copy()

print(f"Time range:     {load_h.index[0]} → {load_h.index[-1]}")
print(f"Records:        {len(load_h):,}")
print(f"Missing values: {load_h['load_mw'].isna().sum():,}")

# %% [markdown]
# ## BDEW feature lookup table
#
# BDEW profiles are quarter-hourly. They are aggregated to hourly means and
# stored in a lookup keyed by:
#
# ```
# (season, day_type, hour_of_day) → {profile: mean value}
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
ref_df["hour"] = ref_df.index.hour

# Build lookup: aggregate 4 quarter-hourly slots per hour, then take mean per cell
lookup = ref_df.groupby(["season", "day_type", "hour"])[PROFILES].mean()

print(f"Lookup table shape: {lookup.shape}  (3 seasons × 3 day-types × 24 hours)")
print(f"Seasons in lookup:  {sorted(lookup.index.get_level_values('season').unique())}")

# %% [markdown]
# ## Feature matrix

# %%
df = load_h.dropna(subset=["load_mw"]).copy()

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
df["hour"] = df.index.hour

# Look up BDEW profile values
bdew_features = df.join(
    lookup, on=["season", "day_type", "hour"], how="left", rsuffix="_bdew"
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
df.drop(columns=["season", "day_type_raw", "day_type", "hour"], inplace=True)
df.dropna(inplace=True)

FEATURE_COLS = [c for c in df.columns if c != "load_mw"]
print(f"Feature matrix: {df.shape}")
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Holiday rows:  {df['is_holiday'].sum():,}")

# %% [markdown]
# ## Train / test split
#
# Train on 2023–2024, evaluate on 2022 and 2025 (out-of-sample in both
# directions).

# %%
train    = df[df.index.year.isin([2023, 2024])]
test_22  = df[df.index.year == 2022]
test_25  = df[df.index.year == 2025]

X_train,   y_train   = train[FEATURE_COLS],   train["load_mw"]
X_test_22, y_test_22 = test_22[FEATURE_COLS], test_22["load_mw"]
X_test_25, y_test_25 = test_25[FEATURE_COLS], test_25["load_mw"]

print(f"Train:     {len(X_train):,} rows  {X_train.index[0].date()} → {X_train.index[-1].date()}")
print(f"Test 2022: {len(X_test_22):,} rows  {X_test_22.index[0].date()} → {X_test_22.index[-1].date()}")
print(f"Test 2025: {len(X_test_25):,} rows  {X_test_25.index[0].date()} → {X_test_25.index[-1].date()}")

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
y_pred_22    = model.predict(X_test_22)
y_pred_25    = model.predict(X_test_25)

# %% [markdown]
# ## Evaluation metrics

# %%
def _metrics(y_true, y_pred, label):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{label:<12} R²={r2:.4f}  MAE={mae:,.0f} MW  RMSE={rmse:,.0f} MW  MAPE={mape:.2f}%")
    return dict(r2=r2, mae=mae, rmse=rmse, mape=mape)

metrics_train = _metrics(y_train,   y_pred_train, "Train")
metrics_22    = _metrics(y_test_22, y_pred_22,    "Test 2022")
metrics_25    = _metrics(y_test_25, y_pred_25,    "Test 2025")

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
ax.set_title("XGBoost feature importances — hourly demand model (2023–2024 training)", fontsize=11)
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
# XGBoost feature importances (gain) for the hourly demand nowcasting model.
# BDEW profiles dominate — H0 (residential) typically ranks highest because
# households make up the largest share of German consumption. The sin/cos month
# features carry the annual heating cycle, while day-type dummies and sin/cos
# hour add intraday level adjustments.
# ```

# %% [markdown]
# ## Actual vs predicted — scatter (test years)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax, year, y_true, y_pred, m in [
    (axes[0], 2022, y_test_22, y_pred_22, metrics_22),
    (axes[1], 2025, y_test_25, y_pred_25, metrics_25),
]:
    ax.scatter(y_true, y_pred, alpha=0.08, s=2, color="#4a90d9", rasterized=True)
    lims = [
        min(float(y_true.min()), float(y_pred.min())),
        max(float(y_true.max()), float(y_pred.max())),
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
    ax.set_xlabel("Actual consumption (MW)")
    ax.set_ylabel("Predicted consumption (MW)")
    ax.set_title(
        f"Demand nowcast — actual vs predicted, test {year}\n"
        f"R²={m['r2']:.4f}  MAE={m['mae']:,.0f} MW  "
        f"RMSE={m['rmse']:,.0f} MW  MAPE={m['mape']:.2f}%",
        fontsize=10,
    )
    ax.legend()

fig.tight_layout()
fig.savefig(paths.images_path / "39_scatter_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/39_scatter_actual_vs_predicted.png
# :name: fig-39-scatter
# Scatter plots of actual vs predicted hourly consumption for the 2022 (left)
# and 2025 (right) test years. Points cluster tightly along the 1:1 line in
# both out-of-sample periods, confirming that BDEW profiles plus temporal
# sin/cos features capture the dominant variance in German electricity demand.
# ```

# %% [markdown]
# ## Sample week time series

# %%
_test_sets = [
    (2022, test_22, pd.Series(y_pred_22, index=test_22.index, name="predicted")),
    (2025, test_25, pd.Series(y_pred_25, index=test_25.index, name="predicted")),
]

for yr, test_df, y_pred_s in _test_sets:
    for season, start_date in [("winter", f"{yr}-01-13"), ("summer", f"{yr}-07-07")]:
        start = pd.Timestamp(start_date)
        end = start + pd.Timedelta(days=7)
        mask = (test_df.index >= start) & (test_df.index < end)
        actual = test_df.loc[mask, "load_mw"]
        predicted = y_pred_s.loc[mask]

        if actual.empty:
            print(f"No data for {season} {yr} — skipping")
            continue

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(actual.index, actual.values, color="#333333", linewidth=1.0, label="Actual")
        ax.plot(predicted.index, predicted.values, color="#e6734a", linewidth=1.0,
                label="Predicted", alpha=0.85)
        ax.set_ylabel("Consumption (MW)")
        ax.set_title(f"Demand nowcast — {season} week {yr} ({start_date})", fontsize=11)
        ax.legend()
        ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(paths.images_path / f"39_sample_week_{yr}_{season}.png", dpi=150, bbox_inches="tight")
        show()

# %% [markdown]
# ```{figure} ../../output/images/39_sample_week_2022_winter.png
# :name: fig-39-week-2022-winter
# Actual vs predicted hourly consumption for a representative winter week
# (January 2022). The model tracks the intraday double-peak (morning/evening)
# and captures the reduced weekend load.
# ```
#
# ```{figure} ../../output/images/39_sample_week_2022_summer.png
# :name: fig-39-week-2022-summer
# Actual vs predicted hourly consumption for a representative summer week
# (July 2022).
# ```
#
# ```{figure} ../../output/images/39_sample_week_2025_winter.png
# :name: fig-39-week-2025-winter
# Actual vs predicted hourly consumption for a representative winter week
# (January 2025).
# ```
#
# ```{figure} ../../output/images/39_sample_week_2025_summer.png
# :name: fig-39-week-2025-summer
# Actual vs predicted hourly consumption for a representative summer week
# (July 2025). The flatter intraday profile and weaker day-type contrast compared
# to winter are reproduced well.
# ```

# %% [markdown]
# ## Residuals by hour of day

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)

for ax, year, y_true, y_pred in [
    (axes[0], 2022, y_test_22, y_pred_22),
    (axes[1], 2025, y_test_25, y_pred_25),
]:
    resid = pd.Series(y_true.values - y_pred, index=y_true.index)
    resid_by_hour = [resid[resid.index.hour == h].values for h in range(24)]
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
    ax.set_title(f"Prediction residuals by hour of day — test {year}", fontsize=11)
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
# Distribution of prediction residuals by hour of day for the 2022 (left) and
# 2025 (right) test periods. Medians near zero indicate unbiased predictions
# across all hours. Wider interquartile ranges during morning ramp-up
# (07:00–09:00) and evening peak (17:00–20:00) reflect higher uncertainty at
# demand transition points.
# ```

# %% [markdown]
# ## Rolling MAE over time

# %%
_WINDOWS = [(24, "#4a90d9", "24 h"), (72, "#e6a817", "72 h"), (240, "#e6734a", "240 h")]

for year, y_true, y_pred in [(2022, y_test_22, y_pred_22), (2025, y_test_25, y_pred_25)]:
    abs_err = pd.Series(np.abs(y_true.values - y_pred), index=y_true.index, name="abs_err")

    fig, ax = plt.subplots(figsize=(14, 4))
    for window, color, label in _WINDOWS:
        rolling_mae = abs_err.rolling(window, center=True, min_periods=1).mean()
        ax.plot(rolling_mae.index, rolling_mae.values, color=color, linewidth=1.2, label=label)

    ax.set_ylabel("Rolling MAE (MW)")
    ax.set_title(f"Rolling mean absolute error — test {year}", fontsize=11)
    ax.legend(title="Window")
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(paths.images_path / f"39_rolling_mae_{year}.png", dpi=150, bbox_inches="tight")
    show()

# %% [markdown]
# ```{figure} ../../output/images/39_rolling_mae_2022.png
# :name: fig-39-rolling-mae-2022
# Rolling mean absolute error for the 2022 test year at three smoothing
# windows (24 h, 72 h, 240 h). Peaks indicate periods where the structural
# BDEW + temporal features are insufficient to capture actual consumption,
# e.g. during unusual weather or public-holiday clusters.
# ```
#
# ```{figure} ../../output/images/39_rolling_mae_2025.png
# :name: fig-39-rolling-mae-2025
# Rolling mean absolute error for the 2025 test year. Comparing the two test
# years reveals whether the model's error structure is stable over time or
# drifts as structural consumption patterns evolve.
# ```

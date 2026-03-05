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
paths.de_load_estimation_path.mkdir(parents=True, exist_ok=True)
model.save_model(paths.de_load_baseline_model_file)
print(f"Saved baseline model → {paths.de_load_baseline_model_file}")

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

# %% [markdown]
# ## Weather features from PECD
#
# Country-level hourly weather for Germany (DE) is read from the processed
# PECD parquet file (script 36). Three raw fields are extracted and converted:
#
# | Feature | Source | Engineering |
# |---|---|---|
# | `temp_c` | 2 m air temperature (K → °C) | raw |
# | `hdh` | `temp_c` | `max(0, 15.5 − T)` — heating degree hours |
# | `cdh` | `temp_c` | `max(0, T − 22)` — cooling degree hours |
# | `wind_speed` | 10 m wind speed (m/s) | raw |
# | `ghi` | Surface downwelling shortwave radiation (W/m²) | raw |
#
# **Timestamp alignment** — PECD/ERA5 uses hour-starting UTC; SMARD uses
# hour-ending CET. Conversion: localize UTC → convert to CET/CEST → shift +1 h
# to obtain the hour-ending label → strip timezone to match the naive SMARD index.

# %%
pecd = pd.read_parquet(paths.pecd_processed_file)

temp_k   = pecd[("2m_air_temperature",                     "value", "DE")]
wind_raw = pecd[("wind_speed_at_10m",                      "value", "DE")]
ghi_raw  = pecd[("surface_downwelling_shortwave_radiation", "value", "DE")]


def pecd_to_smard_index(s: pd.Series) -> pd.Series:
    """Convert PECD hour-starting UTC → hour-ending CET naive index (SMARD convention)."""
    return (
        s.tz_localize("UTC")
         .tz_convert("Europe/Berlin")
         .shift(1, freq="h")
         .tz_localize(None)
    )


weather = pd.DataFrame({
    "temp_c":     pecd_to_smard_index(temp_k) - 273.15,
    "wind_speed": pecd_to_smard_index(wind_raw),
    "ghi":        pecd_to_smard_index(ghi_raw),
})
weather["hdh"] = (15.5 - weather["temp_c"]).clip(lower=0)
weather["cdh"] = (weather["temp_c"] - 22.0).clip(lower=0)

WEATHER_COLS = ["temp_c", "hdh", "cdh", "wind_speed", "ghi"]

print(f"Weather data:  {weather.index[0]} – {weather.index[-1]}  ({len(weather):,} rows)")
print(f"Missing:       {weather[WEATHER_COLS].isna().sum().to_dict()}")
print(weather[WEATHER_COLS].describe().round(2))

# %% [markdown]
# ## Enhanced feature matrix (BDEW + weather)

# %%
df2 = df.join(weather[WEATHER_COLS], how="left")
df2.dropna(inplace=True)

FEATURE_COLS2 = [c for c in df2.columns if c != "load_mw"]
print(f"Enhanced feature matrix: {df2.shape}")
print(f"Features ({len(FEATURE_COLS2)}): {FEATURE_COLS2}")

# %% [markdown]
# ## XGBoost model with weather features

# %%
train2    = df2[df2.index.year.isin([2023, 2024])]
test2_22  = df2[df2.index.year == 2022]
test2_25  = df2[df2.index.year == 2025]

X_train2,   y_train2   = train2[FEATURE_COLS2],   train2["load_mw"]
X_test2_22, y_test2_22 = test2_22[FEATURE_COLS2], test2_22["load_mw"]
X_test2_25, y_test2_25 = test2_25[FEATURE_COLS2], test2_25["load_mw"]

print(f"Train:     {len(X_train2):,} rows  {X_train2.index[0].date()} → {X_train2.index[-1].date()}")
print(f"Test 2022: {len(X_test2_22):,} rows")
print(f"Test 2025: {len(X_test2_25):,} rows")

model2 = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
model2.fit(X_train2, y_train2)
model2.save_model(paths.de_load_weather_model_file)
print(f"Saved weather model  → {paths.de_load_weather_model_file}")

y_pred2_train = model2.predict(X_train2)
y_pred2_22    = model2.predict(X_test2_22)
y_pred2_25    = model2.predict(X_test2_25)

# %% [markdown]
# ## Metrics: baseline vs weather-enhanced

# %%
metrics2_train = _metrics(y_train2,   y_pred2_train, "Train")
metrics2_22    = _metrics(y_test2_22, y_pred2_22,    "Test 2022")
metrics2_25    = _metrics(y_test2_25, y_pred2_25,    "Test 2025")

# Build side-by-side comparison table
_rows = [
    ("Train (in-sample)",  metrics_train,  metrics2_train),
    ("Test 2022 (OOS)",    metrics_22,     metrics2_22),
    ("Test 2025 (OOS)",    metrics_25,     metrics2_25),
]
_records = []
for split, mb, mw in _rows:
    _records.append({
        "Split":         split,
        "R² base":       round(mb["r2"],   4),
        "R² weather":    round(mw["r2"],   4),
        "ΔR²":           round(mw["r2"]   - mb["r2"],   4),
        "MAE base":      round(mb["mae"],  0),
        "MAE weather":   round(mw["mae"],  0),
        "ΔMAE":          round(mw["mae"]  - mb["mae"],  0),
        "MAPE base":     round(mb["mape"], 2),
        "MAPE weather":  round(mw["mape"], 2),
        "ΔMAPE":         round(mw["mape"] - mb["mape"], 2),
    })

cmp = pd.DataFrame(_records).set_index("Split")
print("\nModel comparison — baseline vs weather-enhanced")
print(cmp.to_string())

# %% [markdown]
# ## Feature importance — weather-enhanced model

# %%
importances2 = pd.Series(model2.feature_importances_, index=FEATURE_COLS2).sort_values()

_colors2 = []
for f in importances2.index:
    if f.startswith("bdew_h"):
        _colors2.append("#e6734a")
    elif f.startswith("bdew_g"):
        _colors2.append("#4a90d9")
    elif f.startswith("bdew_l"):
        _colors2.append("#5ab55e")
    elif f in WEATHER_COLS:
        _colors2.append("#e6a817")
    else:
        _colors2.append("#9b59b6")

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(importances2.index, importances2.values, color=_colors2, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Feature importance (gain, normalised)")
ax.set_title("XGBoost feature importances — weather-enhanced model (2023–2024 training)", fontsize=11)
ax.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
legend_elements2 = [
    Patch(facecolor="#e6734a", label="BDEW Residential (H0)"),
    Patch(facecolor="#4a90d9", label="BDEW Commercial (G0–G6)"),
    Patch(facecolor="#5ab55e", label="BDEW Agricultural (L0–L2)"),
    Patch(facecolor="#9b59b6", label="Temporal / day-type"),
    Patch(facecolor="#e6a817", label="Weather (temp_c, HDH, CDH, wind, GHI)"),
]
ax.legend(handles=legend_elements2, loc="lower right")
fig.tight_layout()
fig.savefig(paths.images_path / "39_feature_importance_weather.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/39_feature_importance_weather.png
# :name: fig-39-feature-importance-weather
# XGBoost feature importances for the weather-enhanced model. Temperature-derived
# features (HDH, CDH, temp_c) typically displace the sin/cos month features as
# the dominant seasonal signal because they directly encode the heating and cooling
# response rather than a smooth harmonic proxy. GHI captures the lighting-demand
# reduction on bright days and provides a signal orthogonal to temperature.
# ```

# %% [markdown]
# ## Actual vs predicted — scatter, weather-enhanced

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax, year, y_true, y_pred, m in [
    (axes[0], 2022, y_test2_22, y_pred2_22, metrics2_22),
    (axes[1], 2025, y_test2_25, y_pred2_25, metrics2_25),
]:
    ax.scatter(y_true, y_pred, alpha=0.08, s=2, color="#e6a817", rasterized=True)
    lims = [
        min(float(y_true.min()), float(y_pred.min())),
        max(float(y_true.max()), float(y_pred.max())),
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
    ax.set_xlabel("Actual consumption (MW)")
    ax.set_ylabel("Predicted consumption (MW)")
    ax.set_title(
        f"Weather-enhanced demand nowcast — test {year}\n"
        f"R²={m['r2']:.4f}  MAE={m['mae']:,.0f} MW  "
        f"RMSE={m['rmse']:,.0f} MW  MAPE={m['mape']:.2f}%",
        fontsize=10,
    )
    ax.legend()

fig.tight_layout()
fig.savefig(paths.images_path / "39_scatter_weather.png", dpi=150, bbox_inches="tight")
show()

# %% [markdown]
# ```{figure} ../../output/images/39_scatter_weather.png
# :name: fig-39-scatter-weather
# Scatter plots of actual vs predicted hourly consumption for the 2022 (left) and
# 2025 (right) test years using the weather-enhanced model. Compared to
# {numref}`fig-39-scatter`, tighter clustering along the 1:1 line reflects the
# temperature-driven variance that BDEW structural profiles alone cannot
# capture — particularly at extreme cold spells (upper right) and mild
# shoulder-season hours (centre of the distribution).
# ```

# %% [markdown]
# ## Rolling MAE over time — weather-enhanced model

# %%
for year, y_true, y_pred in [(2022, y_test2_22, y_pred2_22), (2025, y_test2_25, y_pred2_25)]:
    abs_err2 = pd.Series(np.abs(y_true.values - y_pred), index=y_true.index, name="abs_err")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Top panel: both models side by side — align baseline to weather-model index
    _y_true_base = y_test_22  if year == 2022 else y_test_25
    _y_pred_base = y_pred_22  if year == 2022 else y_pred_25
    abs_err_base = (
        pd.Series(np.abs(_y_true_base.values - _y_pred_base), index=_y_true_base.index)
        .reindex(y_true.index)
    )
    for window, color, label in _WINDOWS:
        axes[0].plot(
            abs_err_base.rolling(window, center=True, min_periods=1).mean(),
            color=color, linewidth=1.0, linestyle="--", alpha=0.7, label=f"Base {label}",
        )
        axes[0].plot(
            abs_err2.rolling(window, center=True, min_periods=1).mean(),
            color=color, linewidth=1.2, label=f"Weather {label}",
        )
    axes[0].set_ylabel("Rolling MAE (MW)")
    axes[0].set_title(f"Rolling MAE — baseline (dashed) vs weather-enhanced (solid), test {year}", fontsize=11)
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].yaxis.grid(True, linewidth=0.4, alpha=0.6)
    axes[0].set_axisbelow(True)

    # Bottom panel: improvement (negative = weather model is better)
    for window, color, label in _WINDOWS:
        delta = (
            abs_err2.rolling(window, center=True, min_periods=1).mean()
            - abs_err_base.rolling(window, center=True, min_periods=1).mean()
        )
        axes[1].plot(delta.index, delta.values, color=color, linewidth=1.2, label=label)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("ΔMAE weather − base (MW)")
    axes[1].set_title("MAE improvement from weather features (negative = better)", fontsize=11)
    axes[1].legend(title="Window", fontsize=8)
    axes[1].yaxis.grid(True, linewidth=0.4, alpha=0.6)
    axes[1].set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(paths.images_path / f"39_rolling_mae_weather_{year}.png", dpi=150, bbox_inches="tight")
    show()

# %% [markdown]
# ```{figure} ../../output/images/39_rolling_mae_weather_2022.png
# :name: fig-39-rolling-mae-weather-2022
# Rolling MAE for the 2022 test year. Top: baseline (dashed) vs weather-enhanced
# (solid) at three smoothing windows. Bottom: MAE difference (weather − baseline);
# values below zero indicate hours where adding temperature, HDH/CDH, wind speed
# and GHI reduced absolute error. Persistent negative periods correspond to cold
# spells and temperature-driven demand episodes that BDEW profiles alone
# cannot resolve.
# ```
#
# ```{figure} ../../output/images/39_rolling_mae_weather_2025.png
# :name: fig-39-rolling-mae-weather-2025
# Rolling MAE comparison for the 2025 test year. The improvement pattern reveals
# whether the weather signal remains stable out-of-sample or whether the
# temperature–demand relationship drifts between the training period (2023–2024)
# and the most recent test year.
# ```

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
# # BDEW Standard Load Profiles
#
# The BDEW (Bundesverband der Energie- und Wasserwirtschaft) publishes standard
# load profiles (Standardlastprofile, SLP) that describe typical annual
# electricity demand patterns for different consumer categories in Germany.
# These profiles are used to estimate hourly consumption for metered customers
# who lack smart meters.
#
# Profiles are generated for a reference year using `demandlib`, which
# implements the official BDEW tables with seasonal and day-type differentiation.

# %%
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from demandlib import bdew
from woe.paths import ProjPaths

paths = ProjPaths()

# %%
# Generate quarter-hourly profiles for a representative non-leap year
slp = bdew.ElecSlp(2023)
df = slp.slp_frame
df_hourly = df.resample("h").mean()

print(f"Profiles:    {df.columns.tolist()}")
print(f"Time range:  {df.index[0]} → {df.index[-1]}")
print(f"Resolution:  15-minute (quarter-hourly)")
print(f"Timesteps:   {len(df):,}")

# %% [markdown]
# ## Profile overview
#
# | Profile  | Sector       | Description                                |
# |----------|--------------|--------------------------------------------|
# | H0       | Residential  | Haushalte (households)                     |
# | G0       | Commercial   | Gewerbe allgemein (general commercial)     |
# | G1       | Commercial   | Weekdays 8–18 h (office hours)             |
# | G2       | Commercial   | High evening / night consumption           |
# | G3       | Commercial   | Durchlaufend (continuous 24/7)             |
# | G4       | Commercial   | Laden / Friseur (shop / hairdresser)       |
# | G5       | Commercial   | Bäckerei (bakery with production)          |
# | G6       | Commercial   | Wochenendbetrieb (weekend-heavy operation) |
# | L0       | Agriculture  | Landwirtschaft allgemein (general)         |
# | L1       | Agriculture  | With milking plant                         |
# | L2       | Agriculture  | Other agriculture                          |
# | H0 dyn   | Residential  | Dynamic H0 variant                         |

# %%
PROFILE_LABELS = {
    "h0":    "H0 — Haushalte (residential)",
    "g0":    "G0 — Gewerbe allgemein (general commercial)",
    "g1":    "G1 — Gewerbe 8–18 h (office hours)",
    "g2":    "G2 — Gewerbe Abend/Nacht (evening/night-heavy)",
    "g3":    "G3 — Gewerbe durchlaufend (24/7 continuous)",
    "g4":    "G4 — Laden/Friseur (shop/hairdresser)",
    "g5":    "G5 — Bäckerei (bakery)",
    "g6":    "G6 — Wochenendbetrieb (weekend-heavy)",
    "l0":    "L0 — Landwirtschaft allgemein (general agriculture)",
    "l1":    "L1 — Landwirtschaft Melkanlage (milking plant)",
    "l2":    "L2 — Andere Landwirtschaft (other agriculture)",
    "h0_dyn": "H0 dyn — Dynamisches H0 (dynamic variant)",
}

peak_to_mean = df.max() / df.mean()
peak_to_mean_sorted = peak_to_mean.drop("h0_dyn").sort_values(ascending=True)

print(f"{'Profile':<10} {'Peak-to-mean ratio':>20}")
print("-" * 32)
for col, val in peak_to_mean_sorted.items():
    print(f"{col:<10} {val:>20.2f}")

# %%
fig, ax = plt.subplots(figsize=(10, 6))
colors = [
    "#e6734a" if c.startswith("h") else
    "#4a90d9" if c.startswith("g") else
    "#5ab55e"
    for c in peak_to_mean_sorted.index
]
ax.barh(
    [PROFILE_LABELS[c] for c in peak_to_mean_sorted.index],
    peak_to_mean_sorted.values,
    color=colors,
    edgecolor="white",
    linewidth=0.4,
)
ax.set_xlabel("Peak-to-mean ratio")
ax.set_title("BDEW standard load profiles — peak-to-mean ratio (2023)", fontsize=11)
ax.xaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)

legend_elements = [
    Patch(facecolor="#e6734a", label="Residential"),
    Patch(facecolor="#4a90d9", label="Commercial"),
    Patch(facecolor="#5ab55e", label="Agriculture"),
]
ax.legend(handles=legend_elements, loc="lower right")

fig.tight_layout()
fig.savefig(paths.images_path / "38_profile_overview.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/38_profile_overview.png
# :name: fig-38-profile-overview
# Peak-to-mean ratio for each BDEW standard load profile (2023). Higher values
# indicate more peaked demand patterns. The bakery profile (G5) has the highest
# ratio due to intense early-morning load; residential (H0) shows a moderate
# peak driven by morning and evening consumption spikes. The 24/7 commercial
# profile (G3) is the flattest, reflecting round-the-clock operation.
# ```

# %% [markdown]
# ## Season × day-type profiles for all BDEW load profiles
#
# For each profile: nine curves (three seasons × three day types) on the left,
# and a 3 × 3 heatmap of mean load relative to the minimum cell on the right.

# %%
def _season_of(ts, seasons):
    for name, (sm, sd, em, ed) in seasons.items():
        start = pd.Timestamp(ts.year, sm, sd)
        end   = pd.Timestamp(ts.year, em, ed)
        if start <= ts.normalize() <= end:
            return name.rstrip("12")  # 'winter', 'transition', or 'summer'
    return "unknown"


season_tags = pd.Series(
    [_season_of(ts, slp._seasons) for ts in df_hourly.index],
    index=df_hourly.index,
)
daytype_tags = pd.Series(
    df_hourly.index.dayofweek.map(
        lambda d: "Workday" if d < 5 else ("Saturday" if d == 5 else "Sunday")
    ),
    index=df_hourly.index,
)

SEASON_COLORS  = {"winter": "#4a90d9", "transition": "#e6a817", "summer": "#5ab55e"}
DAYTYPE_STYLES = {"Workday": "-", "Saturday": "--", "Sunday": ":"}

_legend_elements = [
    Line2D([0], [0], color=SEASON_COLORS["winter"],     linewidth=2, label="Winter"),
    Line2D([0], [0], color=SEASON_COLORS["transition"], linewidth=2, label="Transition"),
    Line2D([0], [0], color=SEASON_COLORS["summer"],     linewidth=2, label="Summer"),
    Line2D([0], [0], color="gray", linestyle="-",  linewidth=2, label="Workday"),
    Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label="Saturday"),
    Line2D([0], [0], color="gray", linestyle=":",  linewidth=2, label="Sunday"),
]

# Pre-compute all grids to establish a shared colour scale
_all_grids = {}
for _p in peak_to_mean_sorted.index:
    _s = df_hourly[_p]
    _p9 = (
        pd.DataFrame({
            "val":     _s,
            "season":  season_tags,
            "daytype": daytype_tags,
            "hour":    _s.index.hour,
        })
        .groupby(["season", "daytype", "hour"])["val"]
        .mean()
        .unstack("hour")
    )
    _g = (
        _p9.mean(axis=1)
        .unstack(level=1)
        .loc[["winter", "transition", "summer"], ["Workday", "Saturday", "Sunday"]]
    )
    _g_norm = _g / _g.values.min()
    _all_grids[_p] = (_p9, _g_norm)

HEATMAP_VMIN = 1.0
HEATMAP_VMAX = 2.0  # cap: 1.0 (blue) → 2.0 (red); values above clip to red
_cmap_obj  = plt.colormaps["coolwarm"]
_norm_obj  = mcolors.Normalize(vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)

for profile in peak_to_mean_sorted.index:
    profiles_9, grid_9_norm = _all_grids[profile]

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(16, 5),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # Left: 9 season × day-type lines
    for season in ["winter", "transition", "summer"]:
        for daytype in ["Workday", "Saturday", "Sunday"]:
            ax_l.plot(
                range(24), profiles_9.loc[(season, daytype)].values,
                color=SEASON_COLORS[season],
                linestyle=DAYTYPE_STYLES[daytype],
                linewidth=2,
            )
    ax_l.legend(handles=_legend_elements, fontsize=9, ncol=2)
    ax_l.set_xlabel("Hour of day")
    ax_l.set_ylabel("Relative load (normalised)")
    ax_l.set_xticks(range(0, 24, 2))
    ax_l.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax_l.set_title(
        f"BDEW {PROFILE_LABELS[profile]} — season × day-type profiles (2023)",
        fontsize=11,
    )
    ax_l.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax_l.set_axisbelow(True)

    # Right: 3 × 3 normalised heatmap — shared blue-to-red scale across all profiles
    im = ax_r.imshow(
        grid_9_norm.values, aspect="auto", cmap="coolwarm",
        vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX,
    )
    cbar = plt.colorbar(im, ax=ax_r, pad=0.01)
    cbar.set_label("Load relative to minimum (min = 1.0)")
    ax_r.set_xticks(range(3))
    ax_r.set_xticklabels(["Workday", "Saturday", "Sunday"])
    ax_r.set_yticks(range(3))
    ax_r.set_yticklabels(["Winter", "Transition", "Summer"])
    for row in range(3):
        for col in range(3):
            val = grid_9_norm.values[row, col]
            rgba = _cmap_obj(_norm_obj(val))
            lum  = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax_r.text(
                col, row, f"{val:.2f}×",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="black" if lum > 0.45 else "white",
            )
    ax_r.set_title(
        f"BDEW {profile.upper()} — mean load relative to minimum",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(
        paths.images_path / f"38_{profile}_season_daytype.png",
        dpi=150, bbox_inches="tight",
    )
    plt.show()

# %% [markdown]
# ```{figure} ../../output/images/38_h0_season_daytype.png
# :name: fig-38-h0-season-daytype
# BDEW H0 residential profile: nine season × day-type curves (left) and mean
# load relative to the summer-Sunday minimum (right). The seasonal gradient
# dominates over the day-type gradient; winter workdays carry the highest
# average load.
# ```
#
# ```{figure} ../../output/images/38_g0_season_daytype.png
# :name: fig-38-g0-season-daytype
# BDEW G0 general commercial profile: demand is concentrated in daytime hours
# on workdays and drops sharply on weekends. Seasonal variation is modest
# compared to H0.
# ```
#
# ```{figure} ../../output/images/38_g1_season_daytype.png
# :name: fig-38-g1-season-daytype
# BDEW G1 office-hours commercial profile: load is tightly concentrated between
# 08:00 and 18:00 on workdays and nearly absent on weekends and at night.
# ```
#
# ```{figure} ../../output/images/38_g2_season_daytype.png
# :name: fig-38-g2-season-daytype
# BDEW G2 evening/night-heavy commercial profile: consumption ramps up in the
# late afternoon and peaks in the evening, the inverse of the office-hours shape.
# ```
#
# ```{figure} ../../output/images/38_g3_season_daytype.png
# :name: fig-38-g3-season-daytype
# BDEW G3 continuous 24/7 commercial profile: load is nearly flat across hours,
# day types, and seasons, confirming round-the-clock operation with minimal
# temporal variation.
# ```
#
# ```{figure} ../../output/images/38_g4_season_daytype.png
# :name: fig-38-g4-season-daytype
# BDEW G4 shop/hairdresser profile: daytime-only load with a mid-morning peak,
# active on both weekdays and Saturdays, minimal on Sundays.
# ```
#
# ```{figure} ../../output/images/38_g5_season_daytype.png
# :name: fig-38-g5-season-daytype
# BDEW G5 bakery profile: the most peaked of all BDEW profiles, with intense
# early-morning consumption driven by overnight and pre-dawn production.
# ```
#
# ```{figure} ../../output/images/38_g6_season_daytype.png
# :name: fig-38-g6-season-daytype
# BDEW G6 weekend-heavy commercial profile: load is highest on Saturdays and
# Sundays and substantially lower on weekdays, the inverse of most commercial
# profiles.
# ```
#
# ```{figure} ../../output/images/38_l0_season_daytype.png
# :name: fig-38-l0-season-daytype
# BDEW L0 general agriculture profile: relatively flat load spread across the
# day, with moderate seasonal variation and little day-type differentiation.
# ```
#
# ```{figure} ../../output/images/38_l1_season_daytype.png
# :name: fig-38-l1-season-daytype
# BDEW L1 agriculture with milking plant profile: two distinct peaks aligned
# with morning and evening milking times, visible across all seasons and day
# types.
# ```
#
# ```{figure} ../../output/images/38_l2_season_daytype.png
# :name: fig-38-l2-season-daytype
# BDEW L2 other agriculture profile: broader daytime load than L1, without the
# sharp milking peaks, and with modest seasonal variation.
# ```

# %%

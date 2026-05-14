
# ── 1. IMPORTS ────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path
from linearmodels.panel import PanelOLS
from rdrobust import rdbwselect


# ── 2. CONSTANTS ──────────────────────────────────────────────────────────────
CITY_CENTERS = {
    'houston':        (-95.3698, 29.7604),
    'san antonio':    (-98.4936, 29.4241),
    'dallas':         (-96.7970, 32.7767),
    'austin':         (-97.7431, 30.2672),
    'jacksonville':   (-81.6557, 30.3322),
    'fort worth':     (-97.3308, 32.7555),
    'charlotte':      (-80.8431, 35.2271),
    'el paso':        (-106.4800, 31.7776),
    'washington':     (-77.0369, 38.9072),
    'nashville':      (-86.7816, 36.1627),
    'oklahoma city':  (-97.5164, 35.4676),
    'atlanta':        (-84.3880, 33.7490),
    'virginia beach': (-76.0929, 36.8529),
    'raleigh':        (-78.6382, 35.7796),
    'miami':          (-80.1918, 25.7617),
    'tampa':          (-82.4572, 27.9506),
    'tulsa':          (-95.9928, 36.1540),
    'arlington':      (-97.1081, 32.7357),
    'new orleans':    (-90.0715, 29.9511),
    'corpus christi': (-97.4034, 27.8006),
}

PARK_DIST_CITIES = ['arlington', 'corpus christi', 'miami', 'tulsa']

# Bandwidth / bin parameters (used by models and plots)
BIN_WIDTH_M = 150     # Plot 2: bin width (m)
BIN_MAX_M   = 3_000   # Plot 2: max distance (m)
RDD_BW_M    = None    # Model 4 / Plot 3: set dynamically via rdbwselect
RDD_BIN_W   = 50      # Plot 3: bin width (m)
BW_COOL     = 1_000   # Plot 4: bandwidth (m)
BIN_COOL    = 50      # Plot 4: bin width (m)
FF_M        = 500     # Plot 4: far-field threshold for detrending (m)


# ── 3. HELPER FUNCTIONS ───────────────────────────────────────────────────────
def _dist_to_center(grp):
    key = grp['city'].iloc[0].lower()
    if key not in CITY_CENTERS:
        return pd.Series(np.nan, index=grp.index)
    clon, clat = CITY_CENTERS[key]
    lat_rad = np.radians((grp['latitude'].mean() + clat) / 2)
    dlat_m = (grp['latitude'].values - clat) * 111_000
    dlon_m = (grp['longitude'].values - clon) * 111_000 * np.cos(lat_rad)
    return pd.Series(np.sqrt(dlat_m**2 + dlon_m**2), index=grp.index)



# ── 4. DATA LOAD & PREP ───────────────────────────────────────────────────────
raw = pd.read_csv("/home/wwdelvalle/20_cities_panel_clean.csv", low_memory=False)

raw['dist_to_center_m'] = raw.groupby('city', group_keys=False).apply(_dist_to_center)
print(f"dist_to_center_m — {raw['dist_to_center_m'].notna().sum():,} non-null  "
      f"| mean {raw['dist_to_center_m'].mean():.1f} km  "
      f"| max {raw['dist_to_center_m'].max():.1f} km")

# Model 1 sample: all non-water pixels across all cities
df_all = raw[raw["is_water"] == 0].copy()
print(f"Observations (non-water, all cities): {len(df_all):,}")
df_all = df_all.set_index(["city", "year"])

# Model 4 sample basis: non-water pixels with park distance data
df = raw[(raw["is_water"] == 0) & raw["dist_to_park_m"].notna()].copy()
df = df.set_index(["city", "year"])
park_cities = df.index.get_level_values("city").unique().tolist()


# ── 5. MODELS ─────────────────────────────────────────────────────────────────

# ── Model 1: LST ~ NDVI + near_water | city + year FE ────────────────────────
print("="*80)
print("MODEL 1: LST ~ NDVI + near_water  (city + year FE)")
print("="*80)

model1 = PanelOLS.from_formula(
    "LST ~ NDVI + near_water + EntityEffects + TimeEffects",
    data=df_all
)
result1 = model1.fit(cov_type="clustered", cluster_entity=True)
print(result1.summary)

# ── Model 3: per city — LST ~ log(dist) × park_size_cat + dist_to_center + year FE ──
print("\n" + "="*80)
print("MODEL 3 (per city): LST ~ log(dist_to_park_m) * park_size_cat + dist_to_center_m + year FE")
print("="*80)

results3 = {}
for city_name in PARK_DIST_CITIES:
    df_city = raw[
        (raw['is_water'] == 0) &
        (raw['in_park_patch'] == 0) &
        (raw['dist_to_park_m'].notna()) &
        (raw['city'].str.lower() == city_name) &
        (raw['year'].between(2011, 2021))
    ].copy()
    df_city['park_size_cat'] = pd.Categorical(
        df_city['nearest_park_size'].map({1: 'small', 2: 'medium', 3: 'large'}),
        categories=['small', 'medium', 'large'],
        ordered=True,
    )
    df_city = df_city[df_city['park_size_cat'].notna()]
    if len(df_city) < 2 or df_city['park_size_cat'].nunique() < 2:
        print(f"\n{city_name.title()}: insufficient park size variety — skipped")
        continue
    df_city['log_dist'] = np.log(df_city['dist_to_park_m'] + 1)
    res = smf.ols(
        "LST ~ log_dist * C(park_size_cat) + dist_to_center_m + C(year)",
        data=df_city
    ).fit(cov_type='HC1')
    results3[city_name] = res
    print(f"\n{'='*40}")
    print(f"City: {city_name.title()}  (n={len(df_city):,})")
    print(f"{'='*40}")
    print(res.summary())


# ── Model 4: RDD — park boundary threshold ────────────────────────────────────
# Running variable: signed distance to park boundary
#   negative = inside park (treated), 0 = boundary, positive = outside (control)
# Treatment D: 1 = inside park patch, 0 = outside
# Bandwidth: CCT MSE-optimal, selected via rdbwselect(bwselect="mserd")
# Note: NDVI excluded — mechanically higher inside parks, would absorb the effect.


# ── CCT optimal bandwidth — exterior pixels only ─────────────────────────────
_bw_data = raw[
    (raw["city"].isin(park_cities)) &
    (raw["is_water"] == 0) &
    (raw["in_park_patch"] == 0) &
    raw["dist_to_park_m"].notna() &
    raw["LST"].notna()
].copy()
if len(_bw_data) > 50_000:
    _bw_data = _bw_data.sample(50_000, random_state=42)
_bw_sel = rdbwselect(
    y=_bw_data["LST"].values,
    x=_bw_data["dist_to_park_m"].values,
    c=0,
    bwselect="mserd"
)
RDD_BW_M = int(round(float(_bw_sel.bws.iloc[0, 0])))
print(f"\nCCT optimal bandwidth (exterior pixels only): {RDD_BW_M} m")

# ── Signed running variable: negative inside, 0 = boundary, positive outside ─
raw["_rv"] = np.where(
    raw["in_park_patch"] == 1,
    -raw["interior_dist_m"],
    raw["dist_to_park_m"]
)

print("\n" + "="*80)
print(f"MODEL 4: RDD — park boundary threshold (bandwidth = {RDD_BW_M} m, CCT optimal)")
print("         LST ~ D(in_park) + running_var + near_water  (city + year FE)")
print("="*80)

rdd = raw[
    (raw["is_water"] == 0) &
    (raw["city"].isin(park_cities)) &
    raw["_rv"].notna() &
    (raw["_rv"].abs() <= RDD_BW_M) &
    (raw["year"].between(2011, 2021))
].copy()

rdd["running_var"] = rdd["_rv"]
rdd["D"] = rdd["in_park_patch"].astype(int)             # 1 = inside park
raw.drop(columns=["_rv"], inplace=True)

print(f"RDD observations (bandwidth {RDD_BW_M} m): {len(rdd):,}")
print("Treatment breakdown:\n", rdd["D"].value_counts(), "\n")

rdd = rdd.set_index(["city", "year"])

model4 = PanelOLS.from_formula(
    "LST ~ D + running_var + near_water + EntityEffects + TimeEffects",
    data=rdd
)
result4 = model4.fit(cov_type="clustered", cluster_entity=True)
print(result4.summary)


# ── 6. ANALYSIS TABLES ────────────────────────────────────────────────────────

# ── Near/far comparison ───────────────────────────────────────────────────────
# Compares mean LST across three distance bands, raw and residualized.
# Residualized = within-demean by city-year so city/seasonal differences
# don't drive the result — only the distance gradient matters.
print("\n" + "="*80)
print("NEAR/FAR COMPARISON: Mean LST by distance band")
print("="*80)

_comp = raw[
    (raw["is_water"] == 0) &
    raw["dist_to_park_m"].notna() &
    (raw["dist_to_park_m"] > 0) &
    raw["LST"].notna()
].copy()

_comp["band"] = pd.cut(
    _comp["dist_to_park_m"],
    bins=[0, 300, 600, np.inf],
    labels=["0–300 m (near park)", "300–600 m (transition)", "600 m+ (far from park)"]
)

# Raw means
_raw_tbl = (
    _comp.groupby("band", observed=True)["LST"]
    .agg(mean="mean", std="std", n="count")
)
_raw_tbl["se"]   = _raw_tbl["std"] / np.sqrt(_raw_tbl["n"])
_raw_tbl["ci95"] = 1.96 * _raw_tbl["se"]

# Residualized means (within city-year demean)
_comp["LST_dm"] = (
    _comp["LST"] - _comp.groupby(["city", "year"])["LST"].transform("mean")
)
_resid_tbl = (
    _comp.groupby("band", observed=True)["LST_dm"]
    .agg(mean="mean", std="std", n="count")
)
_resid_tbl["se"]   = _resid_tbl["std"] / np.sqrt(_resid_tbl["n"])
_resid_tbl["ci95"] = 1.96 * _resid_tbl["se"]

print("\nRaw mean LST (°C) by distance band:")
print(_raw_tbl[["mean", "std", "n", "ci95"]].round(3).to_string())

print("\nResidual mean LST (°C) by distance band  [city-year FE removed]:")
print(_resid_tbl[["mean", "std", "n", "ci95"]].round(3).to_string())

# Pairwise difference: near vs far (residualized)
_near = _comp.loc[_comp["band"] == "0–300 m (near park)", "LST_dm"]
_far  = _comp.loc[_comp["band"] == "600 m+ (far from park)", "LST_dm"]
_diff = _far.mean() - _near.mean()
_se_diff = np.sqrt(_near.var() / len(_near) + _far.var() / len(_far))
_t    = _diff / _se_diff
print(f"\nFar minus near (residualized): {_diff:+.3f} °C  (t = {_t:.2f})")
print("Positive t = far pixels are warmer → supports cooling hypothesis")


# ── 7. PLOTS ──────────────────────────────────────────────────────────────────

# ── Plot 1: Residualized scatter (0–500 m bandwidth) ─────────────────────────
# Partial out city-year FE and NDVI via within-demean + OLS, then plot
# the pure distance gradient in the near-park window (< 500 m).
print("\n" + "="*80)
print("PLOT 1: Residualized LST vs dist_to_park_m  (0–500 m, city-year FE + NDVI removed)")
print("="*80)

_res = raw[
    (raw["is_water"] == 0) &
    raw["dist_to_park_m"].notna() &
    (raw["dist_to_park_m"] > 0) &
    (raw["dist_to_park_m"] <= 500) &
    raw["LST"].notna() &
    raw["NDVI"].notna()
].copy()

for col in ["LST", "NDVI"]:
    _res[f"{col}_dm"] = (
        _res[col] - _res.groupby(["city", "year"])[col].transform("mean")
    )

_X = np.column_stack([np.ones(len(_res)), _res["NDVI_dm"].values])
_beta = np.linalg.lstsq(_X, _res["LST_dm"].values, rcond=None)[0]
_res["resid_LST"] = _res["LST_dm"] - (_X @ _beta)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(_res["dist_to_park_m"], _res["resid_LST"],
           alpha=0.05, s=1, color="steelblue", rasterized=True)

_xv = _res["dist_to_park_m"].values
_yv = _res["resid_LST"].values
_m, _b = np.polyfit(_xv, _yv, 1)
_xl = np.linspace(_xv.min(), _xv.max(), 200)
ax.plot(_xl, _m * _xl + _b, color="firebrick", linewidth=1.5,
        label=f"OLS slope: {_m * 1_000:.4f} °C/km")
ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
ax.set_xlabel("Distance to nearest park patch (m)", fontsize=11)
ax.set_ylabel("Residual LST (°C)\n[city-year FE + NDVI removed]", fontsize=10)
ax.set_title("Distance to Park vs Residual LST  (0–500 m bandwidth)", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()

resid_path = Path.home() / "dist_park_vs_resid_lst.png"
plt.savefig(resid_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Residualized scatter saved to: {resid_path}")

# ── Plot 2: Binned means (150 m bins, 0–3 km) ────────────────────────────────
# Collapses vertical scatter into a clean cooling curve.
print("\n" + "="*80)
print("PLOT 2: Mean LST by distance bin  (150 m bins, 0–3 km)")
print("="*80)

_bin = raw[
    (raw["is_water"] == 0) &
    raw["dist_to_park_m"].notna() &
    (raw["dist_to_park_m"] > 0) &
    (raw["dist_to_park_m"] <= BIN_MAX_M) &
    raw["LST"].notna()
].copy()

_bins = np.arange(0, BIN_MAX_M + BIN_WIDTH_M, BIN_WIDTH_M)
_bin["dist_bin"] = pd.cut(_bin["dist_to_park_m"], bins=_bins)

_binned = (
    _bin.groupby("dist_bin", observed=True)["LST"]
    .agg(["mean", "std", "count"])
    .dropna()
)
_binned["ci95"] = 1.96 * _binned["std"] / np.sqrt(_binned["count"])
_bin_centers = np.array([iv.mid for iv in _binned.index]) / 1_000  # km

print(_binned[["mean", "count", "ci95"]].round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(_bin_centers, _binned["mean"], color="steelblue", linewidth=2,
        marker="o", markersize=4)
ax.fill_between(_bin_centers,
                _binned["mean"] - _binned["ci95"],
                _binned["mean"] + _binned["ci95"],
                color="steelblue", alpha=0.25, label="95% CI")
ax.set_xlabel("Distance to nearest park patch (km)", fontsize=11)
ax.set_ylabel("Mean LST (°C)", fontsize=11)
ax.set_title(f"Mean LST by Distance to Park  ({BIN_WIDTH_M} m bins)", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()

binned_path = Path.home() / "dist_park_binned_lst.png"
plt.savefig(binned_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Binned means plot saved to: {binned_path}")

# ── Plot 3: Geographic RDD — discontinuity at the park boundary ───────────────
print("\n" + "="*80)
print(f"PLOT 3: Geographic RDD — LST discontinuity at park boundary ({RDD_BW_M} m bandwidth)")
print("="*80)

_rdd_plot = raw[
    (raw["is_water"] == 0) &
    (raw["city"].isin(park_cities)) &
    ((raw["in_park_patch"] == 1) | (raw["dist_to_park_m"] <= RDD_BW_M)) &
    (raw["year"].between(2011, 2021)) &
    raw["LST"].notna()
].copy()

_rdd_plot["LST_dm"] = (
    _rdd_plot["LST"]
    - _rdd_plot.groupby(["city", "year"])["LST"].transform("mean")
)

_outside = _rdd_plot[
    (_rdd_plot["in_park_patch"] == 0) & _rdd_plot["dist_to_park_m"].notna()
].copy()
_rdd_bins = np.arange(0, RDD_BW_M + RDD_BIN_W, RDD_BIN_W)
_outside["bin"] = pd.cut(_outside["dist_to_park_m"], bins=_rdd_bins)
_out_binned = (
    _outside.groupby("bin", observed=True)["LST_dm"]
    .agg(mean="mean", std="std", count="count")
    .dropna()
)
_out_binned["ci95"] = 1.96 * _out_binned["std"] / np.sqrt(_out_binned["count"])
_out_km = np.array([iv.mid for iv in _out_binned.index]) / 1_000

_m_out, _b_out = np.polyfit(_out_km, _out_binned["mean"].values, 1)

_inside     = _rdd_plot[_rdd_plot["in_park_patch"] == 1]["LST_dm"]
_in_mean    = _inside.mean()
_in_ci      = 1.96 * _inside.std() / np.sqrt(len(_inside))
_tau_hat    = _in_mean - _b_out

print(f"Inside park mean (residualized):      {_in_mean:.3f} °C")
print(f"Outside line extrapolated to x=0:     {_b_out:.3f} °C")
print(f"RDD jump estimate (τ̂):               {_tau_hat:+.3f} °C")

fig, ax = plt.subplots(figsize=(11, 6))
ax.scatter(_out_km, _out_binned["mean"],
           color="steelblue", s=30, zorder=5,
           label="Non-park pixels (50 m bins, ±95% CI)")
ax.errorbar(_out_km, _out_binned["mean"], yerr=_out_binned["ci95"],
            fmt="none", color="steelblue", alpha=0.45, linewidth=1)
_xl = np.linspace(0, RDD_BW_M / 1_000, 300)
ax.plot(_xl, _m_out * _xl + _b_out,
        color="steelblue", linewidth=2, linestyle="--",
        label="Linear fit (non-park side)")
ax.scatter([0], [_b_out], color="steelblue", s=90, marker="o",
           facecolors="none", linewidths=2, zorder=6,
           label=f"Counterfactual at boundary ({_b_out:.2f} °C)")
ax.scatter([-0.04], [_in_mean], color="firebrick", s=90, marker="D",
           zorder=7, label=f"Inside park mean ({_in_mean:.2f} °C)")
ax.errorbar([-0.04], [_in_mean], yerr=_in_ci,
            fmt="none", color="firebrick", linewidth=2)

_mid_y = (_in_mean + _b_out) / 2
ax.annotate("", xy=(-0.04, _b_out), xytext=(-0.04, _in_mean),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax.text(-0.10, _mid_y, f"τ̂ = {_tau_hat:+.2f} °C",
        ha="center", va="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"))

ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Park boundary (cutoff)")
ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
ax.set_xlabel(
    "Distance from park boundary (km)\n[0 = boundary; positive = outside park]",
    fontsize=11)
ax.set_ylabel("Residual LST (°C)\n[city-year FE applied]", fontsize=10)
ax.set_title(
    "Geographic RDD: LST Discontinuity at Park Boundary\n"
    f"Cities: {', '.join(c.title() for c in sorted(park_cities))}  |  bandwidth = {RDD_BW_M} m (CCT optimal)",
    fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()

geo_rdd_path = Path.home() / "geo_rdd_plot.png"
plt.savefig(geo_rdd_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Geographic RDD plot saved to: {geo_rdd_path}")

# ── Plot 4: Cooling Decay — LST relative to park interior ────────────────────
# Y-axis: temperature relative to park interior mean (park = 0)
# Suburban gradient (far-field linear slope) partialled out from outside pixels.
print("\n" + "="*80)
print("PLOT 4: Park cooling decay — LST relative to park interior (suburban gradient removed)")
print("="*80)

_cool = raw[
    (raw["is_water"] == 0) &
    (raw["city"].isin(park_cities)) &
    ((raw["in_park_patch"] == 1) | (raw["dist_to_park_m"] <= BW_COOL)) &
    (raw["year"].between(2011, 2021)) &
    raw["LST"].notna()
].copy()

_cool["LST_dm"] = (
    _cool["LST"]
    - _cool.groupby(["city", "year"])["LST"].transform("mean")
)

_park_ref = _cool.loc[_cool["in_park_patch"] == 1, "LST_dm"].mean()
_cool["LST_rel"] = _cool["LST_dm"] - _park_ref

_out_c = _cool[
    (_cool["in_park_patch"] == 0) & _cool["dist_to_park_m"].notna()
].copy()
_cbins = np.arange(0, BW_COOL + BIN_COOL, BIN_COOL)
_out_c["bin"] = pd.cut(_out_c["dist_to_park_m"], bins=_cbins)
_cbin = (
    _out_c.groupby("bin", observed=True)["LST_rel"]
    .agg(mean="mean", std="std", count="count")
    .dropna()
)
_cbin["ci95"] = 1.96 * _cbin["std"] / np.sqrt(_cbin["count"])
_ckm = np.array([iv.mid for iv in _cbin.index]) / 1_000

_ff_mask = _ckm >= (FF_M / 1_000)
_m_ff = (
    np.polyfit(_ckm[_ff_mask], _cbin["mean"].values[_ff_mask], 1)[0]
    if _ff_mask.sum() > 1 else 0.0
)
_y_dt = _cbin["mean"].values - _m_ff * _ckm

_in_c      = _cool.loc[_cool["in_park_patch"] == 1, "LST_rel"]
_in_mean_c = _in_c.mean()
_in_ci_c   = 1.96 * _in_c.std() / np.sqrt(len(_in_c))

print(f"Park interior mean (should be ~0):    {_in_mean_c:.4f} °C")
print(f"Suburban slope removed:              {_m_ff * 1000:.4f} °C/km")

fig, ax = plt.subplots(figsize=(11, 6))
ax.axvspan(-0.15, 0, color="green", alpha=0.08, zorder=0, label="Park interior")
ax.scatter(_ckm, _y_dt, color="steelblue", s=30, zorder=5,
           label="Non-park pixels (50 m bins, suburban gradient removed)")
ax.errorbar(_ckm, _y_dt, yerr=_cbin["ci95"],
            fmt="none", color="steelblue", alpha=0.4, linewidth=1)
_poly_c = np.polyfit(_ckm, _y_dt, 2)
_xlc = np.linspace(0, BW_COOL / 1_000, 300)
ax.plot(_xlc, np.polyval(_poly_c, _xlc), color="steelblue", linewidth=2,
        label="Quadratic fit (outside)")
ax.scatter([-0.075], [_in_mean_c], color="forestgreen", s=90, marker="D",
           zorder=7, label=f"Park interior (= {_in_mean_c:.2f} °C, reference)")
ax.errorbar([-0.075], [_in_mean_c], yerr=_in_ci_c,
            fmt="none", color="forestgreen", linewidth=2)
ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Park boundary")
ax.axhline(0, color="gray", linewidth=0.7, linestyle=":", alpha=0.7)
ax.set_xlabel(
    "Distance from park boundary (km)\n[negative = inside park, positive = outside]",
    fontsize=11)
ax.set_ylabel(
    "Temperature relative to park interior (°C)\n"
    "[0 = inside park; positive = warmer than park]",
    fontsize=10)
ax.set_title(
    "Park Cooling Decay: LST Relative to Park Interior\n"
    f"Cities: {', '.join(c.title() for c in sorted(park_cities))}  |  "
    "within city-year variation, suburban gradient partialled out",
    fontsize=11)
ax.set_xlim(-0.15, BW_COOL / 1_000)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()

cooling_path = Path.home() / "cooling_decay_plot.png"
plt.savefig(cooling_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Cooling decay plot saved to: {cooling_path}")

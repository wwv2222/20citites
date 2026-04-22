import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path
from linearmodels.panel import PanelOLS
from scipy.spatial import cKDTree

CONLEY_CUTOFF_KM = 5  # spatial autocorrelation cutoff for Conley SEs

def conley_se(res, lat, lon, cutoff_km=CONLEY_CUTOFF_KM):
    """Conley (1999) spatial HAC SEs with Bartlett kernel."""
    coords_rad = np.radians(np.column_stack([lat, lon]))
    cutoff_rad = cutoff_km / 6371.0
    e = res.resid.values
    X = res.model.exog
    Xe = X * e[:, None]
    meat = Xe.T @ Xe  # self terms
    tree = cKDTree(coords_rad)
    pairs = tree.query_pairs(cutoff_rad, output_type='ndarray')
    if len(pairs):
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        d = np.linalg.norm(coords_rad[i_idx] - coords_rad[j_idx], axis=1) / cutoff_rad
        w = 1.0 - d  # Bartlett kernel
        wXe_i = w[:, None] * Xe[i_idx]
        cross = wXe_i.T @ Xe[j_idx]
        meat += cross + cross.T
    bread = np.linalg.inv(X.T @ X)
    V = bread @ meat @ bread
    return np.sqrt(np.diag(V))

# ── Base load ─────────────────────────────────────────────────────────────────
raw = pd.read_csv("/home/wwdelvalle/20_cities_panel_clean.csv", low_memory=False)

# ── City center coordinates (from scraping.py feature collection) ─────────────
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

def _dist_to_center(grp):
    key = grp['city'].iloc[0].lower()
    if key not in CITY_CENTERS:
        return pd.Series(np.nan, index=grp.index)
    clon, clat = CITY_CENTERS[key]
    lat_rad = np.radians((grp['latitude'].mean() + clat) / 2)
    dlat_m = (grp['latitude'].values - clat) * 111_000
    dlon_m = (grp['longitude'].values - clon) * 111_000 * np.cos(lat_rad)
    return pd.Series(np.sqrt(dlat_m**2 + dlon_m**2) / 1_000, index=grp.index)

raw['dist_to_center_km'] = raw.groupby('city', group_keys=False).apply(_dist_to_center)
print(f"dist_to_center_km — {raw['dist_to_center_km'].notna().sum():,} non-null  "
      f"| mean {raw['dist_to_center_km'].mean():.1f} km  "
      f"| max {raw['dist_to_center_km'].max():.1f} km")

# Model 1: all non-water pixels across all cities
df_all = raw[raw["is_water"] == 0].copy()
print(f"Observations (non-water, all cities): {len(df_all):,}")
df_all = df_all.set_index(["city", "year"])

# Model 4 sample prep (park-distance cities, non-water, with park distance)
df = raw[(raw["is_water"] == 0) & raw["dist_to_park_m"].notna()].copy()
df = df.set_index(["city", "year"])

# ── Model 1: LST ~ NDVI + near_water | city + year ───────────────────────────
print("="*80)
print("MODEL 1: LST ~ NDVI + near_water  (city + year FE)")
print("="*80)

model1 = PanelOLS.from_formula(
    "LST ~ NDVI + near_water + EntityEffects + TimeEffects",
    data=df_all
)
result1 = model1.fit(cov_type="clustered", cluster_entity=True)
print(result1.summary)

# ── Models 2 & 3: per-city OLS with year FE ───────────────────────────────────
# Within a single city, EntityEffects reduces to a constant (collinear).
# dist_to_center_km is time-invariant per pixel so pixel FE would absorb it.
# OLS + year dummies (C(year)) with HC1 robust SEs is the right specification.
PARK_DIST_CITIES = ['arlington', 'corpus christi', 'miami', 'oklahoma city', 'tulsa']

# ── Model 2: per city ─────────────────────────────────────────────────────────
print("\n" + "="*80)
print("MODEL 2 (per city): LST ~ dist_to_park_m + dist_to_center_km + NDVI + near_water + year FE")
print("="*80)

results2 = {}
for city_name in PARK_DIST_CITIES:
    df_city = raw[
        (raw['is_water'] == 0) &
        (raw['in_park_patch'] == 0) &
        (raw['city'].str.lower() == city_name) &
        (raw['year'].between(2011, 2021))
    ].copy()
    if len(df_city) == 0:
        print(f"\n{city_name.title()}: no data")
        continue
    df_city['log_dist'] = np.log(df_city['dist_to_park_m'] + 1)
    res = smf.ols(
        "LST ~ log_dist + dist_to_center_km + near_water + C(year)",
        data=df_city
    ).fit(cov_type='HC1')
    results2[city_name] = res
    print(f"\n{'='*40}")
    print(f"City: {city_name.title()}  (n={len(df_city):,})")
    print(f"{'='*40}")
    print(res.summary())
    _idx = res.resid.index
    cse = conley_se(res, df_city.loc[_idx, 'latitude'].values, df_city.loc[_idx, 'longitude'].values)
    se_tbl = pd.DataFrame({
        'coef':      res.params,
        'HC1 SE':    res.bse,
        'Conley SE': cse,
    })
    print(f"\nConley SEs (cutoff={CONLEY_CUTOFF_KM} km):")
    print(se_tbl.to_string())

# ── Model 3: per city — dist_to_park_m × park size ───────────────────────────
print("\n" + "="*80)
print("MODEL 3 (per city): LST ~ dist_to_park_m * park_size_cat + dist_to_center_km + year FE")
print("="*80)

results3 = {}
for city_name in PARK_DIST_CITIES:
    df_city = raw[
        (raw['is_water'] == 0) &
        (raw['city'].str.lower() == city_name) &
        (raw['year'].between(2011, 2021))
    ].copy()
    # Park pixels are inside the park — distance is 0
    df_city.loc[df_city['is_park_pixel'] == 1, 'dist_to_park_m'] = 0.0
    # For in-patch pixels, use their own patch size as the nearest park size
    _size_map = {'small': 1, 'medium': 2, 'large': 3}
    _patch_mask = df_city['in_park_patch'] == 1
    df_city.loc[_patch_mask, 'nearest_park_size'] = (
        df_city.loc[_patch_mask, 'park_size'].map(_size_map)
    )
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
        "LST ~ log_dist * C(park_size_cat) + dist_to_center_km + near_water + C(year)",
        data=df_city
    ).fit(cov_type='HC1')
    results3[city_name] = res
    print(f"\n{'='*40}")
    print(f"City: {city_name.title()}  (n={len(df_city):,})")
    print(f"{'='*40}")
    print(res.summary())
    _idx = res.resid.index
    cse = conley_se(res, df_city.loc[_idx, 'latitude'].values, df_city.loc[_idx, 'longitude'].values)
    se_tbl = pd.DataFrame({
        'coef':      res.params,
        'HC1 SE':    res.bse,
        'Conley SE': cse,
    })
    print(f"\nConley SEs (cutoff={CONLEY_CUTOFF_KM} km):")
    print(se_tbl.to_string())

# ── Model 4: Regression Discontinuity — park boundary threshold ──────────────
# Running variable: distance from park boundary (signed)
#   inside park (in_park_patch == 1)  →  0  (at/inside boundary)
#   outside park                      →  dist_to_park_m (positive)
# Treatment D: 1 = inside park, 0 = outside park
# Bandwidth: 1,000 m (only outside pixels within 1 km of a park boundary)
# Sample: same 5 cities as above + park pixels in those cities
# Note: NDVI is excluded — it is mechanically higher inside parks, so controlling
# for it would absorb the very effect we are estimating. near_water is kept.
print("\n" + "="*80)
print("MODEL 4: RDD — park boundary threshold (bandwidth = 1,000 m)")
print("         LST ~ D(in_park) + running_var + near_water  (city + year FE)")
print("="*80)

# Identify the 5 cities with park distance data
park_cities = df.index.get_level_values("city").unique().tolist()

rdd = raw[
    (raw["is_water"] == 0) &
    (raw["city"].isin(park_cities)) &
    ((raw["in_park_patch"] == 1) | (raw["dist_to_park_m"] <= 1000)) &
    (raw["year"].between(2011, 2021))
].copy()

# Build the signed running variable and treatment indicator
rdd["running_var"] = rdd["dist_to_park_m"].fillna(0)   # park pixels → 0
rdd["D"] = rdd["in_park_patch"].astype(int)             # 1 = inside park

print(f"RDD observations (bandwidth 1km): {len(rdd):,}")
print("Treatment breakdown:\n", rdd["D"].value_counts(), "\n")

rdd = rdd.set_index(["city", "year"])

model4 = PanelOLS.from_formula(
    "LST ~ D + running_var + near_water + EntityEffects + TimeEffects",
    data=rdd
)
result4 = model4.fit(cov_type="clustered", cluster_entity=True)
print(result4.summary)

# Conley SEs for Model 4: two-way within-demean (city + year FE), then OLS
_r4 = rdd[['LST', 'D', 'running_var', 'near_water', 'latitude', 'longitude']].copy().astype(float)
for _col in ['LST', 'D', 'running_var', 'near_water']:
    _city_m  = _r4.groupby(level='city')[_col].transform('mean')
    _year_m  = _r4.groupby(level='year')[_col].transform('mean')
    _grand_m = _r4[_col].mean()
    _r4[f'{_col}_dm'] = _r4[_col] - _city_m - _year_m + _grand_m
_res4_ols = smf.ols(
    "LST_dm ~ D_dm + running_var_dm + near_water_dm - 1",
    data=_r4
).fit()
_cse4 = conley_se(_res4_ols, _r4['latitude'].values, _r4['longitude'].values)
_se4_tbl = pd.DataFrame({
    'coef':         result4.params,
    'Clustered SE': result4.std_errors,
    'Conley SE':    pd.Series(_cse4, index=result4.params.index),
})
print(f"\nConley SEs for Model 4 (cutoff={CONLEY_CUTOFF_KM} km):")
print(_se4_tbl.to_string())

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

# Step 1: within-demean by city-year (removes city-year FE)
for col in ["LST", "NDVI"]:
    _res[f"{col}_dm"] = (
        _res[col] - _res.groupby(["city", "year"])[col].transform("mean")
    )

# Step 2: partial out demeaned NDVI → residual LST
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
# Expect: coolest near boundary, gradual warming, flattening ~600–900 m.
print("\n" + "="*80)
print("PLOT 2: Mean LST by distance bin  (150 m bins, 0–3 km)")
print("="*80)

BIN_WIDTH_M = 150
BIN_MAX_M   = 3_000

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

# ── Near/far comparison table ─────────────────────────────────────────────────
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




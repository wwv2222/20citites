import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path
from linearmodels.panel import PanelOLS
from rdrobust import rdbwselect

# ── Data loading ───────────────────────────────────────────────────────────────
raw = pd.read_csv("/home/wwdelvalle/20_cities_panel_clean.csv", low_memory=False)

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

PARK_DIST_CITIES = ['arlington', 'corpus christi', 'miami', 'oklahoma city', 'tulsa']

# Identify park cities present in the data (same logic as analysis.py)
_df_tmp = raw[(raw["is_water"] == 0) & raw["dist_to_park_m"].notna()].copy()
_df_tmp = _df_tmp.set_index(["city", "year"])
PARK_CITIES_RAW = _df_tmp.index.get_level_values("city").unique().tolist()
del _df_tmp

_SIZE_MAP   = {'small': 1, 'medium': 2, 'large': 3}
_SIZE_LABEL = {1: 'small', 2: 'medium', 3: 'large'}


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1: RDD Bandwidth Sensitivity
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("ROBUSTNESS 1: RDD Bandwidth Sensitivity")
print("  Re-runs Model 4 at BW = 250, 500, 750, 1000, 1500, 2000 m + CCT optimal BW")
print("  Stable coefficient across bandwidths validates the baseline BW=1000m result.")
print("="*80)

# Compute CCT optimal bandwidth using all 5 park-city pixels
_cct_data = raw[
    (raw["is_water"] == 0) &
    (raw["city"].isin(PARK_CITIES_RAW)) &
    raw["LST"].notna()
].copy()
_cct_data["running_var"] = _cct_data["dist_to_park_m"].fillna(0)
_cct_data = _cct_data.dropna(subset=["LST", "running_var"])

CCT_BW = None
try:
    _bw_sel = rdbwselect(
        y=_cct_data["LST"].values,
        x=_cct_data["running_var"].values,
        c=0, bwselect="mserd"
    )
    CCT_BW = int(round(float(_bw_sel.bws["h"].iloc[0])))
    print(f"\nCCT Optimal Bandwidth (Calonico-Cattaneo-Titiunik): {CCT_BW} m")
except Exception as e:
    print(f"CCT bandwidth estimation failed: {e}")

BANDWIDTHS = sorted(set([250, 500, 750, 1000, 1500, 2000] + ([CCT_BW] if CCT_BW else [])))
bw_results = []

for bw in BANDWIDTHS:
    rdd_bw = raw[
        (raw["is_water"] == 0) &
        (raw["city"].isin(PARK_CITIES_RAW)) &
        ((raw["in_park_patch"] == 1) | (raw["dist_to_park_m"] <= bw))
    ].copy()
    rdd_bw["running_var"] = rdd_bw["dist_to_park_m"].fillna(0)
    rdd_bw["D"] = rdd_bw["in_park_patch"].astype(int)
    rdd_bw = rdd_bw.set_index(["city", "year"])

    try:
        res = PanelOLS.from_formula(
            "LST ~ D + running_var + near_water + EntityEffects + TimeEffects",
            data=rdd_bw
        ).fit(cov_type="clustered", cluster_entity=True)
        coef = res.params["D"]
        se   = res.std_errors["D"]
        bw_results.append({
            "bandwidth_m": bw, "n": len(rdd_bw),
            "coef": coef, "se": se,
            "ci_lo": coef - 1.96 * se, "ci_hi": coef + 1.96 * se,
        })
    except Exception as e:
        print(f"  BW={bw}m failed: {e}")
        bw_results.append({"bandwidth_m": bw, "n": 0,
                            "coef": np.nan, "se": np.nan,
                            "ci_lo": np.nan, "ci_hi": np.nan})

bw_df = pd.DataFrame(bw_results)
print("\nRDD coefficient on D (inside park) by bandwidth:")
print(bw_df.round(5).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
bw_ok = bw_df.dropna(subset=["coef"])
ax.plot(bw_ok["bandwidth_m"], bw_ok["coef"], marker="o",
        color="steelblue", linewidth=2)
ax.fill_between(bw_ok["bandwidth_m"], bw_ok["ci_lo"], bw_ok["ci_hi"],
                color="steelblue", alpha=0.25, label="95% CI")
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.axvline(1000, color="firebrick", linewidth=1.2, linestyle=":",
           label="Baseline BW (1000 m)")
if CCT_BW and CCT_BW not in [1000]:
    ax.axvline(CCT_BW, color="forestgreen", linewidth=1.2, linestyle="--",
               label=f"CCT optimal BW ({CCT_BW} m)")
ax.set_xlabel("Bandwidth (m)", fontsize=11)
ax.set_ylabel("RDD coefficient on D (°C)", fontsize=11)
ax.set_title("RDD Bandwidth Sensitivity: Effect of Being Inside Park", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(Path.home() / "robustness_bw_sensitivity.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved: ~/robustness_bw_sensitivity.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2: Placebo / Fake Boundary Test
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("ROBUSTNESS 2: Placebo Fake Boundary Test")
print("  Assigns fake cutoffs at 1500, 2000, 2500 m inside the non-park zone.")
print("  Pixels within 1000 m of the real boundary excluded — beyond the cooling gradient.")
print("  Coefficients near zero = no spatial confound; non-zero = suspect baseline.")
print("="*80)

placebo_results = []

# Baseline: real boundary at BW=1000m
rdd_base = raw[
    (raw["is_water"] == 0) &
    (raw["city"].isin(PARK_CITIES_RAW)) &
    ((raw["in_park_patch"] == 1) | (raw["dist_to_park_m"] <= 1000))
].copy()
rdd_base["running_var"] = rdd_base["dist_to_park_m"].fillna(0)
rdd_base["D"] = rdd_base["in_park_patch"].astype(int)
rdd_base = rdd_base.set_index(["city", "year"])
res_base = PanelOLS.from_formula(
    "LST ~ D + running_var + near_water + EntityEffects + TimeEffects",
    data=rdd_base
).fit(cov_type="clustered", cluster_entity=True)
coef_b = res_base.params["D"]
se_b   = res_base.std_errors["D"]
placebo_results.append({
    "label": "Real boundary (BW=1000m)",
    "coef": coef_b, "se": se_b,
    "ci_lo": coef_b - 1.96 * se_b, "ci_hi": coef_b + 1.96 * se_b,
})

PLACEBO_SHIFTS = [1500, 2000, 2500]
REAL_BOUNDARY_EXCLUSION = 1000

for shift in PLACEBO_SHIFTS:
    placebo = raw[
        (raw["is_water"] == 0) &
        (raw["city"].isin(PARK_CITIES_RAW)) &
        (raw["in_park_patch"] == 0) &
        raw["dist_to_park_m"].notna() &
        (raw["dist_to_park_m"] >= REAL_BOUNDARY_EXCLUSION) &
        (raw["dist_to_park_m"] >= shift - 500) &
        (raw["dist_to_park_m"] <= shift + 500)
    ].copy()

    if len(placebo) < 100:
        print(f"  Fake boundary at +{shift}m: too few obs ({len(placebo)}), skipped")
        continue

    placebo["running_var"] = placebo["dist_to_park_m"] - shift
    placebo["D"] = (placebo["dist_to_park_m"] < shift).astype(int)
    placebo = placebo.set_index(["city", "year"])

    try:
        res_p = PanelOLS.from_formula(
            "LST ~ D + running_var + near_water + EntityEffects + TimeEffects",
            data=placebo
        ).fit(cov_type="clustered", cluster_entity=True)
        coef_p = res_p.params["D"]
        se_p   = res_p.std_errors["D"]
        placebo_results.append({
            "label": f"Fake at +{shift}m (excl. ≤{REAL_BOUNDARY_EXCLUSION}m)",
            "coef": coef_p, "se": se_p,
            "ci_lo": coef_p - 1.96 * se_p, "ci_hi": coef_p + 1.96 * se_p,
        })
    except Exception as e:
        print(f"  Fake boundary at +{shift}m failed: {e}")

placebo_df = pd.DataFrame(placebo_results)
print("\nPlacebo test — coefficient on D:")
print(placebo_df.round(5).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
for i, row in placebo_df.iterrows():
    color = "steelblue" if i == 0 else "darkorange"
    label = "Real boundary" if i == 0 else "Fake boundary"
    ax.errorbar(i, row["coef"], yerr=1.96 * row["se"],
                fmt="o", color=color, capsize=5, markersize=8, label=label)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xticks(range(len(placebo_df)))
ax.set_xticklabels(placebo_df["label"], rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Coefficient on D (°C)", fontsize=11)
ax.set_title("Placebo Test: Fake vs Real Park Boundary", fontsize=12)
handles, labels = ax.get_legend_handles_labels()
ax.legend(dict(zip(labels, handles)).values(),
          dict(zip(labels, handles)).keys(), fontsize=10)
plt.tight_layout()
plt.savefig(Path.home() / "robustness_placebo.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved: ~/robustness_placebo.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3: Leave-One-City-Out (Models 2, 3, 4)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("ROBUSTNESS 3: Leave-One-City-Out")
print("  Drops one city at a time from the 5-city subsample and re-estimates")
print("  Models 2, 3, and 4 on the remaining 4 cities.")
print("  Stable coefficients rule out a single city driving the results.")
print("="*80)

loco_rows = []

for dropped in PARK_DIST_CITIES:
    remaining      = [c for c in PARK_DIST_CITIES if c != dropped]
    remaining_raw  = [c for c in PARK_CITIES_RAW if c.lower() != dropped.lower()]

    # Model 2: pooled OLS across remaining 4 cities
    df_m2 = raw[
        (raw["is_water"] == 0) &
        (raw["in_park_patch"] == 0) &
        (raw["city"].str.lower().isin(remaining)) &
        raw["dist_to_park_m"].notna()
    ].copy()
    try:
        res_m2  = smf.ols(
            "LST ~ dist_to_park_m + dist_to_center_km + near_water + C(year)",
            data=df_m2
        ).fit(cov_type='HC1')
        m2_coef = res_m2.params.get("dist_to_park_m", np.nan)
        m2_se   = res_m2.bse.get("dist_to_park_m", np.nan)
    except Exception:
        m2_coef, m2_se = np.nan, np.nan

    # Model 3: pooled OLS with park-size interaction across remaining 4 cities
    df_m3 = raw[
        (raw["is_water"] == 0) &
        (raw["city"].str.lower().isin(remaining)) &
        raw["dist_to_park_m"].notna()
    ].copy()
    df_m3.loc[df_m3['is_park_pixel'] == 1, 'dist_to_park_m'] = 0.0
    _pm = df_m3['in_park_patch'] == 1
    df_m3.loc[_pm, 'nearest_park_size'] = df_m3.loc[_pm, 'park_size'].map(_SIZE_MAP)
    df_m3['park_size_cat'] = pd.Categorical(
        df_m3['nearest_park_size'].map(_SIZE_LABEL),
        categories=['small', 'medium', 'large'], ordered=True,
    )
    df_m3 = df_m3[df_m3['park_size_cat'].notna()]
    try:
        res_m3  = smf.ols(
            "LST ~ dist_to_park_m * C(park_size_cat) + dist_to_center_km + near_water + C(year)",
            data=df_m3
        ).fit(cov_type='HC1')
        m3_coef = res_m3.params.get("dist_to_park_m", np.nan)
        m3_se   = res_m3.bse.get("dist_to_park_m", np.nan)
    except Exception:
        m3_coef, m3_se = np.nan, np.nan

    # Model 4: RDD on remaining 4 cities
    rdd_loco = raw[
        (raw["is_water"] == 0) &
        (raw["city"].isin(remaining_raw)) &
        ((raw["in_park_patch"] == 1) | (raw["dist_to_park_m"] <= 1000))
    ].copy()
    rdd_loco["running_var"] = rdd_loco["dist_to_park_m"].fillna(0)
    rdd_loco["D"] = rdd_loco["in_park_patch"].astype(int)
    rdd_loco = rdd_loco.set_index(["city", "year"])
    try:
        res_m4  = PanelOLS.from_formula(
            "LST ~ D + running_var + near_water + EntityEffects + TimeEffects",
            data=rdd_loco
        ).fit(cov_type="clustered", cluster_entity=True)
        m4_coef = res_m4.params["D"]
        m4_se   = res_m4.std_errors["D"]
    except Exception:
        m4_coef, m4_se = np.nan, np.nan

    loco_rows.append({
        "dropped":   dropped.title(),
        "M2_coef":   m2_coef,  "M2_se":   m2_se,
        "M3_coef":   m3_coef,  "M3_se":   m3_se,
        "M4_D_coef": m4_coef,  "M4_D_se": m4_se,
    })

loco_df = pd.DataFrame(loco_rows)
print("\nLeave-one-city-out results:")
print("  M2_coef / M3_coef = coefficient on dist_to_park_m")
print("  M4_D_coef         = RDD treatment coefficient (inside park)")
print()
print(loco_df.round(6).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4: Alternative Distance Functional Forms (Models 2 & 3)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("ROBUSTNESS 4: Alternative Distance Functional Forms (Models 2 & 3)")
print("  Tests linear, log, quadratic, and distance-bin specs pooled across 5 cities.")
print("  Consistent sign and significance across forms supports the baseline.")
print("="*80)

# Model 2 data: pooled across all 5 cities
df_m2_pool = raw[
    (raw["is_water"] == 0) &
    (raw["in_park_patch"] == 0) &
    (raw["city"].str.lower().isin(PARK_DIST_CITIES)) &
    raw["dist_to_park_m"].notna()
].copy()
df_m2_pool["log_dist"] = np.log(df_m2_pool["dist_to_park_m"] + 1)
df_m2_pool["dist_sq"]  = df_m2_pool["dist_to_park_m"] ** 2
df_m2_pool["dist_bin"] = pd.cut(
    df_m2_pool["dist_to_park_m"],
    bins=[0, 100, 300, 600, np.inf],
    labels=["0-100m", "100-300m", "300-600m", "600m+"]
)

print(f"\n--- Model 2 alternative specs (n={len(df_m2_pool):,} pooled pixels) ---")

specs_m2 = [
    ("Log (main spec)",
     "LST ~ log_dist + dist_to_center_km + near_water + C(year)",
     "log_dist"),
    ("Linear (robustness)",
     "LST ~ dist_to_park_m + dist_to_center_km + near_water + C(year)",
     "dist_to_park_m"),
    ("Quadratic",
     "LST ~ dist_to_park_m + dist_sq + dist_to_center_km + near_water + C(year)",
     "dist_to_park_m"),
    ("Bins",
     "LST ~ C(dist_bin) + dist_to_center_km + near_water + C(year)",
     None),
]

for spec_name, formula, key_var in specs_m2:
    try:
        res = smf.ols(formula, data=df_m2_pool).fit(cov_type='HC1')
        if key_var:
            coef = res.params.get(key_var, np.nan)
            se   = res.bse.get(key_var, np.nan)
            print(f"  {spec_name:<12}  coef = {coef:+.6f}  se = {se:.6f}  R² = {res.rsquared:.4f}")
        else:
            bin_keys = [k for k in res.params.index if "dist_bin" in k]
            print(f"  {spec_name:<12}  R² = {res.rsquared:.4f}  bin coefs (ref = 0–100m):")
            for bk in bin_keys:
                print(f"               {bk}: {res.params[bk]:+.4f}  (se={res.bse[bk]:.4f})")
    except Exception as e:
        print(f"  {spec_name}: failed — {e}")

# Model 3 data: pooled across all 5 cities
df_m3_pool = raw[
    (raw["is_water"] == 0) &
    (raw["city"].str.lower().isin(PARK_DIST_CITIES)) &
    raw["dist_to_park_m"].notna()
].copy()
df_m3_pool.loc[df_m3_pool['is_park_pixel'] == 1, 'dist_to_park_m'] = 0.0
_pm = df_m3_pool['in_park_patch'] == 1
df_m3_pool.loc[_pm, 'nearest_park_size'] = df_m3_pool.loc[_pm, 'park_size'].map(_SIZE_MAP)
df_m3_pool['park_size_cat'] = pd.Categorical(
    df_m3_pool['nearest_park_size'].map(_SIZE_LABEL),
    categories=['small', 'medium', 'large'], ordered=True,
)
df_m3_pool = df_m3_pool[df_m3_pool['park_size_cat'].notna()]
df_m3_pool["log_dist"] = np.log(df_m3_pool["dist_to_park_m"] + 1)
df_m3_pool["dist_sq"]  = df_m3_pool["dist_to_park_m"] ** 2

print(f"\n--- Model 3 alternative specs (n={len(df_m3_pool):,} pooled pixels) ---")

specs_m3 = [
    ("Log (main spec)",
     "LST ~ log_dist * C(park_size_cat) + dist_to_center_km + near_water + C(year)",
     "log_dist"),
    ("Linear (robustness)",
     "LST ~ dist_to_park_m * C(park_size_cat) + dist_to_center_km + near_water + C(year)",
     "dist_to_park_m"),
    ("Quadratic",
     "LST ~ (dist_to_park_m + dist_sq) * C(park_size_cat) + dist_to_center_km + near_water + C(year)",
     "dist_to_park_m"),
]

for spec_name, formula, key_var in specs_m3:
    try:
        res = smf.ols(formula, data=df_m3_pool).fit(cov_type='HC1')
        coef = res.params.get(key_var, np.nan)
        se   = res.bse.get(key_var, np.nan)
        print(f"  {spec_name:<12}  coef = {coef:+.6f}  se = {se:.6f}  R² = {res.rsquared:.4f}")
    except Exception as e:
        print(f"  {spec_name}: failed — {e}")

print("\nDone.")

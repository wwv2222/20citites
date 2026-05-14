import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sp_connected_components
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %% 
# ── Load panel dataset ────────────────────────────────────────────────────────
LOCAL_PATH = Path.home() / "20_cities_panel.csv"
twenty_cities = pd.read_csv(LOCAL_PATH)

# Drop GEE metadata columns not needed for analysis
twenty_cities = twenty_cities.drop(columns=[c for c in ['system:index', '.geo'] if c in twenty_cities.columns])

# Report all columns so we know what merging.py brought in
print(f"Loaded {len(twenty_cities):,} rows  |  columns: {list(twenty_cities.columns)}")

# %%
# ── Impute missing MNDWI from nearest year (same pixel) ──────────────────────
if 'MNDWI' in twenty_cities.columns:
    _n_before = twenty_cities['MNDWI'].isna().sum()
    if _n_before > 0:
        def _fill_mndwi_nearest(grp):
            grp = grp.sort_values('year')
            vals  = grp['MNDWI'].values.astype(float).copy()
            years = grp['year'].values
            known = ~np.isnan(vals)
            if not known.any():
                return pd.Series(vals, index=grp.index)
            known_years = years[known]
            known_vals  = vals[known]
            for i in np.where(~known)[0]:
                nearest = np.argmin(np.abs(known_years - years[i]))
                vals[i] = known_vals[nearest]
            return pd.Series(vals, index=grp.index)

        twenty_cities['MNDWI'] = (
            twenty_cities
            .groupby(['city', 'latitude', 'longitude'], group_keys=False)
            .apply(_fill_mndwi_nearest)
        )
        _n_after  = twenty_cities['MNDWI'].isna().sum()
        _n_filled = _n_before - _n_after
        print(f"MNDWI imputation: {_n_filled:,} values filled from nearest year")
        if _n_after:
            print(f"  ({_n_after:,} still missing — pixels with no MNDWI in any year)")

# %%
# ── Drop unusable years and low-coverage city-year cells ─────────────────────
# 2012: 100% missing all cities (satellite/acquisition failure)
# 2021: ~50% missing all cities (systematic acquisition gap)
DROP_YEARS      = [2012, 2021]
COVERAGE_THRESH = 90.0   # drop city-years where < this % of master grid is present

_n_before = len(twenty_cities)

# Step 1: drop whole years
twenty_cities = twenty_cities[~twenty_cities['year'].isin(DROP_YEARS)]

# Step 2: compute grid coverage per city-year and drop below threshold
_city_year_px = (
    twenty_cities.groupby(['city', 'year'])
    .apply(lambda x: x[['latitude', 'longitude']].drop_duplicates().shape[0],
           include_groups=False)
    .rename('n_pixels')
)
_city_master = (
    twenty_cities.groupby('city')
    .apply(lambda x: x[['latitude', 'longitude']].drop_duplicates().shape[0],
           include_groups=False)
    .rename('master_size')
)
_coverage = (_city_year_px / _city_master * 100).rename('coverage')
_bad_cy   = _coverage[_coverage < COVERAGE_THRESH].reset_index()[['city', 'year']]

if len(_bad_cy):
    print(f"City-year cells dropped (< {COVERAGE_THRESH}% grid coverage):")
    for _, row in _bad_cy.iterrows():
        pct = _coverage.loc[(row.city, row.year)]
        print(f"  {row.city} {row.year}  ({pct:.1f}%)")
    _bad_cy['_drop'] = True
    twenty_cities = (
        twenty_cities
        .merge(_bad_cy, on=['city', 'year'], how='left')
        .query('_drop != _drop')   # keep rows where _drop is NaN
        .drop(columns='_drop')
    )

twenty_cities = twenty_cities.reset_index(drop=True)
_n_dropped = _n_before - len(twenty_cities)
print(f"\nTotal dropped: {_n_dropped:,} rows  |  Remaining: {len(twenty_cities):,} rows")

# %%
# ── Convert LST from raw Landsat DN to degrees Celsius ───────────────────────
# Landsat Collection 2 formula: LST_K = DN × 0.00341802 + 149.0
# Then: LST_C = LST_K - 273.15
twenty_cities['LST'] = twenty_cities['LST'] * 0.00341802 + 149.0 - 273.15

#%%
# Set physically impossible LST values to NaN (fill/sentinel values from satellite)
LST_MIN, LST_MAX = -5, 65   # °C — physically plausible range for this city panel
bad_lst = (twenty_cities['LST'] < LST_MIN) | (twenty_cities['LST'] > LST_MAX)
print(f"LST values outside [{LST_MIN}, {LST_MAX}]°C set to NaN: {bad_lst.sum():,} rows")
twenty_cities.loc[bad_lst, 'LST'] = float('nan')

#%%
# ── Section 1: Unique spatial points per year ─────────────────────────────────
print("=" * 60)
print("SECTION 1: Unique (latitude, longitude) points per year")
print("=" * 60)
pts_per_year = twenty_cities.groupby('year').apply(
    lambda x: x[['latitude', 'longitude']].drop_duplicates().shape[0]
)
print(pts_per_year.to_string())
print(f"\nMin: {pts_per_year.min()}  Max: {pts_per_year.max()}  "
      f"Consistent: {pts_per_year.nunique() == 1}")

dupes = twenty_cities.duplicated(subset=['city', 'year', 'latitude', 'longitude'])
print(f"Duplicate (city, year, latitude, longitude) rows: {dupes.sum()}")

#%%
# ── Section 1b: Per-city grid consistency ─────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 1b: Grid consistency per city")
print("(% of master grid covered each year — ideally close to 100%)")
print("=" * 60)

for city, grp in twenty_cities.groupby('city'):
    # Master grid = all unique lat/lon seen in ANY year for this city
    master = set(zip(grp['latitude'], grp['longitude']))
    n_master = len(master)

    # Coverage per year
    coverage = grp.groupby('year').apply(
        lambda x: len(set(zip(x['latitude'], x['longitude'])) ) / n_master * 100,
        include_groups=False
    ).round(1)

    print(f"\n{city}  (master grid: {n_master:,} points)")
    print(f"  Min coverage: {coverage.min()}%  (year {coverage.idxmin()})")
    print(f"  Max coverage: {coverage.max()}%  (year {coverage.idxmax()})")
    print(f"  Years < 90% coverage: {sorted(coverage[coverage < 90].index.tolist())}")

# %%
# ── Section 2: Summary statistics ────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2: Summary statistics — LST, NDVI, MNDWI, and similarity")
print("=" * 60)
pd.set_option('display.float_format', '{:.4f}'.format)
stat_cols = [c for c in ['LST', 'NDVI', 'MNDWI', 'similarity'] if c in twenty_cities.columns]
print(twenty_cities[stat_cols].describe())
pd.reset_option('display.float_format')


# %%
# ── Section 3: Missing values per city ───────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3: % Missing by city (sorted worst first)")
print("=" * 60)
miss_cols = [c for c in ['LST', 'NDVI', 'MNDWI'] if c in twenty_cities.columns]
missing_by_city = (
    twenty_cities.groupby('city')[miss_cols]
    .apply(lambda x: x.isnull().mean() * 100)
    .round(2)
)
print(missing_by_city.sort_values('LST', ascending=False).to_string())

# %%
# ── Section 4: Missing values per year ───────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4: % Missing by year (sorted worst first)")
print("=" * 60)
missing_by_year = (
    twenty_cities.groupby('year')[miss_cols]
    .apply(lambda x: x.isnull().mean() * 100)
    .round(2)
)
print(missing_by_year.sort_values('LST', ascending=False).to_string())

# %%
# ── Section 5: Missing value pattern — city × year pivot ─────────────────────
print("\n" + "=" * 60)
print("SECTION 5: % Missing LST — city × year pivot")
print("(High entire row = bad year; high entire col = bad city; scattered = clouds)")
print("=" * 60)
missing_pivot = (
    twenty_cities.groupby(['year', 'city'])['LST']
    .apply(lambda x: x.isnull().mean() * 100)
    .unstack('city')
    .round(1)
)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(missing_pivot.to_string())

# %%
# ── Section 6: Flag city-year cells exceeding 20% missing ────────────────────
print("\n" + "=" * 60)
print("SECTION 6: City-year cells with > 20% missing (LST or NDVI)")
print("=" * 60)
THRESHOLD = 20.0
cell_missing = (
    twenty_cities.groupby(['city', 'year'])[miss_cols]
    .apply(lambda x: x.isnull().mean() * 100)
    .round(2)
)
flag_cols = [c for c in ['LST', 'NDVI', 'MNDWI'] if c in cell_missing.columns]
bad_cells = cell_missing[cell_missing[flag_cols].gt(THRESHOLD).any(axis=1)]
print(f"{len(bad_cells)} city-year cells exceed {THRESHOLD}% missing:\n")
print(bad_cells.sort_values('LST', ascending=False).to_string())

# %%
# ── Section 7: Water classification and proximity control ─────────────────────
print("\n" + "=" * 60)
print("SECTION 7: Water classification and near-water proximity")
print("=" * 60)

WATER_BUFFER_M = 500  # meters; near_water=1 if within this distance

# Step 1 — Classify water pixels
# Prefer MNDWI if available (purpose-built water index: >0 = water).
# Fall back to NDVI < 0.05 if MNDWI is missing.
if 'MNDWI' in twenty_cities.columns and twenty_cities['MNDWI'].notna().any():
    twenty_cities['is_water'] = (twenty_cities['MNDWI'] > 0).astype(int)
    print(f"\nWater classification: MNDWI > 0")
else:
    NDVI_WATER_THRESHOLD = 0.05
    twenty_cities['is_water'] = (twenty_cities['NDVI'] < NDVI_WATER_THRESHOLD).astype(int)
    print(f"\nWater classification: NDVI < {NDVI_WATER_THRESHOLD} (MNDWI not available)")

n_water = twenty_cities['is_water'].sum()
print(f"Total water pixels flagged: {n_water:,} ({n_water / len(twenty_cities) * 100:.1f}% of all rows)")

print("\nWater pixel share by city (% of all city rows):")
water_by_city = (
    twenty_cities.groupby('city')['is_water']
    .mean()
    .mul(100)
    .round(1)
    .sort_values(ascending=False)
    .rename('% water')
)
print(water_by_city.to_string())

# Step 2 — Distance to nearest water pixel (per city-year)
# Converts lat/lon to approximate meters; builds cKDTree on water pixels
# and queries every pixel's distance to its nearest water neighbor.
def _water_dist(group):
    lat_rad = np.radians(group['latitude'].mean())
    lat_m = group['latitude'].values * 111_000
    lon_m = group['longitude'].values * 111_000 * np.cos(lat_rad)
    coords = np.column_stack([lat_m, lon_m])
    water_mask = group['is_water'].values == 1
    if water_mask.sum() == 0:
        return pd.Series(np.inf, index=group.index)
    tree = cKDTree(coords[water_mask])
    dist, _ = tree.query(coords)
    return pd.Series(dist, index=group.index)

twenty_cities['dist_to_water_m'] = (
    twenty_cities
    .groupby(['city', 'year'], group_keys=False)
    .apply(_water_dist)
)

# Step 3 — Binary proximity indicator
twenty_cities['near_water'] = (twenty_cities['dist_to_water_m'] < WATER_BUFFER_M).astype(int)

n_near = twenty_cities['near_water'].sum()
print(f"\nNear-water pixels (within {WATER_BUFFER_M}m): {n_near:,} ({n_near / len(twenty_cities) * 100:.1f}%)")

print(f"\nNear-water share by city (% of all city rows):")
near_by_city = (
    twenty_cities.groupby('city')['near_water']
    .mean()
    .mul(100)
    .round(1)
    .sort_values(ascending=False)
    .rename('% near water')
)
print(near_by_city.to_string())

print("\nNew columns added (LST unchanged):")
print("  is_water       — 1 if NDVI < threshold (exclude from LST regressions)")
print("  dist_to_water_m — distance in meters to nearest water pixel (same city-year)")
print("  near_water     — 1 if dist_to_water_m < buffer (use as regression control)")

# %%
# ── Section 8: Park patches ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 8: Park pixel classification and patch detection")
print("=" * 60)

PARK_SIM_THRESHOLD = 0.85  # similarity >= this → park pixel
PARK_MIN_PIXELS    = 3     # min connected park pixels to form a patch
PARK_ADJACENCY_M   = 220   # meters; covers 8-neighbors on 150m similarity grid
                           # (adjacent=150m, diagonal=212m; next-pixel=300m)

# ── Step 1: flag park pixels on unique (city, lat, lon) ───────────────────────
# Similarity is time-invariant so work on unique pixels, merge back after
unique_px = (
    twenty_cities[['city', 'latitude', 'longitude', 'similarity']]
    .drop_duplicates(subset=['city', 'latitude', 'longitude'])
    .copy()
)

# Bring is_water onto unique_px (take max across years so a pixel is water if
# it was ever classified as water)
_water_flag = (
    twenty_cities.groupby(['city', 'latitude', 'longitude'])['is_water']
    .max()
    .reset_index()
)
unique_px = unique_px.merge(_water_flag, on=['city', 'latitude', 'longitude'], how='left')
unique_px['is_water'] = unique_px['is_water'].fillna(0).astype(int)

unique_px['is_park_pixel'] = (
    unique_px['similarity'].notna() &
    (unique_px['similarity'] >= PARK_SIM_THRESHOLD) &
    (unique_px['is_water'] == 0)          # exclude MNDWI water pixels
).astype(int)

# ── Step 2: connected components per city ─────────────────────────────────────
def _park_patches(city_px):
    city_px = city_px.copy()
    city_px['in_park_patch'] = 0
    city_px['park_patch_id'] = 0   # within-city patch number (0 = not in a patch)
    park = city_px[city_px['is_park_pixel'] == 1]
    if len(park) < PARK_MIN_PIXELS:
        return city_px[['in_park_patch', 'park_patch_id']]

    lat_rad = np.radians(park['latitude'].mean())
    coords  = np.column_stack([
        park['latitude'].values  * 111_000,
        park['longitude'].values * 111_000 * np.cos(lat_rad),
    ])

    pairs = list(cKDTree(coords).query_pairs(PARK_ADJACENCY_M))
    n = len(park)
    if pairs:
        r, c = zip(*pairs)
        adj  = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
        adj  = adj + adj.T
    else:
        adj  = csr_matrix((n, n))

    _, labels = sp_connected_components(adj, directed=False)
    sizes     = np.bincount(labels)
    valid     = sorted(np.where(sizes >= PARK_MIN_PIXELS)[0])

    # Assign 1-indexed patch IDs to valid components
    comp_to_id = {comp: i + 1 for i, comp in enumerate(valid)}
    city_px.loc[park.index, 'in_park_patch'] = [1 if l in comp_to_id else 0 for l in labels]
    city_px.loc[park.index, 'park_patch_id'] = [comp_to_id.get(l, 0) for l in labels]
    return city_px[['in_park_patch', 'park_patch_id']]

_patch_result = (
    unique_px.groupby('city', group_keys=False)
    .apply(_park_patches)
)
unique_px['in_park_patch'] = _patch_result['in_park_patch']
unique_px['park_patch_id'] = _patch_result['park_patch_id']

# ── Step 3: merge back to full panel ──────────────────────────────────────────
twenty_cities = twenty_cities.merge(
    unique_px[['city', 'latitude', 'longitude', 'is_park_pixel', 'in_park_patch', 'park_patch_id']],
    on=['city', 'latitude', 'longitude'],
    how='left',
)
twenty_cities['is_park_pixel'] = twenty_cities['is_park_pixel'].fillna(0).astype(int)
twenty_cities['in_park_patch'] = twenty_cities['in_park_patch'].fillna(0).astype(int)
twenty_cities['park_patch_id'] = twenty_cities['park_patch_id'].fillna(0).astype(int)

# ── Park size classification ──────────────────────────────────────────────────
# Thresholds based on urban green space literature and patch size distribution:
#   Small  :  3–9 px  (0.7–2.0 ha)  — pocket/mini parks
#   Medium : 10–44 px (2.3–9.9 ha)  — neighborhood parks
#   Large  : 45+ px   (≥10.1 ha)    — community/district parks
# Each pixel = 150m × 150m = 0.225 ha
SMALL_MAX_PX  = 9
MEDIUM_MAX_PX = 44

_patch_sizes = (
    unique_px[unique_px['in_park_patch'] == 1]
    .groupby(['city', 'park_patch_id'])
    .size()
    .reset_index(name='patch_size_px')
)
_patch_sizes['park_size'] = pd.cut(
    _patch_sizes['patch_size_px'],
    bins=[0, SMALL_MAX_PX, MEDIUM_MAX_PX, np.inf],
    labels=['small', 'medium', 'large'],
).astype(str)

unique_px = unique_px.merge(_patch_sizes[['city', 'park_patch_id', 'park_size']],
                             on=['city', 'park_patch_id'], how='left')
twenty_cities = twenty_cities.merge(
    unique_px[['city', 'latitude', 'longitude', 'park_size']],
    on=['city', 'latitude', 'longitude'], how='left',
)
# park_size is NaN for non-patch pixels — that's correct

# ── Diagnostics ───────────────────────────────────────────────────────────────
n_park_px  = unique_px['is_park_pixel'].sum()
n_patch_px = unique_px['in_park_patch'].sum()
n_total_px = len(unique_px)
print(f"\nPark pixel threshold : similarity >= {PARK_SIM_THRESHOLD}")
print(f"Patch min size       : >= {PARK_MIN_PIXELS} connected pixels (8-connectivity)")
print(f"Park pixels (unique) : {n_park_px:,} of {n_total_px:,} ({n_park_px/n_total_px*100:.1f}%)")
print(f"In a patch (unique)  : {n_patch_px:,} ({n_patch_px/n_park_px*100:.1f}% of park pixels)")

print("\nPark pixel share by city (% of unique pixels):")
print(
    unique_px.groupby('city')['is_park_pixel']
    .mean().mul(100).round(1)
    .sort_values(ascending=False)
    .rename('% park pixels')
    .to_string()
)
print("\nIn-patch share by city (% of unique pixels):")
print(
    unique_px.groupby('city')['in_park_patch']
    .mean().mul(100).round(1)
    .sort_values(ascending=False)
    .rename('% in park patch')
    .to_string()
)
print("\nPark size breakdown (unique patch pixels):")
_size_counts = (
    unique_px[unique_px['in_park_patch'] == 1]
    .groupby('park_size', observed=True).size()
    .rename('n_pixels')
)
_patch_size_counts = (
    _patch_sizes.groupby('park_size', observed=True).size()
    .rename('n_patches')
)
_size_summary = pd.concat([_patch_size_counts, _size_counts], axis=1)
_size_summary['ha_range'] = ['0.7–2.0 ha', '2.3–9.9 ha', '≥10.1 ha']
print(_size_summary.to_string())

print("\nNew columns added:")
print(f"  is_park_pixel  — 1 if similarity >= {PARK_SIM_THRESHOLD}")
print(f"  in_park_patch  — 1 if part of a connected group of >= {PARK_MIN_PIXELS} park pixels (8-connectivity)")
print("  park_patch_id  — within-city patch number (0 = not in a patch)")
print("  park_size      — 'small' / 'medium' / 'large' (NaN if not in a patch)")

# ── Patch counts per city ─────────────────────────────────────────────────────
patch_counts = (
    unique_px[unique_px['in_park_patch'] == 1]
    .groupby('city')['park_patch_id']
    .nunique()
    .sort_values(ascending=False)
    .rename('n_patches')
)
print("\nPark patch count by city:")
print(patch_counts.to_string())
print(f"\nTotal patches across all cities: {patch_counts.sum():,}")

# ── Map: park patches per city ────────────────────────────────────────────────
cities  = sorted(unique_px['city'].unique())
n_cities = len(cities)
ncols   = 4
nrows   = (n_cities + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
axes_flat = axes.flatten()

for ax, city in zip(axes_flat, cities):
    cpx = unique_px[unique_px['city'] == city]

    # Background: sample of non-park pixels for geographic context
    non_park = cpx[cpx['is_park_pixel'] == 0]
    if len(non_park) > 5_000:
        non_park = non_park.sample(5_000, random_state=42)
    ax.scatter(non_park['longitude'], non_park['latitude'],
               c='lightgray', s=0.1, alpha=0.3, rasterized=True)

    # Isolated park pixels (similarity >= 0.85 but not in a patch)
    isolated = cpx[(cpx['is_park_pixel'] == 1) & (cpx['in_park_patch'] == 0)]
    if len(isolated):
        ax.scatter(isolated['longitude'], isolated['latitude'],
                   c='yellowgreen', s=0.5, alpha=0.6, rasterized=True)

    # Park patch pixels
    patch_px = cpx[cpx['in_park_patch'] == 1]
    if len(patch_px):
        ax.scatter(patch_px['longitude'], patch_px['latitude'],
                   c='darkgreen', s=1.5, alpha=0.9, rasterized=True)

    n_patches = patch_counts.get(city, 0)
    ax.set_title(f"{city}  ({n_patches:,} patches)", fontsize=9)
    ax.tick_params(labelsize=6)
    ax.set_xlabel('lon', fontsize=7)
    ax.set_ylabel('lat', fontsize=7)

for ax in axes_flat[n_cities:]:
    ax.set_visible(False)

legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',   markersize=6, label='Non-park pixel (sample)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellowgreen', markersize=6, label='Park pixel (isolated, < 6 connected)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen',   markersize=6, label='Park patch pixel (≥ 6 connected)'),
]
fig.legend(handles=legend_handles, loc='lower right', fontsize=9, framealpha=0.9)
fig.suptitle(f'Park Patches — similarity ≥ {PARK_SIM_THRESHOLD}, min {PARK_MIN_PIXELS} connected pixels (8-connectivity)', fontsize=12)
plt.tight_layout()
map_path = Path.home() / 'park_patches_map.png'
plt.savefig(map_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nMap saved to: {map_path}")

# %%
# ── Section 9: Distance to nearest park patch ─────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 9: Distance to nearest park patch")
print("=" * 60)

PARK_SIZE_NUM  = {'small': 1, 'medium': 2, 'large': 3}
PARK_DIST_CITIES = {'arlington', 'corpus christi', 'miami', 'oklahoma city', 'tulsa'}

def _park_patch_dist(city_px):
    """
    For pixels that are not park pixels and not water pixels,
    compute distance (m) to nearest park patch pixel and record
    the numeric size category (1=small, 2=medium, 3=large) of that patch.
    """
    city_px = city_px.copy()
    city_px['dist_to_park_m']    = np.nan
    city_px['nearest_park_size'] = np.nan

    # Only compute distances for the specified cities
    city_name = city_px['city'].iloc[0].lower()
    if city_name not in PARK_DIST_CITIES:
        return city_px[['dist_to_park_m', 'nearest_park_size']]

    patch_mask  = city_px['in_park_patch'] == 1
    target_mask = (city_px['in_park_patch'] == 0) & (city_px['is_water'] == 0)

    if patch_mask.sum() == 0 or target_mask.sum() == 0:
        return city_px[['dist_to_park_m', 'nearest_park_size']]

    lat_rad = np.radians(city_px['latitude'].mean())
    lat_m   = city_px['latitude'].values  * 111_000
    lon_m   = city_px['longitude'].values * 111_000 * np.cos(lat_rad)
    coords  = np.column_stack([lat_m, lon_m])

    patch_coords  = coords[patch_mask.values]
    target_coords = coords[target_mask.values]

    tree = cKDTree(patch_coords)
    dist, idx = tree.query(target_coords)

    # Map nearest patch pixel index → park_size → numeric
    patch_px = city_px[patch_mask]
    nearest_size_str = patch_px['park_size'].iloc[idx].values
    nearest_size_num = np.array([PARK_SIZE_NUM.get(s, np.nan) for s in nearest_size_str])

    city_px.loc[target_mask, 'dist_to_park_m']    = dist
    city_px.loc[target_mask, 'nearest_park_size']  = nearest_size_num
    return city_px[['dist_to_park_m', 'nearest_park_size']]


_park_dist_result = (
    unique_px.groupby('city', group_keys=False)
    .apply(_park_patch_dist)
)
unique_px['dist_to_park_m']    = _park_dist_result['dist_to_park_m']
unique_px['nearest_park_size'] = _park_dist_result['nearest_park_size']


def _interior_dist(city_px):
    """
    For park-interior pixels (in_park_patch == 1), compute distance (m) to the
    nearest non-park, non-water pixel. Mirrors _park_patch_dist but in reverse.
    """
    city_px = city_px.copy()
    city_px['interior_dist_m'] = np.nan

    city_name = city_px['city'].iloc[0].lower()
    if city_name not in PARK_DIST_CITIES:
        return city_px[['interior_dist_m']]

    park_mask   = city_px['in_park_patch'] == 1
    target_mask = (city_px['in_park_patch'] == 0) & (city_px['is_water'] == 0)

    if park_mask.sum() == 0 or target_mask.sum() == 0:
        return city_px[['interior_dist_m']]

    lat_rad = np.radians(city_px['latitude'].mean())
    lat_m   = city_px['latitude'].values  * 111_000
    lon_m   = city_px['longitude'].values * 111_000 * np.cos(lat_rad)
    coords  = np.column_stack([lat_m, lon_m])

    tree = cKDTree(coords[target_mask.values])
    dist, _ = tree.query(coords[park_mask.values])

    city_px.loc[park_mask, 'interior_dist_m'] = dist
    return city_px[['interior_dist_m']]


_interior_dist_result = (
    unique_px.groupby('city', group_keys=False)
    .apply(_interior_dist)
)
unique_px['interior_dist_m'] = _interior_dist_result['interior_dist_m']
print(f"interior_dist_m — {unique_px['interior_dist_m'].notna().sum():,} park-interior pixels "
      f"| mean {unique_px['interior_dist_m'].mean():.1f} m")

# Merge back to full panel
twenty_cities = twenty_cities.merge(
    unique_px[['city', 'latitude', 'longitude', 'dist_to_park_m', 'nearest_park_size', 'interior_dist_m']],
    on=['city', 'latitude', 'longitude'],
    how='left',
)

# ── Diagnostics ───────────────────────────────────────────────────────────────
n_eligible  = unique_px[(unique_px['is_park_pixel'] == 0) & (unique_px['is_water'] == 0)].shape[0]
n_with_dist = unique_px['dist_to_park_m'].notna().sum()
print(f"\nEligible pixels (non-park, non-water) : {n_eligible:,}")
print(f"Pixels with distance computed          : {n_with_dist:,}")
print(f"\nDistance to nearest park patch (m) — summary:")
print(unique_px['dist_to_park_m'].describe().round(1).to_string())
print(f"\nNearest park size distribution (unique eligible pixels):")
size_map = {1.0: 'small (1)', 2.0: 'medium (2)', 3.0: 'large (3)'}
size_counts = (
    unique_px['nearest_park_size']
    .value_counts()
    .sort_index()
    .rename(index=size_map)
)
print(size_counts.to_string())
print("\nNew columns added:")
print("  dist_to_park_m    — metres to nearest park patch pixel (NaN for park/water pixels)")
print("  nearest_park_size — 1=small  2=medium  3=large  (NaN for park/water pixels or cities with no patches)")

# ── Histogram: distance to nearest park patch ─────────────────────────────────
_dist_data = (
    unique_px[unique_px['dist_to_park_m'].notna()]
    [['city', 'dist_to_park_m']]
)

cities_with_dist = sorted(_dist_data['city'].unique())
n_cities_dist    = len(cities_with_dist)
fig, axes = plt.subplots(1, n_cities_dist, figsize=(5 * n_cities_dist, 4),
                         sharey=False)
if n_cities_dist == 1:
    axes = [axes]

for ax, city in zip(axes, cities_with_dist):
    vals = _dist_data.loc[_dist_data['city'] == city, 'dist_to_park_m']
    ax.hist(vals / 1_000, bins=40, color='steelblue', edgecolor='white', linewidth=0.4)
    ax.axvline(vals.median() / 1_000, color='firebrick', linewidth=1.2,
               linestyle='--', label=f'Median {vals.median()/1_000:.1f} km')
    ax.set_title(city, fontsize=10)
    ax.set_xlabel('Distance to nearest park patch (km)', fontsize=8)
    ax.set_ylabel('Number of pixels', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

fig.suptitle('Distance to Nearest Park Patch — non-park, non-water pixels', fontsize=12)
plt.tight_layout()
hist_path = Path.home() / 'dist_to_park_histogram.png'
plt.savefig(hist_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nHistogram saved to: {hist_path}")

# %%
# ── Save final panel ──────────────────────────────────────────────────────────
out_path = Path.home() / "20_cities_panel_clean.csv"
twenty_cities.to_csv(out_path, index=False)
print(f"\nFinal panel saved to: {out_path}")

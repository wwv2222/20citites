
import os
import io
import re
import getpass
from pathlib import Path
import pandas as pd
import dropbox

# ── Auth ──────────────────────────────────────────────────────────────────────
token = os.environ.get("DROPBOX_TOKEN") or getpass.getpass("Dropbox access token: ")
dbx = dropbox.Dropbox(token)

DROPBOX_FOLDER = "/20 Cities"   # path inside Dropbox (case-sensitive)
OUTPUT_PATH    = "/20_cities_panel.csv"

# ── Helper: list all entries in a Dropbox folder ──────────────────────────────
def list_folder(dbx, path):
    result = dbx.files_list_folder(path)
    entries = result.entries
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        entries += result.entries
    return entries


def download_csv(entry):
    _, response = dbx.files_download(entry.path_lower)
    return pd.read_csv(io.BytesIO(response.content))


# ── Main loop ─────────────────────────────────────────────────────────────────
all_dfs = []

city_folders = [
    e for e in list_folder(dbx, DROPBOX_FOLDER)
    if isinstance(e, dropbox.files.FolderMetadata)
]
print(f"Found {len(city_folders)} city folders\n")

for city_entry in sorted(city_folders, key=lambda e: e.name):
    city_name = city_entry.name
    city_path = city_entry.path_lower

    csv_files = [
        e for e in list_folder(dbx, city_path)
        if isinstance(e, dropbox.files.FileMetadata) and e.name.endswith(".csv")
    ]

    # ── Classify files ────────────────────────────────────────────────────────
    lst_ndvi_files = [
        e for e in csv_files
        if re.fullmatch(r'.+_\d{4}\.csv', e.name, re.IGNORECASE)
        and 'mndwi'      not in e.name.lower()
        and 'similarity' not in e.name.lower()
        and 'park'       not in e.name.lower()
    ]
    mndwi_files = [
        e for e in csv_files
        if re.search(r'mndwi', e.name, re.IGNORECASE)
    ]
    similarity_files = [
        e for e in csv_files
        if re.search(r'similarity|park', e.name, re.IGNORECASE)
    ]

    # ── Step 1: LST + NDVI base ───────────────────────────────────────────────
    if not lst_ndvi_files:
        print(f"WARNING: no LST/NDVI CSVs in {city_name} — skipping")
        continue

    base_dfs = []
    for entry in sorted(lst_ndvi_files, key=lambda e: e.name):
        match = re.search(r'_(\d{4})\.csv$', entry.name, re.IGNORECASE)
        if not match:
            print(f"  SKIP (no year): {entry.name}")
            continue
        df = download_csv(entry)
        df['year'] = int(match.group(1))
        df['city'] = city_name
        base_dfs.append(df)

    base = pd.concat(base_dfs, ignore_index=True)
    print(f"{city_name}: {len(base_dfs)} LST/NDVI files, {len(base):,} rows")

    # ── Step 2: MNDWI ─────────────────────────────────────────────────────────
    if mndwi_files:
        mndwi_dfs = []
        for entry in sorted(mndwi_files, key=lambda e: e.name):
            match = re.search(r'_(\d{4})\.csv$', entry.name, re.IGNORECASE)
            if not match:
                print(f"  SKIP MNDWI (no year): {entry.name}")
                continue
            df = download_csv(entry)
            df['year'] = int(match.group(1))

            # Auto-detect MNDWI column name (GEE may export it under different names)
            MNDWI_CANDIDATES = ['MNDWI', 'mndwi', 'MNDWI_mean', 'mean']
            mndwi_col = next((c for c in MNDWI_CANDIDATES if c in df.columns), None)
            if mndwi_col is None:
                numeric_cols = [c for c in df.select_dtypes('number').columns
                                if c not in ('latitude', 'longitude', 'year')]
                if len(numeric_cols) == 1:
                    mndwi_col = numeric_cols[0]
                    print(f"  MNDWI column inferred as '{mndwi_col}' in {entry.name}")
                else:
                    print(f"  SKIP MNDWI (cannot identify column): {entry.name} — columns: {list(df.columns)}")
                    continue
            df = df[['latitude', 'longitude', 'year', mndwi_col]].rename(columns={mndwi_col: 'MNDWI'})
            mndwi_dfs.append(df)

        mndwi = pd.concat(mndwi_dfs, ignore_index=True)
        city_panel = base.merge(mndwi, on=['latitude', 'longitude', 'year'], how='left')
        unmatched = city_panel['MNDWI'].isna().sum()
        if unmatched:
            print(f"  WARNING: {unmatched:,} rows had no MNDWI match")
        else:
            print(f"  MNDWI merge: all rows matched")
    else:
        print(f"  WARNING: no MNDWI files found for {city_name}")
        city_panel = base
        city_panel['MNDWI'] = float('nan')

    # ── Step 3: Park similarity (2021 only — propagates to all years via lat/lon merge) ──
    if similarity_files:
        similarity_files_sorted = sorted(
            similarity_files,
            key=lambda e: int(m.group(1)) if (m := re.search(r'rank(\d+)', e.name)) else 999
        )
        sim_entry = similarity_files_sorted[0]
        if len(similarity_files) > 1:
            print(f"  Multiple similarity files; using: {sim_entry.name}")
        sim_df = download_csv(sim_entry)

        # Merge on lat/lon only — year excluded so 2021 score carries across all years
        merge_keys = [k for k in ['latitude', 'longitude'] if k in sim_df.columns]
        drop_cols = [c for c in sim_df.columns if c in city_panel.columns and c not in merge_keys]
        if drop_cols:
            sim_df = sim_df.drop(columns=drop_cols)

        city_panel = city_panel.merge(sim_df, on=merge_keys, how='left')
        sim_cols = [c for c in sim_df.columns if c not in merge_keys]
        unmatched_sim = city_panel[sim_cols].isna().all(axis=1).sum()
        if unmatched_sim:
            print(f"  WARNING: {unmatched_sim:,} rows had no similarity match")
        else:
            print(f"  Similarity merge: all rows matched")
    else:
        print(f"  WARNING: no similarity file found for {city_name}")

    all_dfs.append(city_panel)

if not all_dfs:
    raise RuntimeError("No data loaded. Check the DROPBOX_FOLDER path and file naming.")

# ── Combine & save ────────────────────────────────────────────────────────────
panel = pd.concat(all_dfs, ignore_index=True)

# Upload using chunked session (handles files of any size)
buf = io.BytesIO()
panel.to_csv(buf, index=False)
buf.seek(0)
data = buf.read()
total = len(data)
CHUNK = 100 * 1024 * 1024  # 100 MB chunks

if total <= CHUNK:
    dbx.files_upload(data, OUTPUT_PATH, mode=dropbox.files.WriteMode.overwrite)
else:
    session = dbx.files_upload_session_start(data[:CHUNK])
    cursor  = dropbox.files.UploadSessionCursor(session.session_id, offset=CHUNK)
    offset  = CHUNK
    while offset < total:
        chunk = data[offset: offset + CHUNK]
        offset += len(chunk)
        if offset >= total:
            commit = dropbox.files.CommitInfo(
                path=OUTPUT_PATH,
                mode=dropbox.files.WriteMode.overwrite
            )
            dbx.files_upload_session_finish(chunk, cursor, commit)
        else:
            dbx.files_upload_session_append_v2(chunk, cursor)
            cursor.offset = offset

print(f"\nUploaded to Dropbox: {OUTPUT_PATH}")

# ── Save locally on server for fast access ────────────────────────────────────
LOCAL_PATH = Path.home() / "20_cities_panel.csv"
panel.to_csv(LOCAL_PATH, index=False)
print(f"Also saved locally to {LOCAL_PATH}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nShape  : {panel.shape}")
print(f"Columns: {list(panel.columns)}")
print(f"Cities : {sorted(panel['city'].unique())}")
print(f"Years  : {sorted(panel['year'].unique())}")
print(f"\nMissing values:\n{panel.isnull().sum()}")
print(f"\nSample:\n{panel.head()}")

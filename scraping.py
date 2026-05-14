

# %%
import subprocess
subprocess.run(["pip3", "install", "earthengine-api", "--quiet"])
subprocess.run(["pip3", "install", "dropbox", "--quiet"])
subprocess.run(["pip3", "install", "python-dotenv", "google-api-python-client", "pandas", "--quiet"])
import ee
ee.Authenticate()
ee.Initialize(project='grad-thesis-475918')

# %%
#Getting Citites 
cities_southeast_20 = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([-95.3698, 29.7604]), {'city': 'Houston', 'state': 'TX', 'rank': 1, 'radius_km': 30}),
  ee.Feature(ee.Geometry.Point([-98.4936, 29.4241]), {'city': 'San Antonio', 'state': 'TX', 'rank': 2, 'radius_km': 25}),
  ee.Feature(ee.Geometry.Point([-96.7970, 32.7767]), {'city': 'Dallas', 'state': 'TX', 'rank': 3, 'radius_km': 25}),
  ee.Feature(ee.Geometry.Point([-97.7431, 30.2672]), {'city': 'Austin', 'state': 'TX', 'rank': 4, 'radius_km': 22}),
  ee.Feature(ee.Geometry.Point([-81.6557, 30.3322]), {'city': 'Jacksonville', 'state': 'FL', 'rank': 5, 'radius_km': 30}),
  ee.Feature(ee.Geometry.Point([-97.3308, 32.7555]), {'city': 'Fort Worth', 'state': 'TX', 'rank': 6, 'radius_km': 22}),
  ee.Feature(ee.Geometry.Point([-80.8431, 35.2271]), {'city': 'Charlotte', 'state': 'NC', 'rank': 7, 'radius_km': 18}),
  ee.Feature(ee.Geometry.Point([-106.4800, 31.7776]), {'city': 'El Paso', 'state': 'TX', 'rank': 8, 'radius_km': 20}),
  ee.Feature(ee.Geometry.Point([-77.0369, 38.9072]), {'city': 'Washington', 'state': 'DC', 'rank': 9, 'radius_km': 15}),
  ee.Feature(ee.Geometry.Point([-86.7816, 36.1627]), {'city': 'Nashville', 'state': 'TN', 'rank': 10, 'radius_km': 22}),
  ee.Feature(ee.Geometry.Point([-97.5164, 35.4676]), {'city': 'Oklahoma City', 'state': 'OK', 'rank': 11, 'radius_km': 28}),
  ee.Feature(ee.Geometry.Point([-84.3880, 33.7490]), {'city': 'Atlanta', 'state': 'GA', 'rank': 12, 'radius_km': 20}),
  ee.Feature(ee.Geometry.Point([-76.0929, 36.8529]), {'city': 'Virginia Beach', 'state': 'VA', 'rank': 13, 'radius_km': 22}),
  ee.Feature(ee.Geometry.Point([-78.6382, 35.7796]), {'city': 'Raleigh', 'state': 'NC', 'rank': 14, 'radius_km': 18}),
  ee.Feature(ee.Geometry.Point([-80.1918, 25.7617]), {'city': 'Miami', 'state': 'FL', 'rank': 15, 'radius_km': 15}),
  ee.Feature(ee.Geometry.Point([-82.4572, 27.9506]), {'city': 'Tampa', 'state': 'FL', 'rank': 16, 'radius_km': 18}),
  ee.Feature(ee.Geometry.Point([-95.9928, 36.1540]), {'city': 'Tulsa', 'state': 'OK', 'rank': 17, 'radius_km': 16}),
  ee.Feature(ee.Geometry.Point([-97.1081, 32.7357]), {'city': 'Arlington', 'state': 'TX', 'rank': 18, 'radius_km': 14}),
  ee.Feature(ee.Geometry.Point([-90.0715, 29.9511]), {'city': 'New Orleans', 'state': 'LA', 'rank': 19, 'radius_km': 18}),
  ee.Feature(ee.Geometry.Point([-97.4034, 27.8006]), {'city': 'Corpus Christi', 'state': 'TX', 'rank': 20, 'radius_km': 16})
])

print(f"✓ Loaded {cities_southeast_20.size().getInfo()} cities")

city_list = [
    'Houston', 'San Antonio', 'Dallas', 'Austin', 'Jacksonville',
    'Fort Worth', 'Charlotte', 'El Paso', 'Washington', 'Nashville',
    'Oklahoma City', 'Atlanta', 'Virginia Beach', 'Raleigh', 'Miami',
    'Tampa', 'Tulsa', 'Arlington', 'New Orleans', 'Corpus Christi'
]

print(f"✓ Defined {len(city_list)} cities")


#Function to create sampling grid for a city based on its radius
def create_sampling_grid(feature):
    # Buffer the point into a circular polygon using radius_km
    radius_meters = ee.Number(feature.get('radius_km')).multiply(1000)
    bbox = feature.geometry().buffer(radius_meters)

    grid = ee.Image.pixelLonLat() \
        .reproject(crs='EPSG:4326', scale=150)
    points = grid.sample(
        region=bbox,
        scale=150,
        geometries=True
    )

    city_name = feature.get('city')
    state = feature.get('state')
    rank = feature.get('rank')

    def add_properties(point):

        return point.set({
            'city': city_name,
            'state': state,
            'rank': rank
        })

    return points.map(add_properties)

print("✓ Sampling grid will be built per city on demand")



#cloud masking
def mask_clouds_landsat457(image):
    """Cloud mask for Landsat 4, 5, and 7 using QA_PIXEL band"""
    qa = image.select('QA_PIXEL')
    # Bit 3 is cloud, bit 4 is cloud shadow
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0) \
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(cloud_mask)

def mask_clouds_landsat89(image):
    """Cloud mask for Landsat 8 and 9 using QA_PIXEL band"""
    qa = image.select('QA_PIXEL')
    # Bit 3 is cloud, bit 4 is cloud shadow
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0) \
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(cloud_mask)



#NDVI and LST Processing Functions
def process_landsat457(image):
    """
    Process Landsat 4, 5, or 7 image to calculate NDVI and LST.

    Band references for Landsat 4-5 and 7 Collection 2 Level 2:
    - SR_B4 = Near Infrared (NIR)
    - SR_B3 = Red
    - ST_B6 = Surface Temperature
    """
    # Apply cloud mask
    image = mask_clouds_landsat457(image)
    # Calculate NDVI
    ndvi = image.normalizedDifference(['SR_B4', 'SR_B3']).rename('NDVI')
    # Calculate LST
    lst = image.select('ST_B6') \
      .rename('LST')
    
    # Get date information
    date = ee.Date(image.get('system:time_start'))

    return image.addBands(ndvi).addBands(lst) \
        .set('date', date.format('YYYY-MM-dd')) \
        .set('year', date.get('year')) \
        .set('month', date.get('month')) \
        .set('day', date.get('day')) \
        .set('sensor', 'Landsat_457')

def process_landsat89(image):
    """
    Process Landsat 8 or 9 image to calculate NDVI and LST.

    Band references for Landsat 8-9 Collection 2 Level 2:
    - SR_B5 = Near Infrared (NIR)
    - SR_B4 = Red
    - ST_B10 = Surface Temperature
    """
    # Apply cloud mask
    image = mask_clouds_landsat89(image)

    # Calculate NDVI: (NIR - Red) / (NIR + Red)
    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

    # Calculate LST
    lst = image.select('ST_B10') \
      .rename('LST')

    # Get date information
    date = ee.Date(image.get('system:time_start'))

    return image.addBands(ndvi).addBands(lst) \
        .set('date', date.format('YYYY-MM-dd')) \
        .set('year', date.get('year')) \
        .set('month', date.get('month')) \
        .set('day', date.get('day')) \
        .set('sensor', 'Landsat_89')

print("✓ Defined NDVI/LST processing functions")



def get_landsat_for_city_year(city_geometry, year):
    start_date = f'{year}-06-01'
    end_date = f'{year}-09-30'
    merged = ee.ImageCollection([])

    # Landsat 5: 1984-2012
    if year >= 1984 and year <= 2012:
        landsat5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2') \
            .filterBounds(city_geometry) \
            .filterDate(start_date, end_date) \
            .map(process_landsat457)
        merged = merged.merge(landsat5)

    # Landsat 7 excluded entirely — SLC failure after May 2003 causes ~22% missing pixels.
    # Landsat 8 covers 2013+ so Landsat 7 is not needed.

    # Landsat 8: 2013-present
    if year >= 2013:
        landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(city_geometry) \
            .filterDate(start_date, end_date) \
            .map(process_landsat89)
        merged = merged.merge(landsat8)

    # Landsat 9: 2021-present
    if year >= 2021:
        landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
            .filterBounds(city_geometry) \
            .filterDate(start_date, end_date) \
            .map(process_landsat89)
        merged = merged.merge(landsat9)

    return merged

#Function to Sample Points for One Image
def sample_image_at_points(image, points):
    """
    Sample NDVI and LST values at all points for a single image.
    Returns a FeatureCollection with values and metadata.
    """

    sampled = image.select(['NDVI', 'LST']).sampleRegions(
        collection=points,
        scale=150,
        geometries=False  # Don't include geometry to reduce file size
    )

    # Add image metadata to each sampled point
    date = image.get('date')
    year = image.get('year')
    month = image.get('month')
    day = image.get('day')
    sensor = image.get('sensor')

    def add_metadata(feature):
        return feature.set({
            'date': date,
            'year': year,
            'month': month,
            'day': day,
            'sensor': sensor
        })

    return sampled.map(add_metadata)

print("✓ Defined sampling function")

#Function to Process One City-Year Combination
def process_city_year(city_name, year, poc_sample=None):
    """
    Process all Landsat data for one city and one year.
    Composites all images into a single median image, then samples once.
    Returns a FeatureCollection ready for export.
    """

    # Get city feature and geometry
    city_feature = cities_southeast_20.filter(ee.Filter.eq('city', city_name)).first()
    city_geometry = city_feature.geometry()

    # Build sampling grid on demand for this city only
    city_points = create_sampling_grid(city_feature)
    if poc_sample is not None:
        city_points = city_points.randomColumn('rand').sort('rand').limit(poc_sample)

    # Get all Landsat images for this city-year
    images = get_landsat_for_city_year(city_geometry, year)

    # Composite all images into a single median image, add lat/lon bands, sample once
    composite = images.select(['NDVI', 'LST']).median() \
        .addBands(ee.Image.pixelLonLat())
    sampled = composite.sampleRegions(
        collection=city_points,
        scale=150,
        geometries=False
    ).map(lambda f: f.set('year', year))

    return sampled

print("✓ Defined city-year processing function")


# SECURITY: Never hardcode tokens in source code.
# Store your Dropbox token in a .env file or set it as a shell environment variable:
#   export DROPBOX_TOKEN="your_token_here"
# Then load it here:
import os
from dotenv import load_dotenv
load_dotenv()  # reads from .env file if present

# %%
# --- ONE-TIME SETUP: Run this cell once to generate a Dropbox refresh token ---
# After running, DROPBOX_REFRESH_TOKEN will be saved to your .env automatically.
# You only need to do this once — the refresh token does not expire.
#
#Before running: fill in DROPBOX_APP_KEY and DROPBOX_APP_SECRET in your .env file.
#
#import dropbox as _dbx_setup
#_app_key = os.environ.get('DROPBOX_APP_KEY')
#_app_secret = os.environ.get('DROPBOX_APP_SECRET')
#_auth_flow = _dbx_setup.DropboxOAuth2FlowNoRedirect(_app_key, _app_secret, token_access_type='offline')
#_authorize_url = _auth_flow.start()
#print(f"1. Go to: {_authorize_url}")
#print("2. Click 'Allow' to authorize the app")
#print("3. Copy the authorization code and paste it below")
#_auth_code = input("Enter the authorization code: ").strip()
#_oauth_result = _auth_flow.finish(_auth_code)
#_refresh_token = _oauth_result.refresh_token
#with open('.env', 'a') as _f:
#    _f.write(f'\nDROPBOX_REFRESH_TOKEN={_refresh_token}\n')
#print(f"✓ Refresh token saved to .env")


# %%
# --- FULL PIPELINE: All cities, all years (1984-2024), one CSV per city ---
# Workflow per city-year: GEE → Google Drive → download → accumulate → delete from Drive
# After all years for a city: upload combined CSV to Dropbox
import time
import io
import pandas as pd
import dropbox
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

DRIVE_FOLDER = 'GradThesis'
YEARS = range(1984, 2025)
CHUNK = 150 * 1024 * 1024  # 150 MB — Dropbox chunked upload limit

# Set up Google Drive and Dropbox clients once
_adc_path = os.path.expanduser('~/.config/gcloud/application_default_credentials.json')
if os.path.isfile(_adc_path):
    from google.oauth2.credentials import Credentials as _OAuthCreds
    credentials = _OAuthCreds.from_authorized_user_file(_adc_path, scopes=['https://www.googleapis.com/auth/drive'])
else:
    credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/drive'])
drive = build('drive', 'v3', credentials=credentials)

folder_res = drive.files().list(
    q=f"name='{DRIVE_FOLDER}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
    fields="files(id)"
).execute()
folder_id = folder_res['files'][0]['id']

dbx = dropbox.Dropbox(
    oauth2_refresh_token=os.environ.get('DROPBOX_REFRESH_TOKEN'),
    app_key=os.environ.get('DROPBOX_APP_KEY'),
    app_secret=os.environ.get('DROPBOX_APP_SECRET')
)

SELECTORS = ['city', 'state', 'rank', 'longitude', 'latitude', 'year', 'NDVI', 'LST']


def export_year_to_drive(city_name, year, folder_id):
    """Export one city-year to Drive, return (task, file_prefix)."""
    file_prefix = f"{city_name.lower().replace(' ', '_')}_{year}"
    result = process_city_year(city_name, year)
    task = ee.batch.Export.table.toDrive(
        collection=result,
        description=file_prefix,
        folder=DRIVE_FOLDER,
        fileNamePrefix=file_prefix,
        fileFormat='CSV',
        selectors=SELECTORS
    )
    task.start()
    return task, file_prefix


def download_and_delete_from_drive(file_prefix, folder_id, retries=5):
    """Download a CSV from Drive into memory, delete it, return DataFrame."""
    for attempt in range(retries):
        try:
            file_res = drive.files().list(
                q=f"name='{file_prefix}.csv' and '{folder_id}' in parents and trashed=false",
                fields="files(id)"
            ).execute()
            file_id = file_res['files'][0]['id']

            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, drive.files().get_media(fileId=file_id))
            done = False
            while not done:
                _, done = downloader.next_chunk()

            drive.files().delete(fileId=file_id).execute()

            buf.seek(0)
            return pd.read_csv(buf)
        except Exception as e:
            if attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"\n    [retry {attempt+1}/{retries-1}] {type(e).__name__} — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def upload_to_dropbox(df, dropbox_path):
    """Upload a DataFrame as CSV to Dropbox with chunked upload support."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    data = buf.getvalue()

    if len(data) <= CHUNK:
        dbx.files_upload(data, dropbox_path, mode=dropbox.files.WriteMode.overwrite)
    else:
        session = dbx.files_upload_session_start(data[:CHUNK])
        cursor = dropbox.files.UploadSessionCursor(session_id=session.session_id, offset=CHUNK)
        offset = CHUNK
        while offset < len(data):
            chunk = data[offset:offset + CHUNK]
            if offset + CHUNK >= len(data):
                commit = dropbox.files.CommitInfo(path=dropbox_path, mode=dropbox.files.WriteMode.overwrite)
                dbx.files_upload_session_finish(chunk, cursor, commit)
            else:
                dbx.files_upload_session_append_v2(chunk, cursor)
                cursor = dropbox.files.UploadSessionCursor(session_id=cursor.session_id, offset=cursor.offset + len(chunk))
            offset += CHUNK


# --- Main loop: one city at a time, batch concurrent GEE exports ---
BATCH_SIZE = 15  # GEE concurrent task limit

# %%
for city_name in city_list:
    print(f"\n{'='*50}")
    print(f"Processing {city_name} ({len(YEARS)} years)...")

    # Skip fully completed cities
    if city_name in ['Houston', 'San Antonio', 'Dallas', 'Austin', 'Jacksonville',
                     'Fort Worth', 'Charlotte', 'El Paso', 'Washington', 'Nashville',
                     'Oklahoma City', 'Atlanta', 'Virginia Beach', 'Raleigh', 'Miami',
                     'Tampa', 'Tulsa']:
        print(f"  Skipping — already complete.")
        continue

    # Arlington: resume from 2004 (1984-2003 already exported)
    if city_name == 'Arlington':
        years_remaining = list(range(2004, 2025))
    else:
        years_remaining = list(YEARS)

    while years_remaining:
        # Launch up to BATCH_SIZE tasks at once
        batch = years_remaining[:BATCH_SIZE]
        years_remaining = years_remaining[BATCH_SIZE:]

        active_tasks = []
        for year in batch:
            print(f"  [{city_name}] {year} — submitting...", flush=True)
            task, file_prefix = export_year_to_drive(city_name, year, folder_id)
            active_tasks.append((task, file_prefix, year))

        # Wait for each task in the batch and download/upload as they complete
        for task, file_prefix, year in active_tasks:
            print(f"  [{city_name}] {year} — waiting...", end=' ', flush=True)
            while task.active():
                time.sleep(30)

            status = task.status()
            if status['state'] != 'COMPLETED':
                print(f"FAILED ({status.get('error_message', 'unknown')}) — skipping")
                continue

            df = download_and_delete_from_drive(file_prefix, folder_id)
            dropbox_path = f"/20 Cities/{city_name.lower().replace(' ', '_')}_{year}.csv"
            upload_to_dropbox(df, dropbox_path)
            print(f"✓ {len(df):,} rows → Dropbox")

print("\n✓ All cities complete.")
# %%
# --- RETRY: Arlington 2002-2003 and Fort Worth 1984-2016 ---
RETRY_TARGETS = [
    ('Arlington', [2002, 2003]),
    ('Fort Worth', list(range(1984, 2017))),
]

for city_name, retry_years in RETRY_TARGETS:
    print(f"\n{'='*50}")
    print(f"Retrying {city_name}: {retry_years[0]}–{retry_years[-1]} ({len(retry_years)} years)...")

    years_remaining = list(retry_years)
    while years_remaining:
        batch = years_remaining[:BATCH_SIZE]
        years_remaining = years_remaining[BATCH_SIZE:]

        active_tasks = []
        for year in batch:
            print(f"  [{city_name}] {year} — submitting...", flush=True)
            task, file_prefix = export_year_to_drive(city_name, year, folder_id)
            active_tasks.append((task, file_prefix, year))

        for task, file_prefix, year in active_tasks:
            print(f"  [{city_name}] {year} — waiting...", end=' ', flush=True)
            while task.active():
                time.sleep(30)

            status = task.status()
            if status['state'] != 'COMPLETED':
                print(f"FAILED ({status.get('error_message', 'unknown')}) — skipping")
                continue

            df = download_and_delete_from_drive(file_prefix, folder_id)
            dropbox_path = f"/20 Cities/{city_name.lower().replace(' ', '_')}_{year}.csv"
            upload_to_dropbox(df, dropbox_path)
            print(f"✓ {len(df):,} rows → Dropbox")

print("\n✓ Retry complete.")
# %%
# --- RESCRAPE: All cities, 2012 — replace city subfolder CSVs ---
BATCH_SIZE = 15  # GEE concurrent task limit
RESCRAPE_YEAR = 2012

print(f"\n{'='*50}")
print(f"Rescraping year {RESCRAPE_YEAR} for all {len(city_list)} cities...")

cities_remaining = list(city_list)
while cities_remaining:
    batch = cities_remaining[:BATCH_SIZE]
    cities_remaining = cities_remaining[BATCH_SIZE:]

    active_tasks = []
    for city_name in batch:
        print(f"  [{city_name}] {RESCRAPE_YEAR} — submitting...", flush=True)
        task, file_prefix = export_year_to_drive(city_name, RESCRAPE_YEAR, folder_id)
        active_tasks.append((task, file_prefix, city_name))

    for task, file_prefix, city_name in active_tasks:
        print(f"  [{city_name}] {RESCRAPE_YEAR} — waiting...", end=' ', flush=True)
        while task.active():
            time.sleep(30)

        status = task.status()
        if status['state'] != 'COMPLETED':
            print(f"FAILED ({status.get('error_message', 'unknown')}) — skipping")
            continue

        df = download_and_delete_from_drive(file_prefix, folder_id)
        slug = city_name.lower().replace(' ', '_')
        dropbox_path = f"/20 Cities/{city_name}/{slug}_{RESCRAPE_YEAR}.csv"
        upload_to_dropbox(df, dropbox_path)
        print(f"✓ {len(df):,} rows → Dropbox")

print(f"\n✓ Rescrape of {RESCRAPE_YEAR} complete.")


# --- MNDWI processing functions ---
def process_landsat457_mndwi(image):
    """Landsat 4/5/7: MNDWI = (SR_B2 Green - SR_B5 SWIR1) / (SR_B2 + SR_B5)"""
    image = mask_clouds_landsat457(image)
    mndwi = image.normalizedDifference(['SR_B2', 'SR_B5']).rename('MNDWI')
    date = ee.Date(image.get('system:time_start'))
    return image.addBands(mndwi) \
        .set('date', date.format('YYYY-MM-dd')) \
        .set('year', date.get('year')) \
        .set('sensor', 'Landsat_457')

def process_landsat89_mndwi(image):
    """Landsat 8/9: MNDWI = (SR_B3 Green - SR_B6 SWIR1) / (SR_B3 + SR_B6)"""
    image = mask_clouds_landsat89(image)
    mndwi = image.normalizedDifference(['SR_B3', 'SR_B6']).rename('MNDWI')
    date = ee.Date(image.get('system:time_start'))
    return image.addBands(mndwi) \
        .set('date', date.format('YYYY-MM-dd')) \
        .set('year', date.get('year')) \
        .set('sensor', 'Landsat_89')

def get_landsat_mndwi_for_city_year(city_geometry, year):
    start_date = f'{year}-06-01'
    end_date = f'{year}-09-30'
    merged = ee.ImageCollection([])
    if year >= 1984 and year <= 2012:
        merged = merged.merge(
            ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
                .filterBounds(city_geometry)
                .filterDate(start_date, end_date)
                .map(process_landsat457_mndwi)
        )
    if year >= 2013:
        merged = merged.merge(
            ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                .filterBounds(city_geometry)
                .filterDate(start_date, end_date)
                .map(process_landsat89_mndwi)
        )
    if year >= 2021:
        merged = merged.merge(
            ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
                .filterBounds(city_geometry)
                .filterDate(start_date, end_date)
                .map(process_landsat89_mndwi)
        )
    return merged

def process_city_year_mndwi(city_name, year):
    city_feature = cities_southeast_20.filter(ee.Filter.eq('city', city_name)).first()
    city_points = create_sampling_grid(city_feature)
    images = get_landsat_mndwi_for_city_year(city_feature.geometry(), year)
    composite = images.select('MNDWI').median().addBands(ee.Image.pixelLonLat())
    return composite.sampleRegions(
        collection=city_points,
        scale=150,
        geometries=False
    ).map(lambda f: f.set('year', year))

def export_mndwi_year_to_drive(city_name, year, folder_id):
    file_prefix = f"{city_name.lower().replace(' ', '_')}_mndwi_{year}"
    result = process_city_year_mndwi(city_name, year)
    task = ee.batch.Export.table.toDrive(
        collection=result,
        description=file_prefix,
        folder=DRIVE_FOLDER,
        fileNamePrefix=file_prefix,
        fileFormat='CSV',
        selectors=['city', 'state', 'rank', 'longitude', 'latitude', 'year', 'MNDWI']
    )
    task.start()
    return task, file_prefix


# --- MNDWI: All cities, all years 1984-2024 ---
print(f"\n{'='*50}")
print(f"Collecting MNDWI: {len(city_list)} cities × {len(list(YEARS))} years...")

for city_name in city_list:
    print(f"\n  Processing {city_name}...")
    years_remaining = list(YEARS)
    while years_remaining:
        batch = years_remaining[:BATCH_SIZE]
        years_remaining = years_remaining[BATCH_SIZE:]

        active_tasks = []
        for year in batch:
            print(f"    [{city_name}] {year} — submitting...", flush=True)
            task, file_prefix = export_mndwi_year_to_drive(city_name, year, folder_id)
            active_tasks.append((task, file_prefix, year))

        for task, file_prefix, year in active_tasks:
            print(f"    [{city_name}] {year} — waiting...", end=' ', flush=True)
            while task.active():
                time.sleep(30)
            status = task.status()
            if status['state'] != 'COMPLETED':
                print(f"FAILED ({status.get('error_message', 'unknown')}) — skipping")
                continue
            df = download_and_delete_from_drive(file_prefix, folder_id)
            slug = city_name.lower().replace(' ', '_')
            dropbox_path = f"/20 Cities/{city_name}/{slug}_mndwi_{year}.csv"
            upload_to_dropbox(df, dropbox_path)
            print(f"✓ {len(df):,} rows → Dropbox")

print("\n✓ MNDWI collection complete.")


# --- NLCD processing functions ---
def process_city_nlcd_year(city_name, nlcd_year):
    city_feature = cities_southeast_20.filter(ee.Filter.eq('city', city_name)).first()
    city_points = create_sampling_grid(city_feature)
    nlcd_image = ee.ImageCollection('USGS/NLCD_RELEASES/2021_REL/NLCD') \
        .filter(ee.Filter.calendarRange(nlcd_year, nlcd_year, 'year')) \
        .first()
    composite = nlcd_image.select('landcover').addBands(ee.Image.pixelLonLat())
    return composite.sampleRegions(
        collection=city_points,
        scale=30,
        geometries=False
    ).map(lambda f: f.set('nlcd_year', nlcd_year))

def export_nlcd_year_to_drive(city_name, nlcd_year, folder_id):
    file_prefix = f"{city_name.lower().replace(' ', '_')}_nlcd_{nlcd_year}"
    result = process_city_nlcd_year(city_name, nlcd_year)
    task = ee.batch.Export.table.toDrive(
        collection=result,
        description=file_prefix,
        folder=DRIVE_FOLDER,
        fileNamePrefix=file_prefix,
        fileFormat='CSV',
        selectors=['city', 'state', 'rank', 'longitude', 'latitude', 'nlcd_year', 'landcover']
    )
    task.start()
    return task, file_prefix


# --- NLCD: All cities, all available NLCD years ---
NLCD_YEARS = [2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021]

print(f"\n{'='*50}")
print(f"Collecting NLCD: {len(city_list)} cities × {len(NLCD_YEARS)} years...")

for city_name in city_list:
    print(f"\n  Processing {city_name}...")
    nlcd_remaining = list(NLCD_YEARS)
    while nlcd_remaining:
        batch = nlcd_remaining[:BATCH_SIZE]
        nlcd_remaining = nlcd_remaining[BATCH_SIZE:]

        active_tasks = []
        for nlcd_year in batch:
            print(f"    [{city_name}] NLCD {nlcd_year} — submitting...", flush=True)
            task, file_prefix = export_nlcd_year_to_drive(city_name, nlcd_year, folder_id)
            active_tasks.append((task, file_prefix, nlcd_year))

        for task, file_prefix, nlcd_year in active_tasks:
            print(f"    [{city_name}] NLCD {nlcd_year} — waiting...", end=' ', flush=True)
            while task.active():
                time.sleep(30)
            status = task.status()
            if status['state'] != 'COMPLETED':
                print(f"FAILED ({status.get('error_message', 'unknown')}) — skipping")
                continue
            df = download_and_delete_from_drive(file_prefix, folder_id)
            slug = city_name.lower().replace(' ', '_')
            dropbox_path = f"/20 Cities/{city_name}/{slug}_nlcd_{nlcd_year}.csv"
            upload_to_dropbox(df, dropbox_path)
            print(f"✓ {len(df):,} rows → Dropbox")

print("\n✓ NLCD collection complete.")

# %%

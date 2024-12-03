import os
import time
import logging
import warnings
import argparse
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import stackstac
import xarray as xr
import rioxarray
import pyproj

from scipy import stats
from collections import defaultdict
from pathlib import Path
from einops import reduce
from pystac_client import Client
from shapely.geometry import shape
from shapely.ops import transform

warnings.filterwarnings("ignore")

# pip install geopandas rasterio scikit-learn stackstac xarray einops dask zarr rioxarray pystac_client shapely

# CLAIMED Interface

# Tile id
tile_id = os.getenv('tile_id', '60WVT')
# Output directory
output_dir = os.getenv('output_dir', 'data/processed')
# Output directory
geometry_dir = os.getenv('geometry_dir', 'data/Sentinel-2-Shapefile-Index')
# Pick tiles with cloud cover below threshold. Threshold is increased by 0.1 if no sequences are found.
cloud_threshold = float(os.getenv('cloud_threshold', 0.15))
# Pick tiles with NaN coverage below threshold. Threshold is increased by 0.1 if no sequences are found.
nan_threshold = float(os.getenv('nan_threshold', 0.05))
# Start date
start_date = os.getenv('start_date', '2014-01-01')
# End date
end_date = os.getenv('end_date', '2023-12-31')
# Minimum steps in sampled tile sequence
min_sequence_length = int(os.getenv('min_sequence_length', 4))
# Maximum steps in sampled tile sequence
max_sequence_length = int(os.getenv('max_sequence_length', 16))
# Minimum window size between timestamps in weeks (?W)
min_window_size = os.getenv('min_window_size', '4W')
# Maximum window size between timestamps in weeks (?W). Increases by four weeks if no sequences are found.
max_window_size = os.getenv('max_window_size', '12W')
# Comma-seperated list of bands to pull from HLS S30 and L30. Missing bands are filled with -9999.
bands = os.getenv('bands', 'B02,B03,B04,B05,B06,B07,Fmask')
# Rename S30 bands. Can only be True if shared bands are downloaded.
rename_bands = bool(os.getenv('rename_bands', True))
# Chip size
chip_size = int(os.getenv('chip_size', 256))
# Sample length
sample_length = int(os.getenv('sample_length', 4))
# Step size between to valid samples in the temporal dimension
step_size = int(os.getenv('step_size', 2))
# Minimum ratio of good pixels in each chip
good_pixels_threshold = float(os.getenv('good_pixels_threshold', 0.8))
# Minimum number of chips in a sample with only good pixels (no clouds, no NaN)
min_full_chips = int(os.getenv('min_full_chips', 0))
# Minimum ratio of sequences with only good pixels (no clouds, no NaN)
min_full_sample_ratio = float(os.getenv('min_full_sample_ratio', 0.0))
# Minimum number of samples per tile id
min_samples = int(os.getenv('min_samples', 1500))
# Maximum number of samples per tile id
max_samples = int(os.getenv('max_samples', 3000))
# Maximum number of samples for each patch region
max_samples_per_region = int(os.getenv('max_samples_per_region', 10))
# Chunk size of the datacube
chunk_size = int(os.getenv('chunk_size', 16))
# Threshold for NaNs per sample
nan_sample_threshold = int(os.getenv('nan_sample_threshold', 656))
# Apply interpolation to each sample to fill NaNs
interpolate_na = bool(os.getenv('interpolate_na', True))
# Earthdata token or absolute path to earthdata token with text 'Authorization: Bearer <token>'
earthdata_token = os.getenv('earthdata_token', 'earthdata.txt')


if not os.path.isfile(earthdata_token):
    # Write token to file
    with open('earthdata.txt', 'w') as f:
        f.write(f'Authorization: Bearer {earthdata_token}')
    earthdata_token = os.path.join(os.getcwd(), 'earthdata.txt')

rio_env = rio.Env(
    GDAL_DISABLE_READDIR_ON_OPEN="TRUE",
    GDAL_HTTP_COOKIEFILE=os.path.expanduser("~/cookies.txt"),
    GDAL_HTTP_COOKIEJAR=os.path.expanduser("~/cookies.txt"),
    GDAL_HTTP_HEADER_FILE=earthdata_token,
)
rio_env.__enter__()

# Sentinel & Landsat collections from HLS
COLLECTIONS = [
    "HLSS30.v2.0",
    "HLSL30.v2.0",
]
RESOLUTION = 30  # Resolution of the bands in meters
FILL_VALUE = -9999  # Value to fill missing pixels with
CMR_STAC_URL = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"

L30_to_S30 = {'B01': 'B01',
              'B02': 'B02',
              'B03': 'B03',
              'B04': 'B04',
              'B05': 'B8A',
              'B06': 'B11',
              'B07': 'B12',
              'Fmask': 'Fmask',
              }
S30_TO_L30 = {v: k for k, v in L30_to_S30.items()}


def mgrs_to_geometry(mgrs_tile, geometry_dir):
    """
    Loads geometry of a MGRS tile.

    Args:
    mgrs_tile (str): The MGRS tile identifier (e.g., '17SLA').
    geometry_dir (str): The directory for the shape files.

    Returns:
    shapely.Polygon: Tile geometry (in lat lon)
    """
    geometry_file = os.path.join(geometry_dir, 'sentinel_2_index_shapefile.shp')
    if not os.path.exists(geometry_file):
        # Download MGRS tile shapes
        os.system('git clone https://github.com/justinelliotmeyers/Sentinel-2-Shapefile-Index')
        if not os.path.exists(geometry_file):
            os.makedirs(os.path.dirname(geometry_dir), exist_ok=True)
            os.system(f'mv Sentinel-2-Shapefile-Index {geometry_dir}')
            assert os.path.exists(geometry_file), \
                f"Automatic download of geometry file failed, please check {geometry_dir}."

    mgrs_gdf = gpd.read_file(geometry_file)
    mgrs_gdf = mgrs_gdf.drop_duplicates('Name').set_index('Name')
    geometry = mgrs_gdf.loc[mgrs_tile].geometry

    return geometry


def mgrs_to_epsg(mgrs_tile):
    """
    Convert a MGRS tile to an EPSG code.

    Args:
    mgrs_tile (str): The MGRS tile identifier (e.g., '17SLA').

    Returns:
    int: The corresponding EPSG code.
    """
    # Extract UTM zone from the MGRS tile
    utm_zone = int(mgrs_tile[:2])

    # EPSG code format for UTM is 326 for northern hemisphere and 327 for southern hemisphere
    # Latitude bands 'N' to 'X' are in the northern hemisphere, 'C' to 'M' are in the southern hemisphere
    # 'S' latitude band is in the northern hemisphere
    if "C" <= mgrs_tile[2] <= "M":
        epsg_code = 32700 + utm_zone
    else:
        epsg_code = 32600 + utm_zone

    return epsg_code


def search_catalog(tile_id, geometry, start_date, end_date):
    """
    Search the catalog for items matching the defined criteria.
    Args:
        tile_id (str): Tile id
        geometry (Polygon): tile polygon
        start_date (str): Start date in YYYY-MM-DD
        end_date (str): End date in YYYY-MM-DD

    Returns:
        List of STAC items

    """
    centroid = list(geometry.centroid.coords)[0]
    catalog = Client.open(f"{CMR_STAC_URL}")
    for trial in range(1, 6):
        try:
            hls = catalog.search(
                collections=COLLECTIONS,
                intersects={"type": "Point", "coordinates": centroid},
                datetime=f"{start_date}/{end_date}",
            )
            items = hls.item_collection()
        except Exception as e:
            # Error while downloading STAC items
            logging.exception(e)
            if trial == 5:
                # Abort after 5 trials
                logging.info(f'Five unsuccessful trials for tile {tile_id}')
                raise e
            else:
                time.sleep(30)
                logging.info(f'{trial}. retry')
                continue

        # Found items
        break

    logging.info(
        f"Found {hls.matched()} items for tile {tile_id} from {start_date} to {end_date}",
    )
    return items


def filter_items(items, cloud_threshold, nan_threshold, geometry):
    """
    Filter the items based on cloud cover and nan values.
    Args:
        items (list): The list of items.
        cloud_threshold (float): Maximum cloud cover (0 - 1)
        nan_threshold (float): Maximum NaN pixel ratio (0 - 1)
        geometry (Polygon): tile polygon

    Returns:
        GeoDataFrame: A GeoDataFrame containing the filtered items.
    """
    tiles_info = defaultdict(list)
    for item in items:
        tiles_info["id"].append(item.id)
        tiles_info["time"].append((item.datetime))
        tiles_info["geometry"].append(shape(item.geometry))
        tiles_info["cloud_cover"].append(item.properties["eo:cloud_cover"] / 100)
        non_nan = (shape(item.geometry).area / geometry.area)
        tiles_info["nan_cover"].append(1 - non_nan)
        tiles_info["good_pixels_ratio"].append(non_nan * (1 - item.properties["eo:cloud_cover"] / 100))

    tiles_df = gpd.GeoDataFrame(tiles_info, geometry="geometry")
    tiles_df = tiles_df.sort_values(by=["time", "good_pixels_ratio"])
    # Drop duplicate with smaller good_pixels_ratio
    tiles_df = tiles_df.drop_duplicates('time', keep='last')
    tiles_df.set_index("time", inplace=True)

    tiles_df = tiles_df[tiles_df.nan_cover <= nan_threshold]
    logging.info(f"Filtered to {len(tiles_df)} items with NaN cover <= {nan_threshold}")

    tiles_df = tiles_df[tiles_df.cloud_cover <= cloud_threshold]
    logging.info(f"Filtered to {len(tiles_df)} items with cloud cover <= {cloud_threshold}")

    return tiles_df


def search_sequences(
    df,
    min_sequence_length,
    max_sequence_length,
    min_window_size,
    max_window_size,
    density_based_sampling,
):
    """
    Search for valid sequences in the DataFrame.

    This function iterates over the DataFrame and identifies valid sequences of data.
    A sequence is considered valid if it contains a continuous period of data with no missing observations.
    The function returns a list of these valid sequence candidates.

    Args:
        df (DataFrame): The DataFrame to search.
        min_sequence_length (int): The minimum length of sequence candidates
        max_sequence_length (int): The maximum length of sequence candidates
        min_window_size (str): Minimum window length in weeks (?4)
        max_window_size (str): Maximum window length in weeks (?4)
        density_based_sampling (bool): Whether to sample based on date density within the window size

    Returns:
        list: A list of valid sequence candidates. Each sequence is a DataFrame.
    """

    sequence_candidates = []
    candidate_lengths = []

    if len(df) < min_sequence_length:
        return sequence_candidates

    # Scale total nanoseconds by daily nanoseconds to get daily density
    date_values = df.index.astype(int) / 864e10
    # Calculate kernel for density based sampling
    kernel = stats.gaussian_kde(date_values)

    # Avoid overlapping between sequence candidates
    used_dates = set()

    # Get sequence candidates
    for start in df.index[:-min_sequence_length]:
        # Init sequence with current date
        sequence = [start]
        used_dates.add(start)
        period_start = start + pd.to_timedelta(min_window_size)
        for j in range(max_sequence_length - 1):
            period_end = period_start - pd.to_timedelta(min_window_size) + pd.to_timedelta(max_window_size)

            period_df = df[
                (df.index >= period_start)
                & (df.index <= period_end)
                & (~df.index.isin(used_dates))
                ]

            if not period_df.empty:
                if len(period_df) == 1:
                    idx = period_df.index[0]
                elif density_based_sampling:
                    # Select timestamp based on inverse density within window size
                    date_values = period_df.index.astype(int) / 864e10
                    p = 1 / (kernel(date_values) + 1e-6)
                    p = p / p.sum()
                    idx = np.random.choice(period_df.index, p=p)
                else:
                    # Pick the value with the highest ratio of cloud and NaN-free pixels in the period
                    idx = period_df.sort_values(by="good_pixels_ratio").index[-1]

                sequence.append(idx)
                used_dates.add(idx)
                # Update period_start based on selected timestamp
                period_start = sequence[-1] + pd.to_timedelta(min_window_size)

            elif len(sequence) >= min_sequence_length:
                # No further observations in time window
                break

            else:
                # No further observations in time window and current sequence shorter than threshold
                for timestamp in sequence:
                    # Release dates
                    used_dates.remove(timestamp)
                sequence = []
                break

        if sequence:
            sequence_candidates.append(df.loc[sequence])
            candidate_lengths.append(len(sequence))

    if len(sequence_candidates) == 0:
        logging.info('No valid sequence found')
        return sequence_candidates

    logging.info(f"Found {len(sequence_candidates)} sequences candidates with length between {min(candidate_lengths)} "
                 f"and {max(candidate_lengths)} (window size {min_window_size} to {max_window_size})")

    return sequence_candidates


def select_sequence(sequence_candidates, df, items, *args, **kwargs):
    """
    Select a sequence from the sequence candidates.
    Download the FMask of this sequence and check for good pixel target for each tile.

    Args:
        sequence_candidates (list): List of dataframes.
        df (DateFrame):  DataFrame with all tile info.
        items: List of STAC items
        *args:
        **kwargs:

    Returns:
        DataArray with selected dates
    """

    if len(sequence_candidates) == 1:
        return pull_sequence(items, sequence_candidates[0], *args, **kwargs), []

    # Calculate the density of all days above cloud/nan threshold
    # Scale total nanoseconds by daily nanoseconds to get daily density
    date_values = df.index.astype(int) / 864e10
    kernel = stats.gaussian_kde(date_values)

    # Sample sequence based on inverse daily density to account for uneven distribution across years and seasons
    candidates_time = pd.concat(sequence_candidates).index
    date_values = candidates_time.astype(int) / 864e10
    p_date = 1 / (kernel(date_values) + 1e-6)

    # Multiply with ratio of cloud-free and NaN free pixels
    good_pixels_ratio = df.loc[candidates_time].good_pixels_ratio.values
    p_date = p_date * good_pixels_ratio

    # Aggregate weights by sequence candidate
    candidate_lengths = [len(s) for s in sequence_candidates]
    idx_date = np.concatenate([[i] * l for i, l in enumerate(candidate_lengths)])
    p_sequence = np.zeros(len(sequence_candidates))
    for i, s in enumerate(idx_date):
        p_sequence[s] += p_date[i]
    p_sequence = list(np.array(p_sequence) / np.array(p_sequence).sum())

    # Weighted sampling
    idx = np.random.choice(len(sequence_candidates), p=p_sequence)
    sequence = sequence_candidates[idx]
    tile_da = pull_sequence(items, sequence, *args, **kwargs)
    # Drop sequence from candidates
    _ = sequence_candidates.pop(idx)

    good_pixels_ratio = [round(v, 2) for v in sequence.good_pixels_ratio.values]
    months = [s.strftime('%y-%m') for s in sequence.index]
    logging.info(f'Sampled sequence:\n'
                 f'Length: {len(sequence)}\n'
                 f'Good pixel ratios: {good_pixels_ratio}\n'
                 f'Time stamps: {months}')

    return tile_da, sequence_candidates


def pull_data(items, bands, epsg_code, geometry, chip_size):
    """
    Pull data from STAC

    Args:
        items: STAC items
        bands (list): list of bands
        epsg_code (int): EPSG code
        geometry (Polygon): Shapely polygon of the tile
        chip_size (int): size of patches

    Returns:

    """
    if len(items) == 0:
        return

    tile_da = stackstac.stack(
        items=items,
        assets=bands,
        epsg=epsg_code,
        resolution=RESOLUTION,
        fill_value=FILL_VALUE,
        dtype="int16",
        rescale=False,
    )
    tile_da = tile_da.drop_vars(
        [var for var in tile_da.coords if var not in tile_da.dims and var != 'id'],
    )
    del tile_da.attrs["spec"]

    # Clip by bbox to remove NaN padding
    project = pyproj.Transformer.from_proj(
        pyproj.Proj(init='epsg:4326'),
        pyproj.Proj(init=tile_da.crs))
    bbox = transform(project.transform, geometry)

    tile_da = tile_da.rio.clip_box(*bbox.bounds)

    # Reduce to used area
    num_x = tile_da.sizes['x'] // chip_size
    num_y = tile_da.sizes['y'] // chip_size
    tile_da = tile_da.isel(x=slice(0, chip_size * num_x), y=slice(0, chip_size * num_y))

    return tile_da


def pull_sequence(items, sequence, epsg_code, geometry, bands, rename_bands, chip_size):
    """
    Pull a sequence of items from the catalog.

    This function filters the items based on the sequence, stacks them into a DataArray,
    and then drops unnecessary variables and attributes. The resulting DataArray is then computed.

    Optionally, the corresponding S30 bands for L30 are downloaded, renamed, and merged.

    Args:
        items (list): The list of items.
        sequence (pandas.DataFrame): The sequence of items to pull.
        epsg_code (int): The EPSG code.
        bands (list): list of bands
        rename_bands (bool): Whether to rename S30 bands to L30 bands

    Returns:
        xarray.DataArray: The computed DataArray.
    """

    sequence_ids = set(sequence.id.values)
    sequence_items = list(filter(lambda item: item.id in sequence_ids, items))
    # Drop samples with similar time stamp

    if rename_bands:
        assert all(b in L30_to_S30 for b in bands), (f"Can only process bands shared between L30 and S30 "
                                                     f"({L30_to_S30.keys()}). Provided bands: {bands}")
        # Init separate DataArrays for each product
        l30_sequence_items = list(filter(lambda item: '.L30.' in item.id, sequence_items))
        l30_tile_da = pull_data(l30_sequence_items, bands, epsg_code, geometry, chip_size)

        s30_sequence_items = list(filter(lambda item: '.S30.' in item.id, sequence_items))
        s30_tile_da = pull_data(s30_sequence_items, [L30_to_S30[b] for b in bands], epsg_code, geometry, chip_size)

        if s30_tile_da is not None:
            # Rename S30 to L30 bands
            s30_tile_da.coords['band'] = [S30_TO_L30[b] for b in s30_tile_da['band'].values]

        if l30_tile_da is not None and s30_tile_da is not None:
            # Concat L30 and S30 timestamps
            tile_da = xr.concat([l30_tile_da, s30_tile_da], dim='time')
            tile_da = tile_da.sortby('time')
        elif l30_tile_da is not None:
            # Only L30 present in sequence
            tile_da = l30_tile_da
        elif s30_tile_da is not None:
            # Only S30 present in sequence
            tile_da = s30_tile_da
        else:
            raise ValueError("No date")

    else:
        # Pull data without renaming bands
        tile_da = pull_data(sequence_items, bands, epsg_code, geometry, chip_size)

    return tile_da


def check_sample(sample, nan_sample_threshold):
    """
    Check Fmask in sample for cloud and NaN pixels.
    Args:
        sample (DataArray): sample

    Returns:
        list of ratio between 0 and 1 of good pixels
    """
    # Check for NaN data (threshold: 100 per band)
    if ((sample == FILL_VALUE).sum(dim=['y', 'x']) > nan_sample_threshold).any():
        # Set good pixels ratio to 0 -> skip sample
        return np.array([0.])

    # List of values meeting quality criteria from HLS super
    # https://git.earthdata.nasa.gov/projects/LPDUR/repos/hls-super-script/browse/HLS_PER.py#165
    good_q = [0, 1, 4, 5, 16, 17, 20, 21, 32, 33, 36, 37, 48, 49, 52, 53, 64,
              65, 68, 69, 80, 81, 84, 85, 96, 97, 100, 101, 112, 113, 116,
              117, 128, 129, 132, 133, 144, 145, 148, 149, 160, 161,
              164, 165, 176, 177, 180, 181, 192, 193, 196, 197, 208,
              209, 212, 213, 224, 225, 228, 229, 240, 241, 244, 245]

    pixel_stats = np.isin(sample.sel(band="Fmask"), good_q)
    good_pixels_ratio = reduce(pixel_stats.astype("float"), "t x y -> t", "mean")

    return good_pixels_ratio


def interpolate_samples(da):
    """
    Interpolate values along the x dimension with method nearest. Only 1D interpolation as it is 20 times faster.
    Args:
        da (DataArray): DataArray of the band values

    Returns: Interpolated DataArray

    """

    # def interpolate_slice(data):
    #     # Create a mask of valid (non-nan) entries
    #     mask = data != FILL_VALUE
    #
    #     if mask.all():
    #         # No NaN values
    #         return data
    #
    #     # Coordinates of known (non-nan) and unknown (nan) points
    #     y_known, x_known = np.nonzero(mask)
    #     y_unknown, x_unknown = np.nonzero(~mask)
    #
    #     # Known values
    #     values_known = data[y_known, x_known]
    #
    #     # Construct a KDTree for quick nearest-neighbor lookup
    #     tree = cKDTree(list(zip(y_known, x_known)))
    #
    #     # Find nearest neighbors for each unknown point
    #     distances, indices = tree.query(list(zip(y_unknown, x_unknown)))
    #
    #     # Interpolate (actually just copying the value from the nearest neighbor)
    #     interpolated_data = data.copy()
    #     interpolated_data[y_unknown, x_unknown] = values_known[indices]
    #
    #     return interpolated_data
    #
    # result_da = xr.apply_ufunc(
    #     interpolate_slice,
    #     da,
    #     input_core_dims=[['x', 'y']],
    #     output_core_dims=[['x', 'y']],
    #     vectorize=True
    # )

    da = da.astype(np.float32)
    da.values[da == -9999] = np.nan
    da.interpolate_na(dim='x', method='nearest')
    da = da.astype(np.int16)

    return da


def create_datacube(tile_id, tile_da, chip_size, sample_length, step_size, good_pixels_threshold, min_full_chips,
                    min_full_sample_ratio, drop_fmask, interpolate_na, nan_sample_threshold):
    """
    Create a datacube from a tile DataArray.

    This function processes the tile DataArray to create a datacube. It first crops the tile to an
    even multiple of the chip size. Then it processes each chunk of the tile, selecting only those
    chunks that meet the quality criteria. Finally, it creates a new Dataset with the processed
    chunks and returns it.

    Args:
        tile_id (str): The tile ID
        tile_da (DataArray): The tile DataArray to process.
        chip_size (int): Spatial size of samples
        sample_length (int): Timestamps per sample
        good_pixels_threshold (float): Threshold of good pixels in each sample
        min_full_chips (int): Minimum number of chips per sample with only good pixels
        min_full_sample_ratio (float): Minimum ratio of samples with only good pixels
        drop_fmask (bool): Whether to drop the Fmask band

    Returns:
        xarray.Dataset: The created datacube.
    """
    sequence_length, _, height, width = tile_da.shape

    bands = []
    time_ = []
    x_ = []
    y_ = []
    file_id = []
    prod = []
    pixel_quality = []

    # Iterate over all potential samples for this sequence
    for y_idx in range(0, height - chip_size + 1, chip_size):
        for x_idx in range(0, width - chip_size + 1, chip_size):
            # Iterate over temporal dimension with step size depending on valid samples
            t_idx = 0
            while t_idx <= sequence_length - sample_length:
                sample = tile_da.isel(
                    time=slice(t_idx, t_idx + sample_length),
                    y=slice(y_idx, y_idx + chip_size),
                    x=slice(x_idx, x_idx + chip_size),
                )
                # Only keep sample with a minimum number good pixels (no cloud, no NaN) in each chip and
                #  a minimum number of chips with only good pixels
                good_pixels_ratio = check_sample(sample, nan_sample_threshold)
                keep_sample = (all(good_pixels_ratio >= good_pixels_threshold) and
                               sum(good_pixels_ratio == 1.) >= min_full_chips)

                if keep_sample:
                    bands.append(sample.values)
                    time_.append(sample.time.values)
                    x_.append(sample.x)
                    y_.append(sample.y)
                    prod.append([s.split('.')[1] for s in sample.id.values])
                    pixel_quality.append(good_pixels_ratio)
                    file_id.append(sample.id.values)

                    t_idx += step_size
                else:
                    t_idx += 1

    n_all_samples = (int((sequence_length - sample_length) / step_size + 1)
                     * int(height / chip_size) * int(width / chip_size))
    logging.info(f'Found {len(bands)} valid samples (from {n_all_samples} samples)')

    # Check for no valid samples
    if len(bands) == 0:
        return xr.Dataset({"sample": []})

    bands = xr.DataArray(
        data=np.moveaxis(np.stack(bands), 1, 2),
        dims=("sample", "band", "time", "y", "x"),
        coords={'band': tile_da.band}
    )
    # Split bands and fmask
    fmask = bands.sel(band='Fmask')
    bands = bands.sel(band=[b.values for b in bands.band if b != 'Fmask'])

    time_ = xr.DataArray(data=np.stack(time_), dims=("sample", "time"))
    x_ = xr.DataArray(data=np.stack(x_), dims=("sample", "x"))
    y_ = xr.DataArray(data=np.stack(y_), dims=("sample", "y"))
    tile_id = xr.DataArray(data=np.full(bands.shape[0], tile_id), dims="sample")
    file_id = xr.DataArray(data=np.stack(file_id), dims=("sample", "time"))
    prod = xr.DataArray(data=np.stack(prod), dims=("sample", "time"))
    pixel_quality = xr.DataArray(data=np.stack(pixel_quality), dims=("sample", "time"))

    nan_count = np.isnan(time_.values).any(axis=1).sum()
    if nan_count:
        logging.error(f'Found {nan_count} NaN values in time_ for tile {tile_id}')

    datacube = xr.Dataset(
        data_vars={
            "bands": bands,
            "fmask": fmask,
            "time_": time_,
            "x_": x_,
            "y_": y_,
            "file_id": file_id,
            "prod": prod,
            "tile_id": tile_id,
            "pixel_quality": pixel_quality
        },
        attrs={"no_data": FILL_VALUE},
    )
    if drop_fmask:
        datacube = datacube.drop_vars(["fmask"])

    # Check for full chips in sequences
    perfect_samples = np.nonzero(datacube.pixel_quality.values.sum(axis=1) == sample_length)[0]
    candidates = np.nonzero(datacube.pixel_quality.values.sum(axis=1) < sample_length)[0]
    logging.info(f'{len(candidates) / (len(perfect_samples) + len(candidates)) * 100:.1f}% of samples have clouds')

    if min_full_sample_ratio:
        # Subsample sequences with cloud and NaN values
        max_samples = int(sum(perfect_samples) / min_full_sample_ratio * (1 - min_full_sample_ratio))
        if max_samples < len(candidates):
            selected_indices = np.concatenate([
                perfect_samples,  # Sequences with full chips
                np.random.choice(candidates, size=max_samples, replace=False)  # Samples with cloud / NaN values
            ])
            datacube = datacube.isel(sample=selected_indices)
            logging.info(f'Dropped {len(candidates) - max_samples} samples with clouds and NaNs')

    if interpolate_na:
        datacube['bands'] = interpolate_samples(datacube.bands)

    return datacube


def subsample_regions(datacube, max_samples_per_region):
    """

    Args:
        datacube (DataArray): Samples
        max_samples_per_region (int): Maximum threshold for subsampling

    Returns:
        DataArray with subsampled regions

    """
    if datacube.sizes['sample'] < max_samples_per_region:
        return datacube

    # Get region of each sample
    x_region = datacube.x_.isel(x=0).values.astype(int)
    y_region = datacube.y_.isel(y=0).values.astype(int)

    # Check for shift along x-axis
    x_unique = np.unique(x_region)
    if x_unique[1] - x_unique[0] < 256 * 30:
        # correct shift
        shifts = [(x - x_unique[0]) % (256 * 30) for x in x_unique]
        x_map = {x: x - s if s < (128 * 30) else x + s for x, s in zip(x_unique, shifts)}
        x_region = np.vectorize(lambda x: x_map[x])(x_region)

    # Check for shift along y-axis
    y_unique = np.unique(y_region)
    if y_unique[1] - y_unique[0] < 256 * 30:
        # correct shift
        shifts = [(y - y_unique[0]) % (256 * 30) for y in y_unique]
        y_map = {y: y - s if s < (128 * 30) else y + s for y, s in zip(y_unique, shifts)}
        y_region = np.vectorize(lambda y: y_map[y])(y_region)

    regions = np.stack([datacube.tile_id.values, x_region.astype(str), y_region.astype(str)], axis=-1)
    regions = ['_'.join(t) for t in regions]

    # Subsampling regions to meet threshold
    df = pd.DataFrame({'region': regions,
                       'sample_index': np.arange(len(regions)),
                       'pixel_quality': datacube.pixel_quality.sum('time')})
    # Shuffle samples and select samples with lowest NaN and cloud coverage
    sampled_df = df.groupby('region').apply(
        lambda x: x.sample(frac=1, random_state=42).sort_values('pixel_quality')[-max_samples_per_region:],
        include_groups=False)
    indices = sampled_df['sample_index'].sort_values().values

    datacube = datacube.isel(sample=indices)
    return datacube


def main(
    tile_id,
    output_dir,
    geometry_dir,
    cloud_threshold,
    nan_threshold,
    start_date,
    end_date,
    min_sequence_length,
    max_sequence_length,
    min_window_size,
    max_window_size,
    good_pixels_threshold,
    min_full_chips,
    min_full_sample_ratio,
    chip_size,
    sample_length,
    step_size,
    bands,
    rename_bands,
    chunk_size,
    min_samples,
    max_samples,
    max_samples_per_region,
    interpolate_na,
    nan_sample_threshold,
    density_based_sampling=False,
    drop_fmask=False,
    *args, **kwargs,
):    
    # Create directory to save datacube for the tile
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(bands, str):
        bands = bands.split(',')

    zarr_path = str(output_dir / f"{tile_id}.zarr")
    # Check if zarr for sequence already exists
    if os.path.exists(zarr_path):
        logging.info(f"Skipping tile {tile_id}, already exists")
        return

    tick = time.time()
    logging.info(f'Processing tile {tile_id}')

    epsg_code = mgrs_to_epsg(tile_id)
    geometry = mgrs_to_geometry(tile_id, geometry_dir)
    items = search_catalog(tile_id, geometry, start_date, end_date)
    if len(items) == 0:
        logging.info(f'No items found, abort')
        return

    # Get sequence candidates
    sequence_candidates = []
    while len(sequence_candidates) < 10:
        tiles_df = filter_items(items, cloud_threshold, nan_threshold, geometry)
        sequence_candidates = search_sequences(tiles_df, min_sequence_length, max_sequence_length, min_window_size,
                                               max_window_size, density_based_sampling)
        # Handle tile ids with only cloudy data
        if nan_threshold + cloud_threshold > 0.6:
            if len(sequence_candidates) > 0:
                # Continue with limited sequences
                break
            else:
                logging.info('Cannot find any valid sequences with >50% good pixels.')
                return

        # Increase thresholds if no sequence is found
        cloud_threshold += 0.1
        nan_threshold += 0.1
        max_window_size = f'{int(max_window_size[:-1]) + 4}W'

    # Select sequence, download data and create samples
    datacube = xr.Dataset({"sample": []})
    last_count = 0
    while len(sequence_candidates) and len(datacube.sample) < min_samples:
        tile_da, sequence_candidates = select_sequence(sequence_candidates, tiles_df, items,epsg_code,
                                                       geometry, bands, rename_bands, chip_size)

        logging.info(f"Downloading sequence (size: {tile_da.nbytes / 1e9:.3f} GB)")
        downloaded = False
        for trial in range(1, 4):
            try:
                tile_da = tile_da.compute()
                downloaded = True
            except Exception as e:
                logging.error(e)
                if trial < 3:
                    time.sleep(10)
                    logging.info(f'Error during data download. {trial}. retry')
        if not downloaded:
            logging.info('Error during data download. Skip sequence')
            continue

        samples_da = create_datacube(tile_id, tile_da, chip_size, sample_length, step_size, good_pixels_threshold,
                                     min_full_chips, min_full_sample_ratio, drop_fmask, interpolate_na,
                                     nan_sample_threshold)

        if len(datacube.sample) and len(samples_da):
            # Add to existing datacube
            datacube = xr.concat([datacube, samples_da], dim="sample")
        elif len(samples_da):
            datacube = samples_da

        datacube = subsample_regions(datacube, max_samples_per_region)
        logging.info(f'Number of samples after subsampling: {len(datacube.sample)}')

        if last_count == len(datacube.sample) and len(samples_da) > 10:
            logging.info(f'Cloud not improve sample could due to repeated regions. Skipping further sequences')
            break
        last_count = len(datacube.sample)

    if len(datacube.sample) == 0:
        logging.info(f"No samples found for tile id {tile_id}")
        return

    # Optionally limit number of samples per tile
    if max_samples and len(datacube.sample) > max_samples:
        selected_indices = np.random.choice(len(datacube.sample), size=max_samples, replace=False)
        datacube = datacube.isel(sample=selected_indices)

    # Re-chunk and save datacube (Uneven chunks must be dropped after tile merging and shuffling)
    datacube.chunk({"sample": chunk_size}).to_zarr(zarr_path, mode="w")

    logging.info(f"Processed tile {tile_id} with {len(datacube.sample)} samples in {(time.time() - tick) / 60:.1f} minutes")


def grid_process(batch_id, tile_id, *args, **kwargs):
    """
    Grid process for CLAIMED
    """
    # Set tile id
    tile_id = batch_id
    logging.debug(f"Settings: {kwargs}")
    # Run grid process for selected tile id
    main(tile_id=tile_id, *args, **kwargs)


if __name__ == "__main__":
    # Loading arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=output_dir)
    parser.add_argument('--tile_id', type=str, default=tile_id)
    parser.add_argument('--tile_id_file', type=str)
    parser.add_argument('--geometry_dir', type=str, default=geometry_dir,
                        help='Files are downloaded automatically to this path.')
    parser.add_argument('--cloud_threshold', type=float, default=cloud_threshold,
                        help='Pick tiles with cloud cover below threshold')
    parser.add_argument('--nan_threshold', type=float, default=nan_threshold,
                        help='Pick tiles with NaN coverage below threshold')
    parser.add_argument('--start_date', type=str, default=start_date)
    parser.add_argument('--end_date', type=str, default=end_date)
    parser.add_argument('--min_sequence_length', type=int, default=min_sequence_length,
                        help='Minimum steps in sampled tile sequence')
    parser.add_argument('--max_sequence_length', type=int, default=max_sequence_length,
                        help='Maximum steps in sampled tile sequence')
    parser.add_argument('--min_window_size', type=str, default=min_window_size,
                        help='Minimum window size between timestamps in weeks (?W)')
    parser.add_argument('--max_window_size', type=str, default=max_window_size,
                        help='Maximum window size between timestamps in weeks (?W)')
    parser.add_argument('--density_based_sampling', action='store_true',
                        help='Sample the next timestamp within the window size based on density. '
                             'Selects time with lowest cloud coverage otherwise.')
    parser.add_argument('--bands', type=str, nargs='+',
                        help='Bands to pull from HLS S30 and L30. Missing bands are filled with -9999.',
                        default=bands)
    # All bands -> deactivate rename_bands
    # default=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11",
    # "B12", "Fmask"])
    parser.add_argument('--rename_bands', type=bool, default=rename_bands)
    parser.add_argument('--drop_fmask', action='store_true', help='Drop Fmask in samples')
    parser.add_argument('--chip_size', type=int, default=chip_size)
    parser.add_argument('--sample_length', type=int, default=sample_length)
    parser.add_argument('--step_size', type=int, default=step_size)
    parser.add_argument('--good_pixels_threshold', type=float, default=good_pixels_threshold,
                        help='Minimum ratio of good pixels (no clouds, no NaN) in each chip')
    parser.add_argument('--min_full_chips', type=int, default=min_full_chips,
                        help='Minimum number of chips in a sample with only good pixels (no clouds, no NaN)')
    parser.add_argument('--min_full_sample_ratio', type=float, default=min_full_sample_ratio,
                        help='Minimum ratio of sequences with only good pixels (no clouds, no NaN)')
    parser.add_argument('--min_samples', type=int, default=min_samples)
    parser.add_argument('--max_samples', type=int, default=max_samples)
    parser.add_argument('--max_samples_per_region', type=int, default=max_samples_per_region)
    parser.add_argument('--interpolate_na', type=bool, default=interpolate_na)
    parser.add_argument('--nan_sample_threshold', type=int, default=nan_sample_threshold)
    parser.add_argument('--chunk_size', type=int, default=chunk_size, help='Chunk size of the datacube')
    args = parser.parse_args()

    kwargs = dict(
        output_dir=args.output_dir,
        tile_id=args.tile_id,
        geometry_dir=args.geometry_dir,
        cloud_threshold=args.cloud_threshold,
        nan_threshold=args.nan_threshold,
        start_date=args.start_date,
        end_date=args.end_date,
        max_sequence_length=args.max_sequence_length,
        min_sequence_length=args.min_sequence_length,
        density_based_sampling=args.density_based_sampling,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        good_pixels_threshold=args.good_pixels_threshold,
        min_full_chips=args.min_full_chips,
        min_full_sample_ratio=args.min_full_sample_ratio,
        chip_size=args.chip_size,
        sample_length=args.sample_length,
        step_size=args.step_size,
        bands=args.bands,
        rename_bands=args.rename_bands,
        chunk_size=args.chunk_size,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        max_samples_per_region=args.max_samples_per_region,
        interpolate_na=args.interpolate_na,
        nan_sample_threshold=args.nan_sample_threshold,
        drop_fmask=args.drop_fmask,
    )

    logging.basicConfig(
        level='INFO',
        handlers=[logging.StreamHandler()],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.tile_id is not None:
        main(**kwargs)
    else:
        # Distributed processing of all tiles defined in a comma-seperated txt file.
        # The code uses lock files to enable coordination between processes.

        # Read list of tiles
        with open(args.tile_id_file, 'r') as f:
            tiles = f.read().split(",")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Staggering start to avoid duplicated processing
        time.sleep(np.random.randint(0, 60))
        for tile_id in tiles:
            # Currently running process are marked with a .lock file
            lock_file = output_dir / f'{tile_id}.lock'
            if lock_file.exists():
                logging.info(f'Tile id {tile_id} locked')
                continue
            elif lock_file.with_suffix('.zarr').exists():
                logging.info(f'Tile id {tile_id} is processed')
                continue
            else:
                lock_file.touch()
                # Process tile
                kwargs['tile_id'] = tile_id
                try:
                    main(**kwargs)
                except Exception as e:
                    logging.exception(e)
                    pass

                if lock_file.with_suffix('.zarr').exists():
                    lock_file.unlink()
                else:
                    logging.info(f'Cannot find zarr file, check tile {tile_id}.')
                    # Failed processes are marked with .err files
                    lock_file.with_suffix('.err').touch()

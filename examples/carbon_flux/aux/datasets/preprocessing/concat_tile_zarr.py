
import os
import glob
import logging
import zarr
import xarray as xr
import random
import warnings
import tqdm
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime

# pip install xarray zarr dask tqdm numpy pandas geopandas shapely

# Pattern with wildcard to zarr files
zarr_file_pattern = os.environ.get('zarr_file_pattern')

# Path to save concat zarr file
target_path = os.environ.get('target_path')

# Concat dimension
concat_dim = os.environ.get('concat_dim', 'sample')

# Re-chunk data (no re-chunk when chunk_size=0)
chunk_size = int(os.environ.get('chunk_size', 0))

# Shuffle data along concat_dim
shuffle = bool(os.environ.get('shuffle', True))

# Number of chunks that saved at the same time
max_save_chunks = int(os.environ.get('max_save_chunks', 1000))

# Check coordinates for NaN and drop these samples
drop_nan_coords = bool(os.environ.get('drop_nan_coords', True))

# Comma-seperated list of coords to check
coords_vars = os.environ.get('coords_vars', 'time_,x_,y_')


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


def load_val_gdf(
    geometry_file='data/tiles/sentinel_2_index_shapefile.shp',
    val_tiles_file='datasets/v9_tiles/val_tiles.txt',
):
    mgrs_gdf = gpd.read_file(geometry_file)
    mgrs_gdf = mgrs_gdf.drop_duplicates('Name').set_index('Name')
    with open(val_tiles_file, 'r') as f:
        val_tiles = [t.strip()[:5] for t in f.read().split(',')]
    val_gdf = mgrs_gdf.loc[val_tiles]
    return val_gdf


def drop_overlay(ds, val_gdf):
    xmin = ds.x_.isel(x=0).values
    ymin = ds.y_.isel(y=0).values
    xmax = ds.x_.isel(x=-1).values
    ymax = ds.y_.isel(y=-1).values
    boxes = [shapely.box(*b) for b in zip(xmin, ymin, xmax, ymax)]
    gdf = gpd.GeoDataFrame(geometry=boxes)
    # Project to lat lon
    gdf = gdf.set_crs(mgrs_to_epsg(ds.tile_id.values[0])).to_crs(4326)

    # Drop samples
    drop_index = gdf.sjoin(val_gdf).index
    ds = ds.drop_sel(sample=drop_index)
    return ds


def main(zarr_file_pattern, target_path, concat_dim, chunk_size, shuffle, max_save_chunks, drop_nan_coords, coords_vars,
         **kwargs):
    """
    Concatenate multiple zarr files and saves them as a single zarr file.
    """
    assert target_path.endswith('.zarr'), 'Target path needs to end with .zarr'
    zarr_files = sorted(glob.glob(zarr_file_pattern))
    logging.info(f'Found {len(zarr_files)} zarr files: {str(zarr_files[:3])[:-1]}, ...]')
    assert len(zarr_files) > 1, 'Provide file name pattern for at least two zarr files.'

    ds_list = [xr.open_zarr(file_path, mask_and_scale=False) for file_path in zarr_files]
    logging.info(f'Loaded files to xarray')

    val_gdf = load_val_gdf()
    for i, ds in enumerate(ds_list):

        # Check for corrupted timestamps in downloaded samples
        try:
            if (ds.time_.isnull().any() | (np.datetime64('2013-12-31') > ds.time_).any() |
                    (ds.time_ > np.datetime64('2024-01-01')).any()):
                logging.info(f'Replacing time_ for {i}. ds')
                # Replacing time_ based on file id
                time_ = np.vectorize(lambda s: datetime.strptime(s.split('.')[3], '%Y%jT%H%M%S'))(ds.file_id.values)
                ds_list[i]['time_'] = (('sample', 'time'), time_)
        except:
            logging.error(f'Failed to extract time_ from file_id for {i}. ds: {ds}')
            pass

        # Check for corrupted tile id in downloaded samples
        try:
            if (ds.tile_id == '').any():
                logging.info(f'Replacing tile_id for {i}. ds')
                # Replacing tile_id based on file id
                tile_id = np.array(list(map(lambda s: s[0].split('.')[2], ds.file_id.values)))
                ds_list[i]['tile_id'] = (('sample'), tile_id)
        except:
            logging.error(f'Failed to extract time_ from file_id for {ds.tile_id.values[0]}. ds: {ds}')
            pass

        # Check for overlays between train samples and validation samples
        try:
            init_len = ds_list[i].sizes['sample']
            ds_list[i] = drop_overlay(ds_list[i], val_gdf)
            if ds_list[i].sizes['sample'] != init_len:
                logging.info(f"Dropped overlay in tile {ds.tile_id.values[0]} ({init_len} -> {ds_list[i].sizes['sample']})")
        except Exception as e:
            logging.exception(e)
            logging.error(f'Failed to subsample from file_id for {ds.tile_id.values[0]}. ds: {ds}')
            raise e


    ds = xr.concat(ds_list, dim=concat_dim)
    logging.info(f'Datasets concatenated with {len(ds.sample)} samples')

    if drop_nan_coords:
        logging.info(f'Searching for NaN values')
        for coords_var in coords_vars.split(','):
            nan_idx = ds[coords_var].isnull().any(axis=-1).values
            if nan_idx.sum():
                logging.info(f'Dropping {nan_idx.sum()} samples for {coords_var}')
                ds = ds.sel({concat_dim: ~nan_idx})

    # Drop file_id
    if 'file_id' in ds:
        ds = ds.drop('file_id')

    if shuffle:
        # Shuffled dataset along the concatenated dimension
        logging.info(f'Shuffle dataset')
        random.seed(42)
        warnings.filterwarnings("ignore", message='.*out-of-order index.*')
        shuffled_indices = list(range(ds.sizes[concat_dim]))
        random.shuffle(shuffled_indices)
        ds = ds.isel(**{concat_dim: shuffled_indices})

    ds = ds.chunk(chunks={concat_dim: chunk_size})
    logging.info(f'Chunked with chunk_size {chunk_size}')

    # drop last chunks for lengths chunk_size * 4
    # last_chunk_size = ds.chunks[concat_dim][-1]
    last_chunks = ds.sizes[concat_dim] % (chunk_size * 4)
    if last_chunks:
        drop_idx = list(range(ds.sizes[concat_dim] - last_chunks, ds.sizes[concat_dim]))
        ds = ds.drop_sel({concat_dim: drop_idx})
        logging.info(f'Dropping last chunks with size {last_chunks}')

    logging.info(f'Dataset:\n{ds}')

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
    encoding = {var_name: {"compressor": compressor} for var_name in ds.data_vars}

    logging.info(f'Saving file with {ds.sizes[concat_dim]} samples')
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # Iteratively saves concatenated samples to avoid OOM errors
    for i in tqdm.tqdm(list(range(0, ds.sizes[concat_dim], max_save_chunks * chunk_size))):
        ds_save = ds.isel({concat_dim: slice(i, i + max_save_chunks * chunk_size)})
        if os.path.exists(target_path):
            # Remove attrs for appending to zarr
            for data_var in ds_save.data_vars:
                ds_save[data_var].attrs = {}
            ds_save.to_zarr(target_path, append_dim=concat_dim)
        else:
            ds_save.to_zarr(target_path, encoding=encoding)
        ds_save.close()

    logging.info(f'Saved file to {target_path}')


def grid_process(batch, zarr_file_pattern, target_path, **kwargs):
    zarr_file_pattern += batch
    target_path += batch.replace('*', '').replace('?', '')

    logging.info(f'Running grid process with zarr_file_pattern {zarr_file_pattern} and target_path {target_path}')
    main(zarr_file_pattern, target_path, **kwargs)


if __name__ == '__main__':
    main(zarr_file_pattern, target_path, concat_dim, chunk_size, shuffle, max_save_chunks, drop_nan_coords, coords_vars)

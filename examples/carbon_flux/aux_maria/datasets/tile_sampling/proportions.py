
import time
import tqdm
import multiprocessing
import pandas as pd
import geopandas as gpd
import numpy as np
from functools import partial
import warnings
warnings.filterwarnings("ignore")


def _get_raster_stats(da, index_col, colum_names=None, verbose=False, gdf=None):
    if colum_names is not None and len(colum_names) > 0:
        stats = pd.DataFrame(0, index=gdf[index_col].unique(), columns=colum_names)
    else:
        stats = pd.DataFrame(index=gdf[index_col].unique())
    
    if verbose:
        rows = tqdm.tqdm(gdf.iterrows(), total=len(gdf), mininterval=len(gdf)/100)
    else:
        rows = gdf.iterrows()

    for id, row in rows:
        try:
            patch_da = da.rio.clip_box(*row.geometry.bounds).rio.clip([row.geometry])
            category, counts = np.unique(patch_da.values, return_counts=True)
            for cat, count in zip(category, counts):
                if cat not in stats.columns:
                    stats[cat] = 0
                stats.at[row[index_col], cat] += count
        except:
            # No data in bounds
            pass

    return stats


def raster_statistics(gdf, da, index_col, colum_names=None, num_processes=4, verbose=False):
    start = time.time()

    # Split gdf into batches
    batch_len = int(len(gdf) / num_processes) + 1
    inputs = [gdf.iloc[i:i+batch_len] for i in range(0, len(gdf), batch_len)]

    # Multiprocessing
    parallel_stats = partial(_get_raster_stats, da, index_col, colum_names, verbose)
    with multiprocessing.Pool() as pool:
        results = pool.map(parallel_stats, inputs)

    # Concatenate all results
    stats = pd.concat(results)

    # Merge values by tile id
    stats = stats.reset_index().groupby('index').sum()

    print(f'Statistics computed in {int(time.time() - start)} seconds')

    return stats


def _get_vector_stats(polygons, index_col, vector_col, batch_size=1000, verbose=False, gdf=None):
    stats = []
    if verbose:
        iter = tqdm.tqdm(list(range(0, len(gdf), batch_size)), mininterval=len(gdf)/batch_size/100)
    else:
        iter = range(0, len(gdf), batch_size)

    for i in iter:
        try:
            # Filter polygons for faster processing
            filtered_polygons = gpd.sjoin(polygons, gdf.iloc[i:i+batch_size])
            filtered_polygons.drop(columns=[index_col], inplace=True, errors='ignore')
            filtered_polygons.drop_duplicates(vector_col, inplace=True)

            intersections = gpd.overlay(gdf.iloc[i:i + batch_size], filtered_polygons)
            if len(intersections) > 0:
                intersections = intersections.set_index([index_col, vector_col])
                batch_stats = intersections.area.unstack()
                stats.append(batch_stats)
        except:
            # No data found
            pass

    if len(stats):
        stats = pd.concat(stats)
        return stats
    else:
        # No intersections
        return pd.DataFrame()


def vector_statistics(gdf, polygons, index_col, vector_col=None, num_processes=4, verbose=False, batch_size=1000):
    start = time.time()

    # Split gdf into batches
    batch_len = int(len(gdf) / num_processes) + 1
    inputs = [gdf.iloc[i:i+batch_len] for i in range(0, len(gdf), batch_len)]

    # Multiprocessing
    parallel_stats = partial(_get_vector_stats, polygons, index_col, vector_col, batch_size, verbose)
    with multiprocessing.Pool() as pool:
        results = pool.map(parallel_stats, inputs)

    # Concatenate all results
    stats = pd.concat(results)
    stats = stats.fillna(0)

    # Merge values by tile id
    stats = stats.reset_index().groupby(index_col).sum()

    if verbose:
        print(f'Statistics computed in {int(time.time() - start)} seconds')

    return stats

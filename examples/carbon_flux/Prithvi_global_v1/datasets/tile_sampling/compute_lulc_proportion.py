
import geopandas as gpd
import rioxarray as rxr
import numpy as np
from datasets.tile_sampling.proportions import raster_statistics

lulc_label_map = {
    0: "unknown",
    20: "shrubs",
    30: "herbaceous vegetation",
    40: "managed vegetation",
    50: "urban",
    60: "bare vegetation",
    70: "ice",
    80: "permanent water",
    90: "wetland",
    100: "moss",
    111: "closed forest evergreen needle",
    112: "closed forest evergreen broad",
    113: "closed forest other",
    114: "closed forest deciduous broad",
    115: "closed forest other",
    116: "closed forest other",
    121: "open forest mix",
    122: "open forest mix",
    123: "open forest mix",
    124: "open forest mix",
    125: "open forest mix",
    126: "open forest other",
    200: "sea",
    255: "nodata",
}


def main(polygon_path, raster_path, index_col, num_processes, output_path):
    gdf = gpd.read_file(polygon_path)
    da = rxr.open_rasterio(raster_path)

    lulc_stats = raster_statistics(gdf, da, index_col, colum_names=list(lulc_label_map.keys()),
                                   num_processes=num_processes, verbose=True)

    print(f'Computed LULC statistics for {len(lulc_stats)} grid cells')

    # Drop unknown and nodata
    lulc_stats.drop(columns=[0, 255], inplace=True)

    # Combine forest classes
    lulc_stats['closed forest'] = lulc_stats[[111, 112, 113, 114, 115, 116]].sum(axis=1)
    lulc_stats['open forest'] = lulc_stats[[121, 122, 123, 124, 125, 126]].sum(axis=1)
    lulc_stats.drop(columns=[111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126], inplace=True)

    # Rename columns
    lulc_stats = lulc_stats.rename(columns=lulc_label_map, errors='ignore')

    # Scale stats by pixel count
    lulc_stats = lulc_stats.div(lulc_stats.sum(axis=1) + 1e-6, axis=0)

    lulc_stats.to_csv(output_path)
    print(f'Saved LULC stats to {output_path}')


if __name__ == '__main__':


    main(
        polygon_path='data/tiles/sentinel_2_index_shapefile.shp',
        index_col='Name',
        raster_path='data/LULC/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif',
        output_path='data/tiles/core_grid_cells_lulc_stats.csv',
        num_processes=4,
    )

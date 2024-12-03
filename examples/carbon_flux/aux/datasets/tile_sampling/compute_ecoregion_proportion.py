
import geopandas as gpd
import numpy as np

from datasets.tile_sampling.proportions import vector_statistics


def main(polygon_path, ecoregion_path, index_col, num_processes, output_path):
    gdf = gpd.read_file(polygon_path)

    gdf_ecoregions = gpd.read_file(ecoregion_path)

    ecoregion_stats = vector_statistics(gdf, gdf_ecoregions, index_col, vector_col='ECO_NAME',
                                        num_processes=num_processes, verbose=True, batch_size=100)

    print(f'Found {len(ecoregion_stats.columns)} ecoregions in {len(ecoregion_stats)} grid cells')
    print(f'{len(gdf) - len(ecoregion_stats)} grid cells without overlap to ecoregions')
    print(f'{len(gdf_ecoregions) - len(ecoregion_stats.columns)} ecoregions missing')

    # Scale stats by patch area and add missing ecoregions
    area = gdf.set_index('grid_cell').area
    ecoregion_stats = ecoregion_stats.div(area, axis=0).fillna(0)

    ecoregion_stats.to_csv(output_path)
    print(f'Saved ecoregion stats to {output_path}')


if __name__ == '__main__':

    main(
        polygon_path='data/tiles/sentinel_2_index_shapefile.shp',
        index_col='Name',
        ecoregion_path='data/ecoregions/Biomes_and_Ecoregions_2017.shp',
        output_path='data/tiles/core_grid_cells_ecoregion_stats.csv',
        num_processes=4,
    )

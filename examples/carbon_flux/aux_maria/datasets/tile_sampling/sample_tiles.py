
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy


lulc_class_num = {
    # 'unknown: 0,
    'shrubs': 100,
    'herbaceous vegetation': 100,
    'managed vegetation': 100,
    'urban': 1000,
    'bare vegetation': 100,
    'ice': 100,
    'permanent water': 100,
    'wetland': 100,
    'moss': 100,
    'closed forest': 100,
    'open forest': 100,
    'entropy': 1000,
    # 'sea': 0,
    # 'nodata': 0,
}

lulc_from_highest = {
    # 'unknown: 0,
    'shrubs': 500,
    'herbaceous vegetation': 500,
    'managed vegetation': 500,
    'urban': 1000,
    'bare vegetation': 500,
    'ice': 500,
    'permanent water': 500,
    'wetland': 500,
    'moss': 500,
    'closed forest': 500,
    'open forest': 500,
    'entropy': 2000,
    # 'sea': 0,
    # 'nodata': 0,
}

train_color = 'green'
val_color = 'tab:orange'


def plot_bar(df, output_file, ylabel='Distribution [%]', xticklabels=None, vmax=None, **kwargs):
    fig, ax = plt.subplots(figsize=(8, 6))
    df.plot(kind='bar', ax=ax, width=0.8, color=[train_color, val_color, 'grey'], alpha=0.8, **kwargs)
    ax.set_ylabel(ylabel)
    if vmax:
        ax.set_ylim(0, vmax)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.4)
    plt.legend(loc='upper right')
    plt.savefig(output_file)


def sample_tiles(output_dir,
                 tile_path,
                 lulc_stats_path,
                 biome_stats_path,
                 n_val=0.05,
                 min_biome_tiles=2,
                 random_val=True,
                 seed=1911,
                 ):
    hls_gdf = gpd.read_file(tile_path)
    hls_gdf.to_crs('EPSG:4326')

    lulc_stats = pd.read_csv(lulc_stats_path, index_col=0)
    lulc_stats.columns = lulc_stats.columns.astype(int)
    biome_stats = pd.read_csv(biome_stats_path, index_col=0)

    rng = np.random.default_rng(seed=seed)
    os.makedirs(output_dir, exist_ok=True)

    # Combine forests classes and add entropy as diversity factor
    lulc_stats['entropy'] = lulc_stats.apply(lambda row: entropy(row), axis=1)

    # Sample train tiles
    sampled_tiles = []
    for c, class_num in lulc_class_num.items():
        candidates = lulc_stats[c].sort_values()[-lulc_from_highest[c]:].index
        # n_class = int(class_num / total_samples * n_train)
        sampled_tiles.extend(rng.choice(candidates, size=class_num, replace=False))
    print(f'{len(np.unique(sampled_tiles))} sampled tiles from LULC classes and entropy')
    # Check biomes in train set
    for biome in biome_stats:
        candidates = biome_stats[biome_stats[biome] > 0.1].index
        min_target = min(len(candidates), min_biome_tiles)
        # Sample additional tiles until min target number is reached
        while len(set(candidates) & set(sampled_tiles)) < min_target:
            sampled_tiles.extend(rng.choice(candidates, size=1, replace=False))

    # Drop duplicates
    sampled_tiles = np.unique(sampled_tiles)
    sampled_gdf = hls_gdf.loc[hls_gdf.index.isin(sampled_tiles)]
    print(f'{len(sampled_tiles)} sampled tiles after biome check')

    # Random train test split
    if n_val > 1:
        n_val = n_val / len(sampled_gdf)
    if random_val:
        val_gdf = sampled_gdf.sample(frac=n_val, random_state=42)
    else:
        # Select 5% of the samples as validation set based on the LULC classes.
        val_tiles = []
        lulc_train_idx = list(set(sampled_gdf.index) & set(lulc_stats.index))
        selected_lulc = lulc_stats.loc[lulc_train_idx]
        for c, class_num in lulc_class_num.items():
            candidates = selected_lulc[c].sort_values()[-class_num:].index
            n_class = int(class_num * n_val)
            val_tiles.extend(rng.choice(candidates, size=n_class, replace=False))

        # Add random tiles to match target train val split
        while len(set(val_tiles)) < len(sampled_tiles) * n_val:
            val_tiles.extend(rng.choice(sampled_tiles, size=1, replace=False))

        val_tiles = np.unique(val_tiles)
        val_gdf = hls_gdf.loc[hls_gdf.index.isin(val_tiles)]
    train_gdf = sampled_gdf.drop(val_gdf.index)

    # Drop potential overlapping tiles
    intersections = gpd.overlay(train_gdf.reset_index(), val_gdf.reset_index(), how='intersection')
    intersections = intersections.dissolve(by='Name_1')
    intersections['ratio'] = (intersections.area / train_gdf.area).dropna()
    drop_index = intersections[intersections['ratio'] > 0.3].index

    if len(drop_index):
        print(f'Drop {len(drop_index)} train tiles with >30% overlap with val tiles')
        train_gdf = train_gdf.drop(drop_index)

    # Save comma-seperated tiles
    with open(os.path.join(output_dir, 'train_tiles.txt'), 'w') as f:
        f.write(','.join(train_gdf.index))
    print(f'Saved {len(train_gdf)} train tiles')

    with open(os.path.join(output_dir, 'val_tiles.txt'), 'w') as f:
        f.write(','.join(val_gdf.index))
    print(f'Saved {len(val_gdf)} validation tiles')

    # Compute entropy
    lulc_entropy = entropy(lulc_stats.loc[list(set(train_gdf.index) & set(lulc_stats.index))].sum())
    biome_entropy = entropy(biome_stats.loc[list(set(train_gdf.index) & set(biome_stats.index))].sum())

    print(f'Training Lulc entropy: {lulc_entropy:.2f}')
    print(f'Training biome entropy: {biome_entropy:.2f}')

    # Save some config values
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write(f'{len(train_gdf)} train tiles\n'
                f'{len(val_gdf)} validation tiles\n\n'
                f'Training Lulc entropy: {lulc_entropy:.2f}\n'
                f'Training biome entropy: {biome_entropy:.2f}\n\n'
                f'seed: {seed}\n'
                f'random_val: {random_val}\n'
                f'n_val: {n_val}\n'
                f'min_biome_tiles: {min_biome_tiles}\n'
                f'lulc_stats_path: {lulc_stats_path}\n'
                f'biome_stats_path: {biome_stats_path}\n')

    # Plot locations on a map
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    fig, ax = plt.subplots(figsize=(24, 12))
    world.plot(ax=ax, color="white", edgecolor="black")
    train_gdf.plot(ax=ax, color=train_color)
    val_gdf.plot(ax=ax, color=val_color)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'tile_map.png'))

    # Plot distribution
    lulc_stats.drop(columns=['entropy'], inplace=True)
    lulc_distribution = pd.DataFrame({
        'Training': lulc_stats.loc[list(set(train_gdf.index) & set(lulc_stats.index))].sum(),
        'Validation': lulc_stats.loc[list(set(val_gdf.index) & set(lulc_stats.index))].sum(),
        'Earth': lulc_stats.sum(),
    })
    lulc_coverage = pd.DataFrame({
        'Training': lulc_distribution['Training'] / lulc_distribution['Earth'] * 100,
        'Validation': lulc_distribution['Validation'] / lulc_distribution['Earth'] * 100
    })
    lulc_distribution = lulc_distribution / lulc_distribution.sum() * 100
    plot_bar(lulc_distribution, os.path.join(output_dir, 'sampled_lulc_distribution.png'), 
             ylabel='Distribution [%]', xticklabels=list(lulc_distribution.index), vmax=30)
    plot_bar(lulc_coverage, os.path.join(output_dir, 'sampled_lulc_coverage.png'),
             ylabel='Coverage [%]', xticklabels=list(lulc_distribution.index), vmax=70,
             stacked=True)

    biome_distribution = pd.DataFrame({
        'Training': biome_stats.loc[list(set(train_gdf.index) & set(biome_stats.index))].sum(),
        'Talidation': biome_stats.loc[list(set(val_gdf.index) & set(biome_stats.index))].sum(),
        'Earth': biome_stats.sum(),
    })
    biome_distribution = biome_distribution / biome_distribution.sum() * 100
    plot_bar(biome_distribution, os.path.join(output_dir, 'sampled_biome_distribution.png'), ylabel='Distribution [%]',
             vmax=30)

    # Plot count
    lulc_count = pd.DataFrame({
        'Training': (lulc_stats.loc[list(set(train_gdf.index) & set(lulc_stats.index))] > 0).sum(),
        'Validation': (lulc_stats.loc[list(set(val_gdf.index) & set(lulc_stats.index))] > 0).sum(),
        'Earth': (lulc_stats > 0).sum(),
    })
    plot_bar(lulc_count, os.path.join(output_dir, 'sampled_lulc_count.png'), 
             ylabel='Count', xticklabels=list(lulc_count))

    biome_count = pd.DataFrame({
        'Training': (biome_stats.loc[list(set(train_gdf.index) & set(biome_stats.index))] > 0).sum(),
        'Validation': (biome_stats.loc[list(set(val_gdf.index) & set(biome_stats.index))] > 0).sum(),
        'Earth': (biome_stats > 0).sum(),
    })
    plot_bar(biome_count, os.path.join(output_dir, 'sampled_biome_count.png'), ylabel='Count')
    print(f'Saved plots in {output_dir}')


if __name__ == '__main__':

    sample_tiles(
        output_dir='datasets/v9_tiles',
        tile_path='data/tiles/sentinel_2_index_shapefile.shp',
        biome_stats_path='data/tiles/core_grid_cells_ecoregion_stats.csv',
        lulc_stats_path='data/tiles/core_grid_cells_lulc_stats.csv',
        seed=1911,
        min_biome_tiles=3,
        random_val=True,
        n_val=0.05,
    )



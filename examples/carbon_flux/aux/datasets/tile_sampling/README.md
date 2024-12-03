# HLS v9 tile sampling

## Data

### HLS tile ids

Sentinel-2 shape files (https://github.com/justinelliotmeyers/Sentinel-2-Shapefile-Index)

Download files to `data/tiles`
```shell
git clone https://github.com/justinelliotmeyers/Sentinel-2-Shapefile-Index data/tiles
```

### Biomes

The 847 RESOLVE Ecoregions 2017 can be downloaded from https://hub.arcgis.com/datasets/37ea320eebb647c6838c23f72abae5ef/explore?location=4.703164%2C-139.736838%2C2.65.

Save the shapefiles at `data/ecoregions/Biomes_and_Ecoregions_2017.shp`.


### LULC

The Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019 is available here: https://zenodo.org/records/3939050.

Download the discrete classification map
```shell
mkdir data/LULC
wget https://zenodo.org/records/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif -P data/LULC
```

## Approach

### Preparation

Run [compute_lulc_proportion.py](compute_lulc_proportion.py) to compute the proportions of LULC classes for each tile. 

```shell
python datasets/tile_sampling/compute_lulc_proportion.py
```

Run [compute_ecoregion_proportion.py](compute_ecoregion_proportion.py) to compute the proportions of ecoregions for each tile. 

```shell
python datasets/tile_sampling/compute_ecoregion_proportion.py
```


### Sampling
We sample from all tiles that are either labeled with LULC classes or included in ecoregions (~20k land tiles).

Run [sample_tiles.py](sample_tiles.py) for sampling tile ids.

```shell
python datasets/tile_sampling/sample_tiles.py
```

The sampling approach has three steps:

1. Sample a fixed number of tiles of each LULC class from the top N tiles with the highest proportions of this class. E.g., from the 500 tiles with the highest proportion of forest area, we sample 100.
2. Compute the LULC entropy of each tile and sample additional tiles from tiles with high entropy. This steps should increase the diversity of sampled tiles.
3. Iterate over all 847 ecoregions and check if at least three tiles from this region are present in the sampled tiles. Otherwise, sample further tiles that contain at least 10% of this ecoregion.

The first two are similar to [Clay](https://clay-foundation.github.io/model/data_sampling.html). The third one ensures a global coverage of regions and biodiversity.




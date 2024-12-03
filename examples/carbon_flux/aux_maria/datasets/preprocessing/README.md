# HLS v9 download and preprocessing

The samples are download for each tile separately and then shuffled.
We used [CLAIMED grid wrappers](https://github.com/claimed-framework/c3/blob/main/GettingStarted.md) to parallelize the processes and deploy them on OpenShift.

Downloading code on a tile-level: [hls_preprocess_tiles.py](hls_preprocess_tiles.py)

Concatenation of the tile-based zarr files: [concat_tile_zarr.py](concat_tile_zarr.py)

After data preprocessing, the mean and std values of the training samples can be computed.

Distributed computation: [compute_hls_standardization.py](compute_hls_standardization.py)

Combine standardization values: [combine_standardization_values.py](combine_standardization_values.py)



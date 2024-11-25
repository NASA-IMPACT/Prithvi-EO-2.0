"""
This operator computes mean and std values for multiple zarr files.
"""
# pip install xarray zarr tqdm

import os
import logging
import glob
import json
import tqdm
import xarray as xr
import numpy as np

# glob pattern for all zarr files to process (e.g. path/to/files/**/*.zarr)
file_path_pattern = os.getenv('file_path_pattern')
# comma-separated list of bands
data_var = os.getenv('data_var', "bands")
# json target file path  for saving the statistics
target_path = os.getenv('target_path')
# set no_data values to nan
ignore_no_data_value = bool(os.getenv('ignore_no_data_value', True))
# batch_size of samples to reduce memory usage
batch_size = int(os.getenv('batch_size', 1000))


def combine_standard_scalars(means, variances, sample_counts):
    """
    Combines a list of means and variances, weighted by the sample counts.

    This function is inspired by partial_fit from sklearn.preprocessing.StandardScaler. In contrast, this function does
    not require the actual new data but just the standardization values itself. The current implementation does not
    include error correction which could lead to differences for very large number of samples.


    The algorithm for incremental mean and std is given in Equation 1.5a,b
    in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
    for computing the sample variance: Analysis and recommendations."
    The American Statistician 37.3 (1983): 242-247:

    Args:
        means: list of mean values with shape [n_scalars, m_features]
        variances: list of variance values with shape [n_scalars, m_features]
        sample_counts: list of sample counts that where used to compute the mean and variance with shape [n_scalars]
    Returns:
        numpy arrays for mean, std, var, sample_count with len m_features

    """
    if len(means) == len(variances) == len(sample_counts) == 1:
        logging.debug('Only one value provided to combine_standard_scalars.')
        return np.array(means[0]), np.array(np.sqrt(variances[0])), np.array(variances[0]), np.array(sample_counts[0])

    assert len(means) == len(variances) == len(sample_counts), \
        f"Found different lengths for means ({len(means)}), variances({len(variances)}), and sample_counts({len(sample_counts)})"

    # convert to np arrays
    means = np.array(means)
    variances = np.array(variances)
    sample_counts = np.array(sample_counts)

    # get sums
    sums = means * sample_counts
    var_sums = variances * sample_counts

    # init first values
    last_sum = sums[0]
    last_var = var_sums[0]
    m = sample_counts[0]

    # iterate over all provided standard scalars
    for new_sum, new_var, n in zip(sums[1:], var_sums[1:], sample_counts[1:]):
        # combine previous values with new values (see Equation 1.5b,a in Chan et al.)
        new_var = last_var + new_var + m / (n * (m + n)) * ((n / m) * last_sum - new_sum) ** 2
        new_sum = last_sum + new_sum

        # update values
        last_sum = new_sum
        last_var = new_var
        m = m + n

    # get combined values
    mean = last_sum / m
    var = last_var / m
    std = np.sqrt(var)
    sample_count = m

    return mean, std, var, sample_count


def main(file_path_pattern, data_var, target_path, ignore_no_data_value, batch_size):
    # get file paths from a glob pattern
    file_paths = glob.glob(file_path_pattern)

    logging.info(f'Start processing {len(file_paths)} files')

    # init scalar
    mean_list = []
    variance_list = []
    sample_count_list = []

    for n, file_path in enumerate(file_paths):
        # open a zarr file from COS as xarray dataset
        ds = xr.open_zarr(file_path, mask_and_scale=False)

        # get no data value
        if ignore_no_data_value:
            no_data_value = ds.attrs['no_data']

        # iterate over batches to reduce memory usage
        n_samples = len(ds[data_var])
        logging.info(f"Processing file {file_path.split('/')[-1]} with {n_samples} samples")
        for i in tqdm.tqdm(range(0, n_samples, batch_size)):
            # Get data with shape [band, values]
            values = ds[data_var][i:i+batch_size].values.swapaxes(0, 1)
            values = values.astype(float).reshape(values.shape[0], -1)
            # replace no_data value with nan
            if ignore_no_data_value:
                values[values == no_data_value] = np.nan

            # add values to list
            mean_list.append(np.nanmean(values, axis=1))
            variance_list.append(np.nanvar(values, axis=1))
            sample_count_list.append(np.count_nonzero(~np.isnan(values), axis=1))

    # Combine statistics
    mean, std, var, sample_count = combine_standard_scalars(mean_list, variance_list, sample_count_list)

    statistics = {
        'mean': list(mean.astype(float)),
        'std': list(std.astype(float)),
        'var': list(var.astype(float)),
        'n_samples': list(sample_count.astype(float)),
    }

    # write final results
    logging.debug("Statistics: " + str(statistics))
    if not target_path.endswith('.json'):
        target_path = os.path.join(target_path, 'statistics.json')
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, 'w') as f:
        json.dump(statistics, f)

    logging.info('Finished processing. Statistics are saved in ' + target_path)


# Sub-process for grid wrapper
def grid_process(batch_id, file_path_pattern, data_var, target_path, ignore_no_data_value, batch_size, *args, **kwargs):
    file_path = file_path_pattern.replace('*', batch_id)
    assert os.path.exists(file_path), \
        (f"File path {file_path} for batch id {batch_id} does not exist. "
         f"Expects file_path_pattern to be a wildcard with * marking the tile name, "
         "e.g., fm-geospatial/lchu/geofm_hls_zarr/global_v8/tile_*.zarr.")

    os.makedirs(target_path, exist_ok=True)
    target_path = os.path.join(target_path, f'{batch_id}_statistics.json')

    main(file_path, data_var, target_path, ignore_no_data_value, batch_size)

    return [target_path]


if __name__ == '__main__':
    main(file_path_pattern, data_var, target_path, ignore_no_data_value, batch_size)

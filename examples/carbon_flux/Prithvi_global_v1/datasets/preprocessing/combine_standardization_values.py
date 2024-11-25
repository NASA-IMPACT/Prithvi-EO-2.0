"""
This operator combines mean and std based on num_samples.
"""

# pip install numpy pyyaml

import os
import logging
import glob
import json
import yaml
import numpy as np

# glob pattern for all statistics files to combine (e.g. path/to/files/*.json)
statistics_file_pattern = os.getenv('statistics_file_pattern')
# json target file path  for saving the statistics
target_path = os.getenv('target_path')


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


def main(statistics_file_pattern, target_path):

    statistics_files = glob.glob(statistics_file_pattern)
    assert len(statistics_files) > 0, f'Found no files with statistics_file_pattern {statistics_file_pattern}'
    logging.info(f"Combine statistics from {len(statistics_files)} files, e.g., {statistics_files[0]}")

    # Load statistics
    mean_list = []
    variance_list = []
    sample_count_list = []
    for file in statistics_files:
        with open(file, 'r') as f:
            statistics = json.load(f)

        mean_list.append(statistics['mean'])
        variance_list.append(statistics['var'])
        sample_count_list.append(statistics['n_samples'])

    # Combine statistics
    mean, std, var, sample_count = combine_standard_scalars(mean_list, variance_list, sample_count_list)

    statistics = {
        'mean': list(mean.astype(float)),
        'std': list(std.astype(float)),
        'var': list(var.astype(float)),
        'n_samples': list(sample_count.astype(float)),
    }

    statistics_rounded = {k: list(map(round, v)) for k, v in statistics.items()}
    s = yaml.dump_all([statistics_rounded])
    logging.info("Statistics:\n" + s)

    # write final results
    if not target_path.endswith('.json'):
        target_path = os.path.join(target_path, 'statistics.json')
    with open(target_path, 'w') as f:
        json.dump(statistics, f)

    logging.info(f"Saved statistics in {target_path}")


if __name__ == '__main__':
    main(statistics_file_pattern, target_path)

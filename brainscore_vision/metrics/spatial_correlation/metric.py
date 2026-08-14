from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from brainscore_core import Metric, Score
from brainscore_core.supported_data_standards import NeuroidAssembly, merge_data_arrays
from scipy.spatial.distance import squareform, pdist
from xarray import DataArray

import brainscore_vision
from brainscore_vision.benchmark_helpers.neural_common import average_repetition


def inv_ks_similarity(p, q):
    """
    Inverted ks similarity -> resulting in a score within [0,1], 1 being a perfect match
    """
    import scipy.stats
    return 1 - scipy.stats.ks_2samp(p, q)[0]


class SpatialCorrelationSimilarity(Metric):
    """
    Computes the similarity of two given distributions using a given similarity_function.
    """

    def __init__(self, similarity_function,
                 bin_size_mm: float, num_bootstrap_samples: int, num_sample_arrays: int):
        """
        :param similarity_function: similarity_function to be applied to each bin
            which in turn are created based on a given bin size and the independent variable of the distributions.
            E.g. `inv_ks_similarity`
        :param bin_size_mm: size per bin in mm
        :param num_bootstrap_samples: how many electrode pairs to sample from the data
        :param num_sample_arrays: number of simulated Utah arrays sampled from candidate model tissue
        """
        self.similarity_function = similarity_function
        self.bin_size = bin_size_mm
        self.num_bootstrap_samples = num_bootstrap_samples
        self.num_sample_arrs = num_sample_arrays

    def __call__(self, candidate_assembly: NeuroidAssembly, target_assembly: NeuroidAssembly) -> Score:
        """
        :param candidate_assembly: neural recordings from candidate model
        :param target_assembly: neural recordings from target system.
            Expected to include repetitions to compute electrode ceilings
        :return: a Score representing how similar the two assemblies are with respect to their spatial
            response-correlation.
        """
        # characterize response-correlation for each assembly
        array_size_mm = (np.ptp(target_assembly['tissue_x'].values),
                         np.ptp(target_assembly['tissue_y'].values))
        candidate_statistic = self.sample_global_tissue_statistic(candidate_assembly, array_size_mm=array_size_mm)
        target_statistic = self.compute_global_tissue_statistic_target(target_assembly)
        return self.compare_statistics(candidate_statistic, target_statistic)

    def compare_statistics(self, candidate_statistic: DataArray, target_statistic: DataArray) -> Score:
        # score all bins
        self._bin_min = np.min(target_statistic.distances)
        self._bin_max = np.max(target_statistic.distances)
        bin_scores = []
        for bin_number, (target_mask, candidate_mask) in enumerate(
                self._bin_masks(candidate_statistic, target_statistic)):
            enough_data = target_mask.size > 0 and candidate_mask.size > 0  # both non-zero
            if not enough_data:  # ignore bins with insufficient number of data
                continue
            similarity = self.similarity_function(target_statistic.values[target_mask],
                                                  candidate_statistic.values[candidate_mask])
            similarity = Score([similarity], coords={'bin': [bin_number]}, dims=['bin'])
            bin_scores.append(similarity)
        # aggregate
        bin_scores = merge_data_arrays(bin_scores)
        score = self._aggregate_scores(bin_scores)
        score.attrs['candidate_statistic'] = candidate_statistic
        score.attrs['target_statistic'] = target_statistic
        return score

    def compute_global_tissue_statistic_target(self, assembly: NeuroidAssembly) -> DataArray:
        """
        :return: DataArray with values = correlations; coordinates: distances, source, array
        """
        consistency = brainscore_vision.load_ceiling('internal_consistency')
        neuroid_reliability = consistency(assembly.transpose('presentation', 'neuroid'))

        averaged_assembly = average_repetition(assembly)
        target_statistic_list = []
        for animal in sorted(list(set(averaged_assembly.neuroid.animal.data))):
            for electrode_array in sorted(list(set(averaged_assembly.neuroid.arr.data))):
                sub_assembly = averaged_assembly.sel(animal=animal, arr=electrode_array)
                bootstrap_samples_sub_assembly = int(
                    self.num_bootstrap_samples * (sub_assembly.neuroid.size / averaged_assembly.neuroid.size))

                distances, correlations = self.sample_response_corr_vs_dist(sub_assembly,
                                                                            bootstrap_samples_sub_assembly,
                                                                            neuroid_reliability)
                sub_assembly_statistic = self.to_xarray(correlations, distances, source=animal,
                                                        electrode_array=electrode_array)
                target_statistic_list.append(sub_assembly_statistic)

        target_statistic = xr.concat(target_statistic_list, dim='meta')
        return target_statistic

    def sample_global_tissue_statistic(
            self, candidate_assembly, array_size_mm: Tuple[np.ndarray, np.ndarray]) -> DataArray:
        """
        Simulates placement of multiple arrays in tissue and computes repsonse correlation as a function of distance on
        each of them
        :param array_size_mm: physical size of Utah array in mm
        :param candidate_assembly: NeuroidAssembly
        :return: DataArray with values = correlations; coordinates: distances, source, array
        """
        candidate_statistic_list = []
        bootstrap_samples_per_array = int(self.num_bootstrap_samples / self.num_sample_arrs)
        array_locations = self.sample_array_locations(candidate_assembly.neuroid, array_size_mm=array_size_mm)
        for i, window in enumerate(array_locations):
            distances, correlations = self.sample_response_corr_vs_dist(candidate_assembly[window],
                                                                        bootstrap_samples_per_array)

            array_statistic = self.to_xarray(correlations, distances, electrode_array=str(i))
            candidate_statistic_list.append(array_statistic)

        candidate_statistic = xr.concat(candidate_statistic_list, dim='meta')
        return candidate_statistic

    def sample_array_locations(self, neuroid, array_size_mm: Tuple[np.ndarray, np.ndarray], seed=0):
        """
        Generator: Sample Utah array-like portions from artificial model tissue and generate masks
        :param neuroid: NeuroidAssembly.neuroid, has to contain tissue_x, tissue_y coords
        :param array_size_mm: physical size of Utah array in mm
        :param seed: random seed
        :return: list of masks in neuroid dimension of assembly, usage: assembly[mask] -> neuroids within one array
        """
        bound_max_x, bound_max_y = np.max([neuroid.tissue_x.data, neuroid.tissue_y.data], axis=1) - array_size_mm
        rng = np.random.default_rng(seed=seed)

        lower_corner = np.column_stack((rng.choice(neuroid.tissue_x.data[neuroid.tissue_x.data <= bound_max_x],
                                                   size=self.num_sample_arrs),
                                        rng.choice(neuroid.tissue_y.data[neuroid.tissue_y.data <= bound_max_y],
                                                   size=self.num_sample_arrs)))
        upper_corner = lower_corner + array_size_mm

        # create index masks of neuroids within sample windows
        for i in range(self.num_sample_arrs):
            yield np.logical_and.reduce([neuroid.tissue_x.data <= upper_corner[i, 0],
                                         neuroid.tissue_x.data >= lower_corner[i, 0],
                                         neuroid.tissue_y.data <= upper_corner[i, 1],
                                         neuroid.tissue_y.data >= lower_corner[i, 1]])

    def sample_response_corr_vs_dist(self, assembly, num_samples, neuroid_reliability=None, seed=0):
        """
        1. Samples random pairs from the assembly
        2. Computes distances for all pairs
        3. Computes the response correlation between items of each pair
        (4. Ceils the response correlations by ceiling each neuroid | if neuroid_reliability not None)
        :param assembly: NeuroidAssembly without stimulus repetitions
        :param num_samples: how many random pair you want to be sampled out of the data
        :param neuroid_reliability: if not None: expecting Score object containing reliability estimates of all neuroids
        :param seed: random seed
        :return: [distance, pairwise_correlation_of_neuroids], pairwise correlations can be ceiled
        """
        rng = np.random.default_rng(seed=seed)
        neuroid_pairs = rng.integers(0, assembly.shape[0], (2, num_samples))

        pairwise_distances_all = self.pairwise_distances(assembly)
        pairwise_distance_samples = pairwise_distances_all[(*neuroid_pairs,)]

        response_samples = assembly.data[neuroid_pairs]
        response_correlation_samples = self.corrcoef_rowwise(*response_samples)

        if neuroid_reliability is not None:
            pairwise_neuroid_reliability_all = self.create_pairwise_neuroid_reliability_mat(neuroid_reliability)
            pairwise_neuroid_reliability_samples = pairwise_neuroid_reliability_all[(*neuroid_pairs,)]

            response_correlation_samples = response_correlation_samples / pairwise_neuroid_reliability_samples

        # properly removing nan values
        pairwise_distance_samples = pairwise_distance_samples[~np.isnan(response_correlation_samples)]
        response_correlation_samples = response_correlation_samples[~np.isnan(response_correlation_samples)]

        return np.vstack((pairwise_distance_samples, response_correlation_samples))

    def corrcoef_rowwise(self, a, b):
        # https://stackoverflow.com/questions/41700840/correlation-of-2-time-dependent-multidimensional-signals-signal-vectors
        a_ma = a - a.mean(1)[:, None]
        b_mb = b - b.mean(1)[:, None]
        ssa = np.einsum('ij,ij->i', a_ma, a_ma)  # var A
        ssb = np.einsum('ij,ij->i', b_mb, b_mb)  # var B
        return np.einsum('ij,ij->i', a_ma, b_mb) / np.sqrt(ssa * ssb)  # cov/sqrt(varA*varB)

    def pairwise_distances(self, assembly):
        """
        Convenience function creating a simple lookup table for pairwise distances
        :param assembly: NeuroidAssembly
        :return: square matrix where each entry is the distance between the neuroids at the corresponding indices
        """
        locations = np.stack([assembly.neuroid.tissue_x.data, assembly.neuroid.tissue_y.data]).T

        return squareform(pdist(locations, metric='euclidean'))

    def create_pairwise_neuroid_reliability_mat(self, neuroid_reliability):
        """
        Convenience function creating a simple lookup table for combined reliabilities of neuroid pairs
        :param neuroid_reliability: expects Score object where neuroid_reliability.raw holds [cross validation subset,
            reliability per neuroid]
        :return: square matrix where each entry_ij = sqrt(reliability_i * reliability_j)
        """
        reliability_per_neuroid = np.mean(neuroid_reliability.raw.data, axis=0)
        c_mat = np.zeros((reliability_per_neuroid.size, reliability_per_neuroid.size))
        for i, ci in enumerate(reliability_per_neuroid):
            for j, cj in enumerate(reliability_per_neuroid):
                c_mat[i, j] = np.sqrt(ci * cj)

        return c_mat

    def to_xarray(self, correlations, distances, source='model', electrode_array=None):
        """
        :param correlations: list of data values
        :param distances: list of distance values, each distance value has to correspond to one data value
        :param source: name of monkey
        :param electrode_array: name of recording array
        """
        xarray_statistic = DataArray(
            data=correlations,
            dims=["meta"],
            coords={
                'meta': pd.MultiIndex.from_product([distances, [source], [electrode_array]],
                                                   names=('distances', 'source', 'array'))
            }
        )

        return xarray_statistic

    def _bin_masks(self, candidate_statistic: DataArray, target_statistic: DataArray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generator: Yields masks indexing which elements are within each bin.

        :yield: a triplet where the two elements are masks for a bin over the target and candidate respectively
        """
        bin_step = int(self._bin_max * (1 / self.bin_size) + 1) * 2
        for lower_bound_mm in np.linspace(self._bin_min, self._bin_max, bin_step):
            target_mask = np.where(np.logical_and(target_statistic.distances >= lower_bound_mm,
                                                  target_statistic.distances < lower_bound_mm + self.bin_size))[0]
            candidate_mask = np.where(np.logical_and(candidate_statistic.distances >= lower_bound_mm,
                                                     candidate_statistic.distances < lower_bound_mm + self.bin_size))[0]
            yield target_mask, candidate_mask

    def _aggregate_scores(self, scores: Score, over: str = 'bin') -> Score:
        """
        Aggregates scores into an aggregate Score where `center = mean(scores)` and `error = mad(scores)`
        :param scores: scores over bins
        """
        score = scores.median(dim=over)
        error = abs((scores - scores.median(dim=over))).median(dim=over)  # mean absolute deviation
        score.attrs['error'] = error
        score.attrs[Score.RAW_VALUES_KEY] = scores
        return score

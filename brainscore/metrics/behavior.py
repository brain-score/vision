import logging
from collections import Counter

import itertools
import numpy as np
import scipy.stats
from numpy.random.mtrand import RandomState
from scipy.stats import pearsonr

from brainio_base.assemblies import walk_coords, DataAssembly
from brainscore.metrics import Metric, Score
from brainscore.metrics.transformations import apply_aggregate
from brainscore.utils import fullname


class I2n(Metric):
    """
    Rajalingham & Issa et al., 2018 http://www.jneurosci.org/content/early/2018/07/13/JNEUROSCI.0388-18.2018
    modified by Schrimpf & Kubilius et al., 2018 https://www.biorxiv.org/content/early/2018/09/05/407007:
        - Rajalingham et al. generated model trials by using different train-test splits.
            This implementation fixes the train-test split, and thus computes a fixed response matrix without trials.
        - for computing dprime scores, Rajalingham et al. computed the false-alarms rate across the flat vector of all
            distractor images. This implementation computes the false-alarms rate per object, and then takes the mean.
    """

    def __init__(self, repetitions=2):
        super().__init__()
        self.repetitions = repetitions
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, source_probabilities, target):
        return self._repeat(lambda random_state:
                            self._call_single(source_probabilities, target, random_state=random_state))

    def _call_single(self, source_probabilities, target, random_state):
        self.add_source_meta(source_probabilities, target)
        source_response_matrix = self.target_distractor_scores(source_probabilities)
        source_response_matrix = self.normalized_dprimes(source_response_matrix)

        target_half = self.generate_halves(target, random_state=random_state)[0]
        target_response_matrix = self.build_response_matrix_from_responses(target_half)
        target_response_matrix = self.normalized_dprimes(target_response_matrix)
        correlation = self.correlate(source_response_matrix, target_response_matrix)
        return correlation

    def ceiling(self, assembly, skipna=False):
        return self._repeat(lambda random_state:
                            self.compute_ceiling(assembly, random_state=random_state, skipna=skipna))

    def compute_ceiling(self, assembly, random_state, skipna=False):
        dprime_halves = []
        for half in self.generate_halves(assembly, random_state=random_state):
            half = self.build_response_matrix_from_responses(half)
            half = self.normalized_dprimes(half)
            dprime_halves.append(half)
        return self.correlate(*dprime_halves, skipna=skipna)

    def build_response_matrix_from_responses(self, responses):
        num_choices = [(image_id, choice) for image_id, choice in zip(responses['image_id'].values, responses.values)]
        num_choices = Counter(num_choices)
        num_objects = [[(image_id, sample_obj), (image_id, dist_obj)] for image_id, sample_obj, dist_obj in zip(
            responses['image_id'].values, responses['sample_obj'].values, responses['dist_obj'].values)]
        num_objects = Counter(itertools.chain(*num_objects))

        choices = np.unique(responses)
        image_ids, indices = np.unique(responses['image_id'], return_index=True)
        truths = responses['truth'].values[indices]
        image_dim = responses['image_id'].dims
        coords = {**{coord: (dims, value) for coord, dims, value in walk_coords(responses)},
                  **{'choice': ('choice', choices)}}
        coords = {coord: (dims, value if dims != image_dim else value[indices])  # align image_dim coords with indices
                  for coord, (dims, value) in coords.items()}
        response_matrix = np.zeros((len(image_ids), len(choices)))
        for (image_index, image_id), (choice_index, choice) in itertools.product(
                enumerate(image_ids), enumerate(choices)):
            if truths[image_index] == choice:  # object == choice, ignore
                p = np.nan
            else:
                # divide by number of times where object was one of the two choices (target or distractor)
                p = (num_choices[(image_id, choice)] / num_objects[(image_id, choice)]) \
                    if num_objects[(image_id, choice)] > 0 else np.nan
            response_matrix[image_index, choice_index] = p
        response_matrix = DataAssembly(response_matrix, coords=coords, dims=responses.dims + ('choice',))
        return response_matrix

    def normalized_dprimes(self, response_matrix, cap=5):
        dprime_scores = self.dprime(response_matrix)
        dprime_scores_clipped = dprime_scores.clip(-cap, cap)
        dprime_scores_normalized = self.subtract_mean(dprime_scores_clipped)
        return dprime_scores_normalized

    def target_distractor_scores(self, object_probabilities):
        cached_object_probabilities = self._build_index(object_probabilities, ['image_id', 'choice'])

        def apply(p_choice, image_id, truth, choice, **_):
            if truth == choice:  # object == choice, ignore
                return np.nan
            # probability that something else was chosen rather than object (p_choice == p_distractor after above check)
            p_object = cached_object_probabilities[(image_id, truth)]
            p = p_choice / (p_choice + p_object)
            return p

        result = object_probabilities.multi_dim_apply(['image_id', 'choice'], apply)
        return result

    def dprime(self, response_matrix):
        truth_choice_values = self._build_index(response_matrix, ['truth', 'choice'])

        def apply(false_alarms_rate_images, choice, truth, **_):
            hit_rate = 1 - false_alarms_rate_images
            inverse_choice = truth_choice_values[(choice, truth)]
            false_alarms_rate_objects = np.nanmean(inverse_choice)
            dprime = self.z_score(hit_rate) - self.z_score(false_alarms_rate_objects)
            return dprime

        result = response_matrix.multi_dim_apply(['image_id', 'choice'], apply)
        return result

    def z_score(self, value):
        return scipy.stats.norm.ppf(value)

    def subtract_mean(self, scores):
        result = scores.multi_dim_apply(['truth', 'choice'], lambda group, **_: group - np.nanmean(group))
        return result

    @classmethod
    def correlate(cls, source_response_matrix, target_response_matrix, skipna=False):
        # align
        source_response_matrix = source_response_matrix.sortby('image_id').sortby('choice')
        target_response_matrix = target_response_matrix.sortby('image_id').sortby('choice')
        assert all(source_response_matrix['image_id'].values == target_response_matrix['image_id'].values)
        assert all(source_response_matrix['choice'].values == target_response_matrix['choice'].values)
        # flatten and mask out NaNs
        source, target = source_response_matrix.values.flatten(), target_response_matrix.values.flatten()
        non_nan = ~np.isnan(target)
        non_nan = np.logical_and(non_nan, (~np.isnan(source) if skipna else 1))
        source, target = source[non_nan], target[non_nan]
        assert not any(np.isnan(source))
        correlation, p = pearsonr(source, target)
        return correlation

    @classmethod
    def aggregate(cls, scores):
        center = scores.mean('split')
        error = scores.std('split')
        return Score([center, error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])

    def generate_halves(self, assembly, random_state):
        indices = list(range(len(assembly)))
        random_state.shuffle(indices)
        halves = assembly[indices[:int(len(indices) / 2)]], assembly[indices[int(len(indices) / 2):]]
        return halves

    def _build_index(self, assembly, coords):
        np.testing.assert_array_equal(list(itertools.chain(*[assembly[coord].dims for coord in coords])), assembly.dims)
        aligned_coords = itertools.product(*[assembly[coord].values for coord in coords])
        result = {}
        for coord_values, value in zip(aligned_coords, assembly.values.flatten()):
            if coord_values in result:  # if there are duplicates, make it a list
                previous_value = result[coord_values]
                previous_value = (previous_value if isinstance(previous_value, list) else [previous_value])
                result[coord_values] = previous_value + [value]
            else:
                result[coord_values] = value
        return result

    def _repeat(self, func):
        random_state = self._initialize_random_state()
        repetitions = list(range(self.repetitions))
        scores = [func(random_state=random_state) for repetition in repetitions]
        score = Score(scores, coords={'split': repetitions}, dims=['split'])
        return apply_aggregate(self.aggregate, score)

    def _initialize_random_state(self):
        return RandomState(seed=0)  # fix the seed here so that the same halves are produced for score and ceiling

    def add_source_meta(self, source_probabilities, target):
        image_meta = {image_id: meta_value for image_id, meta_value in
                      zip(target['image_id'].values, target['truth'].values)}
        meta_values = [image_meta[image_id] for image_id in source_probabilities['image_id'].values]
        source_probabilities['truth'] = 'presentation', meta_values

import logging
from collections import Counter

import itertools
import numpy as np
import scipy.stats
import xarray
from numpy.random.mtrand import RandomState
from scipy.stats import pearsonr

from brainio.assemblies import walk_coords, DataAssembly
from brainscore.metrics import Metric, Score
from brainscore.metrics.transformations import apply_aggregate
from brainscore.utils import fullname


def I1(*args, **kwargs):
    return _I(*args, collapse_distractors=True, normalize=False, **kwargs)


def I1n(*args, **kwargs):
    return _I(*args, collapse_distractors=True, normalize=True, **kwargs)


def I2(*args, **kwargs):
    return _I(*args, collapse_distractors=False, normalize=False, **kwargs)


def I2n(*args, **kwargs):
    return _I(*args, collapse_distractors=False, normalize=True, **kwargs)


def O1(*args, **kwargs):
    return _O(*args, collapse_distractors=True, normalize=False, **kwargs)


def O2(*args, **kwargs):
    return _O(*args, collapse_distractors=False, normalize=False, **kwargs)


# From Martin's fork
def _o2(assembly):
    i2n = I2n()
    false_alarm_rates = i2n.target_distractor_scores(assembly)
    false_alarm_rates_object = false_alarm_rates.groupby('truth').mean('presentation')
    false_alarm_rates_object = false_alarm_rates_object.rename({'truth': 'task_left', 'choice': 'task_right'})
    # hit rates are one minus the flipped false alarm rates, e.g. HR(dog, cat) = 1 - FAR(cat, dog)
    hit_rates_object = 1 - false_alarm_rates_object.rename({'task_left': 'task_right', 'task_right': 'task_left'})
    dprime_false_alarms_rates_object = xarray.apply_ufunc(i2n.z_score, false_alarm_rates_object)
    dprime_hit_rates_object = xarray.apply_ufunc(i2n.z_score, hit_rates_object)
    o2 = dprime_hit_rates_object - dprime_false_alarms_rates_object
    return o2


class _Behavior_Metric(Metric):
    """
    Rajalingham & Issa et al., 2018 http://www.jneurosci.org/content/early/2018/07/13/JNEUROSCI.0388-18.2018
    modified by Schrimpf & Kubilius et al., 2018 https://www.biorxiv.org/content/early/2018/09/05/407007:
        - Rajalingham et al. generated model trials by using different train-test splits.
            This implementation fixes the train-test split, and thus computes a fixed response matrix without trials.
        - for computing dprime scores, Rajalingham et al. computed the false-alarms rate across the flat vector of all
            distractor images. This implementation computes the false-alarms rate per object, and then takes the mean.
    """
    def __init__(self, collapse_distractors, normalize=False, repetitions=2):
        super().__init__()
        self._collapse_distractors = collapse_distractors
        self._normalize = normalize
        self._repetitions = repetitions
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, source_probabilities, target):
        return self._repeat(lambda random_state:
                            self._call_single(source_probabilities, target, random_state=random_state))

    def _call_single(self, source_probabilities, target, random_state):
        self.add_source_meta(source_probabilities, target)
        source_response_matrix = self.target_distractor_scores(source_probabilities)
        source_response_matrix = self.dprimes(source_response_matrix)
        if self._collapse_distractors:
            source_response_matrix = self.collapse_distractors(source_response_matrix)

        target_half = self.generate_halves(target, random_state=random_state)[0]
        target_response_matrix = self.build_response_matrix_from_responses(target_half)
        target_response_matrix = self.dprimes(target_response_matrix)
        if self._collapse_distractors:
            target_response_matrix = self.collapse_distractors(target_response_matrix)

        correlation = self.correlate(source_response_matrix, target_response_matrix,
                                     collapse_distractors=self._collapse_distractors)
        return correlation

    def ceiling(self, assembly, skipna=False):
        return self._repeat(lambda random_state:
                            self.compute_ceiling(assembly, random_state=random_state, skipna=skipna))

    def compute_ceiling(self, assembly, random_state, skipna=False):
        dprime_halves = []
        for half in self.generate_halves(assembly, random_state=random_state):
            half = self.build_response_matrix_from_responses(half)
            half = self.dprimes(half)
            if self._collapse_distractors:
                half = self.collapse_distractors(half)
            dprime_halves.append(half)
        return self.correlate(*dprime_halves, skipna=skipna, collapse_distractors=self._collapse_distractors)

    def build_response_matrix_from_responses(self, responses):
        num_choices = [(image_id, choice) for image_id, choice in zip(responses['stimulus_id'].values, responses.values)]
        num_choices = Counter(num_choices)
        num_objects = [[(image_id, sample_obj), (image_id, dist_obj)] for image_id, sample_obj, dist_obj in zip(
            responses['stimulus_id'].values, responses['sample_obj'].values, responses['dist_obj'].values)]
        num_objects = Counter(itertools.chain(*num_objects))

        choices = np.unique(responses)
        image_ids, indices = np.unique(responses['stimulus_id'], return_index=True)
        truths = responses['truth'].values[indices]
        image_dim = responses['stimulus_id'].dims
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

    def dprimes(self, response_matrix, cap=5):
        dprime_scores = self.dprime(response_matrix)
        dprime_scores_clipped = dprime_scores.clip(-cap, cap)
        dprime_scores_clipped = type(dprime_scores)(dprime_scores_clipped)  # make sure type is preserved
        if not self._normalize:
            return dprime_scores_clipped
        else:
            dprime_scores_normalized = self.subtract_mean(dprime_scores_clipped)
            return dprime_scores_normalized

    def target_distractor_scores(self, object_probabilities):
        cached_object_probabilities = self._build_index(object_probabilities, ['stimulus_id', 'choice'])

        def apply(p_choice, stimulus_id, truth, choice, **_):
            if truth == choice:  # object == choice, ignore
                return np.nan
            # probability that something else was chosen rather than object (p_choice == p_distractor after above check)
            p_object = cached_object_probabilities[(stimulus_id, truth)]
            p = p_choice / (p_choice + p_object)
            return p

        result = object_probabilities.multi_dim_apply(['stimulus_id', 'choice'], apply)
        return result

    def z_score(self, value):
        return scipy.stats.norm.ppf(value)

    def subtract_mean(self, scores):
        result = scores.multi_dim_apply(['truth', 'choice'], lambda group, **_: group - np.nanmean(group))
        return result

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
        repetitions = list(range(self._repetitions))
        scores = [func(random_state=random_state) for repetition in repetitions]
        score = Score(scores, coords={'split': repetitions}, dims=['split'])
        return apply_aggregate(self.aggregate, score)

    def _initialize_random_state(self):
        return RandomState(seed=0)  # fix the seed here so that the same halves are produced for score and ceiling

    def add_source_meta(self, source_probabilities, target):
        image_meta = {image_id: meta_value for image_id, meta_value in
                      zip(target['stimulus_id'].values, target['truth'].values)}
        meta_values = [image_meta[image_id] for image_id in source_probabilities['stimulus_id'].values]
        source_probabilities['truth'] = 'presentation', meta_values


class _I(_Behavior_Metric):
    """
    Rajalingham & Issa et al., 2018 http://www.jneurosci.org/content/early/2018/07/13/JNEUROSCI.0388-18.2018
    modified by Schrimpf & Kubilius et al., 2018 https://www.biorxiv.org/content/early/2018/09/05/407007:
        - Rajalingham et al. generated model trials by using different train-test splits.
            This implementation fixes the train-test split, and thus computes a fixed response matrix without trials.
        - for computing dprime scores, Rajalingham et al. computed the false-alarms rate across the flat vector of all
            distractor images. This implementation computes the false-alarms rate per object, and then takes the mean.
    """

    def collapse_distractors(self, response_matrix):
        return response_matrix.mean(dim='choice', skipna=True)

    def dprime(self, response_matrix):
        truth_choice_values = self._build_index(response_matrix, ['truth', 'choice'])

        def apply(false_alarms_rate_images, choice, truth, **_):
            hit_rate = 1 - false_alarms_rate_images
            inverse_choice = truth_choice_values[(choice, truth)]
            false_alarms_rate_objects = np.nanmean(inverse_choice)
            dprime = self.z_score(hit_rate) - self.z_score(false_alarms_rate_objects)
            return dprime

        result = response_matrix.multi_dim_apply(['stimulus_id', 'choice'], apply)
        return result

    @classmethod
    def correlate(cls, source_response_matrix, target_response_matrix, skipna=False, collapse_distractors=False):
        # align
        if collapse_distractors:
            source_response_matrix = source_response_matrix.sortby('stimulus_id')
            target_response_matrix = target_response_matrix.sortby('stimulus_id')
        else:
            source_response_matrix = source_response_matrix.sortby('stimulus_id').sortby('choice')
            target_response_matrix = target_response_matrix.sortby('stimulus_id').sortby('choice')
            assert all(source_response_matrix['choice'].values == target_response_matrix['choice'].values)
        assert all(source_response_matrix['stimulus_id'].values == target_response_matrix['stimulus_id'].values)
        # flatten and mask out NaNs
        source, target = source_response_matrix.values.flatten(), target_response_matrix.values.flatten()
        non_nan = ~np.isnan(target)
        non_nan = np.logical_and(non_nan, (~np.isnan(source) if skipna else 1))
        source, target = source[non_nan], target[non_nan]
        assert not any(np.isnan(source))
        correlation, p = pearsonr(source, target)
        return correlation


class _O(_Behavior_Metric):
    """
    Rajalingham & Issa et al., 2018 http://www.jneurosci.org/content/early/2018/07/13/JNEUROSCI.0388-18.2018
    modified by Schrimpf & Kubilius et al., 2018 https://www.biorxiv.org/content/early/2018/09/05/407007:
        - Rajalingham et al. generated model trials by using different train-test splits.
            This implementation fixes the train-test split, and thus computes a fixed response matrix without trials.
        - for computing dprime scores, Rajalingham et al. computed the false-alarms rate across the flat vector of all
            distractor images. This implementation computes the false-alarms rate per object, and then takes the mean.
    """

    def collapse_distractors(self, response_matrix):
        return response_matrix.mean(dim='task_left', skipna=True)

    def dprime(self, response_matrix):
        false_alarm_rates_object = response_matrix.groupby('truth').mean('presentation')
        false_alarm_rates_object = false_alarm_rates_object.rename({'truth': 'task_left', 'choice': 'task_right'})
        hit_rates_object = 1 - false_alarm_rates_object.rename({'task_left': 'task_right', 'task_right': 'task_left'})

        dprime_false_alarms_rates_object = xarray.apply_ufunc(self.z_score, false_alarm_rates_object)
        dprime_hit_rates_object = xarray.apply_ufunc(self.z_score, hit_rates_object)

        result = dprime_hit_rates_object - dprime_false_alarms_rates_object
        return result

    @classmethod
    def correlate(cls, source_response_matrix, target_response_matrix, skipna=False, collapse_distractors=False):
        # align
        if collapse_distractors:
            source_response_matrix = source_response_matrix.sortby('task_right')
            target_response_matrix = target_response_matrix.sortby('task_right')
        else:
            source_response_matrix = source_response_matrix.sortby('task_right').sortby('task_left')
            target_response_matrix = target_response_matrix.sortby('task_right').sortby('task_left')
            assert all(source_response_matrix['task_left'].values == target_response_matrix['task_left'].values)
        assert all(source_response_matrix['task_right'].values == target_response_matrix['task_right'].values)
        # flatten and mask out NaNs
        source, target = source_response_matrix.values.flatten(), target_response_matrix.values.flatten()
        non_nan = ~np.isnan(target)
        non_nan = np.logical_and(non_nan, (~np.isnan(source) if skipna else 1))
        source, target = source[non_nan], target[non_nan]
        assert not any(np.isnan(source))
        correlation, p = pearsonr(source, target)
        return correlation

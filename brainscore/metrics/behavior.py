import itertools
import logging
from collections import OrderedDict, Counter

import numpy as np
import scipy.stats
import sklearn.linear_model
import sklearn.multioutput

from brainscore.assemblies import walk_coords, array_is_element, DataAssembly
from brainscore.metrics import Metric
from brainscore.metrics.transformations import CrossValidation
from brainscore.utils import fullname


class I2n(Metric):
    """
    Rajalingham & Issa et al., 2018 http://www.jneurosci.org/content/early/2018/07/13/JNEUROSCI.0388-18.2018
    Schrimpf & Kubilius et al., 2018 https://www.biorxiv.org/content/early/2018/09/05/407007
    """

    class MatchToSampleClassifier(object):
        def __init__(self):
            classifier_c = 1e-3
            self._classifier = sklearn.linear_model.LogisticRegression(
                multi_class='multinomial', solver='newton-cg', C=classifier_c)
            self._label_mapping = None
            self._target_class = None

        def fit(self, X, Y):
            X = self._preprocess(X)
            self._target_class = type(Y)
            Y, self._label_mapping = self.labels_to_indices(Y.values)
            self._classifier.fit(X, Y)
            return self

        def predict_proba(self, X):
            assert len(X.shape) == 2, "expected 2-dimensional input"
            proba = self._classifier.predict_proba(X)
            # we take only the 0th dimension because the 1st dimension is just the features
            X_coords = {coord: (dims, value) for coord, dims, value in walk_coords(X)
                        if array_is_element(dims, X.dims[0])}
            proba = self._target_class(proba,
                                       coords={**X_coords, **{'choice': list(self._label_mapping.values())}},
                                       dims=[X.dims[0], 'choice'])
            return proba

        def labels_to_indices(self, labels):
            label2index = OrderedDict()
            indices = []
            for label in labels:
                if label not in label2index:
                    label2index[label] = (max(label2index.values()) + 1) if len(label2index) > 0 else 0
                indices.append(label2index[label])
            index2label = OrderedDict((index, label) for label, index in label2index.items())
            return indices, index2label

        def _preprocess(self, X):
            scaler = sklearn.preprocessing.StandardScaler().fit(X)
            return scaler.transform(X)

    def __init__(self):
        super().__init__()
        self._source_classifier = self.MatchToSampleClassifier()
        self._split = CrossValidation(splits=2, train_size=0.5)
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, source, target):
        source_response_matrix = self.class_probabilities_from_features(source)
        source_response_matrix = self.target_distractor_scores(source_response_matrix)
        source_response_matrix = self.normalize_response_matrix(source_response_matrix)
        target = target.sel(use=True)
        target_response_matrix = self.build_response_matrix_from_responses(target)
        target_response_matrix = self.normalize_response_matrix(target_response_matrix)
        correlations = self._split(source_response_matrix, target_response_matrix, apply=self._split_correlation)
        return correlations

    def class_probabilities_from_features(self, source):
        source_without_behavior = source.sel(use=False)  # images where we don't use behavioral responses
        assert len(source_without_behavior) == 2400 - 240
        target = source_without_behavior['label']
        self._source_classifier.fit(source_without_behavior, target)

        source_with_behavior = source.sel(use=True)
        assert len(source_with_behavior) == 240

        prediction = self._source_classifier.predict_proba(source_with_behavior)
        truth_labels = [source_with_behavior[source_with_behavior['image_id'] == image_id]['label'].values[0]
                        for image_id in prediction['image_id'].values]
        prediction['truth'] = 'presentation', truth_labels
        assert prediction.shape == (240, 24)
        return prediction

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
                p = num_choices[(image_id, choice)] / num_objects[(image_id, choice)]
            response_matrix[image_index, choice_index] = p
        response_matrix = DataAssembly(response_matrix, coords=coords, dims=responses.dims + ('choice',))
        return response_matrix

    def normalize_response_matrix(self, response_matrix):
        assert response_matrix.shape == (240, 24)

        dprime_scores = self.dprime(response_matrix)

        cap = 5
        dprime_scores = dprime_scores.clip(-cap, cap)

        dprime_scores_normalized = self.subtract_mean(dprime_scores)
        assert dprime_scores_normalized.shape == (240, 24)
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
            assert len(inverse_choice) == 10
            false_alarms_rate_objects = np.nanmean(inverse_choice)
            dprime = self.z_score(hit_rate) - self.z_score(false_alarms_rate_objects)
            return dprime

        result = response_matrix.multi_dim_apply(['image_id', 'choice'], apply)
        return result

    def z_score(self, value):
        return scipy.stats.norm.ppf(value)

    def subtract_mean(self, scores):
        def apply(group, **_):
            assert len(group) == 10
            return group - group.mean()

        result = scores.multi_dim_apply(['truth', 'choice'], apply)
        return result

    def _split_correlation(self, source1, target1, source2, target2):
        assert len(source1['image_id']) == len(target1['image_id']) == \
               len(source2['image_id']) == len(target2['image_id'])
        correlation1 = self.correlate(source1, target1)
        correlation2 = self.correlate(source2, target2)
        return DataAssembly((correlation1 + correlation2) / 2)

    def correlate(self, source_response_matrix, target_response_matrix):
        # align
        source_response_matrix = source_response_matrix.sortby('image_id').sortby('choice')
        target_response_matrix = target_response_matrix.sortby('image_id').sortby('choice')
        assert all(source_response_matrix['image_id'].values == target_response_matrix['image_id'].values)
        assert all(source_response_matrix['choice'].values == target_response_matrix['choice'].values)
        # flatten and mask out NaNs
        source, target = source_response_matrix.values.flatten(), target_response_matrix.values.flatten()
        non_nan = ~np.isnan(target)
        source, target = source[non_nan], target[non_nan]
        assert not any(np.isnan(source))
        correlation = np.corrcoef(source, target)
        return correlation[0, 1]

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

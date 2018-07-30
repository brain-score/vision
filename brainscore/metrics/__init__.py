import logging
from collections import Counter

import numpy as np
import scipy

from brainscore.assemblies import NeuroidAssembly, array_is_element, walk_coords, DataAssembly, merge_data_arrays
from brainscore.metrics.transformations import Alignment, Alignment, CartesianProduct, CrossValidation, \
    apply_transformations
from brainscore.utils import fullname
from .utils import collect_coords, collect_dim_shapes, get_modified_coords, merge_dicts


class Metric(object):
    def __call__(self, *args):
        raise NotImplementedError()

    def aggregate(self, scores):
        return scores


class ParametricMetric(Metric):
    class Defaults:
        expected_dims = ['presentation', 'neuroid']
        stimulus_coord = 'image_id'
        neuroid_dim = 'neuroid'
        neuroid_coord = 'neuroid_id'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_neuroid_values = None
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, train_source, train_target, test_source, test_target):
        self._logger.debug("Fitting")
        self.fit(train_source, train_target)
        self._logger.debug("Predicting")
        prediction = self.predict(test_source)
        self._logger.debug("Comparing")
        similarity = self.compare_prediction(prediction, test_target)
        return similarity

    def fit(self, source, target):
        np.testing.assert_array_equal(source.dims, self.Defaults.expected_dims)
        np.testing.assert_array_equal(target.dims, self.Defaults.expected_dims)
        assert all(source[self.Defaults.stimulus_coord].values == target[self.Defaults.stimulus_coord].values)
        self._target_neuroid_values = {}
        for name, dims, values in walk_coords(target):
            if self.Defaults.neuroid_dim in dims:
                assert array_is_element(dims, self.Defaults.neuroid_dim)
                self._target_neuroid_values[name] = dims, values
        self.fit_values(source, target)

    def fit_values(self, source, target):
        raise NotImplementedError()

    def predict(self, source):
        np.testing.assert_array_equal(source.dims, self.Defaults.expected_dims)
        predicted_values = self.predict_values(source)

        # package in assembly
        coords = {name: (dims, values) for name, dims, values in walk_coords(source)
                  if self.Defaults.neuroid_dim not in dims}
        for target_coord, target_dim_value in self._target_neuroid_values.items():
            coords[target_coord] = target_dim_value  # this might overwrite values which is okay
        dims = Counter([dim for name, (dim, values) in coords.items()])
        single_dims = {dim: count == 1 for dim, count in dims.items()}
        result_dims = source.dims
        unstacked_coords = {}
        for name, (dims, values) in coords.items():
            if single_dims[dims] and len(dims) > 0:
                result_dims = [dim if dim != dims[0] else name for dim in result_dims]
                coords[name] = name, values
                unstacked_coords[name] = dims
        prediction = NeuroidAssembly(predicted_values, coords=coords, dims=result_dims)
        for name, dims in unstacked_coords.items():
            assert len(dims) == 1
            prediction = prediction.stack(**{dims[0]: (name,)})
        return prediction

    def predict_values(self, test_source):
        raise NotImplementedError()

    def compare_prediction(self, prediction, target, axis=Defaults.neuroid_coord, correlation=scipy.stats.pearsonr):
        assert all(prediction[self.Defaults.stimulus_coord].values == target[self.Defaults.stimulus_coord].values)
        assert all(prediction[axis].values == target[axis].values)
        correlations = []
        for i in target[axis].values:
            target_activations = target.sel(**{axis: i}).squeeze()
            prediction_activations = prediction.sel(**{axis: i}).squeeze()
            r, p = correlation(target_activations, prediction_activations)
            correlations.append(r)
        return DataAssembly(correlations, coords={axis: target[axis].values}, dims=[axis])


class NonparametricMetric(Metric):
    def __call__(self, source_assembly, target_assembly):
        result = self._apply(source_assembly, target_assembly)
        return DataAssembly(result)

    def _apply(self, source_assembly, target_assembly):
        raise NotImplementedError()


class NonparametricWrapper(ParametricMetric):
    """
    The standard brainscore.metrics.transformations.Transformations will pass a 4-tuple
    (train_source, train_target, test_source, test_target) to the metric
    but non-parametric metrics only operate on the test.
    This wrapper discards the train data and only passes test source and target to the underlying metric.
    """

    def __init__(self, metric, *args, **kwargs):
        super(NonparametricWrapper, self).__init__(*args, **kwargs)
        self._metric = metric

    def __call__(self, train_source, train_target, test_source, test_target):
        # no training, apply directly on test
        return self._metric(test_source, test_target)


class Score(object):
    def __init__(self, values, aggregation):
        self.values = values
        self.aggregation = aggregation

    def __repr__(self):
        return self.__class__.__name__ + "(" + ",".join(
            "{}={}".format(attr, val) for attr, val in self.__dict__.items()) + ")"

    def sel(self, *args, **kwargs):
        values = self.values.sel(*args, **kwargs)
        aggregation = self.aggregation.sel(*args, **kwargs)
        return Score(values, aggregation)

    def expand_dims(self, *args, **kwargs):
        values = self.values.expand_dims(*args, **kwargs)
        aggregation = self.aggregation.expand_dims(*args, **kwargs)
        return Score(values, aggregation)

    def __setitem__(self, key, value):
        self.values[key] = value
        self.aggregation[key] = value

    @classmethod
    def merge(cls, *scores):
        values = merge_data_arrays([score.values for score in scores])
        aggregation = merge_data_arrays([score.aggregation for score in scores])
        return Score(values, aggregation)


def build_score(values, center, error):
    # keep separate from Score class to keep constructor equal to fields (necessary for utils.py#combine_fields)
    center = center.expand_dims('aggregation')
    center['aggregation'] = ['center']
    error = error.expand_dims('aggregation')
    error['aggregation'] = ['error']
    aggregation = merge_data_arrays([center, error])
    return Score(values, aggregation)

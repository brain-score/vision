import logging
from collections import Counter

import numpy as np

from brainscore.assemblies import NeuroidAssembly, walk_coords, DataAssembly, merge_data_arrays
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
        stimulus_coord = 'image_id'

    def __init__(self, *args, expected_source_dims, expected_target_dims,
                 stimulus_coord=Defaults.stimulus_coord, **kwargs):
        super().__init__(*args, **kwargs)
        self._expected_source_dims = expected_source_dims
        self._expected_target_dims = expected_target_dims
        self._stimulus_coord = stimulus_coord
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
        np.testing.assert_array_equal(source.dims, self._expected_source_dims)
        np.testing.assert_array_equal(target.dims, self._expected_target_dims)
        assert all(source[self._stimulus_coord].values == target[self._stimulus_coord].values)
        self.fit_values(source, target)

    def fit_values(self, source, target):
        raise NotImplementedError()

    def predict(self, source):
        np.testing.assert_array_equal(source.dims, self._expected_source_dims)
        predicted_values = self.predict_values(source)
        prediction = self.package_prediction(predicted_values, source=source)
        return prediction

    def package_prediction(self, predicted_values, source):
        # build unstacked version first as to not loose single-value multi-index coords
        coords = {name: (dims, values) for name, dims, values in walk_coords(source)}
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

    def compare_prediction(self, prediction, target):
        raise NotImplementedError()


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

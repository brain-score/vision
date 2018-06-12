import logging
import math
from collections import Counter

import numpy as np
import scipy

from mkgu.assemblies import NeuroidAssembly, array_is_element, walk_coords, DataAssembly, merge_data_arrays
from mkgu.metrics.transformations import Alignment, Alignment, CartesianProduct, CrossValidation, apply_transformations
from mkgu.utils import fullname
from .utils import collect_coords, collect_dim_shapes, get_modified_coords, merge_dicts


class Metric(object):
    def __init__(self, alignment_kwargs=None, cartesian_product_kwargs=None, cross_validation_kwargs=None,
                 alignment_ctr=Alignment, cartesian_product_ctr=CartesianProduct, cross_validation_ctr=CrossValidation):
        alignment_kwargs = alignment_kwargs or {}
        cartesian_product_kwargs = cartesian_product_kwargs or {}
        cross_validation_kwargs = cross_validation_kwargs or {}
        self._transformations = [alignment_ctr(**alignment_kwargs),
                                 cartesian_product_ctr(**cartesian_product_kwargs),
                                 cross_validation_ctr(**cross_validation_kwargs)]

        self._logger = logging.getLogger(fullname(self))

    def __call__(self, source_assembly, target_assembly, **kwargs):
        """
        :param mkgu.assemblies.NeuroidAssembly source_assembly:
        :param mkgu.assemblies.NeuroidAssembly target_assembly:
        :return: mkgu.metrics.Score
        """
        self._logger.debug("Applying metric to transformed assemblies")
        similarity_assembly = apply_transformations(source_assembly, target_assembly,
                                                    transformations=self._transformations, computor=self.apply)
        return self.score(similarity_assembly)

    def score(self, similarity_assembly):
        return MeanScore(similarity_assembly)

    def apply(self, *args, **kwargs):
        raise NotImplementedError()


def standard_error_of_the_mean(values, dim):
    return values.std(dim) / math.sqrt(len(values[dim]))


class Score(object):
    class Defaults:
        aggregation_dim = 'split'

    def __init__(self, values_assembly, aggregation_dim=Defaults.aggregation_dim):
        self.values = values_assembly
        center = self.get_center(self.values, dim=aggregation_dim)
        error = self.get_error(self.values, dim=aggregation_dim)
        center = center.expand_dims('aggregation')
        center['aggregation'] = ['center']
        error = error.expand_dims('aggregation')
        error['aggregation'] = ['error']
        self.aggregation = merge_data_arrays([center, error])

    def get_center(self, values, dim):
        raise NotImplementedError()

    def get_error(self, values, dim):
        return standard_error_of_the_mean(values, dim)

    def __repr__(self):
        return self.__class__.__name__ + "(" + ",".join(
            "{}={}".format(attr, val) for attr, val in self.__dict__.items()) + ")"


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

    def apply(self, train_source, train_target, test_source, test_target):
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
        assert all(source[self.Defaults.stimulus_coord].values
                   == target[self.Defaults.stimulus_coord].values)
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
        # return np.median(correlations)  # median across neuroids TODO


class NonparametricMetric(Metric):
    def apply(self, train_source, train_target, test_source, test_target):
        # no training, apply directly on test
        result = self.compute(test_source, test_target)
        return DataAssembly(result)

    def compute(self, source, target):
        raise NotImplementedError()


class MeanScore(Score):
    def get_center(self, values, dim):
        return values.mean(dim=dim)

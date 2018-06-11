import logging
import math
from collections import Counter

import numpy as np
import scipy

from mkgu.assemblies import NeuroidAssembly, array_is_element, walk_coords
from mkgu.metrics.transformations import Alignment, CartesianProduct, CrossValidation, apply_transformations
from mkgu.utils import fullname
from .utils import collect_coords, collect_dim_shapes, get_modified_coords, merge_dicts


class Metric(object):
    def __init__(self, transformations='default'):
        if transformations == 'default':
            transformations = [Alignment(), CartesianProduct(), CrossValidation()]
        self._transformations = transformations

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


class Score(object):
    def __init__(self, values_assembly, split_dim='split'):
        self.values = values_assembly
        self.center = self.get_center(self.values, dim=split_dim)
        self.error = self.get_error(self.values, dim=split_dim)

    def get_center(self, values, dim):
        raise NotImplementedError()

    def get_error(self, values, dim):
        return values.std(dim) / math.sqrt(len(values[dim]))

    def __repr__(self):
        return self.__class__.__name__ + "(" + ",".join(
            "{}={}".format(attr, val) for attr, val in self.__dict__.items()) + ")"


class ParametricMetric(Metric):
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

    def fit(self, train_source, train_target):
        np.testing.assert_array_equal(train_source.dims, ['presentation', 'neuroid'])
        np.testing.assert_array_equal(train_target.dims, ['presentation', 'neuroid'])
        # assert all(source == target for source, target in zip(train_source['image_id'], train_target['image_id']))
        self._target_neuroid_values = {}
        for name, dims, values in walk_coords(train_target):
            if 'neuroid' in dims:
                assert array_is_element(dims, 'neuroid')
                self._target_neuroid_values[name] = dims, values
        self.fit_values(train_source, train_target)

    def fit_values(self, train_source, train_target):
        raise NotImplementedError()

    def predict(self, test_source):
        np.testing.assert_array_equal(test_source.dims, ['presentation', 'neuroid'])
        predicted_values = self.predict_values(test_source)

        # package in assembly
        coords = {name: (dims, values) for name, dims, values in walk_coords(test_source) if 'neuroid' not in dims}
        for target_coord, target_dim_value in self._target_neuroid_values.items():
            coords[target_coord] = target_dim_value  # this might overwrite values which is okay
        dims = Counter([dim for name, (dim, values) in coords.items()])
        single_dims = {dim: count == 1 for dim, count in dims.items()}
        result_dims = test_source.dims
        unstacked_coords = {}
        for name, (dims, values) in coords.items():
            if single_dims[dims] and len(dims) > 0:
                result_dims = [dim if dim != dims[0] else name for dim in result_dims]
                coords[name] = name, values
                unstacked_coords[name] = dims
        result = NeuroidAssembly(predicted_values, coords=coords, dims=result_dims)
        for name, dims in unstacked_coords.items():
            assert len(dims) == 1
            result = result.stack(**{dims[0]: (name,)})
        return result

    def predict_values(self, test_source):
        raise NotImplementedError()

    def compare_prediction(self, prediction, target, axis='neuroid_id', correlation=scipy.stats.pearsonr):
        assert sorted(prediction['image_id'].values) == sorted(target['image_id'].values)
        assert sorted(prediction[axis].values) == sorted(target[axis].values)
        rs = []
        for i in target[axis].values:
            target_activations = target.sel(**{axis: i}).squeeze()
            prediction_activations = prediction.sel(**{axis: i}).squeeze()
            r, p = correlation(target_activations, prediction_activations)
            rs.append(r)
        return np.median(rs)  # median across neuroids


class NonparametricMetric(Metric):
    def apply(self, train_source, train_target, test_source, test_target):
        # ignore test, apply directly on train
        return self.compute(train_source, train_target)

    def compute(self, source, target):
        raise NotImplementedError()


class MeanScore(Score):
    def get_center(self, values, dim):
        return values.mean(dim=dim)

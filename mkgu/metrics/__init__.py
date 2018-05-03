from __future__ import absolute_import, division, print_function, unicode_literals
import xarray as xr

import functools
import itertools
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from mkgu.assemblies import DataAssembly
from .utils import collect_coords, collect_dim_shapes


class Metric(object):
    def __init__(self, similarity, characterization=None):
        """
        :param Similarity similarity:
        :param Characterization characterization:
        """
        self._similarity = similarity
        self._characterization = characterization or (lambda x: x)

    def __call__(self, source_assembly, target_assembly):
        characterized_source = self._characterization(source_assembly)
        characterized_target = self._characterization(target_assembly)
        return self._similarity(characterized_source, characterized_target)


class Similarity(object, metaclass=ABCMeta):
    def __call__(self, source_assembly, target_assembly):
        """
        :param mkgu.assemblies.NeuroidAssembly source_assembly:
        :param mkgu.assemblies.NeuroidAssembly target_assembly:
        :return: mkgu.metrics.Score
        """
        source_assembly = self.align(source_assembly, target_assembly)
        similarity_assembly = self.apply(source_assembly, target_assembly)
        return self.score(similarity_assembly)

    def align(self, source_assembly, target_assembly, subset_dims=('presentation',)):
        source_assembly = subset(source_assembly, target_assembly, subset_dims=subset_dims)
        return source_assembly

    def score(self, similarity_assembly):
        return MeanScore(similarity_assembly)

    def apply(self, source_assembly, target_assembly):
        raise NotImplementedError()


def subset(source_assembly, target_assembly, subset_dims=None):
    subset_dims = subset_dims or target_assembly.dims
    for dim in subset_dims:
        assert dim in target_assembly.dims
        assert dim in source_assembly.dims
        # we assume here that it does not matter if all levels are present in the source assembly
        # as long as there is at least one level that we can select over
        levels = target_assembly[dim].variable.level_names or [dim]
        assert any(hasattr(source_assembly, level) for level in levels)
        for level in levels:
            if not hasattr(source_assembly, level):
                continue
            level_values = target_assembly[level].values
            indexer = np.array([val in level_values for val in source_assembly[level].values])
            dim_indexes = {_dim: slice(None) if _dim != dim else np.where(indexer)[0] for _dim in source_assembly.dims}
            source_assembly = source_assembly.isel(**dim_indexes)
        # dims match up after selection - cannot make the strong version of equality due to potentially missing levels
        assert len(target_assembly[dim]) == len(source_assembly[dim])
    return source_assembly


class OuterCrossValidationSimilarity(Similarity, metaclass=ABCMeta):
    class Defaults:
        similarity_dims = 'presentation', 'neuroid'
        cross_validation_splits = 10
        cross_validation_data_ratio = .9
        cross_validation_dim = 'presentation'
        stratification_coord = 'object_name'  # cross-validation across images, balancing objects

    def __init__(self, cross_validation_splits=Defaults.cross_validation_splits,
                 cross_validation_data_ratio=Defaults.cross_validation_data_ratio):
        super(OuterCrossValidationSimilarity, self).__init__()
        self._split_strategy = StratifiedShuffleSplit(
            n_splits=cross_validation_splits, train_size=cross_validation_data_ratio)

    def apply(self, source_assembly, target_assembly,
              similarity_dims=Defaults.similarity_dims):
        """
        :param mkgu.assemblies.NeuroidAssembly source_assembly:
        :param mkgu.assemblies.NeuroidAssembly target_assembly:
        :param str similarity_dims: the dimension in both assemblies along which the similarity is to be computed
        :return: mkgu.assemblies.DataAssembly
        """

        # compute similarities over `similarity_dims`, i.e. across all adjacent coords
        def adjacent_selections(assembly):
            adjacent_dims = [dim for dim in assembly.dims if dim not in similarity_dims]
            choices = {dim: np.unique(assembly[dim]) for dim in adjacent_dims}
            combinations = [dict(zip(choices, values)) for values in itertools.product(*choices.values())]
            return combinations

        adjacents1, adjacents2 = adjacent_selections(source_assembly), adjacent_selections(target_assembly)
        similarities = [
            self.cross_apply(source_assembly.sel(**adj1), target_assembly.sel(**adj2))
            for adj1, adj2 in itertools.product(adjacents1, adjacents2)]
        assert all(similarity.shape == similarities[0].shape for similarity in similarities[1:])  # all shapes equal

        # re-shape into adjacent dimensions and split
        assembly_dims = source_assembly.dims + target_assembly.dims + ('split',)
        similarities = [expand(similarity, assembly_dims) for similarity in similarities]
        # https://stackoverflow.com/a/50125997/2225200
        similarities = xr.merge([similarity.rename('z') for similarity in similarities])['z'].rename(None)
        return similarities

    def cross_apply(self, source_assembly, target_assembly,
                    cross_validation_dim=Defaults.cross_validation_dim,
                    stratification_coord=Defaults.stratification_coord,
                    *args, **kwargs):
        assert all(source_assembly[cross_validation_dim] == target_assembly[cross_validation_dim])
        assert all(source_assembly[stratification_coord] == target_assembly[stratification_coord])

        cross_validation_values = target_assembly[cross_validation_dim]
        split_scores = {}
        for split_iterator, (train_indices, test_indices) in enumerate(self._split_strategy.split(
                np.zeros(len(np.unique(source_assembly[cross_validation_dim]))),
                source_assembly[stratification_coord].values)):
            train_values, test_values = cross_validation_values[train_indices], cross_validation_values[test_indices]
            train_source = subset(source_assembly, train_values)
            train_target = subset(target_assembly, train_values)
            test_source = subset(source_assembly, test_values)
            test_target = subset(target_assembly, test_values)
            split_score = self.apply_split(train_source, train_target, test_source, test_target)
            split_scores[split_iterator] = split_score

        # throw away all of the multi-dimensional dims as similarity will be computed over them.
        # we want to keep the adjacent dimensions which are 1-dimensional after the comprehension calling this method
        multi_dimensional_dims = [dim for dim in source_assembly.dims if len(source_assembly[dim]) > 1] + \
                                 [dim for dim in source_assembly.dims if len(target_assembly[dim]) > 1]
        coords = list(source_assembly.coords.keys()) + list(target_assembly.coords.keys())
        duplicate_coords = [coord for coord in coords if coords.count(coord) > 1]
        _collect_coords = functools.partial(collect_coords,
                                            ignore_dims=multi_dimensional_dims, rename_coords_list=duplicate_coords)
        coords = {**_collect_coords(assembly=source_assembly, kind='left'),
                  **_collect_coords(assembly=target_assembly, kind='right'),
                  **{'split': list(split_scores.keys())}}
        split_scores = DataAssembly(list(split_scores.values()), coords=coords, dims=['split'])
        return split_scores

    def apply_split(self, train_source, train_target, test_source, test_target):
        raise NotImplementedError()


def expand(assembly, target_dims):
    def reformat_coord_values(coord, dims, values):
        if coord in target_dims and len(values.shape) == 0:
            values = np.array([values])
            dims = [coord]
        return dims, values

    coords = {coord: reformat_coord_values(coord, values.dims, values.values)
              for coord, values in assembly.coords.items()}
    dim_shapes = OrderedDict((coord, values[1].shape)
                             for coord, values in coords.items() if coord in target_dims)
    shape = [_shape for shape in dim_shapes.values() for _shape in shape]
    values = np.broadcast_to(assembly.values, shape)
    return DataAssembly(values, coords=coords, dims=list(dim_shapes.keys()))


class ParametricCVSimilarity(OuterCrossValidationSimilarity):
    def apply_split(self, train_source, train_target, test_source, test_target):
        self.fit(train_source, train_target)
        prediction = self.predict(test_source)
        return self.compare_prediction(prediction, test_target)

    def fit(self, train_source, train_target):
        raise NotImplementedError()

    def predict(self, test_source):
        raise NotImplementedError()

    def compare_prediction(self, prediction, target):
        raise NotImplementedError()


class NonparametricCVSimilarity(OuterCrossValidationSimilarity):
    def apply_split(self, train_source, train_target, test_source, test_target):
        # ignore test, apply directly on train
        return self.compute(train_source, train_target)

    def compute(self, source, target):
        raise NotImplementedError()


class Characterization(object, metaclass=ABCMeta):
    """A Characterization contains a chain of numerical operations to be applied to a set of
    data to highlight some aspect of the data.  """

    @abstractmethod
    def __call__(self, assembly):
        raise NotImplementedError()


class Score(object):
    def __init__(self, values_assembly, split_dim='split'):
        self.values = values_assembly
        self.center = self.get_center(self.values, dim=split_dim)
        self.error = self.get_error(self.values, dim=split_dim)

    def get_center(self, values, dim):
        raise NotImplementedError()

    def get_error(self, values, dim):
        return values.std(dim)


class MeanScore(Score):
    def get_center(self, values, dim):
        return values.mean(dim=dim)

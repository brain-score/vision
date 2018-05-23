import functools
import itertools
import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, Counter

import numpy as np
import scipy
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from mkgu.assemblies import DataAssembly, NeuroidAssembly, merge_data_arrays, array_is_element, walk_coords
from .utils import collect_coords, collect_dim_shapes, get_modified_coords, merge_dicts


class Metric(object):
    def __init__(self, similarity, characterization=None):
        """
        :param Similarity similarity:
        :param Characterization characterization:
        """
        self._similarity = similarity
        self._characterization = characterization or (lambda x: x)

    def __call__(self, source_assembly, target_assembly, similarity_kwargs=None):
        characterized_source = self._characterization(source_assembly)
        characterized_target = self._characterization(target_assembly)
        return self._similarity(characterized_source, characterized_target, **similarity_kwargs)


class Similarity(object, metaclass=ABCMeta):
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, source_assembly, target_assembly, **kwargs):
        """
        :param mkgu.assemblies.NeuroidAssembly source_assembly:
        :param mkgu.assemblies.NeuroidAssembly target_assembly:
        :return: mkgu.metrics.Score
        """
        kwargs = kwargs or {}
        self._logger.debug("Aligning")
        source_assembly = self.align(source_assembly, target_assembly)
        self._logger.debug("Sorting")
        source_assembly, target_assembly = self.sort(source_assembly), self.sort(target_assembly)
        similarity_assembly = self.apply(source_assembly, target_assembly, **kwargs)
        return self.score(similarity_assembly)

    def align(self, source_assembly, target_assembly, subset_dim='presentation'):
        return subset(source_assembly, target_assembly, subset_dims=[subset_dim])

    def sort(self, assembly):
        return assembly.sortby('image_id')

    def score(self, similarity_assembly):
        return MeanScore(similarity_assembly)

    def apply(self, source_assembly, target_assembly):
        raise NotImplementedError()


def index_efficient(source_values, target_values):
    source_sort_indeces, target_sort_indeces = np.argsort(source_values), np.argsort(target_values)
    source_values, target_values = source_values[source_sort_indeces], target_values[target_sort_indeces]
    indexer = []
    source_index, target_index = 0, 0
    while target_index < len(target_values):
        if source_values[source_index] == target_values[target_index]:
            indexer.append(source_sort_indeces[source_index])
            target_index += 1
        elif source_values[source_index] < target_values[target_index]:
            source_index += 1
        else:  # source_values[source_index] > target_values[target_index]:
            target_index += 1
    return indexer


def subset(source_assembly, target_assembly, subset_dims=None, dims_must_match=True):
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
            target_values = target_assembly[level].values
            source_values = source_assembly[level].values
            indexer = index_efficient(source_values, target_values)
            dim_indexes = {_dim: slice(None) if _dim != dim else indexer for _dim in source_assembly.dims}
            source_assembly = source_assembly.isel(**dim_indexes)
        if dims_must_match:
            # dims match up after selection. cannot compare exact equality due to potentially missing levels
            assert len(target_assembly[dim]) == len(source_assembly[dim])
    return source_assembly


class OuterCrossValidationSimilarity(Similarity, metaclass=ABCMeta):
    class Defaults:
        similarity_dims = 'presentation', 'neuroid'
        adjacent_coords = 'region',
        cross_validation_splits = 10
        cross_validation_data_ratio = .9
        cross_validation_dim = 'image_id'
        stratification_coord = 'object_name'  # cross-validation across images, balancing objects

    def __init__(self, cross_validation_splits=Defaults.cross_validation_splits,
                 cross_validation_data_ratio=Defaults.cross_validation_data_ratio):
        super(OuterCrossValidationSimilarity, self).__init__()
        self._stratified_split = StratifiedShuffleSplit(
            n_splits=cross_validation_splits, train_size=cross_validation_data_ratio)
        self._shuffle_split = ShuffleSplit(
            n_splits=cross_validation_splits, train_size=cross_validation_data_ratio)
        self._logger = logging.getLogger(self.__class__.__name__)

    def apply(self, source_assembly, target_assembly,
              similarity_dims=Defaults.similarity_dims, adjacent_coord_names=Defaults.adjacent_coords,
              adjacent_coord_names_source=(), adjacent_coord_names_target=()):
        """
        :param mkgu.assemblies.NeuroidAssembly source_assembly:
        :param mkgu.assemblies.NeuroidAssembly target_assembly:
        :param str similarity_dims: the dimension in both assemblies along which the similarity is to be computed
        :param [str] adjacent_coord_names:
        :return: mkgu.assemblies.DataAssembly
        """

        # compute similarities over `similarity_dims`, i.e. across all adjacent coords
        def adjacent_selections(assembly, adjacent_coord_names):
            adjacent_coords = [dim for dim in assembly.dims if dim not in similarity_dims] \
                              + [coord for coord in adjacent_coord_names if hasattr(assembly, coord)]
            choices = {coord: np.unique(assembly[coord]) for coord in adjacent_coords}
            combinations = [dict(zip(choices, values)) for values in itertools.product(*choices.values())]
            return combinations

        adjacents_src = adjacent_selections(source_assembly, adjacent_coord_names + tuple(adjacent_coord_names_source))
        adjacents_tgt = adjacent_selections(target_assembly, adjacent_coord_names + tuple(adjacent_coord_names_target))
        # run all adjacent combinations or use the assemblies themselves if no adjacents
        adjacent_combinations = list(itertools.product(adjacents_src, adjacents_tgt)) or [({}, {})]
        similarities = []
        for i, (adj_src, adj_tgt) in enumerate(adjacent_combinations):
            self._logger.debug("adjacents {}/{}: {} | {}".format(i + 1, len(adjacent_combinations), adj_src, adj_tgt))
            source_adj, target_adj = source_assembly.multisel(**adj_src), target_assembly.multisel(**adj_tgt)
            # in single-unit recordings, not all electrodes were recorded for each presentation -> drop non-recorded
            non_nan = ~np.isnan(target_adj.values)
            if not all(non_nan):
                non_nan = non_nan.squeeze()  # FIXME: we assume 2D data with single-value neuroid here
                source_adj, target_adj = source_adj[non_nan], target_adj[non_nan]

            similarity = self.cross_apply(source_adj, target_adj)
            adj_coords = self.merge_adjacents(adj_src, adj_tgt)
            for coord_name, coord_value in adj_coords.items():
                similarity[coord_name] = coord_value
            similarities.append(similarity)
        assert all(similarity.shape == similarities[0].shape for similarity in similarities[1:])  # all shapes equal

        # re-shape into adjacent dimensions and split
        assembly_dims = source_assembly.dims + target_assembly.dims + tuple(adjacent_coord_names) \
                        + tuple(adjacent_coord_names_source) + tuple(adjacent_coord_names_target) \
                        + ('split',)
        similarities = [expand(similarity, assembly_dims) for similarity in similarities]
        similarities = merge_data_arrays(similarities)
        return similarities

    def cross_apply(self, source_assembly, target_assembly,
                    cross_validation_dim=Defaults.cross_validation_dim,
                    stratification_coord=Defaults.stratification_coord):
        assert all(source_assembly[cross_validation_dim].values == target_assembly[cross_validation_dim].values)

        cross_validation_values = target_assembly[cross_validation_dim]
        if hasattr(target_assembly, stratification_coord):
            assert hasattr(source_assembly, stratification_coord)
            assert all(source_assembly[stratification_coord].values == target_assembly[stratification_coord].values)
            splits = list(self._stratified_split.split(np.zeros(len(np.unique(source_assembly[cross_validation_dim]))),
                                                       source_assembly[stratification_coord].values))
        else:
            self._logger.warning("Stratification coord '{}' not found in assembly "
                                 "- falling back to un-stratified splits".format(stratification_coord))
            splits = list(self._shuffle_split.split(np.zeros(len(np.unique(source_assembly[cross_validation_dim])))))

        split_scores = {}
        for split_iterator, (train_indices, test_indices) in enumerate(splits):
            self._logger.debug("split {}/{}".format(split_iterator + 1, len(splits)))
            train_values, test_values = cross_validation_values[train_indices], cross_validation_values[test_indices]
            train_source = subset(source_assembly, train_values)
            train_target = subset(target_assembly, train_values)
            test_source = subset(source_assembly, test_values)
            test_target = subset(target_assembly, test_values)
            split_score = self.apply_split(train_source, train_target, test_source, test_target)
            split_scores[split_iterator] = split_score

        # throw away all of the multi-dimensional dims as similarity will be computed over them.
        # we want to keep the adjacent dimensions which are 1-dimensional after the comprehension calling this method
        # Note that we don't keep adjacent coords yet, which is something we should ultimately do
        multi_dimensional_dims = [dim for dim in source_assembly.dims if len(source_assembly[dim]) > 1] + \
                                 [dim for dim in source_assembly.dims if len(target_assembly[dim]) > 1]
        coords = list(source_assembly.coords.keys()) + list(target_assembly.coords.keys())
        duplicate_coords = [coord for coord in coords if coords.count(coord) > 1]
        _collect_coords = functools.partial(collect_coords,
                                            ignore_dims=multi_dimensional_dims, rename_coords_list=duplicate_coords)
        coords = {**_collect_coords(assembly=source_assembly, kind='source'),
                  **_collect_coords(assembly=target_assembly, kind='target'),
                  **{'split': list(split_scores.keys())}}
        coords = {'split': list(split_scores.keys())}  # FIXME: hack to work around non-matching dims
        split_scores = DataAssembly(list(split_scores.values()), coords=coords, dims=['split'])
        return split_scores

    def apply_split(self, train_source, train_target, test_source, test_target):
        raise NotImplementedError()

    def merge_adjacents(self, adj_left, adj_right):
        coords = list(adj_left.keys()) + list(adj_right.keys())
        duplicates = [coord for coord in coords if coords.count(coord) > 1]
        coords_left = {coord if coord not in duplicates else coord + '-left': coord_value
                       for coord, coord_value in adj_left.items()}
        coords_right = {coord if coord not in duplicates else coord + '-right': coord_value
                        for coord, coord_value in adj_right.items()}
        return {**coords_left, **coords_right}


def expand(assembly, target_dims):
    def strip(coord):
        stripped_coord = coord
        if stripped_coord.endswith('-left'):
            stripped_coord = stripped_coord[:-len('-left')]
        if stripped_coord.endswith('-right'):
            stripped_coord = stripped_coord[:-len('-right')]
        return stripped_coord

    def reformat_coord_values(coord, dims, values):
        stripped_coord = strip(coord)

        if stripped_coord in target_dims and len(values.shape) == 0:
            values = np.array([values])
            dims = [coord]
        return dims, values

    coords = {coord: reformat_coord_values(coord, values.dims, values.values)
              for coord, values in assembly.coords.items()}
    dim_shapes = OrderedDict((coord, values[1].shape)
                             for coord, values in coords.items() if strip(coord) in target_dims)
    shape = [_shape for shape in dim_shapes.values() for _shape in shape]
    # prepare values for broadcasting by adding new dimensions
    values = assembly.values
    for _ in range(sum([dim not in assembly.dims for dim in dim_shapes])):
        values = values[:, np.newaxis]
    values = np.broadcast_to(values, shape)
    return DataAssembly(values, coords=coords, dims=list(dim_shapes.keys()))


class ParametricCVSimilarity(OuterCrossValidationSimilarity):
    def __init__(self, cross_validation_splits=OuterCrossValidationSimilarity.Defaults.cross_validation_splits,
                 cross_validation_data_ratio=OuterCrossValidationSimilarity.Defaults.cross_validation_data_ratio):
        super().__init__(cross_validation_splits, cross_validation_data_ratio)
        self._target_neuroid_values = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def apply_split(self, train_source, train_target, test_source, test_target):
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

        coords = {name: (dims, values) for name, dims, values in walk_coords(test_source) if 'neuroid' not in dims}
        for target_coord, target_dim_value in self._target_neuroid_values.items():
            coords[target_coord] = target_dim_value  # this might overwrite values which is okay
        dims = Counter([dim for name, (dim, values) in coords.items()])
        single_dims = {dim: count == 1 for dim, count in dims.items()}
        result_dims = test_source.dims
        unstacked_coords = {}
        for name, (dims, values) in coords.items():
            if single_dims[dims]:
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
        return values.std(dim) / math.sqrt(len(values[dim]))

    def __repr__(self):
        return self.__class__.__name__ + "(" + ",".join(
            "{}={}".format(attr, val) for attr, val in self.__dict__.items()) + ")"


class MeanScore(Score):
    def get_center(self, values, dim):
        return values.mean(dim=dim)

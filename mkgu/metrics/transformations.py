import functools
import itertools
import logging
from abc import ABCMeta
from collections.__init__ import OrderedDict

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from mkgu.assemblies import merge_data_arrays, DataAssembly
from mkgu.metrics import Metric, collect_coords


def index_efficient(source_values, target_values):
    source_sort_indeces, target_sort_indeces = np.argsort(source_values), np.argsort(target_values)
    source_values, target_values = source_values[source_sort_indeces], target_values[target_sort_indeces]
    indexer = []
    source_index, target_index = 0, 0
    while target_index < len(target_values) and source_index < len(source_values):
        if source_values[source_index] == target_values[target_index]:
            indexer.append(source_sort_indeces[source_index])
            target_index += 1
        elif source_values[source_index] < target_values[target_index]:
            source_index += 1
        else:  # source_values[source_index] > target_values[target_index]:
            target_index += 1
    return indexer


def subset(source_assembly, target_assembly, subset_dims=None, dims_must_match=True, repeat=False):
    """
    :param subset_dims: either dimensions, then all its levels will be used or levels right away
    :param dims_must_match:
    :return:
    """
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
            if repeat:
                indexer = index_efficient(source_values, target_values)
                dim_indexes = {_dim: slice(None) if _dim != dim else indexer for _dim in source_assembly.dims}
            else:
                level_values = target_assembly[level].values
                indexer = np.array([val in level_values for val in source_assembly[level].values])
                dim_indexes = {_dim: slice(None) if _dim != dim else np.where(indexer)[0] for _dim in
                               source_assembly.dims}
            source_assembly = source_assembly.isel(**dim_indexes)
        if dims_must_match:
            # dims match up after selection. cannot compare exact equality due to potentially missing levels
            assert len(target_assembly[dim]) == len(source_assembly[dim])
    return source_assembly


class OuterCrossValidationMetric(Metric, metaclass=ABCMeta):
    class Defaults:
        similarity_dims = 'presentation', 'neuroid'
        adjacent_coords = 'region',
        cross_validation_splits = 10
        cross_validation_data_ratio = .9
        cross_validation_dim = 'image_id'
        stratification_coord = 'object_name'  # cross-validation across images, balancing objects

    def __init__(self, cross_validation_splits=Defaults.cross_validation_splits,
                 cross_validation_data_ratio=Defaults.cross_validation_data_ratio):
        super(OuterCrossValidationMetric, self).__init__()
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
            if not all(non_nan.flatten()):
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
        coords_left = {coord if coord not in duplicates else coord + '_left': coord_value
                       for coord, coord_value in adj_left.items()}
        coords_right = {coord if coord not in duplicates else coord + '_right': coord_value
                        for coord, coord_value in adj_right.items()}
        return {**coords_left, **coords_right}


def expand(assembly, target_dims):
    def strip(coord):
        stripped_coord = coord
        if stripped_coord.endswith('_left'):
            stripped_coord = stripped_coord[:-len('_left')]
        if stripped_coord.endswith('_right'):
            stripped_coord = stripped_coord[:-len('_right')]
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
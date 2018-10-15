import itertools
import logging
import math
from collections import OrderedDict
from enum import Enum

import numpy as np
import xarray as xr
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from brainscore.assemblies import merge_data_arrays, DataAssembly
from brainscore.utils import fullname


class Transformation(object):
    """
    Transforms an incoming assembly into parts/combinations thereof,
    yields them for further processing,
    and packages the results back together.
    """

    def __call__(self, *args):
        raise NotImplementedError()

    def _get_result(self, *args, done):
        """
        Yields the `*args` for further processing by coroutines
        and waits for the result to be sent back.
        :param args: transformed values
        :param bool done: whether this is the last transformation and the next `yield` is the combined result
        :return: the result from processing by the coroutine
        """
        result = yield args  # yield the values to coroutine
        yield done  # wait for coroutine to send back similarity and inform whether result is ready to be returned
        return result

    def aggregate(self, center, error):
        return center, error


class Alignment(Transformation):
    class Defaults:
        order_dimensions = ('presentation', 'neuroid')
        alignment_dim = 'image_id'
        repeat = False

    def __init__(self, order_dimensions=Defaults.order_dimensions, alignment_dim=Defaults.alignment_dim,
                 repeat=Defaults.repeat):
        self._order_dimensions = order_dimensions
        self._alignment_dim = alignment_dim
        self._repeat = repeat
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, source_assembly, target_assembly):
        self._logger.debug("Aligning by {} and {}, {} repeats".format(
            self._order_dimensions, self._alignment_dim, "with" if self._repeat else "no"))
        source_assembly = self.align(source_assembly, target_assembly)
        self._logger.debug("Sorting by {}".format(self._alignment_dim))
        source_assembly, target_assembly = self.sort(source_assembly), self.sort(target_assembly)
        result = yield from self._get_result(source_assembly, target_assembly, done=True)
        yield result

    def align(self, source_assembly, target_assembly):
        dimensions = list(self._order_dimensions) + list(set(source_assembly.dims) - set(self._order_dimensions))
        source_assembly = source_assembly.transpose(*dimensions)
        return subset(source_assembly, target_assembly, subset_dims=[self._alignment_dim], repeat=self._repeat)

    def sort(self, assembly):
        return assembly.sortby(self._alignment_dim)


class CartesianProduct(Transformation):
    """
    Splits an incoming assembly along all dimensions that similarity is not computed over
    as well as along dividing coords that denote separate parts of the assembly.
    """

    class Defaults:
        non_dividing_dims = 'presentation', 'neuroid'
        dividing_coords_target = 'region',

    def __init__(self,
                 non_dividing_dims=Defaults.non_dividing_dims,
                 dividing_coord_names_source=(), dividing_coord_names_target=Defaults.dividing_coords_target):
        super(CartesianProduct, self).__init__()
        self._non_dividing_dims = non_dividing_dims
        self._dividing_coord_names_source = dividing_coord_names_source or ()
        self._dividing_coord_names_target = dividing_coord_names_target or ()

        self._logger = logging.getLogger(fullname(self))

    def dividers(self, assembly, dividing_coord_names):
        return dividers(assembly, dividing_coord_names=dividing_coord_names, non_dividing_dims=self._non_dividing_dims)

    def __call__(self, source_assembly, target_assembly):
        """
        :param brainscore.assemblies.NeuroidAssembly source_assembly:
        :param brainscore.assemblies.NeuroidAssembly target_assembly:
        :return: brainscore.assemblies.DataAssembly
        """
        dividers_source = self.dividers(source_assembly, self._dividing_coord_names_source)
        dividers_target = self.dividers(target_assembly, self._dividing_coord_names_target)
        # run all dividing combinations or use the assemblies themselves if no dividers
        divider_combinations = list(itertools.product(dividers_source, dividers_target)) or [({}, {})]
        similarities = []
        for i, (div_src, div_tgt), done in enumerate_done(divider_combinations):
            self._logger.debug("dividers {}/{}: {} | {}".format(i + 1, len(divider_combinations), div_src, div_tgt))
            source_div, target_div = source_assembly.multisel(**div_src), target_assembly.multisel(**div_tgt)
            similarity = yield from self._get_result(source_div, target_div, done=done)

            divider_coords = self.merge_dividers(div_src, div_tgt)
            for coord_name, coord_value in divider_coords.items():
                similarity = similarity.expand_dims(coord_name)
                similarity[coord_name] = [coord_value]
            similarities.append(similarity)
        assert all(similarity.dims == similarities[0].dims for similarity in similarities[1:])  # all dims equal
        # note that the shapes are likely different between combinations and will be padded with NaNs:
        # for instance, V4 and IT have different numbers of neurons and while the neuroid dimension will be merged,
        # the values for V4 neuroids of an IT combination will be NaN.
        similarities = merge_data_arrays(similarities)
        yield similarities

    def merge_dividers(self, div_source, div_target):
        coords = list(div_source.keys()) + list(div_target.keys())
        duplicates = [coord for coord in coords if coords.count(coord) > 1]
        coords_source = {coord if coord not in duplicates else self.rename_duplicate_coord(coord, self.CoordType.SOURCE)
                         : coord_value for coord, coord_value in div_source.items()}
        coords_target = {coord if coord not in duplicates else self.rename_duplicate_coord(coord, self.CoordType.TARGET)
                         : coord_value for coord, coord_value in div_target.items()}
        return {**coords_source, **coords_target}

    class CoordType(Enum):
        SOURCE = 1
        TARGET = 2

    @classmethod
    def rename_duplicate_coord(cls, coord, coord_type):
        suffix = {cls.CoordType.SOURCE: 'source', cls.CoordType.TARGET: 'target'}
        return coord + '_' + suffix[coord_type]


def dividers(assembly, dividing_coord_names, non_dividing_dims=()):
    """
    divide data along dividing coords and non-central dimensions,
    i.e. dimensions that the metric is not computed over
    """
    dividing_coords = [dim for dim in assembly.dims if dim not in non_dividing_dims] \
                      + [coord for coord in dividing_coord_names if hasattr(assembly, coord)]
    choices = {coord: np.unique(assembly[coord]) for coord in dividing_coords}
    combinations = [dict(zip(choices, values)) for values in itertools.product(*choices.values())]
    return combinations


class CrossValidationSingle(Transformation):
    class Defaults:
        splits = 10
        train_size = .9
        split_coord = 'image_id'
        stratification_coord = 'object_name'  # cross-validation across images, balancing objects
        seed = 1

    def __init__(self,
                 splits=Defaults.splits, train_size=None, test_size=None,
                 split_coord=Defaults.split_coord, stratification_coord=Defaults.stratification_coord,
                 seed=Defaults.seed):
        super().__init__()
        if train_size is None and test_size is None:
            train_size = self.Defaults.train_size
        self._stratified_split = StratifiedShuffleSplit(
            n_splits=splits, train_size=train_size, test_size=test_size, random_state=seed)
        self._shuffle_split = ShuffleSplit(
            n_splits=splits, train_size=train_size, test_size=test_size, random_state=seed)
        self._split_coord = split_coord
        self._stratification_coord = stratification_coord

        self._logger = logging.getLogger(fullname(self))

    def _stratify(self, assembly):
        return self._stratification_coord and hasattr(assembly, self._stratification_coord)

    def _build_splits(self, assembly):
        cross_validation_values, indices = extract_coord(assembly, self._split_coord, return_index=True)
        data_shape = np.zeros(len(cross_validation_values))
        if self._stratify(assembly):
            splits = self._stratified_split.split(data_shape,
                                                  assembly[self._stratification_coord].values[indices])
        else:
            self._logger.warning("Stratification coord '{}' not found in assembly "
                                 "- falling back to un-stratified splits".format(self._stratification_coord))
            splits = self._shuffle_split.split(data_shape)
        return cross_validation_values, list(splits)

    def __call__(self, assembly):
        """
        :param assembly: the assembly to cross-validate over
        """
        cross_validation_values, splits = self._build_splits(assembly)

        split_scores = []
        for split_iterator, (train_indices, test_indices), done in enumerate_done(splits):
            self._logger.debug("split {}/{}".format(split_iterator + 1, len(splits)))
            train_values, test_values = cross_validation_values[train_indices], cross_validation_values[test_indices]
            train = subset(assembly, train_values, dims_must_match=False)
            test = subset(assembly, test_values, dims_must_match=False)

            split_score = yield from self._get_result(train, test, done=done)
            split_score = split_score.expand_dims('split')
            split_score['split'] = [split_iterator]
            split_scores.append(split_score)

        split_scores = merge_data_arrays(split_scores)
        yield split_scores

    def aggregate(self, center, error):
        return center.mean('split'), standard_error_of_the_mean(error, 'split')


class CrossValidation(Transformation):
    """
    Performs multiple splits over a source and target assembly.
    No guarantees are given for data-alignment, use the metadata.
    """

    def __init__(self,
                 splits=CrossValidationSingle.Defaults.splits, split_coord=CrossValidationSingle.Defaults.split_coord,
                 stratification_coord=CrossValidationSingle.Defaults.stratification_coord,
                 train_size=None, test_size=None, seed=CrossValidationSingle.Defaults.seed):
        self._split_coord = split_coord
        self._stratification_coord = stratification_coord
        self._single_crossval = CrossValidationSingle(splits=splits, split_coord=split_coord,
                                                      stratification_coord=stratification_coord,
                                                      train_size=train_size, test_size=test_size, seed=seed)
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, source_assembly, target_assembly):
        # check only for equal values, alignment is given by metadata
        assert sorted(source_assembly[self._split_coord].values) == sorted(target_assembly[self._split_coord].values)
        if self._single_crossval._stratify(target_assembly):
            assert hasattr(source_assembly, self._stratification_coord)
            assert sorted(source_assembly[self._stratification_coord].values) == \
                   sorted(target_assembly[self._stratification_coord].values)
        cross_validation_values, splits = self._single_crossval._build_splits(target_assembly)

        split_scores = []
        for split_iterator, (train_indices, test_indices), done in enumerate_done(splits):
            self._logger.debug("split {}/{}".format(split_iterator + 1, len(splits)))
            train_values, test_values = cross_validation_values[train_indices], cross_validation_values[test_indices]
            train_source = subset(source_assembly, train_values, dims_must_match=False)
            train_target = subset(target_assembly, train_values, dims_must_match=False)
            assert len(train_source[self._split_coord]) == len(train_target[self._split_coord])
            test_source = subset(source_assembly, test_values, dims_must_match=False)
            test_target = subset(target_assembly, test_values, dims_must_match=False)
            assert len(test_source[self._split_coord]) == len(test_target[self._split_coord])

            split_score = yield from self._get_result(train_source, train_target, test_source, test_target,
                                                      done=done)
            split_score = split_score.expand_dims('split')
            split_score['split'] = [split_iterator]
            split_scores.append(split_score)

        split_scores = merge_data_arrays(split_scores)
        yield split_scores


def extract_coord(assembly, coord, return_index=False):
    extracted_assembly, indices = np.unique(assembly[coord].values, return_index=True)
    dims = assembly[coord].dims
    assert len(dims) == 1
    extracted_assembly = xr.DataArray(extracted_assembly, coords={coord: extracted_assembly}, dims=[coord])
    extracted_assembly = extracted_assembly.stack(**{dims[0]: (coord,)})
    return extracted_assembly if not return_index else extracted_assembly, indices


def standard_error_of_the_mean(values, dim):
    return values.std(dim) / math.sqrt(len(values[dim]))


def subset(source_assembly, target_assembly, subset_dims=None, dims_must_match=True, repeat=False):
    """
    :param subset_dims: either dimensions, then all its levels will be used or levels right away
    :param dims_must_match:
    :return:
    """
    subset_dims = subset_dims or target_assembly.dims
    for dim in subset_dims:
        assert hasattr(target_assembly, dim)
        assert hasattr(source_assembly, dim)
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
            else:
                indexer = np.array([val in target_values for val in source_values])
                indexer = np.where(indexer)[0]
            if dim not in target_assembly.dims:
                # not actually a dimension, but rather a coord -> filter along underlying dim
                dim = target_assembly[dim].dims
                assert len(dim) == 1
                dim = dim[0]
            dim_indexes = {_dim: slice(None) if _dim != dim else indexer for _dim in source_assembly.dims}
            source_assembly = source_assembly.isel(**dim_indexes)
        if dims_must_match:
            # dims match up after selection. cannot compare exact equality due to potentially missing levels
            assert len(target_assembly[dim]) == len(source_assembly[dim])
    return source_assembly


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


def expand(assembly, target_dims):
    def strip(coord):
        stripped_coord = coord
        if stripped_coord.endswith('_source'):
            stripped_coord = stripped_coord[:-len('_source')]
        if stripped_coord.endswith('_target'):
            stripped_coord = stripped_coord[:-len('_target')]
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


def enumerate_done(values):
    for i, val in enumerate(values):
        done = i == len(values) - 1
        yield i, val, done


def apply_transformations(*args, transformations, metric):
    """
    Start from first transformation, pass all of the transformed values into second transformation and so forth.
    When no transformations are left, apply the metric on each of the transformed values.
    When values are coming back, go the reverse way:
    collect all computed values, send them to the last transformation.
    Once the last transformation is complete, return its result to the second-last transformation and so forth.

    Note that transformations are expected to yield a transformed value,
    receive the result of computing that transformation,
    and signal whether all transformations are done.
    When the last transformation is done, the generator ought to yield the combined result.
    """

    def recurse(generator, generators):
        for vals in generator:
            if len(generators) > 0:
                gen = generators[0]
                gen = gen(*vals)
                y = recurse(gen, generators[1:])
            else:
                y = metric(*vals)
            done = generator.send(y)
            if done:
                break
        result = next(generator)
        return result

    scores = recurse(transformations[0](*args), transformations[1:])
    return scores


class Transformations(object):
    def __init__(self, transformations):
        self._transformations = transformations

    def __call__(self, *args, metric):
        raw_scores = apply_transformations(*args, transformations=self._transformations, metric=metric)

        scores = metric.aggregate(raw_scores)
        center, error = scores, scores
        for transformation in self._transformations:
            center, error = transformation.aggregate(center, error)
        from brainscore.metrics import build_score  # avoid circular import
        return build_score(raw_scores, center, error)

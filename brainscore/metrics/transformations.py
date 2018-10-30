import itertools
import logging
import math
from collections import OrderedDict

import numpy as np
import xarray as xr
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from tqdm import tqdm

from brainscore.assemblies import merge_data_arrays, DataAssembly, walk_coords
from brainscore.metrics import Score
from brainscore.metrics.utils import unique_ordered
from brainscore.utils import fullname


class Transformation(object):
    """
    Transforms an incoming assembly into parts/combinations thereof,
    yields them for further processing,
    and packages the results back together.
    """

    def __call__(self, *args, apply, aggregate=None, **kwargs):
        values = self._run_pipe(*args, apply=apply, **kwargs)

        score = self._apply_aggregate(aggregate, values) if aggregate is not None else values
        score = self._apply_aggregate(self.aggregate, score)
        return score

    def _run_pipe(self, *args, apply, **kwargs):
        generator = self.pipe(*args, **kwargs)
        for vals in generator:
            y = apply(*vals)
            done = generator.send(y)
            if done:
                break
        result = next(generator)
        return result

    def pipe(self, *args, **kwargs):
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

    def _apply_aggregate(self, aggregate_fnc, values):
        """
        Applies the aggregate while keeping the raw values in the attrs.
        If raw values are already present, keeps them, else they are added.
        """
        score = aggregate_fnc(values)
        if Score.RAW_VALUES_KEY not in score.attrs:
            # check if the raw values are already in the values.
            # if yes, they didn't get copied to the aggregate score and we use those as the "rawest" values.
            raw = values if Score.RAW_VALUES_KEY not in values.attrs else values.attrs[Score.RAW_VALUES_KEY]
            score.attrs[Score.RAW_VALUES_KEY] = raw
        return score

    def aggregate(self, score):
        return Score(score)


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

    def pipe(self, source_assembly, target_assembly):
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

    def __init__(self, dividers=()):
        super(CartesianProduct, self).__init__()
        self._dividers = dividers or ()
        self._logger = logging.getLogger(fullname(self))

    def dividers(self, assembly, dividing_coords):
        """
        divide data along dividing coords and non-central dimensions,
        i.e. dimensions that the metric is not computed over
        """
        non_matched_coords = [coord for coord in dividing_coords if not hasattr(assembly, coord)]
        assert not non_matched_coords, f"{non_matched_coords} not found in assembly"
        choices = {coord: unique_ordered(assembly[coord].values) for coord in dividing_coords}
        combinations = [dict(zip(choices, values)) for values in itertools.product(*choices.values())]
        return combinations

    def pipe(self, assembly):
        """
        :param brainscore.assemblies.NeuroidAssembly assembly:
        :return: brainscore.assemblies.DataAssembly
        """
        dividers = self.dividers(assembly, dividing_coords=self._dividers)
        scores = []
        progress = tqdm(enumerate_done(dividers), total=len(dividers), desc='cartesian product')
        for i, divider, done in progress:
            progress.set_description(str(divider))
            divided_assembly = assembly.multisel(**divider)
            result = yield from self._get_result(divided_assembly, done=done)

            for coord_name, coord_value in divider.items():
                result = result.expand_dims(coord_name)
                result[coord_name] = [coord_value]
            scores.append(result)
        scores = Score.merge(*scores)
        yield scores


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

    def pipe(self, assembly):
        """
        :param assembly: the assembly to cross-validate over
        """
        cross_validation_values, splits = self._build_splits(assembly)

        split_scores = []
        for split_iterator, (train_indices, test_indices), done \
                in tqdm(enumerate_done(splits), total=len(splits), desc='cross-validation'):
            train_values, test_values = cross_validation_values[train_indices], cross_validation_values[test_indices]
            train = subset(assembly, train_values, dims_must_match=False)
            test = subset(assembly, test_values, dims_must_match=False)

            split_score = yield from self._get_result(train, test, done=done)
            split_score = split_score.expand_dims('split')
            split_score['split'] = [split_iterator]
            split_scores.append(split_score)

        split_scores = Score.merge(*split_scores)
        yield split_scores

    def aggregate(self, values):
        center = values.mean('split')
        error = standard_error_of_the_mean(values, 'split')
        return Score([center, error],
                     coords={**{'aggregation': ['center', 'error']},
                             **{coord: (dims, values) for coord, dims, values in walk_coords(center)}},
                     dims=('aggregation',) + center.dims)


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

    def pipe(self, source_assembly, target_assembly):
        # check only for equal values, alignment is given by metadata
        assert sorted(source_assembly[self._split_coord].values) == sorted(target_assembly[self._split_coord].values)
        if self._single_crossval._stratify(target_assembly):
            assert hasattr(source_assembly, self._stratification_coord)
            assert sorted(source_assembly[self._stratification_coord].values) == \
                   sorted(target_assembly[self._stratification_coord].values)
        cross_validation_values, splits = self._single_crossval._build_splits(target_assembly)

        split_scores = []
        for split_iterator, (train_indices, test_indices), done \
                in tqdm(enumerate_done(splits), total=len(splits), desc='cross-validation'):
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

    def aggregate(self, score):
        return self._single_crossval.aggregate(score)


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
            if len(np.unique(source_assembly.dims)) == len(source_assembly.dims):  # no repeated dimensions
                source_assembly = source_assembly.isel(**dim_indexes)
                continue
            # work-around when dimensions are repeated. `isel` will keep only the first instance of a repeated dimension
            positional_dim_indexes = [dim_indexes[dim] for dim in source_assembly.dims]
            source_assembly = type(source_assembly)(
                source_assembly.values[np.ix_(*positional_dim_indexes)],
                coords={coord: (dim, value[dim_indexes[dim]]) for coord, dims, value in walk_coords(source_assembly)},
                dims=source_assembly.dims)
        if dims_must_match:
            # dims match up after selection. cannot compare exact equality due to potentially missing levels
            assert len(target_assembly[dim]) == len(source_assembly[dim])
    return source_assembly


def index_efficient(source_values, target_values):
    source_sort_indices, target_sort_indices = np.argsort(source_values), np.argsort(target_values)
    source_values, target_values = source_values[source_sort_indices], target_values[target_sort_indices]
    indexer = []
    source_index, target_index = 0, 0
    while target_index < len(target_values) and source_index < len(source_values):
        if source_values[source_index] == target_values[target_index]:
            indexer.append(source_sort_indices[source_index])
            # if next source value is greater than target, use next target. else next source.
            # if target values remain the same, we might as well take the next target.
            if (target_index + 1 < len(target_values) and
                target_values[target_index + 1] == target_values[target_index]) or \
                    (source_index + 1 < len(source_values) and
                     source_values[source_index + 1] > target_values[target_index]):
                target_index += 1
            else:
                source_index += 1
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

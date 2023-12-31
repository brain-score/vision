import itertools
import logging
import math
from collections import OrderedDict

import numpy as np
import xarray as xr
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, KFold, StratifiedKFold
from tqdm import tqdm

from brainio.assemblies import DataAssembly
from brainio.transform import subset
from brainscore_vision.metric_helpers.utils import unique_ordered
from brainscore_vision.metrics import Score
from brainscore_vision.utils import fullname


def apply_aggregate(aggregate_fnc, values):
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


class Transformation(object):
    """
    Transforms an incoming assembly into parts/combinations thereof,
    yields them for further processing,
    and packages the results back together.
    """

    def __call__(self, *args, apply, aggregate=None, **kwargs):
        values = self._run_pipe(*args, apply=apply, **kwargs)

        score = apply_aggregate(aggregate, values) if aggregate is not None else values
        score = apply_aggregate(self.aggregate, score)
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

    def aggregate(self, score):
        return Score(score)


class Alignment(Transformation):
    class Defaults:
        order_dimensions = ('presentation', 'neuroid')
        alignment_dim = 'stimulus_id'
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
            # squeeze dimensions if necessary
            for divider_coord in divider:
                dims = assembly[divider_coord].dims
                assert len(dims) == 1
                if dims[0] in divided_assembly.dims and len(divided_assembly[dims[0]]) == 1:
                    divided_assembly = divided_assembly.squeeze(dims[0])
            result = yield from self._get_result(divided_assembly, done=done)

            for coord_name, coord_value in divider.items():
                # expand and set coordinate value. If the underlying raw values already contain that coordinate
                # (e.g. as part of a MultiIndex), don't create and set new dimension on raw values.
                if not hasattr(result, 'raw'):  # no raw values anyway
                    kwargs = {}
                elif not hasattr(result.raw, coord_name):  # doesn't have values yet, we can set (True by default)
                    kwargs = {}
                else:  # has raw values but already has coord in them --> need to prevent setting
                    # this expects the result to accept `_apply_raw` in its `expand_dims` method
                    # which is true for our `Score` class, but not a standard `DataAssembly`.
                    kwargs = dict(_apply_raw=False)
                result = result.expand_dims(coord_name, **kwargs)
                result.__setitem__(coord_name, [coord_value], **kwargs)
            scores.append(result)
        scores = Score.merge(*scores)
        yield scores


class Split:
    class Defaults:
        splits = 10
        train_size = .9
        split_coord = 'stimulus_id'
        stratification_coord = 'object_name'  # cross-validation across images, balancing objects
        unique_split_values = False
        random_state = 1

    def __init__(self,
                 splits=Defaults.splits, train_size=None, test_size=None,
                 split_coord=Defaults.split_coord, stratification_coord=Defaults.stratification_coord, kfold=False,
                 unique_split_values=Defaults.unique_split_values, random_state=Defaults.random_state,
                 preprocess_indices=None):
        super().__init__()
        if train_size is None and test_size is None:
            train_size = self.Defaults.train_size
        if kfold:
            assert (train_size is None or train_size == self.Defaults.train_size) and test_size is None
            if stratification_coord:
                self._split = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
            else:
                self._split = KFold(n_splits=splits, shuffle=True, random_state=random_state)
        else:
            if stratification_coord:
                self._split = StratifiedShuffleSplit(
                    n_splits=splits, train_size=train_size, test_size=test_size, random_state=random_state)
            else:
                self._split = ShuffleSplit(
                    n_splits=splits, train_size=train_size, test_size=test_size, random_state=random_state)
        self._split_coord = split_coord
        self._stratification_coord = stratification_coord
        self._unique_split_values = unique_split_values
        self._preprocess_indices=preprocess_indices

        self._logger = logging.getLogger(fullname(self))

    @property
    def do_stratify(self):
        return bool(self._stratification_coord)

    def build_splits(self, assembly):
        cross_validation_values, indices = extract_coord(assembly, self._split_coord, unique=self._unique_split_values)
        data_shape = np.zeros(len(cross_validation_values))
        args = [assembly[self._stratification_coord].values[indices]] if self.do_stratify else []
        splits = self._split.split(data_shape, *args)
        return cross_validation_values, list(splits)

    @classmethod
    def aggregate(cls, values):
        score = apply_aggregate(lambda scores: scores.mean('split'), values)
        score.attrs['error'] = standard_error_of_the_mean(values, 'split')
        return score


def extract_coord(assembly, coord, unique=False):
    if not unique:
        coord_values = assembly[coord].values
        indices = list(range(len(coord_values)))
    else:
        # need unique values for when e.g. repetitions are heavily redundant and splits would yield equal unique values
        coord_values, indices = np.unique(assembly[coord].values, return_index=True)
    dims = assembly[coord].dims
    assert len(dims) == 1
    extracted_assembly = xr.DataArray(coord_values, coords={coord: coord_values}, dims=[coord])
    extracted_assembly = extracted_assembly.stack(**{dims[0]: (coord,)})
    return extracted_assembly if not unique else extracted_assembly, indices


class TestOnlyCrossValidationSingle:
    def __init__(self, *args, **kwargs):
        self._cross_validation = CrossValidationSingle(*args, **kwargs)

    def __call__(self, *args, apply, **kwargs):
        apply_wrapper = lambda train, test: apply(test)
        return self._cross_validation(*args, apply=apply_wrapper, **kwargs)


class TestOnlyCrossValidation:
    def __init__(self, *args, **kwargs):
        self._cross_validation = CrossValidation(*args, **kwargs)

    def __call__(self, *args, apply, **kwargs):
        apply_wrapper = lambda train1, train2, test1, test2: apply(test1, test2)
        return self._cross_validation(*args, apply=apply_wrapper, **kwargs)


class CrossValidationSingle(Transformation):
    def __init__(self,
                 splits=Split.Defaults.splits, train_size=None, test_size=None,
                 split_coord=Split.Defaults.split_coord, stratification_coord=Split.Defaults.stratification_coord,
                 unique_split_values=Split.Defaults.unique_split_values, random_state=Split.Defaults.random_state):
        super().__init__()
        self._split = Split(splits=splits, split_coord=split_coord,
                            stratification_coord=stratification_coord, unique_split_values=unique_split_values,
                            train_size=train_size, test_size=test_size, random_state=random_state)
        self._logger = logging.getLogger(fullname(self))

    def pipe(self, assembly):
        """
        :param assembly: the assembly to cross-validate over
        """
        cross_validation_values, splits = self._split.build_splits(assembly)

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

    def aggregate(self, score):
        return self._split.aggregate(score)


class CrossValidation(Transformation):
    """
    Performs multiple splits over a source and target assembly.
    No guarantees are given for data-alignment, use the metadata.
    """

    def __init__(self, *args, split_coord=Split.Defaults.split_coord,
                 stratification_coord=Split.Defaults.stratification_coord,
                 preprocess_indices=None, **kwargs):
        self._split_coord = split_coord
        self._stratification_coord = stratification_coord
        self._split = Split(*args, split_coord=split_coord, stratification_coord=stratification_coord, **kwargs)
        self._logger = logging.getLogger(fullname(self))
        self._preprocess_indices = preprocess_indices
        

    def pipe(self, source_assembly, target_assembly):
        # check only for equal values, alignment is given by metadata
        assert sorted(source_assembly[self._split_coord].values) == sorted(target_assembly[self._split_coord].values)
        if self._split.do_stratify:
            assert hasattr(source_assembly, self._stratification_coord), \
                f"Expected stratification coordinate {self._stratification_coord}"
            assert sorted(source_assembly[self._stratification_coord].values) == \
                   sorted(target_assembly[self._stratification_coord].values)
        cross_validation_values, splits = self._split.build_splits(target_assembly)

        split_scores = []
        for split_iterator, (train_indices, test_indices), done \
                in tqdm(enumerate_done(splits), total=len(splits), desc='cross-validation'):
            
            if hasattr(self, '_preprocess_indices'):
                if self._preprocess_indices is not None:
                    train_indices, test_indices = self._preprocess_indices(train_indices, test_indices, source_assembly)

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

        split_scores = Score.merge(*split_scores)
        yield split_scores

    def aggregate(self, score):
        return self._split.aggregate(score)


def standard_error_of_the_mean(values, dim):
    return values.std(dim) / math.sqrt(len(values[dim]))


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

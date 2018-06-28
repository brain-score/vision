import inspect
import itertools
import logging
import os
import pickle
from collections import defaultdict, OrderedDict
from typing import Union

import numpy as np
import xarray as xr

from mkgu.assemblies import merge_data_arrays
from mkgu.utils import fullname


def get_function_identifier(function, call_args):
    function_identifier = os.path.join(function.__module__ + '.' + function.__name__,
                                       ','.join('{}={}'.format(key, value) for key, value in call_args.items()))
    return function_identifier


class _Storage(object):
    def __init__(self, identifier_ignore=()):
        """
        :param [str] identifier_ignore: function parameters to ignore when building the unique function identifier.
            Different versions of the same parameter will result in the same identifier when ignored.
            Useful when the results do not depend on certain parameters.
        """
        self.identifier_ignore = identifier_ignore
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, function):
        def wrapper(*args, **kwargs):
            call_args = self.getcallargs(function, *args, **kwargs)
            function_identifier = self.get_function_identifier(function, call_args)
            if self.is_stored(function_identifier):
                self._logger.debug("Loading from storage: {}".format(function_identifier))
                return self.load(function_identifier)
            result = function(*args, **kwargs)
            self._logger.debug("Saving to storage: {}".format(function_identifier))
            self.save(result, function_identifier)
            return result

        return wrapper

    def getcallargs(self, function, *args, **kwargs):
        call_args = inspect.getcallargs(function, *args, **kwargs)
        argspec = inspect.getfullargspec(function)
        argspec = argspec.args + \
                  ([argspec.varargs] if argspec.varargs else []) + ([argspec.varkw] if argspec.varkw else [])
        sorting = {arg: i for i, arg in enumerate(argspec)}
        return OrderedDict(sorted(call_args.items(), key=lambda pair: sorting[pair[0]]))

    def get_function_identifier(self, function, call_args):
        call_args = {key: value for key, value in call_args.items() if key not in self.identifier_ignore}
        return get_function_identifier(function, call_args)

    def is_stored(self, function_identifier):
        raise NotImplementedError()

    def load(self, function_identifier):
        raise NotImplementedError()

    def save(self, result, function_identifier):
        raise NotImplementedError()


class _DiskStorage(_Storage):
    def __init__(self, storage_directory=os.path.join(os.path.dirname(__file__), '..', 'output'), identifier_ignore=()):
        super().__init__(identifier_ignore=identifier_ignore)
        self.storage_directory = storage_directory

    def storage_path(self, function_identifier):
        return os.path.join(self.storage_directory, function_identifier + '.pkl')

    def save(self, result, function_identifier):
        path = self.storage_path(function_identifier)
        path_dir = os.path.dirname(path)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir, exist_ok=True)
        savepath_part = path + '.filepart'
        self.save_file(result, savepath_part)
        os.rename(savepath_part, path)

    def save_file(self, result, savepath_part):
        with open(savepath_part, 'wb') as f:
            pickle.dump({'data': result}, f)

    def is_stored(self, function_identifier):
        storage_path = self.storage_path(function_identifier)
        return os.path.isfile(storage_path)

    def load(self, function_identifier):
        path = self.storage_path(function_identifier)
        assert os.path.isfile(path)
        return self.load_file(path)

    def load_file(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)['data']


class _XarrayStorage(_DiskStorage):
    """
    All fields in _combine_fields are combined into one file and loaded lazily
    """

    def __init__(self, combine_fields: Union[list, dict], sub_fields=False,
                 map_field_values=None, map_field_values_inverse=None,
                 *args, **kwargs):
        """
        :param combine_fields: fields to consider as primary keys together with the filename
            (i.e. fields not excluded by `identifier_ignore`).
        :param sub_fields: store the result right away (default, False) or only its sub-fields
        """
        super().__init__(*args, **kwargs)
        if not isinstance(combine_fields, dict):  # use identity mapping if list passed
            self._combine_fields = {field: field for field in combine_fields}
        else:
            self._combine_fields = combine_fields
        self._combine_fields_inverse = {value: key for key, value in self._combine_fields.items()}
        self._sub_fields = sub_fields
        if map_field_values:
            assert map_field_values_inverse
        self._map_field_values = map_field_values or (lambda key, value: value)
        self._map_field_values_inverse = map_field_values_inverse or (lambda key, value: value)

    def __call__(self, function):
        def wrapper(*args, **kwargs):
            call_args = self.getcallargs(function, *args, **kwargs)
            infile_call_args = {self._combine_fields[key]: self._map_field_values(self._combine_fields[key], value)
                                for key, value in call_args.items()
                                if key in self._combine_fields}
            function_identifier = self.get_function_identifier(function, call_args)
            stored_result, reduced_call_args = None, call_args
            if self.is_stored(function_identifier):
                self._logger.debug("Loading from storage: {}".format(function_identifier))
                stored_result = self.load(function_identifier)
                missing_call_args = self.filter_coords(infile_call_args, stored_result) if not self._sub_fields \
                    else self.filter_coords(infile_call_args, getattr(stored_result, next(iter(vars(stored_result)))))
                if len(missing_call_args) == 0:
                    # nothing else to run, but still need to filter
                    result = stored_result
                    reduced_call_args = None
                else:
                    # need to run more args
                    non_variable_call_args = {key: value for key, value in call_args.items()
                                              if key not in self._combine_fields}
                    missing_call_args = {self._combine_fields_inverse[key]: self._map_field_values_inverse(key, value)
                                         for key, value in missing_call_args.items()}
                    reduced_call_args = {**non_variable_call_args, **missing_call_args}
                    self._logger.debug("Computing missing: {}".format(reduced_call_args))
            if reduced_call_args:
                # run function if some args are uncomputed
                result = function(**reduced_call_args)
                if stored_result is not None:
                    result = self.merge_results(stored_result, result)
                # only save if new results
                self._logger.debug("Saving to storage: {}".format(function_identifier))
                self.save(result, function_identifier)
            self.ensure_callargs_present(result, infile_call_args)
            result = self.filter_callargs(result, infile_call_args)
            return result

        return wrapper

    def merge_results(self, stored_result, result):
        if not self._sub_fields:
            result = merge_data_arrays([stored_result, result])
        else:
            for field in vars(result):
                setattr(result, field,
                        merge_data_arrays([getattr(stored_result, field), getattr(result, field)]))
        return result

    def ensure_callargs_present(self, result, infile_call_args):
        # make sure coords are set equal to call_args
        if not self._sub_fields:
            assert len(self.filter_coords(infile_call_args, result)) == 0
        else:
            for field in vars(result):
                assert len(self.filter_coords(infile_call_args, getattr(result, field))) == 0

    def filter_callargs(self, result, callargs):
        # filter to what function was called with
        if not self._sub_fields:
            result = self.filter_data(result, callargs)
        else:
            for field in vars(result):
                setattr(result, field, self.filter_data(getattr(result, field), callargs))
        return result

    def filter_coords(self, call_args, result):
        for key, value in call_args.items():
            assert is_iterable(value)
        combinations = [dict(zip(call_args, values)) for values in itertools.product(*call_args.values())]
        uncomputed_combinations = []
        for combination in combinations:
            combination_result = result
            combination_result = self.filter_data(combination_result, combination, single_value=True)
            if combination_result.size == 0:
                uncomputed_combinations.append(combination)
        if len(uncomputed_combinations) == 0:
            return {}
        return self._combine_call_args(uncomputed_combinations)

    def filter_data(self, data, coords, single_value=False):
        for coord, coord_value in coords.items():
            if not hasattr(data, coord):
                raise ValueError("Coord not found in data: {}".format(coord))
            # when called with a combination instantiation, coord_value will be a single item; iterable for check
            indexer = np.array([(val == coord_value) if single_value or not is_iterable(coord_value)
                                else (val in coord_value) for val in data[coord].values])
            coord_dims = data[coord].dims
            dim_indexes = {dim: slice(None) if dim not in coord_dims else np.where(indexer)[0]
                           for dim in data.dims}
            data = data.isel(**dim_indexes)
        data = data.sortby([self._build_sort_array(coord, coord_value, data)
                            for coord, coord_value in coords.items()
                            if is_iterable(coord_value) and len(coord_value) > 1])
        return data

    def _combine_call_args(self, uncomputed_combinations):
        call_args = defaultdict(list)
        for combination in uncomputed_combinations:
            for key, value in combination.items():
                call_args[key].append(value)
        return call_args

    def _build_sort_array(self, coord, coord_value, data):
        dims = data[coord].dims
        assert len(dims) == 1
        s = xr.DataArray(list(range(len(coord_value))), [(coord, coord_value)])
        if dims[0] == 'layer':
            return s
        return s.stack(**{dims[0]: [coord]})


class _MemoryStorage(_Storage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = dict()

    def save(self, result, function_identifier):
        self.cache[function_identifier] = result

    def is_stored(self, function_identifier):
        return function_identifier in self.cache

    def load(self, function_identifier):
        return self.cache[function_identifier]


def is_iterable(x):
    try:
        iter(x)
        if isinstance(x, str):
            return False
        return True
    except TypeError:
        return False


cache = _MemoryStorage
store = _DiskStorage
store_xarray = _XarrayStorage

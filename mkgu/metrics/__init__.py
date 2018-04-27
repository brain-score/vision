from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
from abc import ABCMeta, abstractmethod

import numpy as np

from mkgu.assemblies import DataAssembly


class Benchmark(object):
    """a Benchmark represents the application of a Metric to a specific set of data.  """

    def __init__(self, metric, target_assembly):
        """
        :param Metric metric:
        :param target_assembly:
        """
        self._metric = metric
        self._target_assembly = target_assembly

    def __call__(self, source_assembly):
        return self._metric(self._target_assembly, source_assembly)


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
    def __call__(self, source_assembly, target_assembly, similarity_dim='neuroid'):
        """
        :param mkgu.assemblies.NeuroidAssembly source_assembly:
        :param mkgu.assemblies.NeuroidAssembly target_assembly:
        :param str similarity_dim: the dimension in both assemblies along which the similarity is to be computed
        :return: mkgu.assemblies.DataAssembly
        """

        # compute similarities over `similarity_dim`, i.e. across all adjacent coords
        def iter_indices(assembly):
            dim_indices = np.where(np.array(assembly.dims) == similarity_dim)[0]

            def insert_slice(combination):
                for dim_index in dim_indices:
                    combination.insert(dim_index, slice(None))
                return combination

            adjacent_dims = [dim for dim in assembly.dims if dim != similarity_dim]
            combinations = itertools.product(*[list(range(len(assembly[dim]))) for dim in adjacent_dims])
            return [insert_slice(list(combination)) for combination in combinations]

        indices1, indices2 = iter_indices(source_assembly), iter_indices(target_assembly)
        similarities = np.array([
            self.apply(source_assembly.values[ids1], target_assembly.values[ids2])
            for ids1, ids2 in itertools.product(indices1, indices2)])

        # package in assembly
        coords = list(source_assembly.coords.keys()) + list(target_assembly.coords.keys())
        duplicate_coords = [coord for coord in coords if coords.count(coord) > 1]
        coords = {**self._collect_coords(source_assembly, duplicate_coords, ignore_coord=similarity_dim, kind='left'),
                  **self._collect_coords(target_assembly, duplicate_coords, ignore_coord=similarity_dim, kind='right')}
        dims = {**self._collect_dim_shapes(source_assembly, duplicate_coords, ignore_dim=similarity_dim, kind='left'),
                **self._collect_dim_shapes(target_assembly, duplicate_coords, ignore_dim=similarity_dim, kind='right')}
        similarities = similarities.reshape(list(itertools.chain(*dims.values())))
        return DataAssembly(similarities, coords=coords, dims=dims.keys())

    def _collect_coords(self, assembly, rename_coords, ignore_coord, kind):
        coord_names = {coord: coord if coord not in rename_coords else coord + '-' + kind
                       for coord in assembly.coords}
        return {coord_names[coord]: (tuple(coord_names[dim] for dim in values.dims), values.values)
                for coord, values in assembly.coords.items() if coord is not ignore_coord}

    def _collect_dim_shapes(self, assembly, rename_coords, ignore_dim, kind):
        return {dim if dim not in rename_coords else dim + '-' + kind:
                    assembly[dim].shape for dim in assembly.dims if dim is not ignore_dim}

    @abstractmethod
    def apply(self, source_assembly, target_assembly):
        raise NotImplementedError()


class Characterization(object, metaclass=ABCMeta):
    """A Characterization contains a chain of numerical operations to be applied to a set of
    data to highlight some aspect of the data.  """

    @abstractmethod
    def __call__(self, assembly):
        raise NotImplementedError()

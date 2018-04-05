from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from abc import ABCMeta, abstractmethod

import xarray as xr


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
        return self._metric(source_assembly=source_assembly, target_assembly=self._target_assembly)


class Metric(object):
    def __init__(self, similarity, characterization=None):
        """
        :param Similarity similarity:
        :param Characterization characterization:
        """
        self._similarity = similarity
        self._characterization = characterization or (lambda x: x)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, source_assembly, target_assembly):
        scores = []
        regions = np.unique(target_assembly.region.values)
        for region in regions:  # TODO: all combinations of regions in source and target? (region=layer)
            self._logger.debug("Scoring region {}".format(region))
            region_assembly = target_assembly.sel(region=region)
            score = self.apply(source_assembly, region_assembly)
            scores.append(score)
        return xr.DataArray(scores, coords={'region': regions}, dims=['region'])

    def apply(self, source_assembly, target_assembly):
        characterized_source = self._characterization(source_assembly)
        characterized_target = self._characterization(target_assembly)
        return self._similarity(characterized_source, characterized_target)


class Similarity(object, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, source_assembly, target_assembly):
        raise NotImplementedError()


class Characterization(object, metaclass=ABCMeta):
    """A Characterization contains a chain of numerical operations to be applied to a set of
    data to highlight some aspect of the data.  """

    @abstractmethod
    def __call__(self, assembly):
        raise NotImplementedError()

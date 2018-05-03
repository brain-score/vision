from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABCMeta, abstractmethod


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
    @abstractmethod
    def __call__(self, source_assembly, target_assembly):
        raise NotImplementedError()


class Characterization(object, metaclass=ABCMeta):
    """A Characterization contains a chain of numerical operations to be applied to a set of
    data to highlight some aspect of the data.  """

    @abstractmethod
    def __call__(self, assembly):
        raise NotImplementedError()

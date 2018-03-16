from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABCMeta, abstractmethod


class Benchmark(object):
    """a Benchmark represents the application of a Metric to a specific set of data.  """

    def __init__(self, metric, assembly):
        """
        :param Metric metric:
        :param assembly:
        """
        self._metric = metric
        self._assembly = assembly

    def apply(self, comparison_assembly):
        return self._metric.apply(self._assembly, comparison_assembly)


class Metric(object):
    def __init__(self, similarity, characterization=None):
        """
        :param Similarity similarity:
        :param Characterization characterization:
        """
        self._similarity = similarity
        self._characterization = characterization or IdentityCharacterization()

    def apply(self, assembly1, assembly2):
        characterized_assembly1 = self._characterization.apply(assembly1)
        characterized_assembly2 = self._characterization.apply(assembly2)
        return self._similarity.apply(characterized_assembly1, characterized_assembly2)


class Similarity(object, metaclass=ABCMeta):
    @abstractmethod
    def apply(self, assembly1, assembly2):
        raise NotImplementedError()


class Characterization(object, metaclass=ABCMeta):
    """A Characterization contains a chain of numerical operations to be applied to a set of
    data to highlight some aspect of the data.  """

    @abstractmethod
    def apply(self, assembly):
        raise NotImplementedError()


class IdentityCharacterization(Characterization):
    def apply(self, assembly):
        return assembly

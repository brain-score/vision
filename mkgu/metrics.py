from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

class Metric(object):
    """A Metric contains a chain of numerical operations to be applied to a set of
    neuroscience data to produce a value which quantifies some aspect of the
    system from which the data were recorded.  """
    def __init__(self):
        pass

    def apply(self, assembly):
        return 0


class Benchmark(object):
    """a Benchmark represents the application of a Metric to a specific set of data.  """
    def __init__(self, metric, assembly):
        self.metric = metric
        self.assembly = assembly

    def calculate(self):
        return self.metric.apply(self.assembly)


class RDM(Metric):
    """A Metric for Representational Dissimilarity Matrix.  """
    def __init__(self, **kwargs):
        super(RDM, self).__init__(**kwargs)

    def apply(self, assembly):
        return np.corrcoef(assembly)


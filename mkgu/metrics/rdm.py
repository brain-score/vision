import numpy as np

from mkgu.metrics import Characterization


class RSA(Characterization):
    """A Metric for Representational Similarity Matrix.  """

    def __init__(self, **kwargs):
        super(RSA, self).__init__(**kwargs)

    def apply(self, assembly):
        return np.corrcoef(assembly)


class RDM(RSA):
    def apply(self, assembly):
        return 1 - super(RDM, self).apply(assembly)

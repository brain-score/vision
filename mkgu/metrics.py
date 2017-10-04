from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import scipy.stats

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA


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


def neural_fits(model_feats, neural, labels, n_splits=10, n_components=200, test_size=.25):
    if model_feats.shape[1] > n_components:
        model_feats = PCA(n_components=n_components).fit_transform(model_feats)
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    df = []
    for it, (train_idx, test_idx) in enumerate(skf.split(model_feats, labels)):
        reg = PLSRegression(n_components=25, scale=False)
        reg.fit(model_feats[train_idx], neural[train_idx])
        pred = reg.predict(model_feats[test_idx])
        rs = pearsonr_matrix(neural[test_idx], pred)
        df.extend([(it, site, r) for site, r in enumerate(rs)])
    df = pd.DataFrame(df, columns=['split', 'site', 'fit_r'])
    return df


def pearsonr_matrix(data1, data2, axis=1):
    rs = []
    for i in range(data1.shape[axis]):
        d1 = np.take(data1, i, axis=axis)
        d2 = np.take(data2, i, axis=axis)
        r, p = scipy.stats.pearsonr(d1, d2)
        rs.append(r)
    return np.array(rs)
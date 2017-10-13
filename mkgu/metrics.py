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
        import ipdb; ipdb.set_trace()
        return np.corrcoef(assembly)


def neural_fits(neural,
                model,
                labels,
                sortby=None,
                n_splits=10,
                n_components=200,
                test_size=.25):

    if 'image_id' not in model.presentation: #.indexes['presentation'].names:
        raise ValueError('model must have id')
    if 'image_id' not in neural.indexes['presentation'].names:
        raise ValueError('neural must have id')
    # if model.presentation.size != neural.presentation.size:
    #     raise ValueError('Model features and neural features have an unequal number of features.')

    neural.load()

    if sortby is None:
        sortby = 'image_id'
    neural = neural.sortby('image_id')
    model = model.sortby('image_id')

    neural = neural.groupby('image_id').mean(dim='presentation').squeeze('time_bin').T

    if not np.all(neural['image_id'] == model['image_id']):
        raise ValueError('Image ids do not match')

    if model.coords['neuroid'].size > n_components:
        model = PCA(n_components=n_components).fit_transform(model)

    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    df = []
    for it, (train_idx, test_idx) in enumerate(skf.split(model, model[labels])):
        reg = PLSRegression(n_components=25, scale=False)
        reg.fit(model[train_idx], neural[train_idx])
        pred = reg.predict(model[test_idx])
        rs = pearsonr_matrix(neural[test_idx], pred)
        df.extend([(it, site, r) for site, r in enumerate(rs)])
    df = pd.DataFrame(df, columns=['split', 'site', 'fit_r'])
    return df


def internal_cons(data, niter=10, seed=None):
    rng = np.random.RandomState(seed)
    df = []
    for i in range(niter):
        split1, split2 = splithalf(data, rng=rng)
        r = pearsonr_matrix(split1, split2)
        rc = spearman_brown_correct(r, n=2)
        df.extend([(i, site, rci) for site, rci in enumerate(rc)])
    df = pd.DataFrame(df, columns=['splithalf', 'site', 'internal_cons'])
    return df


def pearsonr_matrix(data1, data2, axis=1):
    rs = []
    for i in range(data1.shape[axis]):
        d1 = np.take(data1, i, axis=axis)
        d2 = np.take(data2, i, axis=axis)
        r, p = scipy.stats.pearsonr(d1, d2)
        rs.append(r)
    return np.array(rs)


def splithalf(data, aggfunc=np.nanmean, rng=None):
    data = np.array(data)
    if rng is None:
        rng = np.random.RandomState(None)
    inds = list(range(data.shape[0]))
    rng.shuffle(inds)
    half = len(inds) // 2
    split1 = aggfunc(data[inds[:half]], axis=0)
    split2 = aggfunc(data[inds[half:2*half]], axis=0)
    return split1, split2


def spearman_brown_correct(pearsonr, n=2):
    pearsonr = np.array(pearsonr)
    return n * pearsonr / (1 + (n-1) * pearsonr)
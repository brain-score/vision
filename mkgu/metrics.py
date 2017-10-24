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
                time_bins=None,
                reg=None,
                cv=None,
                # feat_sel=None,
                sortby='id'
                ):
    """
    Args:
        - neural (DataAssembly)
          Three-dimensional DataAssembly object: neuroid x presentation x time_bin
        - model (DataAssembly)
          Two-dimensional DataAssembly object: neuroid x presentation

    Kwargs:
        - time_bins (int or list)
          Which time bins to use
        - reg
          Regression model from sklearn. If None,
          PLSRegression(n_components=25, scale=False) is used.
        - cv (list)
          Cross-validation strategy, i.e., list of train-test indices.
          If None, StratifiedShuffleSplit(n_splits=10, test_size=.25) is used
    Returns:
        pandas.DataFrame with time bin, split number, site number and
        a Pearson r of the correlation between model predictions and neural data.
    """

    if 'image_id' not in neural.indexes['presentation'].names:
        raise ValueError('neural must have id')
    if 'image_id' not in model.indexes['presentation'].names:
        raise ValueError('model must have id')

    neural.load()

    neural = neural.sortby('image_id')
    model = model.sortby('image_id')

    neural = neural.multi_groupby('image_id').mean(dim='presentation')

    if not np.all(neural['image_id'].values == model['image_id'].values):
        raise ValueError('Image ids do not match')

    if time_bins is None:
        time_bins = np.unique(neural['time_bin_center'])
    elif not isinstance(time_bins, (tuple, list)):
        time_bins = [time_bins]

    if reg is None:
        reg = PLSRegression(n_components=25, scale=False)

    if cv is None:
        skf = StratifiedShuffleSplit(n_splits=10, test_size=.25)
        cv = list(skf.split(model, model['object']))

    dfs = []
    for time in time_bins:
        for it, (train_idx, test_idx) in enumerate(cv):
            reg.fit(model.iloc[train_idx], neural.sel(time_bin_center=time).iloc[train_idx])
            pred = reg.predict(model.iloc[test_idx])
            rs = pearsonr_matrix(neural.sel(time_bin_center=time).iloc[test_idx], pred)
            df = pd.DataFrame([rs], columns=['fit_r'])
            df['time'] = time
            df['split'] = it
            df['neuroid_id'] = neural['neuroid_id']
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df


def internal_cons(neural,
                  time_bins=None,
                  cv=None,
                  niter=10,
                  seed=None
                  ):
    neural.load()

    if time_bins is None:
        time_bins = np.unique(neural['time_bin_center'])
    elif not isinstance(time_bins, (tuple, list)):
        time_bins = [time_bins]

    if cv is None:
        skf = StratifiedShuffleSplit(n_splits=10, test_size=.25)
        cv = list(skf.split(neural.mean(dim='presentation'), neural['object']))

    dfs = []
    for time in time_bins:
        for it, (train_idx, test_idx) in enumerate(cv):
            rng = np.random.RandomState(seed)
            for i in range(niter):
                split1, split2 = splithalf(neural.sel(time_bin_center=time).iloc[:, test_idx], rng=rng)
                r = pearsonr_matrix(split1, split2)
                rc = spearman_brown_correct(r, n=2)
                df = pd.DataFrame([rc], columns=['internal_cons'])
                df['time'] = time
                df['split'] = it
                df['splithalf'] = i
                df['neuroid_id'] = neural['neuroid_id']
                dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
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
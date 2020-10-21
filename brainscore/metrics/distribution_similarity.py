import numpy as np
from brainscore.metrics import Metric, Score

NSAMPLES = 1000


class BootstrapDistributionSimilarity(Metric):
    def __init__(self, similarity_func, property_name, ns=NSAMPLES):
        self.similarity_func = similarity_func
        self.property_name = property_name
        self.ns = ns

    def __call__(self, model_property, data_property):
        bins = data_property.attrs[self.property_name+'_bins']
        data_property = data_property.loc[:, self.property_name].values
        model_property = model_property.loc[:, self.property_name].values

        data_property[data_property < bins[0]] = bins[0]
        data_property[data_property > bins[-1]] = bins[-1]
        model_property[model_property < bins[0]] = bins[0]
        model_property[model_property > bins[-1]] = bins[-1]

        n_neurons = data_property.shape[0]

        data_hist = np.histogram(data_property, bins=bins)[0]
        data_hist = data_hist / data_hist.sum()

        model_hist = np.zeros((self.ns, data_hist.shape[0]))
        dist_similarity = np.zeros(self.ns)

        for s in range(self.ns):
            sample = np.random.choice(model_property, n_neurons)
            model_hist[s, :] = np.histogram(sample, bins=bins)[0]
            model_hist[s, :] = model_hist[s, :] / model_hist[s, :].sum()
            dist_similarity[s] = self.similarity_func(model_hist[s, :], data_hist)

        if np.isnan(dist_similarity).sum() > NSAMPLES / 10:
            center = 0
            error = 0
        else:
            center = np.nanmean(dist_similarity)
            error = np.nanstd(dist_similarity)

        score = Score([center, error], coords={'aggregation': ['center', 'error']}, dims=('aggregation',))
        score.attrs[Score.RAW_VALUES_KEY] = dist_similarity

        return score


def ks_similarity(p, q):
    z1 = np.zeros_like(p)
    z2 = np.zeros_like(p)
    z1[0] = 1
    z2[-1] = 1

    kolm_raw = np.max(np.abs(np.cumsum(p)-np.cumsum(q)))
    kolm_max1 = np.max(np.abs(np.cumsum(z1)-np.cumsum(q)))
    kolm_max2 = np.max(np.abs(np.cumsum(z2)-np.cumsum(q)))
    kolm_max = np.max(np.array([kolm_max1, kolm_max2]))
    kolm_ceiled = kolm_raw / kolm_max

    return 1 - kolm_ceiled


import numpy as np


def gen_sample(hist, bins, scale='linear'):
    random_sample = np.empty([0,1])

    if scale == 'log2':
        bins = np.log2(bins)
    if scale == 'log10':
        bins = np.log10(bins)

    for b_ind, b in enumerate(bins[:-1]):
        random_sample = np.concatenate((random_sample, (np.random.random_sample(hist[b_ind]) *
                                        (bins[b_ind+1]-bins[b_ind]) + bins[b_ind]).reshape((-1, 1))), axis=0)
    if scale == 'log2':
        random_sample = 2 ** random_sample
    if scale == 'log10':
        random_sample = 10 ** random_sample

    return random_sample

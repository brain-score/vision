import numpy as np
from scipy.stats import spearmanr, pearsonr

def compute_spearman_rho_pair(activations):
    """ Computes the spearman corrleation between the activations """
    return spearmanr(activations[0].ravel(), activations[1].ravel())[0]

def compute_pearson_r_pair(activations):
    """ Computes the spearman correlation between the activations """
    return pearsonr(activations[0].ravel(), activations[1].ravel())[0]

def compute_l2_pair(activations):
    """ Computes the l2 distance between the activation """
    return np.linalg.norm(activations[0].ravel()-activations[1].ravel())

def compute_snr_db(activations):
    """
    Computes the dB SNR, treating the first activation as the "signal" and
    the difference as the "noise"
    """
    norm_signal = np.linalg.norm(activations[0].ravel())
    norm_noise = np.linalg.norm(activations[0].ravel()-activations[1].ravel())
    dBSNR = 10 * np.log10(norm_signal / norm_noise)
    return dBSNR, norm_signal, norm_noise

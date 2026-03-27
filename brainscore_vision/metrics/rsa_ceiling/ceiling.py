import numpy as np
from scipy.stats import spearmanr

from brainscore_core.metrics import Score
from brainscore_core.supported_data_standards.brainio.assemblies import DataAssembly
from brainscore_vision.metrics import Ceiling
from brainscore_vision.metrics.rdm.metric import RDM


class RSACeiling(Ceiling):
    """Inter-subject RDM consistency ceiling (Spearman).

    Computes two ceiling estimates and returns the upper bound (Nili et al.,
    2014) for ceiling-normalization, with the LOO lower bound stored in
    ``score.attrs['lower_bound_loo']``.

    Upper bound: each subject's RDM vs. the mean RDM across *all* subjects
    (including itself).  This is a hard upper limit that structurally matches
    model evaluation and guarantees ceiled scores <= 1.

    Lower bound (LOO): each subject's RDM vs. the mean of the remaining N-1
    subjects.  Biased low for finite subject counts because the mean-of-(N-1)
    is a noisier estimate of the shared signal.

    Unlike InternalConsistency,no Spearman-Brown correction is applied.  
    InternalConsistency splits a single subject's repetitions into halves, so
    each half underestimates the full-data reliability and the correction 
    compensates for that.  Here, each subject's RDM is already computed 
    from all their data (all voxels, all stimuli, repetitions already 
    averaged), so there is no split-induced underestimation to correct for.

    Requires the assembly to have a ``subject`` coordinate on the neuroid
    dimension.

    :param rdm: callable that converts a (presentation x neuroid) assembly
        into a dissimilarity matrix.  Defaults to :class:`RDM`.
    :param similarity_func: function(array, array) -> (correlation, pvalue).
        Defaults to :func:`scipy.stats.spearmanr`.
    """

    def __init__(self, rdm: RDM = None, similarity_func=None):
        self._rdm = rdm or RDM()
        self._similarity_func = similarity_func or spearmanr

    def __call__(self, assembly: DataAssembly) -> Score:
        subjects = np.unique(assembly['subject'].values)

        subject_rdms = {}
        for subject in subjects:
            neural_subj = assembly.sel(neuroid=assembly['subject'] == subject)
            subject_rdms[subject] = self._rdm(neural_subj).values

        rdm_stack = np.array([subject_rdms[s] for s in subjects])
        mean_all_rdm = np.mean(rdm_stack, axis=0)
        mask = np.triu(np.ones_like(mean_all_rdm, dtype=bool), k=1)

        loo_correlations = []
        upper_correlations = []
        for i, subject in enumerate(subjects):
            subj_triu = subject_rdms[subject][mask]

            # LOO: mean of N-1 others (lower bound)
            other_rdms = np.delete(rdm_stack, i, axis=0)
            mean_other_rdm = np.mean(other_rdms, axis=0)
            loo_corr, _ = self._similarity_func(subj_triu, mean_other_rdm[mask])
            loo_correlations.append(loo_corr)

            # Upper: mean of all N subjects (Nili et al., 2014)
            upper_corr, _ = self._similarity_func(subj_triu, mean_all_rdm[mask])
            upper_correlations.append(upper_corr)

        upper = Score(np.mean(upper_correlations))
        lower = Score(np.mean(loo_correlations))

        upper.attrs['lower_bound_loo'] = float(lower)
        return upper

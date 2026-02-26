import numpy as np
from scipy.stats import spearmanr

from brainscore_core.metrics import Score
from brainscore_core.supported_data_standards.brainio.assemblies import DataAssembly
from brainscore_vision.metrics import Ceiling
from brainscore_vision.metrics.rdm.metric import RDM


class RSACeiling(Ceiling):
    """Leave-one-out inter-subject RDM consistency (Spearman).

    For each held-out subject, computes the Spearman rank correlation between
    that subject's RDM upper triangle and the mean of all other subjects'
    RDM upper triangles.  Returns the mean correlation across subjects.

    Unlike InternalConsistency,no Spearman-Brown correction is applied.  
    InternalConsistency splits a single subject's repetitions into halves, so
    each half underestimates the full-data reliability and the correction 
    compensates for that.  Here, each subject's RDM is already computed 
    from all their data (all voxels, all stimuli, repetitions already 
    averaged), so there is no split-induced underestimation to correct for.

    The ceiling value increases with subject count because the mean-of-(N-1)
    RDM becomes more stable with more subjects.  This reflects genuinely better
    estimation of the shared representational structure.

    RSA ceilings are not comparable to ridge ceilings.  RSA ceilings tend to
    be high (e.g. ~0.8 for IT) because categorical structure is consistent
    across subjects; ridge ceilings reflect per-voxel signal reliability and
    are typically lower (e.g. ~0.4).

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

        correlations = []
        for subject in subjects:
            other_rdms = [subject_rdms[s] for s in subjects if s != subject]
            mean_other_rdm = np.mean(other_rdms, axis=0)

            mask = np.triu(np.ones_like(subject_rdms[subject], dtype=bool), k=1)
            subj_triu = subject_rdms[subject][mask]
            other_triu = mean_other_rdm[mask]

            corr, _ = self._similarity_func(subj_triu, other_triu)
            correlations.append(corr)

        return Score(np.mean(correlations))

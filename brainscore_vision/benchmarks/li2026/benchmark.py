import numpy as np

from brainscore_core import Score
from brainscore_vision import load_metric, load_dataset
from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, RSABenchmark
from brainscore_vision.metrics.regression_correlation.metric import (
    CrossRegressedCorrelation, dual_ridge_cv_regression, pearsonr_correlation,
)

VISUAL_DEGREES = 11
NUMBER_OF_TRIALS = 1
RELIABILITY_THRESHOLD = 0.4

BIBTEX = """@article{li2026triplen,
    title = {Triple-N dataset: large-scale fMRI-guided dense recordings of nonhuman
             primate neural responses to natural scenes},
    author = {Li, Yipeng and Liu, Xieyi and Li, Wanru and Yang, Jia and Gong, Baoqi
              and Jin, Wei and Gong, Zhengxin and Wang, Kesheng and Luo, Jingqiu
              and Zhao, Zishuo and Bao, Pinglei},
    journal = {Nature Neuroscience},
    year = {2026},
    doi = {10.1038/s41593-026-02322-z},
    url = {https://doi.org/10.1038/s41593-026-02322-z},
}"""

# Natural scenes have no single object category, so cross-validate without stratification.
_CV_KWARGS = dict(stratification_coord=None)


def _metric(metric_type: str):
    if metric_type == 'ridgecv':  # only registered for the train/test-split path, so build it here
        # dual form: predicts through an (n_test, n_train) projection rather than a
        # (n_targets, n_features) coefficient matrix, which at IT's ~21k units dominates peak memory
        return CrossRegressedCorrelation(regression=dual_ridge_cv_regression(),
                                         correlation=pearsonr_correlation(),
                                         crossvalidation_kwargs=_CV_KWARGS)
    return load_metric(metric_type, crossvalidation_kwargs=_CV_KWARGS)


def _reliability_ceiling(assembly, coord: str = 'reliability', n_bootstraps: int = 1000) -> Score:
    """Noise ceiling: median across neuroids of the per-neuroid split-half reliability,
    with a bootstrap error term.

    Reliability is a Spearman-Brown-corrected split-half correlation (``2r / (1 + r)``):
    a unit's 1000-image response profile is split in half, the halves correlated, and the
    result SB-corrected -- the full-data reliability expected if all trials were included.

    The ceiling must be measured on the SAME response the metric scores. The static
    benchmark scores the fixed 70-170 ms window, so it ceils on ``reliability_window``
    (split-half SB recomputed at 70-170 ms; see build_li2026_reliability_70_170ms.py).
    ``reliability`` (paper-canonical best-window) is retained on the assembly as provenance.

    :param assembly: reliability-filtered neural assembly.
    :param coord: which reliability coord to use (``reliability`` or ``reliability_window``).
    :param n_bootstraps: number of neuroid resamples for the error term.
    :return: scalar ceiling Score; ``attrs['error']`` is the bootstrap SE of the median
        (persisted as ``BenchmarkInstance.ceiling_error``) and ``attrs['raw']`` holds the
        per-neuroid reliabilities.
    """
    rel = assembly[coord].values
    finite = rel[np.isfinite(rel)]
    rng = np.random.RandomState(0)
    boot = [np.median(rng.choice(finite, size=finite.size, replace=True)) for _ in range(n_bootstraps)]
    ceiling = Score(float(np.nanmedian(rel)))
    ceiling.attrs['error'] = float(np.std(boot))
    ceiling.attrs['raw'] = Score(rel, coords={'neuroid_id': ('neuroid', assembly['neuroid_id'].values)},
                                 dims=['neuroid'])
    return ceiling


def load_assembly(region: str):
    assembly = load_dataset('Li2026')
    stimulus_set = assembly.attrs['stimulus_set']
    assembly = assembly.squeeze('time_bin')
    # Select on the window-matched reliability (70-170 ms), since that is the response scored.
    mask = (assembly['region'].values == region) & (assembly['reliability_window'].values > RELIABILITY_THRESHOLD)
    assembly = assembly.isel(neuroid=np.where(mask)[0])  # region already uniform after this filter
    assembly.load()
    assembly = assembly.transpose('presentation', 'neuroid', ...)
    assembly.attrs['stimulus_set'] = stimulus_set
    return assembly


def _Li2026Region(region: str, metric_type: str, parent: str = None) -> NeuralBenchmark:
    assembly = load_assembly(region)
    return NeuralBenchmark(
        identifier=f'Li2026.{region}-{metric_type}', version=1,
        assembly=assembly, similarity_metric=_metric(metric_type),
        visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
        ceiling_func=lambda: _reliability_ceiling(assembly, coord='reliability_window'),
        parent=parent, bibtex=BIBTEX)


# Scored leaves: one per region, so Li2026 contributes each unit population to the
# tree once. RidgeCV matches the other high-neuroid-count datasets (Allen2022,
# Hebart2023, Papale2025); no PLS benchmark in the suite runs above ~1k neuroids.
def Li2026V1RidgeCV(): return _Li2026Region('V1', 'ridgecv', parent='V1')
def Li2026V2RidgeCV(): return _Li2026Region('V2', 'ridgecv', parent='V2')
def Li2026V4RidgeCV(): return _Li2026Region('V4', 'ridgecv', parent='V4')
def Li2026ITRidgeCV(): return _Li2026Region('IT', 'ridgecv', parent='IT')


# Unparented: runnable for MajajHong comparability, but out of the scored tree.
def Li2026V1PLS(): return _Li2026Region('V1', 'pls')
def Li2026V2PLS(): return _Li2026Region('V2', 'pls')
def Li2026V4PLS(): return _Li2026Region('V4', 'pls')
def Li2026ITPLS(): return _Li2026Region('IT', 'pls')


def load_rsa_assembly(region: str):
    """Same units as :func:`load_assembly`, with ``animal`` exposed as ``subject``.

    :class:`RSABenchmark` builds one RDM per subject and averages, so it reads a
    ``subject`` coord. Li2026 records the animal as ``animal``. It is added as a
    MultiIndex level rather than a plain coord, matching the Allen2022 and
    laion_fmri convention: levels are invisible to ``assembly.coords.items()``,
    which keeps the RDM metric's coord filtering out of the picture.

    :param region: brain region to load.
    :return: assembly whose neuroid index carries a ``subject`` level.
    """
    assembly = load_assembly(region)
    stimulus_set = assembly.attrs['stimulus_set']
    animals = assembly['animal'].values
    levels = [level for level in assembly.indexes['neuroid'].names if level != 'neuroid']
    assembly = assembly.reset_index('neuroid')
    assembly = assembly.assign_coords(subject=('neuroid', animals))
    assembly = assembly.set_index(neuroid=levels + ['subject'])
    assembly.attrs['stimulus_set'] = stimulus_set
    return assembly


# IT only: RSACeiling compares each subject against the mean of all, so it needs
# several. IT has 5 animals; V1/V2/V4 have 2, where the leave-one-out bound
# degenerates to a single A-vs-B correlation and the ceiling is not estimable.
def Li2026ITRDM() -> RSABenchmark:
    return RSABenchmark(
        identifier='Li2026.IT-rdm', version=1,
        assembly=load_rsa_assembly('IT'), region='IT',
        visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
        parent='IT', bibtex=BIBTEX)



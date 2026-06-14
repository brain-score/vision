import numpy as np

from brainscore_core import Score
from brainscore_vision import load_metric, load_dataset
from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark

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
}"""

# Natural scenes have no single object category, so cross-validate without stratification.
_CV_KWARGS = dict(stratification_coord=None)


def _metric(metric_type: str):
    return load_metric(metric_type, crossvalidation_kwargs=_CV_KWARGS)


def _reliability_ceiling(assembly) -> Score:
    """Population split-half reliability = median of per-neuroid reliability_best
    (the dataset's split-half Spearman-Brown values). This matches what
    InternalConsistency computes from repetitions (median across neuroids), since
    reliability_best is exactly the per-neuroid split-half SB reliability.
    Replaced by the canonical repetition-based ceiling once trial data is packaged."""
    rel = assembly['reliability'].values
    ceiling = Score(float(np.median(rel)))
    ceiling.attrs['raw'] = Score(rel, coords={'neuroid_id': ('neuroid', assembly['neuroid_id'].values)},
                                 dims=['neuroid'])
    return ceiling


def load_assembly(region: str):
    assembly = load_dataset('Li2026')
    stimulus_set = assembly.attrs['stimulus_set']
    assembly = assembly.squeeze('time_bin')
    mask = (assembly['region'].values == region) & (assembly['reliability'].values > RELIABILITY_THRESHOLD)
    assembly = assembly.isel(neuroid=np.where(mask)[0])  # region already uniform after this filter
    assembly.load()
    assembly = assembly.transpose('presentation', 'neuroid', ...)
    assembly.attrs['stimulus_set'] = stimulus_set
    return assembly


def _Li2026Region(region: str, metric_type: str) -> NeuralBenchmark:
    assembly = load_assembly(region)
    return NeuralBenchmark(
        identifier=f'Li2026.{region}-{metric_type}', version=1,
        assembly=assembly, similarity_metric=_metric(metric_type),
        visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
        ceiling_func=lambda: _reliability_ceiling(assembly),
        parent=region, bibtex=BIBTEX)


def Li2026V1PLS():   return _Li2026Region('V1', 'pls')
def Li2026V2PLS():   return _Li2026Region('V2', 'pls')
def Li2026V4PLS():   return _Li2026Region('V4', 'pls')
def Li2026ITPLS():   return _Li2026Region('IT', 'pls')
def Li2026V1Ridge(): return _Li2026Region('V1', 'ridge')
def Li2026V2Ridge(): return _Li2026Region('V2', 'ridge')
def Li2026V4Ridge(): return _Li2026Region('V4', 'ridge')
def Li2026ITRidge(): return _Li2026Region('IT', 'ridge')


def load_temporal_assembly(region: str):
    assembly = load_dataset('Li2026.temporal')
    stimulus_set = assembly.attrs['stimulus_set']
    # IT-session units outside a named patch ('IT-other') are still IT, matching the
    # static assembly convention (all IT-session units pooled as 'IT').
    regions = np.where(assembly['region'].values == 'IT-other', 'IT', assembly['region'].values)
    mask = (regions == region) & (assembly['reliability'].values > RELIABILITY_THRESHOLD)
    assembly = assembly.isel(neuroid=np.where(mask)[0])
    assembly.load()
    # make the region coord uniform (fold IT-other into IT) while preserving the neuroid index
    assembly = assembly.reset_index('neuroid')
    assembly = assembly.assign_coords(region=('neuroid', [region] * assembly.sizes['neuroid']))
    neuroid_coords = [c for c in assembly.coords if assembly[c].dims == ('neuroid',)]
    assembly = assembly.set_index(neuroid=neuroid_coords)
    assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')
    assembly.attrs['stimulus_set'] = stimulus_set
    return assembly


def _Li2026RegionTemporal(region: str) -> NeuralBenchmark:
    assembly = load_temporal_assembly(region)
    return NeuralBenchmark(
        identifier=f'Li2026.{region}-temporal-pls', version=1,
        assembly=assembly, similarity_metric=load_metric('spantime_pls', crossvalidation_kwargs=_CV_KWARGS),
        visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
        ceiling_func=lambda: _reliability_ceiling(assembly),
        parent=region, bibtex=BIBTEX)


def Li2026V1Temporal(): return _Li2026RegionTemporal('V1')
def Li2026V2Temporal(): return _Li2026RegionTemporal('V2')
def Li2026V4Temporal(): return _Li2026RegionTemporal('V4')
def Li2026ITTemporal(): return _Li2026RegionTemporal('IT')

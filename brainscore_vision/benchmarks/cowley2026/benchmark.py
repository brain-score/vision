import numpy as np

from brainscore_core import Score
from brainscore_vision import load_metric, load_ceiling, load_dataset
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, average_repetition

VISUAL_DEGREES = 11.2
NUMBER_OF_TRIALS = 14  # mode of the per-image repeat counts
SESSIONS = ['190923', '201025', '210225', '211022']  # 4 sessions from 3 subjects (one recorded twice)
BIBTEX = """@article{cowley2026compact,
  title={Compact deep neural network models of the visual cortex},
  author={Cowley, Benjamin R and Stan, Patricia L and Pillow, Jonathan W and Smith, Matthew A},
  journal={Nature},
  volume={652},
  number={8111},
  pages={947--954},
  year={2026},
  publisher={Nature Publishing Group}}"""

# no object categories -> plain random CV splits, not object_name stratification
pls_metric = lambda: load_metric('pls', crossvalidation_kwargs=dict(stratification_coord=None))


class _CombinedNeuralBenchmark(BenchmarkBase):
    """Mean ceiled score across the Cowley2026 sessions. Each session has its own
    images and neurons, so they are scored separately and averaged (no pooling)."""

    def __init__(self, identifier, sessions, parent, bibtex, version):
        self._sessions = [_session_benchmark(session) for session in sessions]
        super().__init__(identifier=identifier, version=version, parent=parent, bibtex=bibtex,
                         ceiling_func=lambda: Score(np.mean([b.ceiling.values for b in self._sessions])))

    def __call__(self, candidate):
        scores = [benchmark(candidate) for benchmark in self._sessions]
        combined = Score(np.mean([score.values for score in scores]))
        combined.attrs['ceiling'] = self.ceiling
        combined.attrs['session_scores'] = {b.identifier: float(s.values)
                                             for b, s in zip(self._sessions, scores)}
        return combined

    def preallocate_memory(self, candidate, raise_if_oom: bool = True):
        # sessions score sequentially, so peak memory is the single largest session
        largest = max(self._sessions, key=lambda b: b._assembly.sizes['presentation'])
        return largest.preallocate_memory(candidate, raise_if_oom=raise_if_oom)


def _session_benchmark(session):
    identifier = f'Cowley2026.{session}'
    assembly_repetition = alternate_repetition_halves(load_assembly(identifier, average_repetitions=False))
    assembly = load_assembly(identifier, average_repetitions=True)
    return NeuralBenchmark(
        identifier=f'{identifier}.V4-pls', version=1,
        assembly=assembly, similarity_metric=pls_metric(),
        visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
        ceiling_func=lambda: load_ceiling('internal_consistency')(assembly_repetition),
        parent='V4', bibtex=BIBTEX)


def alternate_repetition_halves(assembly):
    """Relabel repetitions to even/odd halves so the split-half ceiling balances per image."""
    names = list(assembly.indexes['presentation'].names)
    half = (assembly['repetition'].values % 2).astype(int)
    assembly = assembly.reset_index('presentation')
    assembly['repetition'] = 'presentation', half
    return assembly.set_index(presentation=names)


def load_assembly(identifier: str, average_repetitions: bool):
    assembly = load_dataset(identifier)
    assembly = assembly.sel(region='V4')
    assembly = assembly.stack(neuroid=['neuroid_id'])  # work around xarray multiindex issues
    assembly['region'] = 'neuroid', ['V4'] * len(assembly['neuroid'])
    assembly.load()
    if 'time_bin' not in assembly.dims and 'time_bin' not in assembly.coords:
        # only 190923 was packaged with the (50, 150) ms window; add it to the others
        assembly = assembly.expand_dims('time_bin_start').expand_dims('time_bin_end')
        assembly['time_bin_start'], assembly['time_bin_end'] = [50], [150]
        assembly = assembly.stack(time_bin=['time_bin_start', 'time_bin_end'])
    assembly = assembly.squeeze('time_bin').transpose('presentation', 'neuroid')
    if average_repetitions:
        assembly = average_repetition(assembly)
    return assembly


def Cowley2026V4PLS():
    return _CombinedNeuralBenchmark(identifier='Cowley2026.V4-pls', sessions=SESSIONS,
                                    parent='V4', bibtex=BIBTEX, version=1)

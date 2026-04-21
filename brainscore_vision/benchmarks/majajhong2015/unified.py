"""Unified-interface variants of MajajHong2015 benchmarks.

These call `model.process()` directly instead of legacy `model.look_at()`.
Scores must match the legacy variants bit-for-bit (modulo non-determinism in
layer search) — this is a regression test for the unified interface.
"""

import numpy as np

from brainscore_core import Score
from brainscore_vision import load_metric, load_ceiling
from brainscore_vision.benchmark_helpers.neural_common import (
    NeuralBenchmark, timebins_from_assembly,
)
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from .benchmark import (
    BIBTEX, VISUAL_DEGREES, NUMBER_OF_TRIALS, load_assembly,
)

crossvalidation_kwargs = dict(stratification_coord='object_name')


class UnifiedNeuralBenchmark(NeuralBenchmark):
    """Variant of NeuralBenchmark that calls process() instead of look_at().

    The unified `process()` method takes the same StimulusSet input and returns
    the same NeuroidAssembly. The recording layer is configured via
    start_recording() exactly as before.
    """

    def __call__(self, candidate) -> Score:
        candidate.start_recording(self.region, time_bins=self.timebins)
        stimulus_set = place_on_screen(
            self._assembly.stimulus_set,
            target_visual_degrees=candidate.visual_degrees(),
            source_visual_degrees=self._visual_degrees,
        )
        # The only line that differs from NeuralBenchmark.__call__:
        source_assembly = candidate.process(stimulus_set)
        if 'time_bin' in source_assembly.dims and source_assembly.sizes['time_bin'] == 1:
            source_assembly = source_assembly.squeeze('time_bin')
        raw_score = self._similarity_metric(source_assembly, self._assembly)
        from brainscore_vision.benchmark_helpers.neural_common import explained_variance
        ceiled_score = explained_variance(raw_score, self.ceiling)
        return ceiled_score


def _build_majajhong_unified(region: str, access: str):
    pls_metric = load_metric('pls', crossvalidation_kwargs=crossvalidation_kwargs)
    assembly_repetition = load_assembly(average_repetitions=False, region=region, access=access)
    assembly = load_assembly(average_repetitions=True, region=region, access=access)
    benchmark_identifier = f'MajajHong2015.{region}' + ('.public' if access == 'public' else '')
    return UnifiedNeuralBenchmark(
        identifier=f'{benchmark_identifier}-pls-unified',
        version=4,
        assembly=assembly,
        similarity_metric=pls_metric,
        visual_degrees=VISUAL_DEGREES,
        number_of_trials=NUMBER_OF_TRIALS,
        ceiling_func=lambda: load_ceiling('internal_consistency')(assembly_repetition),
        parent=region,
        bibtex=BIBTEX,
    )


def MajajHongV4PublicUnified():
    return _build_majajhong_unified(region='V4', access='public')


def MajajHongITPublicUnified():
    return _build_majajhong_unified(region='IT', access='public')


def DicarloMajajHong2015V4PLSUnified():
    return _build_majajhong_unified(region='V4', access='private')


def DicarloMajajHong2015ITPLSUnified():
    return _build_majajhong_unified(region='IT', access='private')

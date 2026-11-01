"""Legacy vision benchmarks run every subject through the permanent facade.

Scoring wraps the candidate only for benchmarks the loader marked as legacy.
Each legacy call reaches a legacy model exactly as the benchmark made it.
"""
import numpy as np
import pandas as pd
import pytest

import brainscore_vision
from brainscore_core.legacy_facade import LegacyBenchmarkFacade
from brainscore_core.metrics import Score
from brainscore_core.model_interface import BrainScoreModel, Subject
from brainscore_core.streaming import StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly, NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_vision.benchmarks.majajhong2015.unified import UnifiedNeuralBenchmark
from brainscore_vision.compat.unified_adapter import VisionModelAdapter

from tests.test_unified_adapter import _make_legacy_model, _vision_neural_assembly, _vision_stimulus_set


def _score(monkeypatch, model, benchmark):
    monkeypatch.setattr(brainscore_vision, 'import_plugin', lambda *a, **k: None)
    monkeypatch.setitem(brainscore_vision.benchmark_registry, 'legacy-test', lambda: benchmark)
    monkeypatch.setattr(brainscore_vision, 'load_model', lambda _: model)
    return brainscore_vision._run_score('m', 'legacy-test', check_mem=False)


class _NeuralBenchmark:
    """What a legacy NeuralBenchmark does to its candidate."""
    identifier = 'legacy-neural'
    region = 'IT'

    def __init__(self, trials=1, require_variance=False, time_bins=((70, 170),)):
        self.trials, self.require_variance, self.time_bins = trials, require_variance, list(time_bins)
        self.stimuli = _vision_stimulus_set()

    def preallocate_memory(self, *a, **k):
        pass

    def __call__(self, candidate):
        candidate.start_recording('IT', time_bins=self.time_bins)
        degrees = candidate.visual_degrees()
        settings = {'require_variance': True} if self.require_variance else {}
        output = candidate.look_at(self.stimuli, number_of_trials=self.trials, **settings)
        score = Score([1.0], coords={'aggregation': ['center']}, dims=['aggregation'])
        score.attrs.update(candidate=type(candidate).__name__, degrees=degrees, output=output)
        return score


class _BehavioralBenchmark:
    identifier = 'legacy-behavior'

    def __init__(self):
        self.fitting = _vision_stimulus_set()
        self.test = _vision_stimulus_set()

    def preallocate_memory(self, *a, **k):
        pass

    def __call__(self, candidate):
        candidate.start_task('probabilities', fitting_stimuli=self.fitting)
        output = candidate.look_at(self.test, number_of_trials=3)
        score = Score([1.0], coords={'aggregation': ['center']}, dims=['aggregation'])
        score.attrs.update(candidate=type(candidate).__name__, output=output)
        return score


def test_adapted_legacy_model_receives_the_exact_legacy_calls(monkeypatch):
    legacy = _make_legacy_model(region_layer_map={'IT': 'layer4'}, visual_degrees=12)
    legacy.look_at.return_value = _vision_neural_assembly()
    benchmark = _NeuralBenchmark(trials=7, time_bins=[(10, 30), (30, 50)])
    score = _score(monkeypatch, VisionModelAdapter(legacy), benchmark)
    assert score.attrs['candidate'] == 'LegacyBenchmarkFacade'
    assert score.attrs['degrees'] == 12
    assert score.attrs['output'] is legacy.look_at.return_value
    legacy.start_recording.assert_called_once_with('IT', [(10, 30), (30, 50)])
    legacy.look_at.assert_called_once()
    args, kwargs = legacy.look_at.call_args
    assert args[0] is benchmark.stimuli and args[1] == 7 and kwargs == {}
    legacy.start_task.assert_not_called()
    assert score.attrs['compatibility']['match'] == 'full'


def test_variance_request_is_forwarded_only_when_asked(monkeypatch):
    legacy = _make_legacy_model(region_layer_map={'IT': 'layer4'})
    legacy.look_at.return_value = _vision_neural_assembly()
    _score(monkeypatch, VisionModelAdapter(legacy), _NeuralBenchmark(trials=2, require_variance=True))
    args, kwargs = legacy.look_at.call_args
    assert args[1] == 2 and kwargs == {'require_variance': True}


def test_adapted_legacy_model_runs_a_behavioral_benchmark(monkeypatch):
    legacy = _make_legacy_model(region_layer_map={'IT': 'layer4'})
    legacy.look_at.return_value = BehavioralAssembly(np.array(['cat', 'dog']), dims=['presentation'])
    benchmark = _BehavioralBenchmark()
    score = _score(monkeypatch, VisionModelAdapter(legacy), benchmark)
    legacy.start_task.assert_called_once_with('probabilities', benchmark.fitting)
    args, kwargs = legacy.look_at.call_args
    assert args[0] is benchmark.test and args[1] == 3 and kwargs == {}
    legacy.start_recording.assert_not_called()
    assert score.attrs['output'] is legacy.look_at.return_value


class _Extractor:
    def __call__(self, stimuli, layers=None, **kwargs):
        n = len(stimuli)
        return NeuroidAssembly(
            np.arange(n * 2, dtype=float).reshape(n, 2),
            coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                    'neuroid_id': ('neuroid', [0, 1]), 'layer': ('neuroid', [layers[0]] * 2)},
            dims=['presentation', 'neuroid'])


def test_typed_model_scores_identically_to_process(monkeypatch):
    model = BrainScoreModel('typed', region_layer_map={'IT': 'layer'},
                            extractor={'vision': _Extractor()}, visual_degrees=8)
    benchmark = _NeuralBenchmark()
    score = _score(monkeypatch, model, benchmark)
    assert score.attrs['candidate'] == 'LegacyBenchmarkFacade'
    assert score.attrs['degrees'] == 8
    model.start_recording('IT', time_bins=[(70, 170)])
    np.testing.assert_array_equal(score.attrs['output'].values, model.process(benchmark.stimuli).values)


class _NativeSubject(Subject):
    """Interact-only: no look_at, start_recording or process."""
    identifier = 'native'
    in_channels = {'vision'}
    out_channels = {'neural:IT'}
    required_channels = set()

    def __init__(self, degrees=6):
        self.sessions = []
        self._degrees = degrees

    def visual_degrees(self):
        return self._degrees

    def interact(self, session):
        self.sessions.append(session)
        n = 0
        while session.next_input() is not None:
            n += 1
        output = NeuroidAssembly(np.ones((n, 2)), dims=['presentation', 'neuroid'],
                                 coords={'stimulus_id': ('presentation', list(range(n))),
                                         'neuroid_id': ('neuroid', [0, 1])})
        session.respond(StreamEvent('neural:IT', output, None))


def test_interact_only_subject_runs_a_legacy_benchmark(monkeypatch):
    subject = _NativeSubject(degrees=6)
    score = _score(monkeypatch, subject, _NeuralBenchmark(trials=4, time_bins=[(70, 170)]))
    assert score.attrs['candidate'] == 'LegacyBenchmarkFacade'
    assert score.attrs['degrees'] == 6
    assert score.attrs['output'].sizes['presentation'] == 2
    session, = subject.sessions
    assert session.time_bins == [(70, 170)]
    assert session.settings['number_of_trials'] == 4


class _NoFieldSubject(_NativeSubject):
    visual_degrees = None  # declares no field of view at all


def test_subject_without_visual_field_fails_clearly(monkeypatch):
    subject = _NoFieldSubject()
    with pytest.raises(NotImplementedError, match='declares no visual field'):
        _score(monkeypatch, subject, _NeuralBenchmark())


def test_typed_benchmark_variant_is_not_wrapped():
    from types import SimpleNamespace
    from brainscore_core.legacy_facade import mark_legacy_benchmark, subject_for_benchmark
    assert UnifiedNeuralBenchmark.subject_api == 'typed'
    benchmark = mark_legacy_benchmark(SimpleNamespace(subject_api=UnifiedNeuralBenchmark.subject_api))
    model = object()
    assert subject_for_benchmark(model, benchmark) is model


# --- setup once, measure many times ------------------------------------------

class _MalaniaLikeBenchmark:
    """Malania2007: fitting settings on start_task, variance on every look_at."""
    identifier = 'legacy-malania'

    def __init__(self):
        self.fitting = _vision_stimulus_set()
        self.conditions = [_vision_stimulus_set(), _vision_stimulus_set()]

    def preallocate_memory(self, *a, **k):
        pass

    def __call__(self, candidate):
        candidate.start_task('probabilities', fitting_stimuli=self.fitting,
                             number_of_trials=2, require_variance=True)
        outputs = [candidate.look_at(stimuli, number_of_trials=2, require_variance=True)
                   for stimuli in self.conditions]
        score = Score([1.0], coords={'aggregation': ['center']}, dims=['aggregation'])
        score.attrs.update(outputs=outputs)
        return score


def test_fitting_settings_reach_the_legacy_model_once(monkeypatch):
    legacy = _make_legacy_model(region_layer_map={'IT': 'layer4'})
    legacy.look_at.return_value = BehavioralAssembly(np.array(['cat', 'dog']), dims=['presentation'])
    benchmark = _MalaniaLikeBenchmark()
    _score(monkeypatch, VisionModelAdapter(legacy), benchmark)
    legacy.start_task.assert_called_once_with('probabilities', benchmark.fitting,
                                              number_of_trials=2, require_variance=True)
    assert legacy.look_at.call_count == 2
    for call, stimuli in zip(legacy.look_at.call_args_list, benchmark.conditions):
        assert call.args[0] is stimuli and call.args[1] == 2
        assert call.kwargs == {'require_variance': True}


class _TwoMeasurementNeuralBenchmark(_NeuralBenchmark):
    def __call__(self, candidate):
        candidate.start_recording('IT', time_bins=self.time_bins)
        first = candidate.look_at(self.stimuli, number_of_trials=self.trials)
        second = candidate.look_at(self.stimuli, number_of_trials=self.trials)
        score = Score([1.0], coords={'aggregation': ['center']}, dims=['aggregation'])
        score.attrs.update(output=first, second=second)
        return score


def test_recording_setup_is_applied_once_for_several_measurements(monkeypatch):
    legacy = _make_legacy_model(region_layer_map={'IT': 'layer4'}, visual_degrees=12)
    legacy.look_at.return_value = _vision_neural_assembly()
    _score(monkeypatch, VisionModelAdapter(legacy), _TwoMeasurementNeuralBenchmark())
    legacy.start_recording.assert_called_once_with('IT', [(70, 170)])
    assert legacy.look_at.call_count == 2


def test_reset_lets_setup_be_applied_again():
    legacy = _make_legacy_model(region_layer_map={'IT': 'layer4'})
    adapter = VisionModelAdapter(legacy)
    adapter.start_recording('IT', time_bins=[(70, 170)])
    adapter.start_recording('IT', time_bins=[(70, 170)])
    assert legacy.start_recording.call_count == 1
    adapter.start_recording('V4', time_bins=[(70, 170)])
    assert legacy.start_recording.call_count == 2
    adapter.reset()
    adapter.start_recording('V4', time_bins=[(70, 170)])
    assert legacy.start_recording.call_count == 3


def test_fitting_table_reaches_the_legacy_model_with_declared_inputs_only():
    legacy = _make_legacy_model(region_layer_map={'IT': 'layer4'})
    adapter = VisionModelAdapter(legacy)
    fitting = StimulusSet(pd.DataFrame({'stimulus_id': ['a', 'b'], 'image_path': ['a.png', 'b.png'],
                                        'sentence': ['one', 'two'], 'image_label': ['x', 'y']}))
    fitting.identifier = 'fit'
    adapter.start_task('probabilities', fitting)
    seen = legacy.start_task.call_args.args[1]
    assert 'sentence' not in seen.columns and list(seen['image_label']) == ['x', 'y']
    assert seen.identifier == 'fit' and 'sentence' in fitting.columns

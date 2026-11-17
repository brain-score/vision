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
from brainscore_core.contract import Subject
from brainscore_core.streaming import StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly, NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
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


def _labelled_fitting():
    fitting = _vision_stimulus_set()
    fitting['image_label'] = ['cat', 'dog']
    return fitting


def _probability_rows():
    """The answer of an old probabilities task: rows over the fitted labels, float64."""
    return BehavioralAssembly(np.array([[0.75, 0.25], [0.375, 0.625]]), dims=['presentation', 'choice'],
                              coords={'stimulus_id': ('presentation', ['s0', 's1']),
                                      'image_path': ('presentation', ['s0.png', 's1.png']),
                                      'object_name': ('presentation', ['cat', 'dog']),
                                      'choice': ['cat', 'dog']})


class _BehavioralBenchmark:
    identifier = 'legacy-behavior'

    def __init__(self):
        self.fitting = _labelled_fitting()
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
    import xarray as xr
    legacy = _make_legacy_model(region_layer_map={'IT': 'layer4'})
    legacy.look_at.return_value = _probability_rows()
    benchmark = _BehavioralBenchmark()
    score = _score(monkeypatch, VisionModelAdapter(legacy), benchmark)
    legacy.start_task.assert_called_once_with('probabilities', benchmark.fitting)
    args, kwargs = legacy.look_at.call_args
    assert args[0] is benchmark.test and args[1] == 3 and kwargs == {}
    legacy.start_recording.assert_not_called()
    xr.testing.assert_identical(score.attrs['output'], legacy.look_at.return_value)


class _NativeSubject(Subject):
    """Participate-only: no look_at, start_recording or process."""
    identifier = 'native'
    in_channels = {'vision:image'}
    out_channels = {'neural:IT'}
    required_channels = set()

    def __init__(self, degrees=6):
        self.sessions = []
        self._degrees = degrees

    def visual_degrees(self):
        return self._degrees

    def participate(self, session):
        self.sessions.append(session)
        n = 0
        while (event := session.next_input()) is not None:
            n += event.channel != 'control'
        output = NeuroidAssembly(np.ones((n, 2)), dims=['presentation', 'neuroid'],
                                 coords={'stimulus_id': ('presentation', list(range(n))),
                                         'neuroid_id': ('neuroid', [0, 1])})
        session.respond(StreamEvent('neural:IT', output, None))


def test_participate_only_subject_runs_a_legacy_benchmark(monkeypatch):
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


# --- setup once, measure many times ------------------------------------------

class _MalaniaLikeBenchmark:
    """Malania2007: fitting settings on start_task, variance on every look_at."""
    identifier = 'legacy-malania'

    def __init__(self):
        self.fitting = _labelled_fitting()
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
    legacy.look_at.return_value = _probability_rows()
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


class _LabelBenchmark:
    """What an old label benchmark does: a task with a list of labels, one look_at."""
    identifier = 'legacy-label'

    def __init__(self):
        from tests.test_unified_adapter import _twenty_stimuli
        self.stimuli = _twenty_stimuli('entry-point-label')

    def preallocate_memory(self, *a, **k):
        pass

    def __call__(self, candidate):
        candidate.start_task('label', ['real', 'pseudo'])
        candidate.look_at(self.stimuli)
        return Score(0.5)


def _label_model(n_outside):
    from tests.test_unified_adapter import _LegacyLabels

    class _Answers(_LegacyLabels):
        def look_at(self, stimuli, number_of_trials=1, require_variance=False):
            ids = list(stimuli['stimulus_id'])
            answers = ['real'] * (len(ids) - n_outside) + ['unlisted'] * n_outside
            return BehavioralAssembly([answers], dims=['choice', 'presentation'],
                                      coords={'stimulus_id': ('presentation', ids)})
    return VisionModelAdapter(_Answers([]))


def test_the_score_of_a_label_benchmark_carries_the_count_of_old_labels_outside_the_list(monkeypatch):
    from brainscore_core.legacy_facade import OLD_LABEL_POLICY
    score = _score(monkeypatch, _label_model(1), _LabelBenchmark())
    assert score.attrs['n_invalid'] == 1
    assert score.attrs['answer_policy'] == OLD_LABEL_POLICY


def test_the_score_carries_a_zero_when_no_old_label_is_outside_the_list(monkeypatch):
    from brainscore_core.legacy_facade import OLD_LABEL_POLICY
    score = _score(monkeypatch, _label_model(0), _LabelBenchmark())
    assert score.attrs['n_invalid'] == 0
    assert score.attrs['answer_policy'] == OLD_LABEL_POLICY


class _LabelsThenProbabilities:
    """An old benchmark that asks one old model for labels and then for probabilities in one
    score call, with or without reset() between the two tasks."""
    identifier = 'legacy-labels-then-probabilities'

    def __init__(self, reset):
        from tests.test_unified_adapter import _choice_stimuli
        self.reset = reset
        self.stimuli = _choice_stimuli()

    def preallocate_memory(self, *a, **k):
        pass

    def __call__(self, candidate):
        candidate.start_task('label', ['real', 'pseudo'])
        candidate.look_at(self.stimuli)
        if self.reset:
            candidate.reset()
        candidate.start_task('probabilities', fitting_stimuli=self.stimuli)
        candidate.look_at(self.stimuli)
        return Score(0.5)


@pytest.mark.parametrize('reset', [False, True])
def test_one_benchmark_that_asks_for_labels_and_probabilities_gets_a_score_with_both_policies(
        monkeypatch, reset):
    """The facade keeps every policy it applied, in the order of first use; the count is the sum."""
    from brainscore_core.legacy_facade import OLD_LABEL_POLICY, REPLACED_ANSWER_POLICY
    from tests.test_unified_adapter import _LegacyForcedChoice, _LegacyLabels

    class _Both(_LegacyLabels):
        def look_at(self, stimuli, number_of_trials=1, require_variance=False):
            if self.tasks and self.tasks[-1] == 'probabilities':
                return _LegacyForcedChoice.look_at(self, stimuli)
            return super().look_at(stimuli, number_of_trials)

    score = _score(monkeypatch, VisionModelAdapter(_Both(['real', 'unlisted', 'pseudo'])),
                   _LabelsThenProbabilities(reset))
    assert score.attrs['n_invalid'] == 1
    assert score.attrs['answer_policy'] == f'{OLD_LABEL_POLICY}; {REPLACED_ANSWER_POLICY}'

"""Tests for pre-flight checks wired into the vision scoring pipeline."""

import math

import pytest
from unittest.mock import patch, MagicMock

from brainscore_core.compatibility import CompatibilityError, ExecutionNotFeasibleError
from brainscore_core.memory import MemoryError
from brainscore_core.contract import Subject


class _Subject(Subject):
    """A small real subject: it has what it declares and nothing else.

    A MagicMock answers every attribute, so a pre-flight that starts to read a
    new declaration would pass on mock values; this subject does not.
    """
    identifier = 'test-model'
    in_channels = frozenset()
    out_channels = frozenset()
    required_channels = frozenset()
    region_layer_map = {}

    def __init__(self, in_channels):
        self.in_channels = frozenset(in_channels)

    def participate(self, session):
        raise AssertionError('the stand-in benchmark never runs a session')


class TestPreflightInRunScore:
    """Verify that _run_score calls check_compatibility and check_memory."""

    def _make_model_and_benchmark(self, model_in_channels, bench_required):
        """A real subject and a stand-in benchmark for pre-flight testing."""
        model = _Subject(model_in_channels)

        benchmark = MagicMock(spec=['identifier', 'required_modalities', '__call__', 'preallocate_memory'])
        benchmark.identifier = 'test-bench'
        benchmark.required_modalities = bench_required
        return model, benchmark

    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_incompatible_model_returns_na_before_compute(
        self, mock_load_model, mock_load_benchmark
    ):
        from brainscore_vision import _run_score

        model, benchmark = self._make_model_and_benchmark(
            model_in_channels={'text'},
            bench_required={'vision'},
        )
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        result = _run_score('test-model', 'test-bench', check_mem=False)

        assert math.isnan(float(result))
        assert result.attrs['status'] == 'N/A'
        assert "takes in none of the inputs" in result.attrs['reason']
        assert result.attrs['model_identifier'] == 'test-model'
        assert result.attrs['benchmark_identifier'] == 'test-bench'
        benchmark.assert_not_called()

    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_score_metadata_attrs_are_stamped(
        self, mock_load_model, mock_load_benchmark
    ):
        from brainscore_vision import _run_score

        model, benchmark = self._make_model_and_benchmark(
            model_in_channels={'vision:image'},
            bench_required={'vision'},
        )
        model.out_channels = {'neural:IT'}
        model.region_layer_map = {'IT': 'it-layer'}
        benchmark.region = 'IT'
        score = MagicMock()
        score.attrs = {'existing': 'preserved'}
        benchmark.return_value = score
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        result = _run_score('test-model', 'test-bench', check_mem=False)

        benchmark.assert_called_once_with(model)   # a compatible pair runs once
        assert result is score
        assert score.attrs['existing'] == 'preserved'
        assert score.attrs['in_channels'] == ('vision:image',)
        assert score.attrs['out_channels'] == ('neural:IT',)
        assert score.attrs['response'] == ()
        assert 'protocol' not in score.attrs
        assert score.attrs['subject_id'] == 'test-model'
        assert score.attrs['compatibility']['match'] == 'full'
        assert score.attrs['harness_id'] == 'brainscore_vision'
        # No session answered in this fake run; the plan and versions are still recorded.
        assert score.attrs['delivery'] is None
        assert score.attrs['regions_per_pass'] is None   # no subject said which plan it used
        assert score.attrs['software']['python']
        assert 'brainscore_core' in score.attrs['code_commits']
        assert score.attrs['provenance'] == ()   # an old model's responses carry no provenance
        # The core stamp writes the identifier and the trial memory (one step in all entry points).
        assert score.attrs['benchmark_identifier'] == 'test-bench'
        assert score.attrs['trial_memory'] == 'none'

    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_a_score_that_names_another_benchmark_is_refused(
            self, mock_load_model, mock_load_benchmark):
        from brainscore_vision import _run_score

        model, benchmark = self._make_model_and_benchmark(
            model_in_channels={'vision:image'}, bench_required={'vision'})
        model.out_channels = {'neural:IT'}
        model.region_layer_map = {'IT': 'it-layer'}
        benchmark.region = 'IT'
        score = MagicMock()
        score.attrs = {'benchmark_identifier': 'another-bench'}
        benchmark.return_value = score
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        with pytest.raises(ValueError, match='must match the requested identifier'):
            _run_score('test-model', 'test-bench', check_mem=False)

    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_behavior_response_record_comes_from_recorded_answers(
            self, mock_load_model, mock_load_benchmark):
        from brainscore_vision import _run_score
        from brainscore_core.streaming import Session, StreamEvent
        from brainscore_core.supported_data_standards.brainio.assemblies import (
            BehavioralAssembly,
        )

        model, benchmark = self._make_model_and_benchmark(
            model_in_channels={'vision:image'},
            bench_required={'vision'},
        )
        model.out_channels = {'behavior'}
        benchmark.parent = 'behavior'
        benchmark.requested_output_channels = {'behavior'}
        score = MagicMock()
        score.attrs = {}

        from brainscore_core.response_formats import response_record
        record = response_record('behavior', 'generation', parsed_by='model')

        def run(candidate):
            # A subject's behavioral response, accepted by a session.
            Session((), ['behavior']).respond(
                StreamEvent('behavior', BehavioralAssembly(
                    [[1.0, 0.0]], dims=['presentation', 'choice'],
                    attrs={'response': record}), None))
            return score

        benchmark.side_effect = run
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        _run_score('test-model', 'test-bench', check_mem=False)

        assert score.attrs['response'] == (record,)
        assert 'measure' not in score.attrs and 'protocol' not in score.attrs

    def test_an_older_core_fails_with_a_clear_message(self):
        from brainscore_vision import _run_score
        import brainscore_core.score_metadata as score_metadata

        saved = score_metadata.record_measures
        del score_metadata.record_measures
        try:
            with pytest.raises(ImportError, match='core ad7288a or later'):
                _run_score('test-model', 'test-bench', check_mem=False)
        finally:
            score_metadata.record_measures = saved

    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_channel_mismatch_names_the_channel_in_the_na_result(
        self, mock_load_model, mock_load_benchmark
    ):
        from brainscore_vision import _run_score

        model, benchmark = self._make_model_and_benchmark(
            model_in_channels={'vision:image'},
            bench_required={'vision'},
        )
        benchmark.required_input_channels = {'text'}
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        result = _run_score('test-model', 'test-bench', check_mem=False)

        assert result.attrs['status'] == 'N/A'
        assert "text" in result.attrs['reason']
        assert result.attrs['missing_channels'] == sorted(benchmark.required_input_channels)

        benchmark.assert_not_called()

    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_a_refusal_at_session_start_is_an_error(
        self, mock_load_model, mock_load_benchmark
    ):
        from brainscore_vision import _run_score
        from brainscore_core.streaming import Session

        model, benchmark = self._make_model_and_benchmark(
            model_in_channels={'vision:image'},
            bench_required={'vision'},
        )
        model.out_channels = {'neural:IT'}
        model.region_layer_map = {'IT': 'it-layer'}
        benchmark.region = 'IT'

        def refuse(session):
            raise ExecutionNotFeasibleError('no one part answers', channels=session.outputs)
        model.participate = refuse
        benchmark.side_effect = lambda candidate: candidate.participate(Session([], ['neural:IT']))
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        with pytest.raises(ExecutionNotFeasibleError, match='no one part answers') as caught:
            _run_score('test-model', 'test-bench', check_mem=False)

        assert not isinstance(caught.value, CompatibilityError)   # an error, never N/A

        assert caught.value.channels == set(['neural:IT'])

    @patch('brainscore_core.memory.check_memory')
    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_check_mem_false_skips_memory_check(
        self, mock_load_model, mock_load_benchmark, mock_check_memory
    ):
        from brainscore_vision import _run_score

        model, benchmark = self._make_model_and_benchmark(
            model_in_channels={'vision:image'},
            bench_required={'vision'},
        )
        score = MagicMock()
        score.attrs = {}
        benchmark.return_value = score
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        _run_score('test-model', 'test-bench', check_mem=False)
        mock_check_memory.assert_not_called()

    @patch('brainscore_core.memory.check_memory')
    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_memory_error_propagates(
        self, mock_load_model, mock_load_benchmark, mock_check_memory
    ):
        from brainscore_vision import _run_score

        model, benchmark = self._make_model_and_benchmark(
            model_in_channels={'vision:image'},
            bench_required={'vision'},
        )
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark
        mock_check_memory.side_effect = MemoryError("not enough memory")

        with pytest.raises(MemoryError, match="not enough memory"):
            _run_score('test-model', 'test-bench', check_mem=True)

    def test_load_benchmark_sets_legacy_vision_required_modalities(self):
        import brainscore_vision

        benchmark = MagicMock()
        del benchmark.required_modalities
        del benchmark.accepted_modalities
        del benchmark.required_input_channels

        with patch.object(brainscore_vision, 'benchmark_registry',
                          {'legacy-bench': lambda: benchmark}):
            with patch('brainscore_vision.import_plugin'):
                loaded = brainscore_vision.load_benchmark('legacy-bench')

        assert loaded.required_modalities == {'vision'}


class TestRefusalDuringTheRunIsAnError:
    """N/A comes only from the declarations; a refusal during the run raises."""

    @staticmethod
    def _legacy_model():
        import numpy as np
        from unittest.mock import PropertyMock
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
        legacy = MagicMock()
        type(legacy).identifier = PropertyMock(return_value='mock-legacy')
        legacy.visual_degrees.return_value = 8
        legacy.layer_model.region_layer_map = {'IT': 'layer4'}
        legacy.look_at.return_value = NeuroidAssembly(
            np.array([[1.0], [2.0]]), dims=['presentation', 'neuroid'],
            coords={'stimulus_id': ('presentation', ['s0', 's1']),
                    'neuroid_id': ('neuroid', ['n0']), 'layer': ('neuroid', ['layer4'])})
        return legacy

    @staticmethod
    def _stimuli():
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
        stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['s0', 's1'],
                                            'image_path': ['s0.png', 's1.png']}))
        stimuli.identifier = 'preflight-stimuli'
        return stimuli

    def _benchmark(self, run, outputs):
        benchmark = MagicMock(spec=['identifier', 'required_input_channels',
                                    'requested_output_channels', '__call__', 'preallocate_memory'])
        benchmark.identifier = 'test-bench'
        benchmark.required_input_channels = {'vision:image'}
        benchmark.requested_output_channels = set(outputs)
        benchmark.side_effect = run
        return benchmark

    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_a_legacy_model_asked_for_behavior_and_a_region_together_is_an_error(
            self, mock_load_model, mock_load_benchmark):
        from brainscore_vision import _run_score
        from brainscore_vision.compat.unified_adapter import VisionModelAdapter
        from brainscore_core.streaming_helpers import stimulus_session
        legacy = self._legacy_model()
        outputs = ('behavior', 'neural:IT')
        mock_load_model.return_value = VisionModelAdapter(legacy)
        mock_load_benchmark.return_value = self._benchmark(
            lambda subject: subject.participate(stimulus_session(self._stimuli(), outputs)), outputs)

        with pytest.raises(ExecutionNotFeasibleError, match='passed the declared compatibility check') as caught:
            _run_score('test-model', 'test-bench', check_mem=False)

        assert caught.value.channels == {'behavior', 'neural:IT'}
        legacy.look_at.assert_not_called()
        legacy.start_task.assert_not_called()

    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_a_refusal_after_an_earlier_session_computed_is_an_error(
            self, mock_load_model, mock_load_benchmark):
        from brainscore_vision import _run_score
        from brainscore_vision.compat.unified_adapter import VisionModelAdapter
        from brainscore_core.streaming_helpers import stimulus_session
        legacy = self._legacy_model()

        def run(subject):
            subject.participate(stimulus_session(self._stimuli(), ['neural:IT']))
            subject.participate(stimulus_session(self._stimuli(), ['behavior', 'neural:IT']))
        mock_load_model.return_value = VisionModelAdapter(legacy)
        mock_load_benchmark.return_value = self._benchmark(run, ['behavior', 'neural:IT'])

        with pytest.raises(ExecutionNotFeasibleError, match='Session refused'):
            _run_score('test-model', 'test-bench', check_mem=False)
        assert legacy.look_at.call_count == 1

    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_a_runtime_error_passes_through(self, mock_load_model, mock_load_benchmark):
        from brainscore_vision import _run_score
        model, benchmark = TestPreflightInRunScore()._make_model_and_benchmark(
            model_in_channels={'vision:image'}, bench_required={'vision'})

        def crash(candidate):
            raise RuntimeError('model crashed')
        benchmark.side_effect = crash
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        with pytest.raises(RuntimeError, match='model crashed'):
            _run_score('test-model', 'test-bench', check_mem=False)

    @patch('brainscore_vision.load_benchmark')
    @patch('brainscore_vision.load_model')
    def test_a_malformed_benchmark_declaration_is_an_error(self, mock_load_model, mock_load_benchmark):
        from brainscore_vision import _run_score
        model, benchmark = TestPreflightInRunScore()._make_model_and_benchmark(
            model_in_channels={'vision:image'}, bench_required={'vision'})
        benchmark.required_input_channels = {'vision'}       # bare vision: names no form
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        with pytest.raises(ValueError):
            _run_score('test-model', 'test-bench', check_mem=False)
        benchmark.assert_not_called()


class _LeavesALesion:
    """A benchmark that applies an intervention and, unless told to, leaves it active."""
    identifier = 'leaves-a-lesion'
    version = 1
    required_input_channels = {'text', 'lesion'}    # the lesion record answers the lesion input
    requested_output_channels = {'perturbation'}

    def __init__(self, undo):
        self.undo = undo

    def preallocate_memory(self, candidate):
        pass

    def __call__(self, subject):
        from brainscore_core.events import Perturbation, Selection, StateChange
        from brainscore_core.metrics import Score
        from brainscore_core.streaming_helpers import apply_state_change
        applied = apply_state_change(subject, StateChange(
            kind='ablation', target=Selection(layer='x'), perturbation=Perturbation(kind='zero')))
        if self.undo:
            apply_state_change(subject, StateChange(kind='reset', handle_id=applied.handle_id))
        return Score(1.0)


class _LesionSubject(Subject):
    identifier = 'lesion-subject'
    in_channels = frozenset({'text', 'lesion'})
    out_channels = frozenset({'perturbation'})

    def participate(self, session):
        from brainscore_core.events import PerturbationApplied
        from brainscore_core.streaming import StreamEvent
        change = session.next_input().payload
        session.respond(StreamEvent(channel='perturbation', t_ms=0.0, payload=PerturbationApplied(
            change.handle_id or 'h1', change.target, change.perturbation)))


@pytest.mark.parametrize('undo', [False, True])
def test_a_benchmark_that_leaves_an_intervention_active_is_an_error(undo):
    """The release check of brainscore.score runs here too, before a result is returned."""
    from brainscore_vision import _run_score
    with patch('brainscore_vision.load_benchmark', return_value=_LeavesALesion(undo)), \
            patch('brainscore_vision.load_model', return_value=_LesionSubject()):
        if undo:
            assert float(_run_score('lesion-subject', 'leaves-a-lesion', check_mem=False)) == 1.0
        else:
            with pytest.raises(RuntimeError, match=r"left intervention handle\(s\) \['h1'\]"):
                _run_score('lesion-subject', 'leaves-a-lesion', check_mem=False)

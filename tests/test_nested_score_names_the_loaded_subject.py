"""A score run inside another score names the subject it loaded, not its lookup key.

A model can be registered under a second key and load a subject with its own identifier.
The sessions of the inner score belong to that loaded subject, also in the outer score.
"""
from brainscore_core.contract import Subject
from brainscore_core.metrics import Score
from brainscore_core.score_metadata import record_measures, stamp_score_metadata
from brainscore_core.streaming import Session, StreamEvent

import brainscore_vision


class _Loaded(Subject):
    """Answers each text input; it states no parts."""
    identifier = 'loaded-identifier'
    in_channels = {'text'}
    out_channels = {'behavior:text'}

    def participate(self, session):
        while (event := session.next_input()) is not None:
            session.respond(StreamEvent('behavior:text', 'word', event.t_ms, {}))


class _Inner:
    identifier = 'inner-benchmark'
    required_input_channels = frozenset({'text'})
    requested_output_channels = frozenset({'behavior:text'})

    def preallocate_memory(self, *a, **k):     # vision's score_benchmark calls it first
        pass

    def __call__(self, subject):
        subject.participate(Session([StreamEvent('text', 'a', 0.0)], ['behavior:text']))
        return Score(0.5)


def test_a_nested_score_names_the_loaded_subject(monkeypatch):
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    monkeypatch.setattr(brainscore_vision, 'import_plugin', lambda *a, **k: None)
    monkeypatch.setitem(brainscore_vision.model_registry, 'second-key', _Loaded)
    monkeypatch.setattr(brainscore_vision, 'load_benchmark', lambda _: _Inner())

    class _Outer:
        identifier = 'outer-subject'
        in_channels, traits = {'text'}, set()

    with record_measures(subject='outer-subject') as outer:
        inner = brainscore_vision._run_score('second-key', 'inner-benchmark', check_mem=False)
    assert inner.attrs['model_identifier'] == 'second-key'      # the requested key stays
    stamped = stamp_score_metadata(Score(0.5), _Outer(), requested_channels={'behavior:text'},
                                   entry_point='test', measures=outer)
    assert [entry['subject'] for entry in stamped.attrs['provenance']] == ['loaded-identifier']

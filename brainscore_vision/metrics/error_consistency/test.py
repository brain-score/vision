import numpy as np
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore_vision import load_metric


def test_score():
    assembly = _make_data()
    metric = load_metric('error_consistency')
    score = metric(assembly.sel(subject='A'), assembly)
    assert score == approx(0.2)


def test_has_error():
    assembly = _make_data()
    metric = load_metric('error_consistency')
    score = metric(assembly.sel(subject='A'), assembly)
    assert hasattr(score, 'error')


def test_raw_subjects():
    assembly = _make_data()
    metric = load_metric('error_consistency')
    score = metric(assembly.sel(subject='A'), assembly)
    subject_scores = score.raw.mean('condition')
    assert subject_scores.sel(subject='A') == 1


def _make_data():
    return BehavioralAssembly(['dog', 'cat', 'chair', 'cat', 'dog', 'dog', 'dog', 'dog', 'chair',  # subject A
                               'cat', 'cat', 'chair', 'cat', 'dog', 'cat', 'chair', 'cat', 'cat',  # subject B
                               'dog', 'cat', 'chair', 'dog', 'cat', 'chair', 'dog', 'cat', 'chair'  # subject C
                               ],
                              coords={'stimulus_id': ('presentation', np.resize(np.arange(9), 9 * 3)),
                                      'truth': ('presentation', np.resize(['dog', 'cat', 'chair'], 9 * 3)),
                                      'condition': ('presentation', np.resize([1, 1, 1, 2, 2, 2, 3, 3, 3], 9 * 3)),
                                      'subject': ('presentation', ['A'] * 9 + ['B'] * 9 + ['C'] * 9)},
                              dims=['presentation'])

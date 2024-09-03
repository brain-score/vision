from pytest import approx

from brainio.assemblies import PropertyAssembly
from brainscore_vision import load_metric


def test_threshold_score_from_thresholds():
    assembly = _make_threshold_data()
    # independent_variable is not used since we compute from thresholds, and do not need to fit them
    metric = load_metric('threshold', independent_variable='placeholder')
    score = metric(float(assembly.sel(subject='A').values), assembly)
    assert score == approx(0.5625)


def test_threshold_elevation_score_from_threshold_elevations():
    assembly = _make_threshold_elevation_data()
    # independent_variable is not used since we compute from thresholds, and do not need to fit them
    metric = load_metric('threshold_elevation',
                         independent_variable='placeholder',
                         baseline_condition='placeholder',
                         test_condition='placeholder')
    score = metric(float(assembly.sel(subject='A').values), assembly)
    assert score == approx(0.525)


def test_threshold_has_error():
    assembly = _make_threshold_data()
    metric = load_metric('threshold', independent_variable='placeholder')
    score = metric(float(assembly.sel(subject='A').values), assembly)
    assert hasattr(score, 'error')


def test_threshold_elevation_has_error():
    assembly = _make_threshold_elevation_data()
    metric = load_metric('threshold_elevation',
                         independent_variable='placeholder',
                         baseline_condition='placeholder',
                         test_condition='placeholder')
    score = metric(float(assembly.sel(subject='A').values), assembly)
    assert hasattr(score, 'error')


def test_threshold_has_raw():
    assembly = _make_threshold_data()
    metric = load_metric('threshold', independent_variable='placeholder')
    score = metric(float(assembly.sel(subject='A').values), assembly)
    assert hasattr(score, 'raw')


def test_threshold_elevation_has_raw():
    assembly = _make_threshold_elevation_data()
    metric = load_metric('threshold_elevation',
                         independent_variable='placeholder',
                         baseline_condition='placeholder',
                         test_condition='placeholder')
    score = metric(float(assembly.sel(subject='A').values), assembly)
    assert hasattr(score, 'raw')


def _make_threshold_data():
    # Subjects have thresholds of 10, 20, 40, and 20 respectively.
    return PropertyAssembly([10.0, 20.0, 40.0, 20.0],
                            coords={'subject': ('subject', ['A', 'B', 'C', 'D'])},
                            dims=['subject'])


def _make_threshold_elevation_data():
    # Subjects have threshold elevations of 3, 2, 1.5, and 5 respectively.
    return PropertyAssembly([3.0, 2.0, 1.5, 5.0],
                            coords={'subject': ('subject', ['A', 'B', 'C', 'D'])},
                            dims=['subject'])

from pathlib import Path

import pytest
from pytest import approx

from brainscore_vision import benchmark_registry
from brainscore_vision.benchmark_helpers.test_helper import StandardizedTests, PrecomputedTests, NumberOfTrialsTests
from brainscore_vision.data_helpers import s3

standardized_tests = StandardizedTests()
precomputed_test = PrecomputedTests()
num_trials_test = NumberOfTrialsTests()


@pytest.mark.parametrize('benchmark', [
    'dicarlo.Sanghavi2020.V4-pls',
    'dicarlo.Sanghavi2020.IT-pls',
    'dicarlo.SanghaviJozwik2020.V4-pls',
    'dicarlo.SanghaviJozwik2020.IT-pls',
    'dicarlo.SanghaviMurty2020.V4-pls',
    'dicarlo.SanghaviMurty2020.IT-pls',
])
def test_benchmark_registry(benchmark):
    assert benchmark in benchmark_registry


@pytest.mark.memory_intense
@pytest.mark.private_access
@pytest.mark.parametrize('benchmark, expected', [
    ('dicarlo.Sanghavi2020.V4-pls', approx(.8892049, abs=.001)),
    ('dicarlo.Sanghavi2020.IT-pls', approx(.868293, abs=.001)),
    ('dicarlo.SanghaviJozwik2020.V4-pls', approx(.9630336, abs=.001)),
    ('dicarlo.SanghaviJozwik2020.IT-pls', approx(.860352, abs=.001)),
    ('dicarlo.SanghaviMurty2020.V4-pls', approx(.9666086, abs=.001)),
    ('dicarlo.SanghaviMurty2020.IT-pls', approx(.875714, abs=.001)),
])
def test_ceilings(benchmark, expected):
    standardized_tests.ceilings_test(benchmark, expected)


@pytest.mark.memory_intense
@pytest.mark.private_access
@pytest.mark.parametrize('benchmark, visual_degrees, expected', [
    ('dicarlo.Sanghavi2020.V4-pls', 8, approx(.9727137, abs=.001)),
    ('dicarlo.Sanghavi2020.IT-pls', 8, approx(.890062, abs=.001)),
    ('dicarlo.SanghaviJozwik2020.V4-pls', 8, approx(.9739177, abs=.001)),
    ('dicarlo.SanghaviJozwik2020.IT-pls', 8, approx(.9999779, abs=.001)),
    ('dicarlo.SanghaviMurty2020.V4-pls', 5, approx(.978581, abs=.001)),
    ('dicarlo.SanghaviMurty2020.IT-pls', 5, approx(.9997532, abs=.001)),
])
def test_self_regression(benchmark, visual_degrees, expected):
    standardized_tests.self_regression_test(benchmark, visual_degrees, expected)


@pytest.mark.memory_intense
@pytest.mark.private_access
@pytest.mark.slow
@pytest.mark.parametrize('benchmark, filename, expected', [
    ('dicarlo.Sanghavi2020.V4-pls', 'alexnet-sanghavi2020-features.12.nc', approx(.551135, abs=.015)),
    ('dicarlo.Sanghavi2020.IT-pls', 'alexnet-sanghavi2020-features.12.nc', approx(.611347, abs=.015)),
    ('dicarlo.SanghaviJozwik2020.V4-pls', 'alexnet-sanghavijozwik2020-features.12.nc', approx(.49235, abs=.005)),
    ('dicarlo.SanghaviJozwik2020.IT-pls', 'alexnet-sanghavijozwik2020-features.12.nc', approx(.590543, abs=.005)),
    ('dicarlo.SanghaviMurty2020.V4-pls', 'alexnet-sanghavimurty2020-features.12.nc', approx(.357461, abs=.015)),
    ('dicarlo.SanghaviMurty2020.IT-pls', 'alexnet-sanghavimurty2020-features.12.nc', approx(.53006, abs=.015)),
])
def test_model_features(benchmark, filename, expected):
    filepath = Path(__file__).parent / filename
    s3.download_file_if_not_exists(local_path=filepath,
                                   bucket='brainio-brainscore', remote_filepath=f'tests/test_benchmarks/{filename}')
    precomputed_test.run_test(benchmark=benchmark, precomputed_features_filepath=filepath, expected=expected)


@pytest.mark.private_access
@pytest.mark.parametrize('benchmark_identifier', [
    'dicarlo.Sanghavi2020.V4-pls',
    'dicarlo.Sanghavi2020.IT-pls',
    'dicarlo.SanghaviJozwik2020.V4-pls',
    'dicarlo.SanghaviJozwik2020.IT-pls',
    'dicarlo.SanghaviMurty2020.V4-pls',
    'dicarlo.SanghaviMurty2020.IT-pls',
])
def test_repetitions(benchmark_identifier):
    num_trials_test.repetitions_test(benchmark_identifier)

# current active tests are checking that 'dicarlo.Sanghavi2020.V4-pls' and 'dicarlo.Sanghavi2020.IT-pls'
# are in set(evaluation_benchmark_pool.keys())
import pytest
from pytest import approx
from brainscore_vision.benchmarks.test_helper import TestStandardized, TestPrecomputed, TestNumberOfTrials

standardized_tests = TestStandardized()
precomputed_test = TestPrecomputed()
num_trials_test = TestNumberOfTrials()


@pytest.mark.parametrize('benchmark, expected', [
    pytest.param('dicarlo.Sanghavi2020.V4-pls', approx(.8892049, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param('dicarlo.Sanghavi2020.IT-pls', approx(.868293, abs=.001),
                 marks=pytest.mark.memory_intense),
])
def test_ceilings(benchmark, expected):
    standardized_tests.test_ceilings(benchmark, expected)


@pytest.mark.parametrize('benchmark, visual_degrees, expected', [
    pytest.param('dicarlo.Sanghavi2020.V4-pls', 8, approx(.9727137, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param('dicarlo.Sanghavi2020.IT-pls', 8, approx(.890062, abs=.001),
                 marks=pytest.mark.memory_intense),
])
def test_self_regression(benchmark, visual_degrees, expected):
    standardized_tests.test_self_regression(benchmark, visual_degrees, expected)


@pytest.mark.memory_intense
@pytest.mark.slow
@pytest.mark.parametrize('benchmark, expected', [
    ('dicarlo.Sanghavi2020.V4-pls', approx(.551135, abs=.015)),
    ('dicarlo.Sanghavi2020.IT-pls', approx(.611347, abs=.015)),
])
def test_Sanghavi2020(benchmark, expected):
    precomputed_test.run_test(benchmark=benchmark, file='alexnet-sanghavi2020-features.12.nc', expected=expected)


@pytest.mark.private_access
@pytest.mark.parametrize('benchmark_identifier', [
    'dicarlo.Sanghavi2020.V4-pls',
    'dicarlo.Sanghavi2020.IT-pls',
])
def test_repetitions(benchmark_identifier):
    num_trials_test.test_repetitions(benchmark_identifier)


# @pytest.mark.private_access
# class TestStandardized:
#     @pytest.mark.parametrize('benchmark, expected', [
#         pytest.param('dicarlo.Sanghavi2020.V4-pls', approx(.8892049, abs=.001),
#                      marks=pytest.mark.memory_intense),
#         pytest.param('dicarlo.Sanghavi2020.IT-pls', approx(.868293, abs=.001),
#                      marks=pytest.mark.memory_intense),
#     ])
#     def test_ceilings(self, benchmark, expected):
#         benchmark = load_benchmark(benchmark)
#         ceiling = benchmark.ceiling
#         assert ceiling.sel(aggregation='center') == expected
#
#     @pytest.mark.parametrize('benchmark, visual_degrees, expected', [
#         pytest.param('dicarlo.Sanghavi2020.V4-pls', 8, approx(.9727137, abs=.001),
#                      marks=pytest.mark.memory_intense),
#         pytest.param('dicarlo.Sanghavi2020.IT-pls', 8, approx(.890062, abs=.001),
#                      marks=pytest.mark.memory_intense),
#     ])
#     def test_self_regression(self, benchmark, visual_degrees, expected):
#         benchmark = load_benchmark(benchmark)
#         score = benchmark(PrecomputedFeatures(benchmark._assembly, visual_degrees=visual_degrees)).raw
#         assert score.sel(aggregation='center') == expected
#         raw_values = score.attrs['raw']
#         assert hasattr(raw_values, 'neuroid')
#         assert hasattr(raw_values, 'split')
#         assert len(raw_values['split']) == 10
#
#     @pytest.mark.memory_intense
#     @pytest.mark.slow
#     @pytest.mark.parametrize('benchmark, expected', [
#         ('dicarlo.Sanghavi2020.V4-pls', approx(.551135, abs=.015)),
#         ('dicarlo.Sanghavi2020.IT-pls', approx(.611347, abs=.015)),
#     ])
#     def test_Sanghavi2020(self, benchmark, expected):
#         self.run_test(benchmark=benchmark, file='alexnet-sanghavi2020-features.12.nc', expected=expected)

# class TestNumberOfTrials:
#     @pytest.mark.private_access
#     @pytest.mark.parametrize('benchmark_identifier', ['dicarlo.Sanghavi2020.V4-pls', 'dicarlo.Sanghavi2020.IT-pls']
#


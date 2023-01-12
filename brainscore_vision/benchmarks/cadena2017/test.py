import pytest
from pytest import approx
from brainscore_vision.benchmark_helpers.test_helper import TestStandardized, TestVisualDegrees

# should these be in function definitions
standardized_tests = TestStandardized()
visual_degrees_test = TestVisualDegrees()


@pytest.mark.parametrize('benchmark, visual_degrees, expected', [
    pytest.param('tolias.Cadena2017-pls', 2, approx(.577474, abs=.005),
                 marks=pytest.mark.private_access),
])
def test_self_regression(benchmark, visual_degrees, expected):
    standardized_tests.self_regression_test(benchmark, visual_degrees, expected)


@pytest.mark.parametrize('benchmark, candidate_degrees, image_id, expected', [
    pytest.param('tolias.Cadena2017-pls', 14, '0fe27ddd5b9ea701e380063dc09b91234eba3551', approx(.32655, abs=.0001),
                 marks=[pytest.mark.private_access]),
    pytest.param('tolias.Cadena2017-pls', 6, '0fe27ddd5b9ea701e380063dc09b91234eba3551', approx(.29641, abs=.0001),
                 marks=[pytest.mark.private_access]),
])
def test_amount_gray(benchmark, candidate_degrees, image_id, expected, brainio_home, resultcaching_home,
                     brainscore_home):
    visual_degrees_test.amount_gray_test(benchmark, candidate_degrees, image_id, expected, brainio_home,
                                         resultcaching_home, brainscore_home)

import pytest
from pytest import approx
from brainscore_vision.benchmark_helpers.test_helper import StandardizedTests, VisualDegreesTests

standardized_tests = StandardizedTests()
visual_degrees_test = VisualDegreesTests()


@pytest.mark.private_access
def test_self_regression():
    standardized_tests.self_regression_test(
        benchmark='tolias.Cadena2017-pls', visual_degrees=2, expected=approx(.577474, abs=.005))


@pytest.mark.parametrize('benchmark, candidate_degrees, image_id, expected', [
    pytest.param('tolias.Cadena2017-pls', 14, '0fe27ddd5b9ea701e380063dc09b91234eba3551', approx(.32655, abs=.0001),
                 marks=[pytest.mark.private_access]),
    pytest.param('tolias.Cadena2017-pls', 6, '0fe27ddd5b9ea701e380063dc09b91234eba3551', approx(.29641, abs=.0001),
                 marks=[pytest.mark.private_access]),
])
def test_amount_gray(benchmark: str, candidate_degrees: int, image_id: str, expected: float):
    visual_degrees_test.amount_gray_test(benchmark, candidate_degrees, image_id, expected)

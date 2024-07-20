import pytest
from pytest import approx

from brainscore_vision.benchmark_helpers import check_standard_format
from brainscore_vision.benchmark_helpers.test_helper import StandardizedTests, VisualDegreesTests
from brainscore_vision.benchmarks.cadena2017.benchmark import AssemblyLoader

standardized_tests = StandardizedTests()
visual_degrees_test = VisualDegreesTests()


@pytest.mark.private_access
class TestAssembly:
    def test(self):
        loader = AssemblyLoader()
        assembly = loader()
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_identifier'] == 'Cadena2017'
        assert len(assembly['presentation']) == 6249
        assert len(assembly['neuroid']) == 166


@pytest.mark.private_access
def test_self_regression():
    standardized_tests.self_regression_test(
        benchmark='Cadena2017-pls', visual_degrees=2, expected=approx(.577474, abs=.005))


@pytest.mark.private_access
@pytest.mark.parametrize('candidate_degrees, image_id, expected', [
    (14, '0fe27ddd5b9ea701e380063dc09b91234eba3551', approx(.32655, abs=.0001)),
    (6, '0fe27ddd5b9ea701e380063dc09b91234eba3551', approx(.29641, abs=.0001)),
])
def test_amount_gray(candidate_degrees: int, image_id: str, expected: float):
    visual_degrees_test.amount_gray_test('Cadena2017-pls', candidate_degrees, image_id, expected)

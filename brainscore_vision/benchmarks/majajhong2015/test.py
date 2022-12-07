import pytest
from pytest import approx

from brainscore_vision.metrics.ceiling import InternalConsistency
from .benchmark import load_assembly


@pytest.mark.private_access
def test_IT_ceiling():
    assembly_repetitions = load_assembly(average_repetitions=False, region='IT')
    ceiler = InternalConsistency()
    ceiling = ceiler(assembly_repetitions)
    assert ceiling.sel(aggregation='center') == approx(.82, abs=.01)

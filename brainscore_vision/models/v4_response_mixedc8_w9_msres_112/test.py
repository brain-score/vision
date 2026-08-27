import pytest

from brainscore_vision import score

from .model import IDENTIFIER, LAYERS, get_model


def test_model_loads():
    model = get_model()
    assert model is not None
    assert LAYERS[-1] == "taps.head4"


@pytest.mark.memory_intense
def test_public_objectome_score():
    actual = score(
        model_identifier=IDENTIFIER,
        benchmark_identifier="Rajalingham2018public-i2n",
    )
    assert float(actual) == pytest.approx(0.3474, abs=0.002)

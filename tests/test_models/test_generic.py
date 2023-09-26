import pytest

import brainscore_vision


@pytest.mark.parametrize('identifier',
                         # TODO: replace with generic collection of all models
                         # ['sketch_model-4o-ep10'])
                         ['pixels'])
def test_has_identifier(identifier):
    model = brainscore_vision.load_model(identifier)
    assert model.identifier == identifier

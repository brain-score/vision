# ##
# generic tests for model plugins
# ##

import pytest

import brainscore_vision


@pytest.mark.travis_slow
@pytest.mark.parametrize('identifier',
                         ['pixels'])
def test_has_identifier(identifier):
    model = brainscore_vision.load_model(identifier)
    assert model.identifier == identifier

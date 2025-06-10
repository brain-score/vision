import pytest
import brainscore_vision

@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('briaai_rmbg_1_4')
    assert model.identifier == 'briaai_rmbg_1_4'
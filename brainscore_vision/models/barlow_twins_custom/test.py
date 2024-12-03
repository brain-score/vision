import pytest
import brainscore_vision

@pytest.mark.travis_slow
def test_barlow_twins_custom():
    model = brainscore_vision.load_model('barlow_twins_custom')
    assert model.identifier == 'barlow_twins_custom'



# AssertionError: No registrations found for resnet18_random
# ⚡ master ~/vision python -m brainscore_vision score --model_identifier='resnet50_tutorial' --benchmark_identifier='MajajHong2015public.IT-pls'
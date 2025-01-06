import pytest
import brainscore_vision

@pytest.mark.travis_slow
def test_resnet18_random():
    model = brainscore_vision.load_model('resnet18_random')
    assert model.identifier == 'resnet18_random'



# AssertionError: No registrations found for resnet18_random
# ⚡ master ~/vision python -m brainscore_vision score --model_identifier='resnet50_tutorial' --benchmark_identifier='MajajHong2015public.IT-pls'
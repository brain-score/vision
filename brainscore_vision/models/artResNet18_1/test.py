import pytest
import brainscore_vision

@pytest.mark.travis_slow
def test_artResNet18_1():
    model = brainscore_vision.load_model('artResNet18_1')
    assert model.identifier == 'artResNet18_1'



# AssertionError: No registrations found for resnet18_random
# ⚡ master ~/vision python -m brainscore_vision score --model_identifier='resnet50_tutorial' --benchmark_identifier='MajajHong2015public.IT-pls'
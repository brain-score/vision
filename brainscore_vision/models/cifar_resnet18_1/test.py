import pytest
import brainscore_vision

@pytest.mark.travis_slow
def test_cifar_resnet18_1():
    model = brainscore_vision.load_model('cifar_resnet18_1')
    assert model.identifier == 'cifar_resnet18_1'




import pytest
from brainscore_vision import load_stimulus_set


@pytest.mark.private_access
@pytest.mark.slow
class TestImagenetC:
    def test_noise(self):
        stimulus_set = load_stimulus_set('imagenet_c.noise')
        assert len(stimulus_set) == 3 * 5 * 50000
        assert len(set(stimulus_set['synset'])) == 1000

    def test_blur(self):
        stimulus_set = load_stimulus_set('imagenet_c.blur')
        assert len(stimulus_set) == 4 * 5 * 50000
        assert len(set(stimulus_set['synset'])) == 1000

    def test_weather(self):
        stimulus_set = load_stimulus_set('imagenet_c.weather')
        assert len(stimulus_set) == 4 * 5 * 50000
        assert len(set(stimulus_set['synset'])) == 1000

    def test_digital(self):
        stimulus_set = load_stimulus_set('imagenet_c.digital')
        assert len(stimulus_set) == 4 * 5 * 50000
        assert len(set(stimulus_set['synset'])) == 1000

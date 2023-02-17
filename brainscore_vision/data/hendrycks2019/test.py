import pytest
from brainscore_vision import load_stimulus_set


@pytest.mark.private_access
@pytest.mark.slow
class TestDietterichHendrycks2019:
    def test_noise(self):
        stimulus_set = load_stimulus_set('dietterich.Hendrycks2019.noise')
        assert len(stimulus_set) == 3 * 5 * 50000
        assert len(set(stimulus_set['synset'])) == 1000

    def test_blur(self):
        stimulus_set = load_stimulus_set('dietterich.Hendrycks2019.blur')
        assert len(stimulus_set) == 4 * 5 * 50000
        assert len(set(stimulus_set['synset'])) == 1000

    def test_weather(self):
        stimulus_set = load_stimulus_set('dietterich.Hendrycks2019.weather')
        assert len(stimulus_set) == 4 * 5 * 50000
        assert len(set(stimulus_set['synset'])) == 1000

    def test_digital(self):
        stimulus_set = load_stimulus_set('dietterich.Hendrycks2019.digital')
        assert len(stimulus_set) == 4 * 5 * 50000
        assert len(set(stimulus_set['synset'])) == 1000
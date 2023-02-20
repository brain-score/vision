import pytest

from candidate_models.model_commitments import brain_translated_pool


@pytest.mark.memory_intense
@pytest.mark.private_access
class TestBestLayers:
    @pytest.mark.parametrize(['model_identifier', 'expected'], [
        ('alexnet', {'V1': 'features.2', 'V2': 'features.7', 'V4': 'features.7', 'IT': 'features.12'}),
    ])
    def test(self, model_identifier, expected):
        model = brain_translated_pool[model_identifier]
        for region in expected:  # need to first initialize mapping (only run lazily)
            model.start_recording(region, [(70, 170)])
        assert model.layer_model.region_layer_map == expected

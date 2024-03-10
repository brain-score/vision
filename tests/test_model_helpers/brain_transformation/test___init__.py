from unittest.mock import Mock

from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.utils import fullname


class TestVisualDegrees:
    def test_standard_commitment(self):
        # create mock ActivationsExtractorHelper with a mock set_visual_degrees to avoid failing set_visual_degrees()
        mock_extractor = Mock()
        mock_extractor.set_visual_degrees = Mock()
        mock_activations_model = Mock()
        mock_activations_model._extractor = mock_extractor

        # Initialize ModelCommitment with the mock activations_model
        brain_model = ModelCommitment(identifier=fullname(self), activations_model=mock_activations_model,
                                      layers=['dummy'])
        assert brain_model.visual_degrees() == 8

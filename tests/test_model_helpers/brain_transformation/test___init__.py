from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.utils import fullname


class TestVisualDegrees:
    def test_standard_commitment(self):
        brain_model = ModelCommitment(identifier=fullname(self), activations_model=None,
                                      layers=['dummy'])
        assert brain_model.visual_degrees() == 8

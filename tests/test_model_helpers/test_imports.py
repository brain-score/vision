# noinspection PyUnresolvedReferences
class TestImport:
    def test_model_helpers(self):
        import brainscore_vision.model_helpers

    def test_activations_extractor(self):
        from brainscore_vision.model_helpers.activations.core import ActivationsExtractorHelper

    def test_model_commitment(self):
        from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

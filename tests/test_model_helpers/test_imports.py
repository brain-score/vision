# noinspection PyUnresolvedReferences
class TestImport:
    def test_model_tools(self):
        import model_tools

    def test_activations_extractor(self):
        from model_tools.activations.core import ActivationsExtractorHelper

    def test_model_commitment(self):
        from model_tools.brain_transformation import ModelCommitment

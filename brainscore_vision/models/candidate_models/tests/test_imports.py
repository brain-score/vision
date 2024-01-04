# noinspection PyUnresolvedReferences
class TestImport:
    def test_base_models(self):
        from candidate_models.base_models import base_model_pool

    def test_brain_translated(self):
        from candidate_models.model_commitments import brain_translated_pool

    def test_cornet(self):
        from candidate_models.base_models.cornet import cornet

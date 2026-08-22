from .descriptor import FoveatedGaborFilterEntropyExtractor


def test_feature_count():
    extractor = FoveatedGaborFilterEntropyExtractor()
    assert len(extractor.feature_names) == 2916

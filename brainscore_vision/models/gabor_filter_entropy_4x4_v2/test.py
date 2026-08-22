from .descriptor import FoveatedGaborFilterEntropyExtractor


def test_feature_count():
    extractor = FoveatedGaborFilterEntropyExtractor()
    n_filters = len(extractor.config.spatial_frequencies) * extractor.config.n_orientations
    n_positions = len(range(0, extractor.config.image_size - extractor.mid_window + 1, extractor.base_stride))
    assert n_filters * n_positions * n_positions == 2916

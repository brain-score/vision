from .model import get_layers


def test_layers():
    layers = get_layers("deepeisnn_imagenet_resnet18_lif")
    assert len(layers) > 0
    assert all(isinstance(layer, str) for layer in layers)

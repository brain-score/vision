from .model import IDENTIFIER, LAYERS, get_model


def test_model_loads():
    model = get_model()
    assert model is not None
    assert IDENTIFIER == "task_only_mixedc8_w9_msres_112"
    assert LAYERS[-1] == "taps.head4"

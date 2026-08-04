import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model("persistence_memory")
    assert model.identifier == "persistence_memory"

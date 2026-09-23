from brainscore_vision import load_model


def test_load_model():
    model = load_model('jem_imagenet_p0p0_s1')
    assert model.identifier == 'jem_imagenet_p0p0_s1'
    # Brain-Score reads 'logits' from the last registered leaf module: it must be the 1000-way classifier
    activations_model = model.activations_model
    assert activations_model._output_layer() is activations_model._model.jem.class_output

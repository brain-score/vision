import numpy as np
from PIL import Image

from brainscore_vision import load_model
from brainscore_vision.model_interface import BrainModel

from .model import IDENTIFIER, LAYERS, preprocess_images


def test_preprocessing(tmp_path):
    image_path = tmp_path / "solid_rgb.png"
    Image.new("RGB", (360, 256), (64, 128, 255)).save(image_path)
    batch = preprocess_images([str(image_path)])
    assert batch.shape == (1, 3, 224, 224)
    assert batch.dtype == np.float32
    np.testing.assert_allclose(batch[0, :, 0, 0], [64 / 255, 128 / 255, 1], rtol=1e-6)


def test_brainmodel_interfaces(tmp_path):
    image_path = tmp_path / "stimulus.png"
    Image.new("RGB", (360, 256), (64, 128, 255)).save(image_path)
    model = load_model(IDENTIFIER)
    assert model.identifier == IDENTIFIER
    assert model.visual_degrees() == 8
    assembly = model.activations_model(
        [str(image_path)], layers=LAYERS + ["features.avgpool", "logits"],
    )
    assert assembly.sizes["presentation"] == 1
    assert set(assembly["layer"].values) == set(LAYERS + ["features.avgpool", "logits"])
    assert np.isfinite(assembly.values).all()
    model.start_task(BrainModel.Task.label, "imagenet")
    labels = model.look_at([str(image_path)])
    assert labels.sizes["presentation"] == 1
    assert str(labels.values.ravel()[0]).startswith("n")

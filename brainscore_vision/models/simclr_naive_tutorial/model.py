from brainscore_vision.model_helpers.check_submission import check_models
import functools
import torchvision.models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import timm
# This is an example implementation for submitting resnet-50 as a pytorch model

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.


def get_model(name):
    assert name == 'simclr_naive_tutorial'
    model = timm.create_model(
        model_name="hf-hub:1aurent/resnet50.tcga_brca_simclr",
        pretrained=True,
    ).eval()

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='simclr_naive_tutorial', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == 'simclr_naive_tutorial'
    return ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']

if __name__ == '__main__':
    check_models.check_base_models(__name__)
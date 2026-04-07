from brainscore_vision.model_helpers.check_submission import check_models
import functools
import timm
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

def get_model(name):
    assert name == 'convnext_base_fb_in22k_ft_in1k'
    model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='convnext_base_fb_in22k_ft_in1k', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == 'convnext_base_fb_in22k_ft_in1k'
    return ['stem', 'stages.0', 'stages.1', 'stages.2', 'stages.3', 'head']

def get_bibtex(model_identifier):
    return """"""

if __name__ == '__main__':
    check_models.check_base_models(__name__)

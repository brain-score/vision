from brainscore_vision.model_helpers.check_submission import check_models
import functools
import timm
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

def get_model(name):
    assert name == 'deit3_base_patch16_224_fb_in22k_ft_in1k'
    model = timm.create_model('deit3_base_patch16_224.fb_in22k_ft_in1k', pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='deit3_base_patch16_224_fb_in22k_ft_in1k', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == 'deit3_base_patch16_224_fb_in22k_ft_in1k'
    return ['patch_embed', 'blocks.0', 'blocks.2', 'blocks.4', 'blocks.6', 'blocks.8', 'blocks.10', 'blocks.11', 'norm']

def get_bibtex(model_identifier):
    return """"""

if __name__ == '__main__':
    check_models.check_base_models(__name__)

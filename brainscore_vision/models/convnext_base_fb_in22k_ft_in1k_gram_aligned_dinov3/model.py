from brainscore_vision.model_helpers.check_submission import check_models
import functools
import timm
import torch
from huggingface_hub import hf_hub_download
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

def get_model(name):
    assert name == 'convnext_base_fb_in22k_ft_in1k_gram_aligned_dinov3'
    model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=False)
    checkpoint_path = hf_hub_download(
        repo_id='HosseinAdeli/fusion_models',
        filename='gram_aligned/convnext_base.fb_in22k_ft_in1k_gram_aligned_dinov3/checkpoint.pth'
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='convnext_base_fb_in22k_ft_in1k_gram_aligned_dinov3', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == 'convnext_base_fb_in22k_ft_in1k_gram_aligned_dinov3'
    return ['stem', 'stages.0', 'stages.1', 'stages.2', 'stages.3', 'head']

def get_bibtex(model_identifier):
    return """"""

if __name__ == '__main__':
    check_models.check_base_models(__name__)

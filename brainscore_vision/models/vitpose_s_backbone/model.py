import functools, timm
from brainscore_vision.model_helpers.activations.pytorch import (
    PytorchWrapper, load_preprocess_images
)

def get_model():
    vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0, global_pool='')
    pre = functools.partial(load_preprocess_images, image_size=224)
    w = PytorchWrapper(identifier='vitpose_s_backbone', model=vit, preprocessing=pre)
    w.image_size = 224
    return w

def get_layers():
    return ['blocks.2', 'blocks.6', 'blocks.10', 'norm']

import functools, torch
from torchvision.models import alexnet
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
def get_model():
    m = alexnet(weights='IMAGENET1K_V1'); m.classifier[-1] = torch.nn.Identity()
    pre = functools.partial(load_preprocess_images, image_size=224)
    w = PytorchWrapper(identifier='alex_hello', model=m, preprocessing=pre); w.image_size=224
    return w
def get_layers(): return ['features.3','features.6','features.8','classifier.5']

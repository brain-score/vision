import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from pytorch_pretrained_vit import ViT
import ssl


ssl._create_default_https_context = ssl._create_unverified_context


# Visual Transformer (ViT)
# Using PyTorch implementation and converted weights from https://github.com/lukemelas/PyTorch-Pretrained-ViT

def get_model(name):
    name = name[4:]
    model = ViT(name, pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=model.image_size[0])
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = model.image_size[0]
    return wrapper


def get_layers(name):
    name = name[4:]
    number_of_blocks = 12 if name.startswith('B') else 24
    return [f'transformer.blocks.{i}.pwff.fc2' for i in range(number_of_blocks)]
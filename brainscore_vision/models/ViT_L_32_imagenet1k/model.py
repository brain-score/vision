import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
from pytorch_pretrained_vit import ViT
import ssl


ssl._create_default_https_context = ssl._create_unverified_context


# Visual Transformer (ViT)
# Using PyTorch implementation and converted weights from https://github.com/lukemelas/PyTorch-Pretrained-ViT

def get_model_list():
    return ['ViT_L_32_imagenet1k']


def get_model(name):
    assert name == 'ViT_L_32_imagenet1k'
    name = name[4:]
    model = ViT(name, pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=model.image_size[0])
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = model.image_size[0]
    return wrapper


def get_layers(name):
    assert name == 'ViT_L_32_imagenet1k'
    name = name[4:]
    number_of_blocks = 12 if name.startswith('B') else 24
    return [f'transformer.blocks.{i}.pwff.fc2' for i in range(number_of_blocks)]


def get_bibtex(model_identifier):
    return """@inproceedings{
                dosovitskiy2021an,
                title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
                author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
                booktitle={International Conference on Learning Representations},
                year={2021},
                url={https://openreview.net/forum?id=YicbFdNTTy}
                }"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
import functools


from pytorch_pretrained_vit import ViT
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images

# Visual Transformer (ViT)
# Using PyTorch implementation and converted weights from https://github.com/lukemelas/PyTorch-Pretrained-ViT

from model_tools.check_submission import check_models


def get_model_list():
    return ['B_16_imagenet1k', 'B_32_imagenet1k', 'L_16_imagenet1k', 'L_32_imagenet1k', 'B_16', 'B_32', 'L_32']
    #  the non-imagenet1k have 21k outputs; no pretrained weights available for L16


def get_model(name):
    assert name in ['B_16', 'B_32', 'L_32', 'B_16_imagenet1k', 'B_32_imagenet1k', 'L_16_imagenet1k', 'L_32_imagenet1k']
    model = ViT(name, pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=model.image_size[0])
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = model.image_size[0]
    return wrapper


def get_layers(name):
    assert name in ['B_16', 'B_32', 'L_32', 'B_16_imagenet1k', 'B_32_imagenet1k', 'L_16_imagenet1k', 'L_32_imagenet1k']
    number_of_blocks = 12 if name.startswith('B') else 24
    return [f'transformer.blocks.{i}.pwff.fc2' for i in range(number_of_blocks)]

def get_bibtex(model_identifier):
    return """@misc{dosovitskiy2020image,
                    title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale}, 
                    author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
                    year={2020},
                    eprint={2010.11929},
                    archivePrefix={arXiv},
                    primaryClass={cs.CV}
                    }"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)

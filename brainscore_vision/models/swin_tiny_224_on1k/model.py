from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_images, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
import ssl
from transformers import AutoModelForImageClassification
import functools

ssl._create_default_https_context = ssl._create_unverified_context

'''
Can be found on huggingface: https://huggingface.co/microsoft/swin-tiny-patch4-window7-224
'''

MODEL = AutoModelForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")


def get_model(name):
    assert name == 'swin_tiny_224_on1k'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='swin_tiny_224_on1k', model=MODEL,
                             preprocessing=preprocessing,
                             batch_size=4)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'swin_tiny_224_on1k'
    layer_names = []

    for name, module in MODEL.named_modules():
        layer_names.append(name)

    print(layer_names)
    print(f"Number of layers:{len(layer_names)}")

    return layer_names[2:]


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@article{DBLP:journals/corr/abs-2103-14030,
                  author    = {Ze Liu and
                               Yutong Lin and
                               Yue Cao and
                               Han Hu and
                               Yixuan Wei and
                               Zheng Zhang and
                               Stephen Lin and
                               Baining Guo},
                  title     = {Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
                  journal   = {CoRR},
                  volume    = {abs/2103.14030},
                  year      = {2021},
                  url       = {https://arxiv.org/abs/2103.14030},
                  eprinttype = {arXiv},
                  eprint    = {2103.14030},
                  timestamp = {Thu, 08 Apr 2021 07:53:26 +0200},
                  biburl    = {https://dblp.org/rec/journals/corr/abs-2103-14030.bib},
                  bibsource = {dblp computer science bibliography, https://dblp.org}
                }
              """


if __name__ == '__main__':
    check_models.check_base_models(__name__)

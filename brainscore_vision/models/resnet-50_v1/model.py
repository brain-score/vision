import functools
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.s3 import load_weight_file
import torch
import importlib.util


# def load_module(module_name, module_path):
#     spec = importlib.util.spec_from_file_location(module_name, module_path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module
#
#
# model_path = load_weight_file(bucket="brainscore-vision", folder_name="models",
#                                     relative_path="resnet-50_v1/tf_resnet_to_pth.py",
#                                     version_id="EvvsqSCrjI3yIBRtPdm.sG971Ne6LMzZ",
#                                     sha1="c1ae529e0368e0c1804b2d6ab2feea443734023f")
# model_weight_path = load_weight_file(bucket="brainscore-vision", folder_name="models",
#                                     relative_path="resnet-50_v1/tf_resnet_to_pth.pth",
#                                     version_id="29SKJxBWqkwARadLKKH5pg9yS4pGi2HL",
#                                     sha1="11bf09095fbcbf6b6ad109a574c691c12b339374")
# MainModel = load_module('MainModel', model_path.as_posix())
# globals()['MainModel'] = MainModel
# model = torch.load(model_weight_path.as_posix())

def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'resnet-50_v1'
    preprocessing = functools.partial(load_preprocess_images, image_size=224, preprocess_type='vgg')
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'resnet-50_v1'
    units = [3, 4, 6, 3]
    layer_names = ['resnet_v1_50_conv1_Conv2D'] + [
        f'resnet_v1_50_block{block + 1}_unit_{unit + 1}_bottleneck_v1_conv1_Conv2d' for block, block_units in
        enumerate(units) for unit in range(block_units)] + ['resnet_v1_50_logits_Conv2D']
    return layer_names

def get_bibtex(name):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return '''
@article{DBLP:journals/corr/ZophVSL17,
  author       = {Barret Zoph and
                  Vijay Vasudevan and
                  Jonathon Shlens and
                  Quoc V. Le},
  title        = {Learning Transferable Architectures for Scalable Image Recognition},
  journal      = {CoRR},
  volume       = {abs/1707.07012},
  year         = {2017},
  url          = {http://arxiv.org/abs/1707.07012},
  eprinttype    = {arXiv},
  eprint       = {1707.07012},
  timestamp    = {Mon, 13 Aug 2018 16:48:00 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/ZophVSL17.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
'''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
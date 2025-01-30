import functools
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.s3 import load_weight_file
from brainscore_vision.models.mobilenet_v2_0_5_192.model import MobilenetPytorchWrapper
import torch
import imp

model_path = load_weight_file(bucket="brainscore-storage", folder_name="brainscore-vision/models",
                                    relative_path="mobilenet_v2_0.75_192/mobilenet_v2_0.py",
                                    version_id="null",
                                    sha1="8d253c2faad210834b4d39b9ccc644165ed8e3f6")
model_weight_path = load_weight_file(bucket="brainscore-storage", folder_name="brainscore-vision/models",
                                    relative_path="mobilenet_v2_0.75_192/mobilenet_v2_0.75_192_frozen.pth",
                                    version_id="null",
                                    sha1="af063236e83cb92fd78ed3eb7d9d2d4a65d794ab")
MainModel = imp.load_source('MainModel', model_path.as_posix())


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'mobilenet_v2_0_75_192'
    preprocessing = functools.partial(load_preprocess_images, image_size=192, preprocess_type='inception')
    model = torch.load(model_weight_path.as_posix(), weights_only=False)
    wrapper = MobilenetPytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 192
    return wrapper


def get_layers(name):
    assert name == 'mobilenet_v2_0_75_192'
    layer_names = (['MobilenetV2_Conv_Conv2D'] +
                   [f'MobilenetV2_expanded_conv_{i}_expand_Conv2D' for i in range(1, 17)] +
                   ['MobilenetV2_Conv_1_Conv2D'])
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

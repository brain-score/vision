from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from torchvision.models import resnet50
from brainscore_vision.model_helpers.s3 import load_weight_file
import functools
import torch
import os
dir_path = os.path.dirname(os.path.realpath(__file__))


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'resnet50-vicregl0p75'
    #model = torchvision.models.resnet50(pretrained=True)
    model = resnet50(pretrained=False).eval()
    weights_path = load_weight_file(bucket="brainscore-storage", folder_name="brainscore-vision/models",
                                   relative_path="resnet50-vicregl0p75/resnet50_alpha0p75.pth",
                                   version_id="null",
                                   sha1="37a5e0d06050260d73ffbeb3ad163974e4361435")
    ckpt= torch.load(weights_path, map_location=torch.device('cpu'))
    ckpt['fc.weight'] = model.fc.weight
    ckpt['fc.bias'] = model.fc.bias
    model.load_state_dict(ckpt)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='resnet50-vicregl0p75', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'resnet50-vicregl0p75'
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """
    return ['conv1', 'layer1.0', 'layer1.1', 'layer1.2', 
        'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3',
        'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3',
         'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2','avgpool']


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return '''@article{DBLP:journals/corr/HeZRS15,
  author       = {Kaiming He and
                  Xiangyu Zhang and
                  Shaoqing Ren and
                  Jian Sun},
  title        = {Deep Residual Learning for Image Recognition},
  journal      = {CoRR},
  volume       = {abs/1512.03385},
  year         = {2015},
  url          = {http://arxiv.org/abs/1512.03385},
  eprinttype    = {arXiv},
  eprint       = {1512.03385},
  timestamp    = {Wed, 25 Jan 2023 11:01:16 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/HeZRS15.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}'''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)

from brainscore_vision.model_helpers.check_submission import check_models
from .helpers.resnet_helpers import res2net50_26w_4s
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images



def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """

    assert name == 'Res2Net50_26w_4s'

    model = res2net50_26w_4s(pretrained=False)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='Res2Net50_26w_4s', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


# get_layers method to tell the code what layers to consider. If you are submitting a custom
# model, then you will most likley need to change this method's return values.
def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """

    assert name == 'Res2Net50_26w_4s'
    return ['maxpool', 'layer1.0', 'layer1.1', 'layer1.2',
            'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3',
            'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3',
            'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2', 'avgpool', 'fc']


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """

    # from pytorch.py:
    return '''@article{DBLP:journals/corr/abs-1904-01169,
  author       = {Shanghua Gao and
                  Ming{-}Ming Cheng and
                  Kai Zhao and
                  Xinyu Zhang and
                  Ming{-}Hsuan Yang and
                  Philip H. S. Torr},
  title        = {Res2Net: {A} New Multi-scale Backbone Architecture},
  journal      = {CoRR},
  volume       = {abs/1904.01169},
  year         = {2019},
  url          = {http://arxiv.org/abs/1904.01169},
  eprinttype    = {arXiv},
  eprint       = {1904.01169},
  timestamp    = {Thu, 25 Apr 2019 10:24:54 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1904-01169.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}'''

# Main Method: In submitting a custom model, you should not have to mess with this.
if __name__ == '__main__':
    check_models.check_base_models(__name__)
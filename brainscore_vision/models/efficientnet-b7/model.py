import functools
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
from transformers import AutoModelForImageClassification


model = AutoModelForImageClassification.from_pretrained("google/efficientnet-b7")
def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'efficientnet-b7'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='name', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'efficientnet-b7'
    layer_names = []
    for name, module in model.named_modules():
        layer_names.append(name)

    return layer_names[-50:]


def get_bibtex(name):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return '''
@article{DBLP:journals/corr/abs-1905-11946,
  author       = {Mingxing Tan and
                  Quoc V. Le},
  title        = {EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  journal      = {CoRR},
  volume       = {abs/1905.11946},
  year         = {2019},
  url          = {http://arxiv.org/abs/1905.11946},
  eprinttype    = {arXiv},
  eprint       = {1905.11946},
  timestamp    = {Mon, 03 Jun 2019 13:42:33 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1905-11946.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
'''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
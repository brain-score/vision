import functools
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.s3 import load_weight_file
import torch
import imp

model_path = load_weight_file(bucket="brainscore-vision", folder_name="models",
                                    relative_path="mobilenet_v2_1.0_160/mobilenet_v2_1.py",
                                    version_id="b9SbmlqjH2p34zjET3TZAo.hPyefB7wU",
                                    sha1="cbf3a3e34652f2eb4836929eba848b9dcad26a5c")
model_weight_path = load_weight_file(bucket="brainscore-vision", folder_name="models",
                                    relative_path="mobilenet_v2_1.0_160/mobilenet_v2_1.0_160_frozen.pth",
                                    version_id="uzE3iSMBhnT.nPEj7rVy5mmeoIA9rQc7",
                                    sha1="c588bc1589f49c62dd4e6de014c02f34b358fe04")
MainModel = imp.load_source('MainModel',model_path.as_posix())
model = torch.load(model_weight_path.as_posix()) 
    
def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'mobilenet_v2_1_0_160'
    preprocessing = functools.partial(load_preprocess_images, image_size=160)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 160
    return wrapper


def get_layers(name):
    assert name == 'mobilenet_v2_1_0_160'
    return list(dict(model.named_modules()).keys())[1:]


def get_bibtex(name):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return '''
@article{DBLP:journals/corr/abs-1801-04381,
  author       = {Mark Sandler and
                  Andrew G. Howard and
                  Menglong Zhu and
                  Andrey Zhmoginov and
                  Liang{-}Chieh Chen},
  title        = {Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification,
                  Detection and Segmentation},
  journal      = {CoRR},
  volume       = {abs/1801.04381},
  year         = {2018},
  url          = {http://arxiv.org/abs/1801.04381},
  eprinttype    = {arXiv},
  eprint       = {1801.04381},
  timestamp    = {Tue, 12 Jan 2021 15:30:06 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1801-04381.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
'''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
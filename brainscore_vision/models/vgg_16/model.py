from torchvision.models import vgg16
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
import functools



def get_model(name):
    assert name == 'vgg-16'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    model = vgg16(pretrained = True)
    wrapper = PytorchWrapper(identifier='vgg-16', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'vgg-16'
    return ['features', 'avgpool', 'classifier']

def get_bibtex(model_identifier):
    assert model_identifier == 'vgg-16'
    return """
    @article{DBLP:journals/corr/SimonyanZ14a,
  added-at = {2016-11-19T13:14:27.000+0100},
  author = {Simonyan, Karen and Zisserman, Andrew},
  bibsource = {dblp computer science bibliography, http://dblp.org},
  biburl = {https://www.bibsonomy.org/bibtex/20ee0434e0a70b329d5518f43f1742f7a/albinzehe},
  interhash = {4e6fa56cb7cf99400d5701543ee228de},
  intrahash = {0ee0434e0a70b329d5518f43f1742f7a},
  journal = {CoRR},
  keywords = {cnn ma-zehe neuralnet},
  timestamp = {2016-11-19T13:14:27.000+0100},
  title = {Very Deep Convolutional Networks for Large-Scale Image Recognition},
  url = {http://arxiv.org/abs/1409.1556},
  volume = {abs/1409.1556},
  year = 2014
}"""

    

if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)

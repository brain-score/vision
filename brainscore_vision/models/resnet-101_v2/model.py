from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_images, preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.s3 import load_weight_file
import functools
import onnx
import numpy as np
from onnx2pytorch import ConvertModel
import os

onnx_model_path = load_weight_file(bucket="brainscore-vision", folder_name="models",
                                   relative_path="resnet-101_v2/resnet-101_v2.onnx",
                                   version_id="JPmn3lo1W.VI4lOP5mAZaoBQthE_MDZ1",
                                   sha1="81798740ba5451ee3c4d71eafb7e569eb288edb5")
onnx_model = onnx.load(onnx_model_path.as_posix())
model = ConvertModel(onnx_model)

def load_preprocess_images(image_filepaths, image_size, **kwargs):
    images = load_images(image_filepaths)
    images = preprocess_images(images, image_size=image_size, **kwargs)
    images = np.transpose(images, (0,2,3,1))
    return images

class Pytorch_wrapper(PytorchWrapper):
    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            target_dict[name] = PytorchWrapper._tensor_to_numpy(output = output[0])
        hook = layer.register_forward_hook(hook_function)
        return hook
    
def get_model(name):
    assert name == 'resnet-101_v2'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = Pytorch_wrapper(identifier='resnet-101_v2', model=model, preprocessing=preprocessing,batch_size=1)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == 'resnet-101_v2'
    return [layer for layer, _ in model.named_modules()][1:]

def get_bibtex(model_identifier):
    assert model_identifier == 'resnet-101_v2'
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

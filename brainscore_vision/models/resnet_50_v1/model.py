from torchvision.models import resnet50
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
import functools


model = resnet50(weights='IMAGENET1K_V1')

def get_model(name):
    assert name == 'resnet_50_v1'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='resnet_50_v1', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == 'resnet_50_v1'
    units = [3, 4, 6, 3]
    layer_names = ['conv1'] + [f'layer{block+1}.{unit}' for block, block_units in
                               enumerate(units) for unit in range(block_units)] + ['avgpool']
    return layer_names


def get_bibtex(model_identifier):
    assert model_identifier == 'resnet_50_v1'
    return """
    @inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}"""



if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
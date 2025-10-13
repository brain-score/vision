
# model.py - ResNet-50 구현 예시
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import torchvision.models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

def get_model(name):
    assert name == 'hk_model_1'
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    # Reduce batch size to avoid OOM (Out of Memory) errors
    wrapper = PytorchWrapper(identifier='hk_model_1', model=model, preprocessing=preprocessing, batch_size=8)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == 'hk_model_1'
    # Use more granular layer names to avoid alignment issues
    return ['maxpool', 'layer1.0', 'layer1.1', 'layer1.2',
            'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3',
            'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3',
            'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2',
            'avgpool', 'fc']

def get_bibtex(model_identifier):
    return """@article{he2016deep,
        title={Deep residual learning for image recognition},
        author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
        journal={CVPR},
        year={2016}
    }"""

if __name__ == '__main__':
    check_models.check_base_models(__name__)
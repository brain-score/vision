
# model.py - Self-supervised ResNet-50 (SwAV) implementation
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

def get_model(name):
    assert name == 'hk_model_2'

    # Load SwAV pretrained ResNet-50 from torch hub
    # SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
    model = torch.hub.load('facebookresearch/swav:main', 'resnet50')

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    # Reduce batch size to avoid OOM (Out of Memory) errors
    wrapper = PytorchWrapper(identifier='hk_model_2', model=model, preprocessing=preprocessing, batch_size=8)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == 'hk_model_2'
    # Use simple layer structure following hk_model_1 pattern
    return ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']

def get_bibtex(model_identifier):
    return """@inproceedings{caron2020unsupervised,
        title={Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
        author={Caron, Mathilde and Misra, Ishan and Mairal, Julien and Goyal, Priya and Bojanowski, Piotr and Joulin, Armand},
        booktitle={Advances in Neural Information Processing Systems},
        year={2020}
    }"""

if __name__ == '__main__':
    check_models.check_base_models(__name__)

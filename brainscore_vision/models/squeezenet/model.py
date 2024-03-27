import functools

import torch
import torchvision.models
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

BIBTEX = """@article{iandola2016squeezenet,
    title={SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size},
    author={Iandola, Forrest N and Han, Song and Moskewicz, Matthew W and Ashraf, Khalid and Dally, William J and Keutzer, Kurt},
    journal={arXiv preprint arXiv:1602.07360},
    year={2016}
}"""

net_constructors = {
    "squeezenet1_0": torchvision.models.squeezenet1_0,
    "squeezenet1_1": torchvision.models.squeezenet1_1,
}

# these layer choices were not investigated in any depth, we blindly
# picked all high-level blocks
LAYERS = [
    'classifier',
    *{f'features.{i}' for i in range(13)}
]


def get_model(net):
    assert net in net_constructors, f"Could not find SqueezeNet network: {net}"

    model = net_constructors[net](pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(
        identifier=net,
        model=model,
        preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


# Main Method: In submitting a custom model, you should not have to mess
# with this.
if __name__ == "__main__":
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)

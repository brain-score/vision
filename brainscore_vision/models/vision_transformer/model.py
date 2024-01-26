import functools

import torch
import torchvision.models
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images


BIBTEX = """@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}"""

net_constructors = {
    "vit_b_16": torchvision.models.vit_b_16,
    "vit_b_32": torchvision.models.vit_b_32,
    "vit_l_16": torchvision.models.vit_l_16,
    "vit_l_32": torchvision.models.vit_l_32,
}

include_layer_names = {
    "encoder.ln",
    *{f"encoder.layers.encoder_layer_{i}" for i in range(1, 24)},
}


def get_layers(net):
    assert net in net_constructors, f"Could not find ViT network: {net}"
    model = net_constructors[net](pretrained=True)

    # Reverse layer order to avoid channel_x affecting transformer layers
    return [layer for layer, _ in model.named_modules()
            if layer in include_layer_names][::-1]


def get_model(net):
    assert net in net_constructors, f"Could not find ViT network: {net}"

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

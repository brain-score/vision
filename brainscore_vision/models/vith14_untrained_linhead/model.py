"""
Model definition of an untrained ViT-H/14 model for submission to Brain-Score.
"""

import functools
import math
from collections import OrderedDict

import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import IJepaConfig, IJepaModel

# from transformers import AutoModel
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.

NUM_LAYERS = 32
NUM_CLASSES = 1000
REPO_ID = "niels-c/vith14_untrained_linhead"
FILENAME = "vith14_untrained_linhead.safetensors"
REVISION = "11a552a822cce283537351d091130152261c7d59"

# Configuration of facebook/ijepa_vith14_1k
CONFIG = IJepaConfig(
    hidden_size=1280,
    intermediate_size=5120,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=16,
    patch_size=14,
    image_size=224,
    layer_norm_eps=1e-6,
)


class IJepaClassifier(IJepaModel):
    """
    I-JEPA encoder with a linear head (fc) on the average of the final patch tokens, returning logits.
    """

    def __init__(self, config: IJepaConfig):
        super().__init__(config)
        self.fc = nn.Linear(config.hidden_size, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(super().forward(pixel_values=images).last_hidden_state.mean(dim=1))


def load_model() -> nn.Module:
    """
    Load the network from its checkpoint on the Hugging Face Hub.

    Returns:
        nn.Module: A Sequential of "backbone", the encoder whose layer fc is the linear head (logits), in eval mode.
    """

    weights = load_file(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, revision=REVISION))
    backbone = IJepaClassifier(CONFIG)
    backbone.load_state_dict(weights)
    return nn.Sequential(OrderedDict(backbone=backbone)).eval()


class TokenGrid(nn.Module):
    """
    Arrange patch tokens (batch, tokens, dim) as a channels-first grid (batch, dim, side, side).

    Brain-Score names the coordinates of token activations channel and embedding, whereas the mask regression of
    Cadena2017-mask needs channel, channel_x and channel_y, as for a convolutional feature map. The features themselves
    are unchanged.
    """

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        side = math.isqrt(tokens.shape[1])
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)


def add_token_grids(model: nn.Module) -> None:
    """
    Give every transformer block a "grid" submodule that reproduces its output as a TokenGrid.

    The grid is computed from a forward hook that returns nothing, so the tokens still flow to the next block.

    Args:
        model (nn.Module): The network, with blocks at backbone.layers.
    """

    def compute_grid(block, _inputs, output):
        block.grid(output)

    for i in range(NUM_LAYERS):
        block = model.get_submodule(f"backbone.layers.{i}")
        block.add_module("grid", TokenGrid())
        block.register_forward_hook(compute_grid)


def get_model(name):
    assert name == "vith14_untrained_linhead"
    image_size = 224
    model = load_model()  # head at backbone.fc, logits
    add_token_grids(model)
    # I-JEPA was pretrained with ImageNet mean and std, the default of load_preprocess_images.
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = PytorchWrapper(identifier="vith14_untrained_linhead", model=model, preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper


def get_layers(name):
    assert name == "vith14_untrained_linhead"
    # Outputs of the transformer blocks as grids, named as in transformers>=5.
    return [f"backbone.layers.{i}.grid" for i in range(NUM_LAYERS)] + ["backbone.fc"]


def get_bibtex(model_identifier):
    return """"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)

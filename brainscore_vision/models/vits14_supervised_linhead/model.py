"""
Model definition of a supervised ViT-S/14 model for submission to Brain-Score.
"""

import functools
import math
from collections import OrderedDict
from pathlib import Path

import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
from timm.models.vision_transformer import VisionTransformer
from torch import nn

# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.

NUM_LAYERS = 12
NUM_CLASSES = 1000
IMAGE_SIZE = 224
REPO_ID = "niels-c/vits14_supervised_linhead"
FILENAME = "vits14_supervised_linhead.safetensors"


class ViTS(VisionTransformer):
    """
    timm ViT-S/14 without class token, with a linear head (fc) on the average of the final patch tokens, returning
    logits.

    It is the backbone of the training runs, built there with timm.create_model("vit_small_patch16_224",
    patch_size=14, class_token=False, global_pool="avg", fc_norm=False).
    """

    def __init__(self):
        super().__init__(
            img_size=IMAGE_SIZE,
            patch_size=14,
            embed_dim=384,
            depth=NUM_LAYERS,
            num_heads=6,
            num_classes=0,
            class_token=False,
            global_pool="avg",
            fc_norm=False,
        )
        self.fc = nn.Linear(self.embed_dim, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(super().forward(images))


def load_model() -> nn.Module:
    """
    Load the network from its checkpoint on the Hugging Face Hub.

    Returns:
        nn.Module: A Sequential of "backbone", the ViT-S/14 whose layer fc is the linear head (logits), in eval mode.
    """

    weights = load_file(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))
    backbone = ViTS()
    backbone.load_state_dict(weights["backbone"])
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
        model (nn.Module): The network, with blocks at backbone.blocks.
    """

    def compute_grid(block, _inputs, output):
        block.grid(output)

    for i in range(NUM_LAYERS):
        block = model.get_submodule(f"backbone.blocks.{i}")
        block.add_module("grid", TokenGrid())
        block.register_forward_hook(compute_grid)


def get_model(name):
    assert name == "vits14_supervised_linhead"
    model = load_model()  # head at backbone.fc, logits
    add_token_grids(model)
    # The weights were trained with the ImageNet mean and std, the default of load_preprocess_images.
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE)
    wrapper = PytorchWrapper(identifier="vits14_supervised_linhead", model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == "vits14_supervised_linhead"
    # Outputs of the transformer blocks as grids.
    return [f"backbone.blocks.{i}.grid" for i in range(NUM_LAYERS)] + ["backbone.fc"]


def get_bibtex(model_identifier):
    return """@inproceedings{touvron2021training,
            title={Training data-efficient image transformers \\& distillation through attention},
            author={Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and J{\\'e}gou, Herv{\\'e}},
            booktitle={International Conference on Machine Learning},
            pages={10347--10357},
            year={2021},
            organization={PMLR}
            }"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)

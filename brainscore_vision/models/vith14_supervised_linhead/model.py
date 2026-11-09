"""
Model definition of a supervised ViT-H/14 model for submission to Brain-Score.
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


NUM_LAYERS = 32
NUM_CLASSES = 1000
IMAGE_SIZE = 224
REPO_ID = "niels-c/vith14_supervised_linhead"
FILENAME = "vith14_supervised_linhead.safetensors"
REVISION = "bc432d274f592e3212639f41a1b69af2c83c522d"

class ViTH(VisionTransformer):
    """
    timm ViT-H/14 of DeiT III (class token and LayerScale) with a linear head (fc) on the average of the final patch
    tokens, returning logits.

    It is the backbone of the training run, built there with timm.create_model("deit3_huge_patch14_224.fb_in1k",
    global_pool="avg", fc_norm=False).
    """

    def __init__(self):
        super().__init__(
            img_size=IMAGE_SIZE,
            patch_size=14,
            embed_dim=1280,
            depth=NUM_LAYERS,
            num_heads=16,
            no_embed_class=True,
            init_values=1e-6,
            num_classes=0,
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
        nn.Module: A Sequential of "backbone", the encoder whose layer fc is the linear head (logits), in eval mode.
    """

    weights = load_file(
        hf_hub_download(repo_id=REPO_ID, filename=FILENAME, revision=REVISION)
    )
    backbone = ViTH()
    backbone.load_state_dict(weights)
    return nn.Sequential(OrderedDict(backbone=backbone)).eval()


class TokenGrid(nn.Module):
    """
    Arrange the patch tokens (batch, 1 + tokens, dim) as a channels-first grid (batch, dim, side, side).

    The class token is dropped. Brain-Score names the coordinates of token activations channel and embedding, whereas
    the mask regression of Cadena2017-mask needs channel, channel_x and channel_y, as for a convolutional feature map.
    The features themselves are unchanged.
    """

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        patches = tokens[:, 1:]
        side = math.isqrt(patches.shape[1])
        return patches.transpose(1, 2).reshape(patches.shape[0], patches.shape[2], side, side)


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
    assert name == "vith14_supervised_linhead"
    model = load_model()  # head at backbone.fc, logits
    add_token_grids(model)
    # The weights were trained with the ImageNet mean and std, the default of load_preprocess_images.
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE)
    wrapper = PytorchWrapper(identifier="vith14_supervised_linhead", model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == "vith14_supervised_linhead"
    # Outputs of the transformer blocks as grids of their patch tokens.
    return [f"backbone.blocks.{i}.grid" for i in range(NUM_LAYERS)] + ["backbone.fc"]


def get_bibtex(model_identifier):
    return """@misc{touvron2022deitiiirevengevit,
            title={DeiT III: Revenge of the ViT}, 
            author={Hugo Touvron and Matthieu Cord and Hervé Jégou},
            year={2022},
            eprint={2204.07118},
            archivePrefix={arXiv},
            primaryClass={cs.CV},
            url={https://arxiv.org/abs/2204.07118}, 
        }"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)

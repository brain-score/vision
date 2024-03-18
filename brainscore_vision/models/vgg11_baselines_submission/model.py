# Custom Pytorch model from:
# https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb

from brainscore_vision.model_helpers.check_submission import check_models
import numpy as np
import torch
from torch import nn
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper

# from brainscore import score_model
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

# from brainscore import score_model
from pathlib import Path
from brainscore_vision.model_helpers import download_weights

BIBTEX = ""
LAYERS = [
    "early_layer",
    "middle_layer",
    "final_layer",
    "early_layer_conv",
    "middle_layer_conv",
    "final_layer_conv",
]


# define your custom model here:
class MLP(nn.Module):
    """
    Simple module for projection MLPs
    """

    def __init__(self, input_dim=256, hidden_dim=2048, output_dim=256, no_biases=False):
        """
        :param input_dim: number of input units
        :param hidden_dim: number of hidden units
        :param output_dim: number of output units
        """
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=not no_biases),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=not no_biases),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ConvBlock(nn.Module):
    """
    Simple convolutional block with 3x3 conv filters used for VGG-like architectures
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        pooling=True,
        kernel_size=3,
        padding=1,
        stride=1,
        groups=1,
        no_biases=False,
    ):
        """
        :param in_channels (int):
        :param out_channels (int):
        :param pooling (bool):
        """

        super(ConvBlock, self).__init__()

        conv_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=not no_biases,
            groups=groups,
        )

        if pooling:
            pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            pool_layer = nn.Identity()

        self.module = nn.Sequential(conv_layer, nn.ReLU(inplace=True), pool_layer)

    def forward(self, x):
        return self.module(x)


class VGG11Encoder(nn.Module):
    """
    Custom implementation of VGG11 encoder with added support for greedy training
    """

    def __init__(
        self,
        train_end_to_end=False,
        projector_mlp=False,
        projection_size=256,
        hidden_layer_size=2048,
        base_image_size=32,
        no_biases=False,
    ):
        """
        :param train_end_to_end (bool): Enable backprop between conv blocks
        :param projector_mlp (bool): Whether to project representations through an MLP before calculating loss
        :param projection_size (int): Only used when projection mlp is enabled
        :param hidden_layer_size (int): Only used when projection mlp is enabled
        :param base_image_size (int): input image size (eg. 32 for cifar, 96 for stl10)
        """
        super(VGG11Encoder, self).__init__()

        # VGG11 conv layers configuration
        self.channel_sizes = [3, 64, 128, 256, 256, 512, 512, 512, 512]
        pooling = [True, True, False, True, False, True, False, True]

        # Configure end-to-end/layer-local architecture with or without projection MLP(s)
        self.layer_local = not train_end_to_end
        self.num_trainable_hooks = 1 if train_end_to_end else 8
        self.projection_sizes = (
            [projection_size] * self.num_trainable_hooks
            if projector_mlp
            else self.channel_sizes[-self.num_trainable_hooks :]
        )

        # Conv Blocks
        self.blocks = nn.ModuleList([])

        # Projector(s) - identity modules by default
        self.projectors = nn.ModuleList([])
        self.flattened_feature_dims = []

        # Pooler
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))

        feature_map_size = base_image_size
        for i in range(8):
            if pooling[i]:
                feature_map_size /= 2
            self.blocks.append(
                ConvBlock(
                    self.channel_sizes[i],
                    self.channel_sizes[i + 1],
                    pooling=pooling[i],
                    no_biases=no_biases,
                )
            )
            input_dim = self.channel_sizes[i + 1]
            # Attach a projector MLP if specified either at every layer for layer-local training or just at the end
            if projector_mlp and (self.layer_local or i == 7):
                projector = MLP(
                    input_dim=int(input_dim),
                    hidden_dim=hidden_layer_size,
                    output_dim=projection_size,
                    no_biases=no_biases,
                )
                self.flattened_feature_dims.append(projection_size)
            else:
                projector = nn.Identity()
                self.flattened_feature_dims.append(
                    input_dim * feature_map_size * feature_map_size
                )
            self.projectors.append(projector)

        self.early_layer_conv = self.blocks[3]
        self.middle_layer_conv = self.blocks[5]
        self.final_layer_conv = self.blocks[-1]

        self.early_layer = self.projectors[3]
        self.middle_layer = self.projectors[5]
        self.final_layer = self.projectors[-1]

    def forward(self, x):
        z = []
        feature_maps = []
        for i, block in enumerate(self.blocks):
            x = block(x)

            # For layer-local training, record intermediate feature maps and pooled layer activities z (after projection if specified)
            # Also make sure to detach layer outputs so that gradients are not backproped
            if self.layer_local:
                x_pooled = self.pooler(x).view(x.size(0), -1)
                z.append(self.projectors[i](x_pooled))
                feature_maps.append(x)
                x = x.detach()

        x_pooled = self.pooler(x).view(x.size(0), -1)

        # Get outputs for end-to-end training
        if not self.layer_local:
            z.append(self.projectors[-1](x_pooled))
            feature_maps.append(x)

        return x_pooled, feature_maps, z


# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.
def get_model():
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :return: the model instance
    """
    # init the model and the preprocessing:
    preprocessing = functools.partial(load_preprocess_images, image_size=96)

    # Create network and load weights
    neg_samples_e2e_network = VGG11Encoder()
    weights_path = Path(__file__).parent / "neg_samples_e2e.pt"
    download_weights(
        bucket="brainscore-vision",
        folder_path="models/vgg11_baselines_submission copy",
        filename_version_sha=[
            (
                "neg_samples_e2e.pt",
                "li2DPVfciHhJfKfO4ZRwhNnKPtPH9tGw",
                "6476ba16b9d62f7da382feca617b9e8aac9fe27d",
            )
        ],
        save_directory=Path(__file__).parent,
    )
    state_dict = torch.load(weights_path)
    neg_samples_e2e_network.load_state_dict(state_dict, strict=False)

    # get an activations model from the Pytorch Wrapper
    activations_model_neg_samples_e2e = PytorchWrapper(
        identifier="neg_samples_e2e",
        model=neg_samples_e2e_network,
        preprocessing=preprocessing,
    )
    wrapper = activations_model_neg_samples_e2e
    wrapper.image_size = 96
    return wrapper


# Main Method: In submitting a custom model, you should not have to mess with this.
if __name__ == "__main__":
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)

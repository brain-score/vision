import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

from collections import OrderedDict
import torch as torch
from torch import nn
import torch.utils.model_zoo

HASH = "5c427c9c"


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_Z(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x


def Dorinet_cornet_z():
    model = nn.Sequential(
        OrderedDict(
            [
                ("V1", CORblock_Z(3, 64, kernel_size=7, stride=2)),
                ("V2", CORblock_Z(64, 128)),
                ("V4", CORblock_Z(128, 256)),
                ("IT", CORblock_Z(256, 512)),
                (
                    "decoder",
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("avgpool", nn.AdaptiveAvgPool2d(1)),
                                ("flatten", Flatten()),
                                ("linear", nn.Linear(512, 1000)),
                                ("output", Identity()),
                            ]
                        )
                    ),
                ),
            ]
        )
    )

    # weight initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model


def get_model_list():
    return ["dorinet_cornet_z"]


def get_model():
    model = Dorinet_cornet_z()
    # model = torch.nn.DataParallel(model)
    # if pretrained:
    url = "https://s3.amazonaws.com/cornet-models/cornet_z-5c427c9c.pth"
    torch.hub.load_state_dict_from_url(url)
    # ckpt_data = torch.hub.load_state_dict_from_url(url)
    # ckpt_data = torch.utils.model_zoo.load_url(url, map_location=None)
    # model.load_state_dict(ckpt_data['state_dict'])
    # V1.conv.weight = module.V1.conv.weight
    # ckpt_data.state_dict.
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(
        identifier="dorinet_cornet_z", model=model, preprocessing=preprocessing
    )
    wrapper.image_size = 224
    return wrapper


def get_layers():
    return ["V1", "V2", "V4", "IT", "decoder"]


BIBTEX = """@incollection{,
                title = {PPP-Tech},
                author = {},
                booktitle = {},
                editor = {Oriz DoronR},
                pages = {10-11},
                year = {2020},
                publisher = {},
                url = {}
}"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)

__all__ = ["dorinet_cornet_z", "Dorinet_cornet_z"]

model_urls = {
    "dorinet_cornet_z": "https://s3.amazonaws.com/cornet-models/cornet_z-5c427c9c.pth",
}

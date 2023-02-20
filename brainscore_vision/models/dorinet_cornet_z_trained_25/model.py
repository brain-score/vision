import functools
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.check_submission import check_models
import  torch.utils.model_zoo
from collections import OrderedDict
import torch
from torch import nn
from google_drive_downloader import GoogleDriveDownloader as gdd

# HASH = '5c427c9c'


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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, out_shape=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=kernel_size // 2)
        self.norm_input = nn.GroupNorm(32, out_channels)
        self.nonlin_input = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

    # def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
    #     super().__init__()
    #     self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
    #                           stride=stride, padding=kernel_size // 2)
    #     self.nonlin = nn.ReLU(inplace=True)
    #     self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    #     self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        # x = self.conv_input(inp)
        x = self.conv_input(inp)
        x = self.norm_input(x)
        x = self.nonlin_input(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlin1(x)

        # x = self.conv(inp)
        # x = self.nonlin(x)
        # x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x


def Dorinet_cornet_z():
    model = nn.Sequential(OrderedDict([
        ('V1', CORblock_Z(3, 64, kernel_size=7, stride=4, out_shape=56)),
        ('V2', CORblock_Z(64, 128, stride=2, out_shape=28)),
        ('V4', CORblock_Z(128, 256, stride=2, out_shape=14)),
        ('IT', CORblock_Z(256, 512, stride=2, out_shape=7)),

        # ('V1', CORblock_Z(3, 64, kernel_size=7, stride=2)),
        # ('V2', CORblock_Z(64, 128)),
        # ('V4', CORblock_Z(128, 256)),
        # ('IT', CORblock_Z(256, 512)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

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
    return ['dorinet_cornet_z']



def get_model(name):
    assert name == 'dorinet_cornet_z'
    model = Dorinet_cornet_z()
    file_id_loc = '1qWLm-VFw7xItqcXpb7IiRakdm5B_cI3I'


    download_path = './trytrytry.pth'
    gdd.download_file_from_google_drive(file_id=file_id_loc, dest_path=download_path,unzip=False)
    ckpt_data = torch.load(download_path,map_location=torch.device('cpu'))
    model.load_state_dict(ckpt_data['state_dict'])
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='dorinet_cornet_z', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'dorinet_cornet_z'
    return ['V1', 'V2', 'V4', 'IT', 'decoder']


def get_bibtex(model_identifier):
    return """@incollection{,
                title = {PPP-Tech},
                author = {},
                booktitle = {},
                editor = {Oriz DoronR},
                pages = {10-11},
                year = {2020},
                publisher = {},
                url = {}
                }"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)

__all__ = ['dorinet_cornet_z', 'Dorinet_cornet_z']

model_urls = {
    'dorinet_cornet_z': 'www.technion.org',
}
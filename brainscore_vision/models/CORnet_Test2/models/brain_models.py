import torch
from torch import nn
from collections import OrderedDict
import functools
from candidate_models.model_commitments.cornets import CORnetCommitment, CORNET_S_TIMEMAPPING, _build_time_mappings
from candidate_models.base_models.cornet import TemporalPytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.brain_transformation import ModelCommitment
from model_tools.check_submission import check_models
import torch.utils.model_zoo




def get_model_list():
    #return ['CornetVanilla3', 'CornetCustom3']
    return ['CORnet_Test2']

def get_model(model_name):
    model = CORnet_S()
    model = torch.nn.DataParallel(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    url = 'https://www.dropbox.com/s/xcck6czcfpne8li/checkpoint.pth?dl=1'
    ckpt_data = torch.utils.model_zoo.load_url(url, map_location=device)
    model.load_state_dict(ckpt_data['state_dict'])
    model = model.module

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = TemporalPytorchWrapper(identifier=model_name, model=model, preprocessing=preprocessing, separate_time=True)
    activations_model.image_size = 224
    
    time_mappings = CORNET_S_TIMEMAPPING
    model_commitment = CORnetCommitment(identifier=model_name, activations_model=activations_model,
                                        layers=['V1.output-t0'] +
                                               [f'{area}.output-t{timestep}'
                                                for area, timesteps in [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                                                for timestep in timesteps] +
                                               ['decoder.avgpool-t0'],
                                        time_mapping=_build_time_mappings(time_mappings))

    
    return model_commitment


def get_bibtex(model_identifier):
    return """@incollection{NIPS2012_4824,
                title = {ImageNet Classification with Deep Convolutional Neural Networks},
                author = {Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E},
                booktitle = {Advances in Neural Information Processing Systems 25},
                editor = {F. Pereira and C. J. C. Burges and L. Bottou and K. Q. Weinberger},
                pages = {1097--1105},
                year = {2012},
                publisher = {Curran Associates, Inc.},
                url = {http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf}
                }"""

class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = nn.Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


def CORnet_S():
    model = nn.Sequential(OrderedDict([
        ('V1', nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.ReLU(inplace=True)),
            ('output', nn.Identity())
        ]))),
        ('V2', CORblock_S(64, 128, times=2)),
        ('V4', CORblock_S(128, 256, times=4)),
        ('IT', CORblock_S(256, 512, times=2)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', nn.Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', nn.Identity())
        ])))
    ]))
    
    return model

if __name__ == '__main__':
    check_models.check_brain_models(__name__)
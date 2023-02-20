import math
from collections import OrderedDict
from torch import nn
import torch.utils.model_zoo as model_zoo
from .custom_modules import SequentialWithArgs, FakeReLU

HASH = '1d3f7974'


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
        # Last ReLU not inplace, because it might be fake for metamer generation
        self.nonlin3 = nn.ReLU(inplace=False)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp, fake_relu=False):
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
            # If it is the last part of the block have the option of returning a 
            # FakeReLU
            if (fake_relu==True) and (t==self.times-1):
                x = FakeReLU.apply(x)
            else:
                x = self.nonlin3(x)
            output = self.output(x)

        return output


class CORnet_V1_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)
        self.norm1 = nn.BatchNorm2d(64)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                                         bias=False)
        self.norm2 = nn.BatchNorm2d(64)
        # Last ReLU not inplace, because it might be fake for metamer generation
        self.nonlin2 = nn.ReLU(inplace=False)
        self.output = Identity()

    def forward(self, x, fake_relu=False):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlin1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if fake_relu==True:
            x = FakeReLU.apply(x)
        else:
            x = self.nonlin2(x)
            x = self.output(x)
        return x 


class CORnet_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = Flatten()
        self.linear = nn.Linear(512, 1000)
        self.output = Identity()

    def forward(self, x, fake_relu=False):
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)
        return x 


class CORnet_S_Architecture(nn.Module):
    def __init__(self, num_classes=1000):
        super(CORnet_S_Architecture, self).__init__()
        self.module = nn.Sequential(OrderedDict([
                      ('V1', CORnet_V1_Block()), 
#                      ('V1', nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
#                          ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                          bias=False)),
#                          ('norm1', nn.BatchNorm2d(64)),
#                          ('nonlin1', nn.ReLU(inplace=True)),
#                          ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#                          ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
#                                          bias=False)),
#                          ('norm2', nn.BatchNorm2d(64)),
#                          ('nonlin2', nn.ReLU(inplace=True)),
#                          ('output', Identity())
#                      ]))),
                     ('V2', CORblock_S(64, 128, times=2)),
                     ('V4', CORblock_S(128, 256, times=4)),
                     ('IT', CORblock_S(256, 512, times=2)),
                     ('decoder', CORnet_decoder()),
#                      ('decoder', SequentialWithArgs(OrderedDict([
#                          ('avgpool', nn.AdaptiveAvgPool2d(1)),
#                          ('flatten', Flatten()),
#                          ('linear', nn.Linear(512, 1000)),
#                          ('output', Identity())
#                      ])))
                 ]))
    
        self.layer_names = ['V1', 'V2', 'V4', 'IT', 'decoder']

        # weight initialization
        for m in self.module.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # nn.Linear is missing here because I originally forgot
            # to add it during the training of this network
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        del no_relu # Does not do anything for this network
        all_outputs = {}
        all_outputs['input_after_preproc'] = x
        # TODO(jfeather): add in fake_relu parameters
        for layer, layer_name in list(zip(self.module, self.layer_names)): 
            if fake_relu:
                # When generating metamers, we want to have a ReLU in the forward
                # pass but the end layer to have gradients of 1 in the backwards
                # pass. As implemented here, this has a large memory footprint
                # because we double the number of layers in the model. 
                all_outputs[layer_name + '_fake_relu'] = layer(x, fake_relu=fake_relu)
            x = layer(x)
            all_outputs[layer_name] = x

        all_outputs['final'] = x

        # TODO(jfeather): we are using a sequential for the full model, so 
        # the latent isn't returned in the typical fashion
        if with_latent:
            return x, None, all_outputs
        else:
            return x

def CORnet_S(pretrained=False, **kwargs):
    model = CORnet_S_Architecture(**kwargs)
    model_letter = 'S'
    model_hash = HASH
#     model = torch.nn.DataParallel(model)

    if pretrained:
        url = f'https://s3.amazonaws.com/cornet-models/cornet_{model_letter.lower()}-{model_hash}.pth'
        ckpt_data = model_zoo.load_url(url)
        model.load_state_dict(ckpt_data['state_dict'])

    return model


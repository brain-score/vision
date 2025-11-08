from collections import OrderedDict
from torch import nn
import math
import torch.nn.functional as F
import torch
import numpy as np
from timm.models.registry import register_model

HASH = '5c427c9c'

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
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

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


class CORnet_S(nn.Module):
    def __init__(self, num_classes=1000):
        super(CORnet_S, self).__init__()
        self.num_classes = num_classes
        self.contrastive_loss = False

        self.model = nn.Sequential(OrderedDict([
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
                        ('output', Identity())
                    ]))),
                    ('V2', CORblock_S(64, 128, times=2)),
                    ('V4', CORblock_S(128, 256, times=4)),
                    ('IT', CORblock_S(256, 512, times=2)),
                    ('decoder', nn.Sequential(OrderedDict([
                        ('avgpool', nn.AdaptiveAvgPool2d(1)),
                        ('flatten', Flatten()),
                        ('linear', nn.Linear(512, 1000)),
                        ('output', Identity())
                    ])))
                ]))

        # weight initialization
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # nn.Linear is missing here because I originally forgot 
            # to add it during the training of this network
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.model(x)



# copy from HMAX.py     
def get_ip_scales(num_scale_bands, base_image_size, scale=4):
    """
    Generate a list of image scales for multi-scale pyramid input.

    Args:
        num_scale_bands (int): Number of scale bands (not including center).
        base_image_size (int): Input image size (e.g., 224).
        scale (int): Scaling divisor for the exponent.

    Returns:
        List[int]: List of scale sizes sorted from smallest to largest.
    """
    if num_scale_bands % 2 == 1:
        # Odd bands => symmetric around 0
        image_scales = np.arange(-num_scale_bands//2 + 1, num_scale_bands//2 + 2)
    else:
        # Even bands => slightly asymmetric
        image_scales = np.arange(-num_scale_bands//2, num_scale_bands//2 + 1)

    # Compute scaled sizes
    image_scales = [int(np.ceil(base_image_size / (2 ** (i / scale)))) for i in image_scales]
    image_scales.sort()

    # Sanity check
    if num_scale_bands > 2:
        assert len(image_scales) == num_scale_bands + 1, "Mismatch in number of scales generated"

    return image_scales


class C(nn.Module):
    #Spatial then Scale
    def __init__(self,
                  pool_func1 = nn.MaxPool2d(kernel_size = 3, stride = 2),
                  pool_func2 = nn.MaxPool2d(kernel_size = 4, stride = 3),
                  global_scale_pool=False):
        super(C, self).__init__()
        ## TODO: Add arguments for kernel_sizes
        self.pool1 = pool_func1
        self.pool2 = pool_func2
        self.global_scale_pool = global_scale_pool

    def forward(self,x_pyramid):
        # if only one thing in pyramid, return

        out = []
        if self.global_scale_pool:
            if len(x_pyramid) == 1:
                return self.pool1(x_pyramid[0])

            out = [self.pool1(x) for x in x_pyramid]
            # resize everything to be the same size
            final_size = out[len(out) // 2].shape[-2:]
            out = F.interpolate(out[0], final_size, mode='bilinear')
            for x in x_pyramid[1:]:
                temp = F.interpolate(x, final_size, mode='bilinear')
                out = torch.max(out, temp)  # Out-of-place operation to avoid in-place modification
                del temp  # Free memory immediately

        else: # not global pool

            if len(x_pyramid) == 1:
                return [self.pool1(x_pyramid[0])]

            for i in range(0, len(x_pyramid) - 1):
                x_1 = x_pyramid[i]
                x_2 = x_pyramid[i+1]
                #spatial pooling
                x_1 = self.pool1(x_1)
                x_2 = self.pool2(x_2)
                # Then fix the sizing interpolating such that feature points match spatially
                if x_1.shape[-1] > x_2.shape[-1]:
                    x_2 = F.interpolate(x_2, size = x_1.shape[-2:], mode = 'bilinear')
                else:
                    x_1 = F.interpolate(x_1, size = x_2.shape[-2:], mode = 'bilinear')
                x = torch.stack([x_1, x_2], dim=4)

                to_append, _ = torch.max(x, dim=4)

                out.append(to_append)
        return out
    

class CORnet_S_MultiScale(nn.Module):
    def __init__(self, num_classes=1000, ip_scale_bands=1, base_size=224):
        super().__init__()
        self.num_classes = num_classes
        self.ip_scale_bands = ip_scale_bands
        self.base_size = base_size

        self.V1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.ReLU(inplace=True)),
        ]))
        self.V2 = CORblock_S(64, 128, times=2)
        self.V4 = CORblock_S(128, 256, times=4)
        self.IT = CORblock_S(256, 512, times=2)

        self.C1 = C(
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            global_scale_pool=False
        )
        self.C2 = C(
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            global_scale_pool=False
        )
        self.C4 = C(
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            global_scale_pool=True  # Use global pooling at the final stage
        )

        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, num_classes)),
        ]))

        self._weight_init(self.V1)
        self._weight_init(self.V2)
        self._weight_init(self.V4)
        self._weight_init(self.IT)
    

    def _weight_init(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_ip(self, x, num_scale_bands):
        """
        Build an image pyramid.
        num_scale_bands = number of images in the pyramid - 1
        """
        base_image_size = int(x.shape[-1])
        scale_factor = 4  # exponent factor for scaling
        image_scales = get_ip_scales(num_scale_bands, base_image_size, scale_factor)
        
        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interp_img = F.interpolate(x, size=(i_s, i_s), mode='bilinear', align_corners=False)
                image_pyramid.append(interp_img)
            return image_pyramid
        else:
            return [x]

    def forward(self, x):
        if self.ip_scale_bands > 1:
            x_pyramid = self.make_ip(x, self.ip_scale_bands)
        else:
            x_pyramid = [x]

        # Pass through V1
        x_pyramid = [self.V1(x) for x in x_pyramid]
        x_pyramid = self.C1(x_pyramid)

        x_pyramid = [self.V2(x) for x in x_pyramid]
        x_pyramid = self.C2(x_pyramid)

        x_pyramid = [self.V4(x) for x in x_pyramid]

        x_pyramid = [self.IT(x) for x in x_pyramid]
        x_pyramid = self.C4(x_pyramid)

        out = self.decoder(x_pyramid)
        return out


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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x


def CORnet_Z():
    model = nn.Sequential(OrderedDict([
        ('V1', CORblock_Z(3, 64, kernel_size=7, stride=2)),
        ('V2', CORblock_Z(64, 128)),
        ('V4', CORblock_Z(128, 256)),
        ('IT', CORblock_Z(256, 512)),
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

@register_model
def cornet_s(pretrained=False, **kwargs):
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    if pretrained:
       raise ValueError("No pretrained model available for CORnet_S")
    model = CORnet_S()
    return model

@register_model
def cornet_z(pretrained=False, **kwargs):
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    if pretrained:
       raise ValueError("No pretrained model available for CORnet_Z")
    model = CORnet_Z(**kwargs)
    return model

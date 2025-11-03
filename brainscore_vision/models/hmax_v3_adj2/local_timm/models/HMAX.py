
"""
An implementation of HMAX:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy as sp
import time

from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs



def get_gabor(l_size, la, si, n_ori, aspect_ratio):
    """generate the gabor filters

    Args
    ----
        l_size: float
            gabor sizes
        la: float
            lambda
        si: float
            sigma
        n_ori: type integer
            number of orientations
        aspect_ratio: type float
            gabor aspect ratio

    Returns
    -------
        gabor: type nparray
            gabor filter

    """

    gs = l_size

    # TODO: inverse the axes in the begining so I don't need to do swap them back
    # thetas for all gabor orientations
    th = np.array(range(n_ori)) * np.pi / n_ori + np.pi / 2.
    th = th[sp.newaxis, sp.newaxis, :]
    hgs = (gs - 1) / 2.
    yy, xx = sp.mgrid[-hgs: hgs + 1, -hgs: hgs + 1]
    xx = xx[:, :, sp.newaxis]
    yy = yy[:, :, sp.newaxis]

    # x = xx * np.cos(th) - yy * np.sin(th)
    # y = xx * np.sin(th) + yy * np.cos(th)
    x = xx * np.cos(th) + yy * np.sin(th)
    y = - xx * np.sin(th) + yy * np.cos(th)

    filt = np.exp(-(x ** 2 + (aspect_ratio * y) ** 2) / (2 * si ** 2)) * np.cos(2 * np.pi * x / la)
    filt[np.sqrt(x ** 2 + y ** 2) > gs / 2.] = 0

    # gabor normalization (following cns hmaxgray package)
    for ori in range(n_ori):
        filt[:, :, ori] -= filt[:, :, ori].mean()
        filt_norm = fastnorm(filt[:, :, ori])
        if filt_norm != 0: filt[:, :, ori] /= filt_norm

    filt_c = np.array(filt, dtype='float32').swapaxes(0, 2).swapaxes(1, 2)

    filt_c = torch.Tensor(filt_c)
    filt_c = filt_c.view(n_ori, 1, gs, gs)
    # filt_c = filt_c.repeat((1, 3, 1, 1))

    return filt_c


def fastnorm(in_arr):
    arr_norm = np.dot(in_arr.ravel(), in_arr.ravel()).sum() ** (1. / 2.)
    return arr_norm

def fastnorm_tensor(in_arr):
    arr_norm = torch.dot(in_arr.ravel(), in_arr.ravel()).sum() ** (1. / 2.)
    return arr_norm


def get_sp_kernel_sizes_C(scales, num_scales_pooled, scale_stride):
    '''
    Recursive function to find the right relative kernel sizes for the spatial pooling performed in a C layer.
    The right relative kernel size is the average of the scales that will be pooled. E.g, if scale 7 and 9 will be
    pooled, the kernel size for the spatial pool is 8 x 8

    Parameters
    ----------
    scales
    num_scales_pooled
    scale_stride

    Returns
    -------
    list of sp_kernel_size

    '''

    if len(scales) < num_scales_pooled:
        return []
    else:
        average = int(sum(scales[0:num_scales_pooled]) / len(scales[0:num_scales_pooled]))
        return [average] + get_sp_kernel_sizes_C(scales[scale_stride::], num_scales_pooled, scale_stride)


def pad_to_size(a, size):
    current_size = (a.shape[-2], a.shape[-1])
    total_pad_h = size[0] - current_size[0]
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top

    total_pad_w = size[1] - current_size[1]
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    a = nn.functional.pad(a, (pad_left, pad_right, pad_top, pad_bottom))

    return a

"""
general get ip scales function (cleaner code)
scale: the denominator of the exponent. ie. if you want to scale by 2^(1/4), this should be 4
"""
def get_ip_scales(num_scale_bands, base_image_size, scale=4):
    if num_scale_bands % 2 == 1:
        # If x is odd, create an array centered at 0
        image_scales = np.arange(-num_scale_bands//2 + 1, num_scale_bands//2 + 2)
    else:
        # If x is even, shift one extra integer to the positive side
        image_scales = np.arange(-num_scale_bands//2, num_scale_bands//2 + 1)

    image_scales = [np.ceil(base_image_size/(2**(i/scale))) for i in image_scales]
    image_scales.sort()
    if num_scale_bands > 2:
        assert(len(image_scales) == num_scale_bands + 1)
    
    return image_scales
    


def check_for_nans(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaNs found in {name}")


class S1(nn.Module):
    def __init__(self):
        super(S1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU())
    
    def forward(self, x_pyramid):
        # get dimensions
        return [self.layer1(x) for x in x_pyramid]

class S2b(nn.Module):
    def __init__(self):
        super(S2b, self).__init__()
        ## bypass layers
        self.s2b_kernel_size=[4,8,12,16]
        self.s2b_seqs = nn.ModuleList()
        for size in self.s2b_kernel_size:
            self.s2b_seqs.append(nn.Sequential(
                nn.Conv2d(96, 64, kernel_size=size, stride=1, padding=size//2),
                nn.BatchNorm2d(64, 1e-3),
                nn.ReLU(True)
            ))
    
    def forward(self, x_pyramid):
        # get dimensions
        bypass = [torch.cat([seq(out) for seq in self.s2b_seqs], dim=1) for out in x_pyramid]
        return bypass

# lots of repeated code
class S2(nn.Module):
    def __init__(self):
        super(S2, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU())
    
    def forward(self, x_pyramid):
        # get dimensions
        return [self.layer(x) for x in x_pyramid]

# lots of repeated code
class S3(nn.Module):
    def __init__(self):
        super(S3, self).__init__()
        self.layer = nn.Sequential(
            #layer3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            #layer 4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            #layer5
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
    
    def forward(self, x_pyramid):
        # get dimensions
        return [self.layer(x) for x in x_pyramid]

# class C(nn.Module):
#     ## Scale then Spatial
#     def __init__(self, pool_func = nn.MaxPool2d(kernel_size = 3, stride = 2)):
#         super(C, self).__init__()
#         ## TODO: Add arguments for kernel_sizes
#         self.pool = pool_func
#     def forward(self,x_pyramid):
#         # if only one thing in pyramid, return

#         if len(x_pyramid) == 1:
#             return [self.pool(x_pyramid[0])]

#         out = []

#         for i in range(0, len(x_pyramid) - 1):
#             x_1 = x_pyramid[i]
#             x_2 = x_pyramid[i+1]
#             # First interpolating such that feature points match spatially
#             if x_1.shape[-1] > x_2.shape[-1]:
#                 x_2 = F.interpolate(x_2, size = x_1.shape[-2:], mode = 'bilinear')
#             else:
#                 x_1 = F.interpolate(x_1, size = x_2.shape[-2:], mode = 'bilinear')

#             x = torch.stack([x_1, x_2], dim=4)
#             to_append, _ = torch.max(x, dim=4)
            
#             #spatial pooling
#             to_append = self.pool(to_append)
#             out.append(to_append)

#         return out


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


class HMAX_from_Alexnet(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3, ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False):
        self.num_classes = num_classes
        self.in_chans = in_chans
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands
        # contrastive_loss: boolean for if the model should be using contrastive loss or not
        self.contrastive_loss = contrastive_loss
        super(HMAX_from_Alexnet, self).__init__()

        self.layer1 = S1()
        self.pool1 = C()
        self.S2b = S2b()
        self.C2b = C(nn.MaxPool2d(kernel_size = 10, stride = 5), nn.MaxPool2d(kernel_size = 12, stride = 6), global_scale_pool=True)

        self.layer2 = S2()
        self.pool2 = C()
        self.layer3 = S3()
        self.pool3 =  C(global_scale_pool=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_size, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))


    def make_ip(self, x):
        ## num_scale_bands = num images in IP - 1
        num_scale_bands = self.ip_scale_bands
        base_image_size = int(x.shape[-1])
        scale = 4 ## factor in exponenet
        
        image_scales = get_ip_scales(num_scale_bands, base_image_size, scale)

        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear').clamp(min=0, max=1)
                
                image_pyramid.append(interpolated_img)
            return image_pyramid
        else:
            return [x]
    
    def forward(self, x):

        out = self.make_ip(x)
        ## should make SxBxCxHxW

        out = self.layer1(out) #s1
        out = self.pool1(out) #c1

        #bypass layers
        bypass = self.S2b(out)
        bypass = self.C2b(bypass)
        
        # main
        out = self.layer2(out) #s2 
        out = self.pool2(out) # c2
        out = self.layer3(out)
        out = self.pool3(out) #c3?
        out = out.reshape(out.size(0), -1)
        bypass = bypass.reshape(bypass.size(0), -1)

        ## merge here
        out = torch.cat([bypass, out], dim=1)
        del bypass

        if self.contrastive_loss:
            # cl means contrastive loss
            cl_feats = out

        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        if self.contrastive_loss:
            return out, cl_feats
        return out
#########################################################################################################

class HMAX_from_Alexnet_bypass(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3, ip_scale_bands=1, classifier_input_size=4096, contrastive_loss=False):
        self.num_classes = num_classes
        self.in_chans = in_chans
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands
        # contrastive_loss: boolean for if the model should be using contrastive loss or not
        self.classifier_input_size = classifier_input_size
        self.contrastive_loss = contrastive_loss
        super(HMAX_from_Alexnet_bypass, self).__init__()

        self.layer1 = S1()
        self.pool1 = C()
        self.S2b = S2b()
        self.C2b = C(nn.MaxPool2d(kernel_size = 10, stride = 5), nn.MaxPool2d(kernel_size = 12, stride = 6), global_scale_pool=True)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.classifier_input_size, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))


    def make_ip(self, x):
        ## num_scale_bands = num images in IP - 1
        num_scale_bands = self.ip_scale_bands
        base_image_size = int(x.shape[-1])
        scale = 4 ## factor in exponenet
        
        image_scales = get_ip_scales(num_scale_bands, base_image_size, scale)

        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear').clamp(min=0, max=1)
                
                image_pyramid.append(interpolated_img)
            return image_pyramid
        else:
            return [x]
    
    def forward(self, x):

        bypass = self.make_ip(x)
        ## should make SxBxCxHxW

        bypass = self.layer1(bypass) #s1
        bypass = self.pool1(bypass) #c1

        #bypass layers
        bypass = self.S2b(bypass)
        bypass = self.C2b(bypass)
        if self.contrastive_loss: 
            c2b_feats = bypass
        bypass = bypass.reshape(bypass.size(0), -1)

        bypass = self.fc(bypass)
        bypass = self.fc1(bypass)
        bypass = self.fc2(bypass)
        if self.contrastive_loss:
            return bypass, c2b_feats
        return bypass

import random
import torchvision

class CHMAX(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3, ip_scale_bands=1, classifier_input_size=13312, hmax_type="full", **kwargs):
        super(CHMAX, self).__init__()

        # the below line is so that the training script calculates the loss correctly
        self.contrastive_loss = kwargs['contrastive_loss']
        if hmax_type == "full":
            self.model_backbone = HMAX_from_Alexnet(num_classes=num_classes,
                                                        in_chans=in_chans,
                                                        ip_scale_bands=ip_scale_bands,
                                                        classifier_input_size=classifier_input_size,
                                                        contrastive_loss=self.contrastive_loss)
        elif hmax_type == "bypass":
            self.model_backbone = HMAX_from_Alexnet_bypass(num_classes=num_classes,
                                                        in_chans=in_chans,
                                                        ip_scale_bands=ip_scale_bands,
                                                        classifier_input_size=classifier_input_size,
                                                        contrastive_loss=self.contrastive_loss)
        else:
            raise(NotImplementedError)

        self.num_classes = num_classes
        self.in_chans = in_chans
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands

    def forward(self, x):

        # stream 1
        stream_1_output, stream_1_c2b_feats = self.model_backbone(x)

        # stream 2
        scale_factor_list = [0.707, 0.841, 1, 1.189, 1.414]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw*scale_factor)
        x_rescaled = F.interpolate(x, size = (new_hw, new_hw), mode = 'bilinear').clamp(min=0, max=1)
        if new_hw <= img_hw:
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
        elif new_hw > img_hw:
            center_crop = torchvision.transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)
        
        stream_2_output, stream_2_c2b_feats = self.model_backbone(x_rescaled)

        correct_scale_loss = torch.mean(torch.abs(stream_1_c2b_feats - stream_2_c2b_feats))
        return stream_1_output, correct_scale_loss


def checkpoint_filter_fn(state_dict, model: nn.Module):
    out_dict = {}
    for k, v in state_dict.items():
        out_dict[k] = v
    return out_dict


@register_model
def hmax_from_alexnet(pretrained=False, **kwargs):
    ## there are some weird things in the kwargs by default that I don't care about
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = HMAX_from_Alexnet(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def hmax_from_alexnet_bypass(pretrained=False, **kwargs):
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = HMAX_from_Alexnet_bypass(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def chmax(pretrained=False, **kwargs):
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = CHMAX(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torchvision
import cv2
import os
import _pickle as pickle
import random

from .ALEXMAX3_optimized import C_scoring2_optimized

def visualize_map(map):
    map = map.detach().numpy()
    plt.imshow(map)


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
    th = th[np.newaxis, np.newaxis, :]
    hgs = (gs - 1) / 2.
    yy, xx = np.mgrid[-hgs: hgs + 1, -hgs: hgs + 1]
    xx = xx[:, :, np.newaxis]
    yy = yy[:, :, np.newaxis]

    # x = xx * np.cos(th) - yy * np.sin(th)
    # y = xx * np.sin(th) + yy * np.cos(th)
    x = xx * np.cos(th) + yy * np.sin(th)
    y = - xx * np.sin(th) + yy * np.cos(th)

    filt = np.exp(-(x ** 2 + (aspect_ratio * y) ** 2) / (2 * si ** 2)) * np.cos(2 * np.pi * x / la)
    filt[np.sqrt(x ** 2 + y ** 2) > gs / 2.] = 0

    # gabor normalization (following cns hmaxgray package)
    for ori in range(n_ori):
        filt[:, :, ori] -= filt[:, :, ori].mean()
        # filt[:, :, ori] = (filt[:, :, ori] - filt[:, :, ori].mean()) / (filt[:, :, ori].std() + 1e-3)
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


def pad_to_size(a, size, pad_mode = 'constant'):
    current_size = (a.shape[-2], a.shape[-1])
    total_pad_h = size[0] - current_size[0]
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top

    total_pad_w = size[1] - current_size[1]
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    a = nn.functional.pad(a, (pad_left, pad_right, pad_top, pad_bottom), mode = pad_mode)

    return a


class S1(nn.Module):
    # TODO: make more flexible such that forward will accept any number of tensors
    def __init__(self, scale, n_ori, padding, trainable_filters, la, si, visualize_mode = False, prj_name = None, MNIST_Scale = None):

        super(S1, self).__init__()

        self.scale = scale
        self.la = la
        self.si = si
        self.visualize_mode = visualize_mode
        self.prj_name = prj_name
        self.MNIST_Scale = MNIST_Scale
        self.padding = padding

        # setattr(self, f's_{scale}', nn.Conv2d(1, n_ori, scale, padding=padding))
        # s1_cell = getattr(self, f's_{scale}')
        self.gabor_filter = get_gabor(l_size=scale, la=la, si=si, n_ori=n_ori, aspect_ratio=0.3)  # ??? What is aspect ratio
        self.batchnorm = nn.BatchNorm2d(n_ori, 1e-3)

        ######################
        # self.noise_mode = 'gaussian'
        self.noise_mode = 'none'
        self.k_exc = 1
        self.noise_scale = 1
        self.noise_level = 1

    def forward(self, x_pyramid, MNIST_Scale = None, batch_idx = None, prj_name = None, category = None, save_rdms = None, plt_filters = None):
        self.MNIST_Scale = MNIST_Scale
        s1_maps = []
        s1_maps_rdm = []
        # Loop over scales, normalizing.
        for p_i in range(len(x_pyramid)):

            x = x_pyramid[p_i]
            s1_map = F.conv2d(x, self.gabor_filter.to(device='cuda'), None, 1, self.padding)

            
            ###############################################################
            if self.noise_mode == 'neuronal':
                eps = 10e-5
                s1_map *= self.k_exc
                s1_map *= self.noise_scale
                s1_map += self.noise_level
                
                s1_map += torch.distributions.normal.Normal(torch.zeros_like(s1_map), scale=1).rsample() * torch.sqrt(F.relu(s1_map.clone()) + eps)

                s1_map -= self.noise_level
                s1_map /= self.noise_scale

            if self.noise_mode == 'gaussian':
                s1_map += torch.distributions.normal.Normal(torch.zeros_like(s1_map), scale=1).rsample() * self.noise_scale

            s1_map = self.batchnorm(s1_map)
            s1_map = torch.abs(s1_map)

            s1_maps.append(s1_map)

            # Padding (to get s1_maps in same size) ---> But not necessary for us
            ori_size = (x.shape[-2], x.shape[-1])
            # ori_size = (360,360)

            s1_maps[p_i] = pad_to_size(s1_maps[p_i], ori_size)
            # s1_maps[p_i] = pad_to_size(s1_maps[p_i], (96,96))

        return s1_maps

class C(nn.Module):
    # TODO: make more flexible such that forward will accept any number of tensors
    def __init__(self, global_pool, sp_kernel_size=[10, 8], sp_stride_factor=None, n_in_sbands=None,
                 num_scales_pooled=2, scale_stride=1,image_subsample_factor=1, visualize_mode = False, \
                 c1_bool = False, prj_name = None, MNIST_Scale = None, c2_bool = False, c3_bool = False, \
                 c2b_bool = False, attn_mech = False):

        super(C, self).__init__()

        self.c1_bool = c1_bool
        self.c2_bool = c2_bool
        self.c3_bool = c3_bool
        self.c2b_bool = c2b_bool
        self.prj_name = prj_name
        self.MNIST_Scale = MNIST_Scale
        self.visualize_mode = visualize_mode
        self.global_pool = global_pool
        self.sp_kernel_size = sp_kernel_size
        self.num_scales_pooled = num_scales_pooled
        self.sp_stride_factor = sp_stride_factor
        self.scale_stride = scale_stride
        self.n_in_sbands = n_in_sbands
        self.n_out_sbands = int(((n_in_sbands - self.num_scales_pooled) / self.scale_stride) + 1)
        self.img_subsample = image_subsample_factor 

        self.attn_mech = attn_mech

        if self.attn_mech:
            self.scale_lin_1 = nn.Sequential(nn.Linear(512, 64), nn.ReLU())
            self.scale_lin_2 = nn.Sequential(nn.Linear(64, 1))

        if not self.global_pool:
            if self.sp_stride_factor is None:
                self.sp_stride = [1] * len(self.sp_kernel_size)
            else:
                self.sp_stride = [int(np.ceil(self.sp_stride_factor * kernel_size)) for kernel_size in self.sp_kernel_size]
                # self.sp_stride = [int(np.ceil(0.5 + kernel_size/self.sp_kernel_size[len(self.sp_kernel_size)//2])) for kernel_size in self.sp_kernel_size]

    def forward(self, x_pyramid, x_input = None, MNIST_Scale = None, batch_idx = None, category = None, \
                prj_name = None, same_scale_viz = None, base_scale = None, c1_sp_kernel_sizes = None, \
                c2_sp_kernel_sizes = None, image_scales = None, overall_max_scale_index = False, save_rdms = None, \
                plt_filters = None, scale_loss = False, argmax_bool = False):
        # TODO - make this whole section more memory efficient

        # print('####################################################################################')
        # print('####################################################################################')

        max_scale_index = [0]
        x = 0
        c_maps_scale = 0
        correct_scale_loss = 0

        c_maps = []
        if same_scale_viz:
            ori_size = (base_scale, base_scale)
        else:
            ori_size = x_pyramid[0].shape[2:4]

        # print('ori_size : ',ori_size)
        # print('len(x_pyramid) : ',len(x_pyramid))

        # Single scale band case --> While Training
        if len(x_pyramid) == 1:
            if not self.global_pool:
                x = F.max_pool2d(x_pyramid[0], self.sp_kernel_size[0], self.sp_stride[0])
                x = pad_to_size(x, ori_size)

                c_maps.append(x)
            else:
                s_m = F.max_pool2d(x_pyramid[0], x_pyramid[0].shape[-1], 1)
                c_maps.append(s_m)

            c_maps_scale = c_maps

        # Multi Scale band case ---> While Testing
        else:
            #####################################################
            if not self.global_pool:
                if same_scale_viz:
                    ori_size = (base_scale, base_scale)
                else:
                    if len(x_pyramid) == 2:
                        ori_size = x_pyramid[0].shape[2:4]
                    else:
                        ori_size = x_pyramid[-int(np.ceil(len(x_pyramid)/2))].shape[2:4]

                ####################################################
                # # MaxPool for C1 with 2 scales being max pooled over at a time with overlap
                for p_i in range(len(x_pyramid)-1):
                    # print('############################')
                    x_1 = F.max_pool2d(x_pyramid[p_i], self.sp_kernel_size[0], self.sp_stride[0])
                    x_2 = F.max_pool2d(x_pyramid[p_i+1], self.sp_kernel_size[1], self.sp_stride[1])


                    # First interpolating such that feature points match spatially
                    if x_1.shape[-1] > x_2.shape[-1]:
                        # x_2 = pad_to_size(x_2, x_1.shape[-2:])
                        x_2 = F.interpolate(x_2, size = x_1.shape[-2:], mode = 'bilinear')
                    else:
                        # x_1 = pad_to_size(x_1, x_2.shape[-2:])
                        x_1 = F.interpolate(x_1, size = x_2.shape[-2:], mode = 'bilinear')

                    # Then padding
                    x_1 = pad_to_size(x_1, ori_size)
                    x_2 = pad_to_size(x_2, ori_size)

                    ##################################
                    # Maxpool over scale groups
                    x = torch.stack([x_1, x_2], dim=4)

                    to_append, _ = torch.max(x, dim=4)
                    c_maps.append(to_append)

            # ####################################################
            # # # # # # MaxPool over all positions first then scales (C2b)
            elif not(argmax_bool):
                scale_max = []
                # Global MaxPool over positions for each scale separately
                for p_i in range(len(x_pyramid)):
                    s_m = F.max_pool2d(x_pyramid[p_i], x_pyramid[p_i].shape[-1], 1)
                    scale_max.append(s_m)

                # ####################################################
                # ####################################################
                if scale_loss:
                    # Option 1
                    # 0-1 1-2 2-3 3-4 4-5 5-6 6-7 7-8 8-9 9-10 10-11 11-12 12-13 13-14 14-15 15-16 
                    # scale_max Shape --> [Scales, Batch, Channels]
                    # Extra loss for penalizing if correct scale does not have max activation (ScaleBand 7 or 8)
                    correct_scale_l_loss = torch.tensor([0.], device = scale_max[0].device)
                    correct_scale_u_loss = torch.tensor([0.], device = scale_max[0].device)

                    middle_scaleband = int(len(scale_max)/2)

                    middle_scaleband_list = [middle_scaleband-1, middle_scaleband, middle_scaleband+1]
                    scaleband_loss_weight = [0.5, 1, 0.5]

                    for sm_i in range(len(scale_max)):
                        if sm_i not in [middle_scaleband-1, middle_scaleband]:
                            if len(scale_max) % 2 == 0:
                                # When overlap of 1 is done we'll always get a even no else when no. overlap we'll get odd no.
                                correct_scale_l_loss = correct_scale_l_loss + F.relu(scale_max[sm_i] - scale_max[middle_scaleband-1])
                            
                            for mid_sb, sb_weight in zip(middle_scaleband_list, scaleband_loss_weight):
                                correct_scale_u_loss = correct_scale_u_loss + (sb_weight*F.relu(scale_max[sm_i] - scale_max[mid_sb]))

                    correct_scale_loss = (torch.mean(correct_scale_l_loss) + torch.mean(correct_scale_u_loss))

                ####################################################
                ####################################################

                # Option 1:: Global Max Pooling over Scale i.e, Maxpool over scale groups
                x = torch.stack(scale_max, dim=4)

                if self.attn_mech:
                    x_prime = x.squeeze().permute(2,0,1) # [No. of scales, Batch, Channel]
                    # x_prime_clone = x_prime.clone()
                    x_prime_attn = self.scale_lin_1(x_prime)  # [No. of scales, Batch, hidden_channels]
                    x_prime_attn = self.scale_lin_2(x_prime_attn)  # [No. of scales, Batch, 1]
                    attention_weights = F.softmax(x_prime_attn, dim=0)  # Normalize weights across scales

                    # Multiply the input tensor by the attention_weights
                    output = x_prime * attention_weights
                    # print('output : ',output.shape)
                    x = output.permute(1,2,0)[:,:,None,None,:]
                    # print('x : ',x.shape)

                c_maps_scale = x

                to_append, _ = torch.max(x, dim=4)
                c_maps.append(to_append)

            else:
                scale_max = []
                for p_i in range(len(x_pyramid)):
                    s_m = x_pyramid[p_i]
                    s_m = F.max_pool2d(s_m, s_m.shape[-1], 1)
                    s_m = s_m.reshape(*x_pyramid[p_i].shape[:2])
                    s_m = torch.sort(s_m, dim = -1)[0]
                    scale_max.append(s_m)

                # scale_max shape --> Scale x B x C
                scale_max = torch.stack(scale_max, dim=0) # Shape --> Scale x B x C

                if self.attn_mech:
                    x_prime = scale_max # [No. of scales, Batch, Channel]
                    x_prime_attn = self.scale_lin_1(x_prime)  # [No. of scales, Batch, hidden_channels]
                    x_prime_attn = self.scale_lin_2(x_prime_attn)  # [No. of scales, Batch, 1]
                    attention_weights = F.softmax(x_prime_attn, dim=0)  # Normalize weights across scales

                    # Multiply the input tensor by the attention_weights
                    output = x_prime * attention_weights
                    # print('output : ',output.shape)
                    scale_max = output
                    # print('x : ',x.shape)

                # print('image_scales : ',image_scales)

                max_scale_index = [0]*scale_max.shape[1]
                # max_scale_index = torch.tensor(max_scale_index).cuda()
                for p_i in range(1, len(x_pyramid)):
                    
                    for b_i in range(scale_max.shape[1]):
                        scale_max_argsort = torch.argsort(torch.stack([scale_max[max_scale_index[b_i]][b_i], scale_max[p_i][b_i]], dim=0), dim = 0) # Shape --> 2 x C
                        sum_scale_batch = torch.sum(scale_max_argsort, dim = 1) # SHape --> 2 x 1]
                        if sum_scale_batch[0] < sum_scale_batch[1]:
                            max_scale_index[b_i] = p_i

                to_append = []
                for b_i in range(scale_max.shape[1]):
                    to_append_batch = F.max_pool2d(x_pyramid[max_scale_index[b_i]][b_i][None], x_pyramid[max_scale_index[b_i]][b_i][None].shape[-1], 1) # Shape --> 1 x C x 1 x 1
                    to_append.append(to_append_batch)

                to_append = torch.cat(to_append, dim = 0)
                c_maps.append(to_append)

        
        if not self.global_pool: 
            return c_maps #, overall_max_scale_index
        else:
            return c_maps, c_maps_scale, max_scale_index, correct_scale_loss


class S2(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride, s2b_bool = False):
        super(S2, self).__init__()

        self.s2b_bool = s2b_bool

        if type(kernel_size) == int:
            self.kernel_size = [kernel_size]

            setattr(self, f's_0', nn.Sequential(nn.Conv2d(channels_in, channels_out, kernel_size, stride),
                                                nn.BatchNorm2d(channels_out, 1e-3),
                                                nn.ReLU(True)
                                                ))

        elif type(kernel_size) == list:
            self.kernel_size = kernel_size
            self.kernel_size.sort()
            for i in range(len(kernel_size)):
                
                setattr(self, f's_{i}', nn.Sequential(nn.Conv2d(channels_in, channels_out, kernel_size[i], stride, dilation = 1),
                                                     nn.BatchNorm2d(channels_out, 1e-3),
                                                     nn.ReLU(True)
                                                    ))

        self.batchnorm = nn.BatchNorm2d(channels_out, 1e-3)


    def forward(self, x_pyramid, prj_name = None, MNIST_Scale = None, category = None, x_input = None, save_rdms = None, plt_filters = None):
        
        s_maps_per_k = []
        for k in range(len(self.kernel_size)):
            s_maps_per_i = []
            layer = getattr(self, f's_{k}')

            for i in range(len(x_pyramid)):  # assuming S is last dimension

                x = x_pyramid[i]
                s_map = layer(x)
                # TODO: think about whether the resolution gets too small here
                ori_size = x.shape[2:4]
                s_map = pad_to_size(s_map, ori_size)
                s_maps_per_i.append(s_map)

            s_maps_per_k.append(s_maps_per_i)

        if len(s_maps_per_k) == 1:
            s_maps = s_maps_per_k[0]
        else:
            s_maps = []
            for i in range(len(x_pyramid)):
                k_list = [s_maps_per_k[j][i] for j in range(len(s_maps_per_k))]
                temp_maps = torch.cat(k_list, dim=1)
                s_maps.append(temp_maps)

        return s_maps


class S3(S2):
    # S3 does the same thing as S2
    pass

"""This is just HMAX oldm, now with new C layer for your choice"""
class HMAX_IP_basic_single_band_deeper(nn.Module):
    def __init__(self,
                 ip_scales = 18,
                 s1_scale=13, #25 #23 #21 #19 #17 #15 #13 #11 # 7, #5
                 s1_la=6.8, #14.1 #12.7 #11.5 #10.3 #9.1 #7.9 #6.8 #5.6 # 3.5, # 2.5
                 s1_si=5.4, #11.3 #10.2 #9.2 #8.2 #7.3 #6.3 #5.4 #4.5 # 2.8, # 2
                 n_ori=4,
                 num_classes=1000,
                 s1_trainable_filters=False,
                 visualize_mode = False,
                 prj_name = None,
                 MNIST_Scale = None,
                 category = None,
                 single_scale_bool = True,
                 use_c_scoring2_optimized = False
                 ):
        super(HMAX_IP_basic_single_band_deeper, self).__init__()
        self.ip_scales = 1
        self.ip_scale_bands = ip_scales
        self.single_scale_bool = False
        self.make_ip_2_bool = False

        self.argmax_bool = False

        # A few settings
        self.s1_scale = s1_scale
        self.s1_la = s1_la
        self.s1_si = s1_si
        self.n_ori = n_ori
        self.num_classes = num_classes
        self.s1_trainable_filters = s1_trainable_filters
        self.MNIST_Scale = MNIST_Scale
        self.category = category
        self.prj_name = prj_name
        self.scale = 4

        self.same_scale_viz = None
        self.base_scale = None
        self.orcale_bool = None
        self.save_rdms = []
        self.plt_filters = []

        self.force_const_size_bool = False

        self.c1_sp_kernel_sizes = [12,10]
        
        self.use_c_scoring2_optimized = use_c_scoring2_optimized

        ########################################################
        ########################################################
        # Reverese
        # self.c1_sp_kernel_sizes.reverse()

        print('c1_sp_kernel_sizes : ',self.c1_sp_kernel_sizes)
    
        ########################################################
        ########################################################
        # Setting the scale stride and number of scales pooled at a time
        self.c_scale_stride = 1
        self.c_num_scales_pooled = 2

        self.c1_scale_stride = self.c_scale_stride
        self.c1_num_scales_pooled = self.c_num_scales_pooled
        
        # Global pooling (spatially)
        self.c2b_scale_stride = self.c_scale_stride
        self.c2b_num_scales_pooled = ip_scales-1 #len(self.c1_sp_kernel_sizes)  # all of them


        ########################################################
        # Feature extractors (in the order of the table in Figure 1)
        self.s1 = S1(scale=self.s1_scale, n_ori=n_ori, padding='valid', trainable_filters = True, #s1_trainable_filters,
                     la=self.s1_la, si=self.s1_si, visualize_mode = visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
        self.c1 = C(global_pool = False, sp_kernel_size=self.c1_sp_kernel_sizes, sp_stride_factor=0.5, n_in_sbands=ip_scales,
                        num_scales_pooled=self.c1_num_scales_pooled, scale_stride=self.c1_scale_stride, visualize_mode = visualize_mode, \
                        c1_bool = True, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
        
        self.s2b_before_1 = S2(channels_in=n_ori, channels_out=128, kernel_size=3, stride=1)
        self.s2b_before_2 = S2(channels_in=128, channels_out=128, kernel_size=3, stride=1)
        self.s2b_before_3 = S2(channels_in=128, channels_out=128, kernel_size=3, stride=1)

        self.s2b = S2(channels_in=128, channels_out=128, kernel_size=[4, 8, 12, 16], stride=1, s2b_bool = True)
        self.c2b = C(global_pool = True, sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=ip_scales-1,
                     num_scales_pooled=self.c2b_num_scales_pooled, scale_stride=self.c2b_scale_stride, c2b_bool = True, prj_name = self.prj_name)
        
        # if not self.use_c_scoring2_optimized:
        #     self.c1 = C(global_pool = False, sp_kernel_size=self.c1_sp_kernel_sizes, sp_stride_factor=0.5, n_in_sbands=ip_scales,
        #                 num_scales_pooled=self.c1_num_scales_pooled, scale_stride=self.c1_scale_stride, visualize_mode = visualize_mode, \
        #                 c1_bool = True, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
        #     self.c2b = C(global_pool = True, sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=ip_scales-1,
        #                 num_scales_pooled=self.c2b_num_scales_pooled, scale_stride=self.c2b_scale_stride, c2b_bool = True, prj_name = self.prj_name)
        # else:
        #     self.c1 = C_scoring2_optimized(
        #         num_channels=n_ori,
        #         pool_func1=nn.MaxPool2d(kernel_size=self.c1_sp_kernel_sizes[0], stride=int(np.ceil(0.5 * self.c1_sp_kernel_sizes[0]))),
        #         pool_func2=nn.MaxPool2d(kernel_size=self.c1_sp_kernel_sizes[1], stride=int(np.ceil(0.5 * self.c1_sp_kernel_sizes[1]))),
        #         skip=self.c1_scale_stride,
        #         global_scale_pool=False
        #     )
        self.c2b = C(global_pool = True, sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=ip_scales-1,
                    num_scales_pooled=self.c2b_num_scales_pooled, scale_stride=self.c2b_scale_stride, c2b_bool = True, prj_name = self.prj_name)
            
            # global pooling handles differently, need to set up the pooling functions correctly
            # self.c2b = C_scoring2_optimized(
            #     num_channels=len(self.s2b.kernel_size) * 128,  # Calculate based on s2b output channels
            #     pool_func1=nn.AdaptiveMaxPool2d(1),  # Global pooling
            #     pool_func2=nn.AdaptiveMaxPool2d(1),  # Global pooling
            #     skip=self.c2b_scale_stride,
            #     global_scale_pool=True
            # )
            
    
        ########################################################

        # # Classifier
        self.classifier = nn.Sequential(
                                        nn.Linear(self.get_s4_in_channels(), 256),  # fc1
                                        nn.Dropout(0.2),
                                        nn.Linear(256, num_classes)  # fc3
                                        )

        # self.overall_max_scale_index = []


    def get_s4_in_channels(self):
       
        c2b_out = len(self.s2b.kernel_size) * self.s2b.s_0[0].weight.shape[0]
        s4_in = c2b_out

        return s4_in


    def make_ip(self, x, same_scale_viz = None, base_scale = None, ip_scales = None, scale = None):

        if ip_scales and scale:
            # print("In right condition")
            ip_scales = ip_scales
            scale = scale
            const_size_bool = True or self.force_const_size_bool
        else:
            ip_scales = self.ip_scales
            scale = self.scale #5
            const_size_bool = False or self.force_const_size_bool

        # if self.MNIST_Scale == 1000:
        #     center_crop = torchvision.transforms.CenterCrop(140)
        #     x = center_crop(x)

        base_image_size = int(x.shape[-1]) 
        # print('base_image_size : ',base_image_size)
        
        if ip_scales == 1:
            image_scales_down = [base_image_size]
            image_scales_up = []
        elif ip_scales == 2:
            image_scales_up = []
            image_scales_down = [np.ceil(base_image_size/(2**(1/scale))), base_image_size]
        else:
            image_scales_down = [np.ceil(base_image_size/(2**(i/scale))) for i in range(int(np.ceil(ip_scales/2)))]
            image_scales_up = [np.ceil(base_image_size*(2**(i/scale))) for i in range(1, int(np.ceil(ip_scales/2)))]
        

        image_scales = image_scales_down + image_scales_up
        index_sort = np.argsort(image_scales)
        index_sort = index_sort[::-1]
        self.image_scales = [image_scales[i_s] for i_s in index_sort]


        if const_size_bool:
            base_image_size = 112
        else:
            base_image_size = int(x.shape[-1]) 


        # print('base_image_size : ',base_image_size)
        # print('self.image_scales : ',self.image_scales)

        if len(self.image_scales) > 1:
            # print('Right Hereeeeeee: ', self.image_scales)
            image_pyramid = []
            for i_s in self.image_scales:
                i_s = int(i_s)
                # print('i_s : ',i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear').clamp(min=0, max=1)

                if const_size_bool:
                    # Padding or Cropping
                    if i_s <= base_image_size:
                        interpolated_img = pad_to_size(interpolated_img, (base_image_size, base_image_size))
                    elif i_s > base_image_size:
                        center_crop = torchvision.transforms.CenterCrop(base_image_size)
                        interpolated_img = center_crop(interpolated_img)
                
                # print('interpolated_img : ',interpolated_img.shape,' ::: i_s : ',i_s,' ::: base_image_size : ',base_image_size)
                image_pyramid.append(interpolated_img)

                # # print('image_pyramid : ',image_pyramid[-1].shape,' ::: i_s : ',i_s,' ::: base_image_size : ',base_image_size)

            # print('image_pyramid len : ',len(image_pyramid))

            return image_pyramid
        else:
            # print('Hereeeeeee')
            ##############################################################################
            if self.orcale_bool:
                # FOr oracle:
                if x.shape[-1] > 224:
                    center_crop = torchvision.transforms.CenterCrop(224)
                    x = center_crop(x)
                elif x.shape[-1] < 224:
                    x = pad_to_size(x, (224, 224))
            ##############################################################################

            return [x]

    def make_ip_2(self, x, same_scale_viz = None, base_scale = None, scale = None):

        base_image_size = int(x.shape[-1]) 

        if scale > 1:
            image_scales_up = [np.ceil(base_image_size*scale), base_image_size]
            image_scales_down = []
        else:
            image_scales_up = []
            image_scales_down = [base_image_size, np.ceil(base_image_size/scale)]

        image_scales = image_scales_down + image_scales_up
        index_sort = np.argsort(image_scales)
        index_sort = index_sort[::-1]
        self.image_scales = [image_scales[i_s] for i_s in index_sort]


        if len(self.image_scales) > 1:
            # print('Right Hereeeeeee: ', self.image_scales)
            image_pyramid = []
            for i_s in self.image_scales:
                i_s = int(i_s)
                # print('i_s : ',i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear').clamp(min=0, max=1)

                image_pyramid.append(interpolated_img)

            return image_pyramid

    def forward(self, x, batch_idx = None, contrastive_scale_loss = False, contrastive_2_bool = False, ip_scales = None, scale = None):

        if x.shape[1] == 3:
            # print('0 1 : ',torch.equal(x[:,0:1], x[:,1:2]))
            # print('1 2 : ',torch.equal(x[:,1:2], x[:,2:3]))
            x = x[:,0:1]

        correct_scale_loss = 0

        ###############################################
        if not self.make_ip_2_bool:
            x_pyramid = self.make_ip(x, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale, ip_scales = ip_scales, scale = scale) # Out 17 Scales x BxCxHxW --> C = 3
        else:
            x_pyramid = self.make_ip_2(x, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale) # Out 17 Scales x BxCxHxW --> C = 3

        # print('x_pyramid : ',len(x_pyramid))
        ###############################################
        s1_maps = self.s1(x_pyramid, self.MNIST_Scale, batch_idx, prj_name = self.prj_name, category = self.category, save_rdms = self.save_rdms, plt_filters = self.plt_filters) # Out 17 Scales x BxCxHxW --> C = 4
        
        # Testing the optimized version C Layer
        if self.use_c_scoring2_optimized:
            c1_maps = self.c1(s1_maps)
        else:
            c1_maps = self.c1(s1_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale, c1_sp_kernel_sizes = self.c1_sp_kernel_sizes, image_scales = self.image_scales, save_rdms = self.save_rdms, plt_filters = self.plt_filters)  # Out 16 Scales x BxCxHxW --> C = 4

        ###############################################
        s2b_bef_maps_1 = self.s2b_before_1(c1_maps, MNIST_Scale = self.MNIST_Scale, prj_name = self.prj_name, category = self.category, x_input = x_pyramid, save_rdms = self.save_rdms, plt_filters = self.plt_filters) # Out 15 Scales x BxCxHxW --> C = 2000
        s2b_bef_maps_2 = self.s2b_before_2(s2b_bef_maps_1, MNIST_Scale = self.MNIST_Scale, prj_name = self.prj_name, category = self.category, x_input = x_pyramid, save_rdms = self.save_rdms, plt_filters = self.plt_filters) # Out 15 Scales x BxCxHxW --> C = 2000
        s2b_bef_maps_3 = self.s2b_before_3(s2b_bef_maps_2, MNIST_Scale = self.MNIST_Scale, prj_name = self.prj_name, category = self.category, x_input = x_pyramid, save_rdms = self.save_rdms, plt_filters = self.plt_filters) # Out 15 Scales x BxCxHxW --> C = 2000

        # c2b_bef_maps = []
        # for s2b_bef_i in range(len(s2b_bef_maps_2)):
        #     c2b_bef_maps.append(self.c2b_before(s2b_bef_maps_2[s2b_bef_i]))

        ###############################################
        # ByPass Route
        s2b_maps = self.s2b(s2b_bef_maps_3, MNIST_Scale = self.MNIST_Scale, prj_name = self.prj_name, category = self.category, x_input = x_pyramid, save_rdms = self.save_rdms, plt_filters = self.plt_filters) # Out 15 Scales x BxCxHxW --> C = 2000
        
        # Testing the optimized version C Layer
        # if self.use_c_scoring2_optimized:
        #     c2b_maps = self.c2b(s2b_maps)
        # else:
        c2b_maps, c2b_scale_maps, max_scale_index, correct_scale_loss = self.c2b(s2b_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, \
                                                                        base_scale = self.base_scale, image_scales = self.image_scales, save_rdms = self.save_rdms, plt_filters = self.plt_filters, \
                                                                        scale_loss = False, argmax_bool = self.argmax_bool) # Overall x BxCx1x1 --> C = 2000
        

        ###############################################
        c2b_maps_flatten = torch.flatten(c2b_maps[0], 1) # Shape --> 1 x B x 400 x 1 x 1 

        if contrastive_2_bool:
            output = c2b_maps_flatten
        else:
            # Classify
            output = self.classifier(c2b_maps_flatten)

        if contrastive_2_bool:
            return output, c2b_scale_maps, max_scale_index, correct_scale_loss
        else:
            return output, c2b_maps[0].squeeze(), max_scale_index, correct_scale_loss


class HMAX_IP_basic_single_band_alex_deep(nn.Module):
    def __init__(self,
                 ip_scales = 18,
                 s1_scale=13, #25 #23 #21 #19 #17 #15 #13 #11 # 7, #5
                 s1_la=6.8, #14.1 #12.7 #11.5 #10.3 #9.1 #7.9 #6.8 #5.6 # 3.5, # 2.5
                 s1_si=5.4, #11.3 #10.2 #9.2 #8.2 #7.3 #6.3 #5.4 #4.5 # 2.8, # 2
                 n_ori=96,
                 num_classes=1000,
                 s1_trainable_filters=False,
                 visualize_mode = False,
                 prj_name = None,
                 MNIST_Scale = None,
                 category = None,
                 single_scale_bool = True,
                 ):
        super(HMAX_IP_basic_single_band_alex_deep, self).__init__()
        self.ip_scales = 1
        self.ip_scale_bands = ip_scales
        self.single_scale_bool = False
        self.make_ip_2_bool = False

        self.argmax_bool = False

        # A few settings
        self.s1_scale = s1_scale
        self.s1_la = s1_la
        self.s1_si = s1_si
        self.n_ori = n_ori
        self.num_classes = num_classes
        self.s1_trainable_filters = s1_trainable_filters
        self.MNIST_Scale = MNIST_Scale
        self.category = category
        self.prj_name = prj_name
        self.scale = 96

        self.same_scale_viz = None
        self.base_scale = None
        self.orcale_bool = None
        self.save_rdms = []
        self.plt_filters = []

        self.force_const_size_bool = False

        self.c1_sp_kernel_sizes = [12,10]

        ########################################################
        ########################################################
        # Reverese
        # self.c1_sp_kernel_sizes.reverse()

        print('c1_sp_kernel_sizes : ',self.c1_sp_kernel_sizes)
    
        ########################################################
        ########################################################
        # Setting the scale stride and number of scales pooled at a time
        self.c_scale_stride = 1
        self.c_num_scales_pooled = 2

        self.c1_scale_stride = self.c_scale_stride
        self.c1_num_scales_pooled = self.c_num_scales_pooled
        
        # Global pooling (spatially)
        self.c2b_scale_stride = self.c_scale_stride
        self.c2b_num_scales_pooled = ip_scales-1 #len(self.c1_sp_kernel_sizes)  # all of them


        ########################################################
        # Feature extractors (in the order of the table in Figure 1)
        self.s1 = S1(scale=self.s1_scale, n_ori=n_ori, padding='valid', trainable_filters = True, #s1_trainable_filters,
                     la=self.s1_la, si=self.s1_si, visualize_mode = visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
        self.c1 = C(global_pool = False, sp_kernel_size=self.c1_sp_kernel_sizes, sp_stride_factor=0.5, n_in_sbands=ip_scales,
                    num_scales_pooled=self.c1_num_scales_pooled, scale_stride=self.c1_scale_stride, visualize_mode = visualize_mode, \
                    c1_bool = True, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)

        self.s2b_before_1 = S2(channels_in=n_ori, channels_out=256, kernel_size=3, stride=1)
        self.s2b_before_2 = S2(channels_in=256, channels_out=384, kernel_size=3, stride=1)
        self.s2b_before_3 = S2(channels_in=384, channels_out=256, kernel_size=3, stride=1)

        self.s2b = S2(channels_in=256, channels_out=256, kernel_size=[4, 8, 12, 16], stride=1, s2b_bool = True)
        self.c2b = C(global_pool = True, sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=ip_scales-1,
                     num_scales_pooled=self.c2b_num_scales_pooled, scale_stride=self.c2b_scale_stride, c2b_bool = True, prj_name = self.prj_name)
    
        ########################################################

        # # Classifier
        self.classifier = nn.Sequential(
                                        nn.Linear(self.get_s4_in_channels(), 256),  # fc1
                                        nn.Dropout(0.2),
                                        nn.Linear(256, num_classes)  # fc3
                                        )

        # self.overall_max_scale_index = []


    def get_s4_in_channels(self):
       
        c2b_out = len(self.s2b.kernel_size) * self.s2b.s_0[0].weight.shape[0]
        s4_in = c2b_out

        return s4_in


    def make_ip(self, x, same_scale_viz = None, base_scale = None, ip_scales = None, scale = None):

        if ip_scales and scale:
            # print("In right condition")
            ip_scales = ip_scales
            scale = scale
            const_size_bool = True or self.force_const_size_bool
        else:
            ip_scales = self.ip_scales
            scale = self.scale #5
            const_size_bool = False or self.force_const_size_bool

        # if self.MNIST_Scale == 1000:
        #     center_crop = torchvision.transforms.CenterCrop(140)
        #     x = center_crop(x)

        base_image_size = int(x.shape[-1]) 
        # print('base_image_size : ',base_image_size)
        
        if ip_scales == 1:
            image_scales_down = [base_image_size]
            image_scales_up = []
        elif ip_scales == 2:
            image_scales_up = []
            image_scales_down = [np.ceil(base_image_size/(2**(1/scale))), base_image_size]
        else:
            image_scales_down = [np.ceil(base_image_size/(2**(i/scale))) for i in range(int(np.ceil(ip_scales/2)))]
            image_scales_up = [np.ceil(base_image_size*(2**(i/scale))) for i in range(1, int(np.ceil(ip_scales/2)))]
        

        image_scales = image_scales_down + image_scales_up
        index_sort = np.argsort(image_scales)
        index_sort = index_sort[::-1]
        self.image_scales = [image_scales[i_s] for i_s in index_sort]


        if const_size_bool:
            base_image_size = 112
        else:
            base_image_size = int(x.shape[-1]) 


        # print('base_image_size : ',base_image_size)
        # print('self.image_scales : ',self.image_scales)

        if len(self.image_scales) > 1:
            # print('Right Hereeeeeee: ', self.image_scales)
            image_pyramid = []
            for i_s in self.image_scales:
                i_s = int(i_s)
                # print('i_s : ',i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear').clamp(min=0, max=1)

                if const_size_bool:
                    # Padding or Cropping
                    if i_s <= base_image_size:
                        interpolated_img = pad_to_size(interpolated_img, (base_image_size, base_image_size))
                    elif i_s > base_image_size:
                        center_crop = torchvision.transforms.CenterCrop(base_image_size)
                        interpolated_img = center_crop(interpolated_img)
                
                # print('interpolated_img : ',interpolated_img.shape,' ::: i_s : ',i_s,' ::: base_image_size : ',base_image_size)
                image_pyramid.append(interpolated_img)

                # # print('image_pyramid : ',image_pyramid[-1].shape,' ::: i_s : ',i_s,' ::: base_image_size : ',base_image_size)

            # print('image_pyramid len : ',len(image_pyramid))

            return image_pyramid
        else:
            # print('Hereeeeeee')
            ##############################################################################
            if self.orcale_bool:
                # FOr oracle:
                if x.shape[-1] > 224:
                    center_crop = torchvision.transforms.CenterCrop(224)
                    x = center_crop(x)
                elif x.shape[-1] < 224:
                    x = pad_to_size(x, (224, 224))
            ##############################################################################

            return [x]

    def make_ip_2(self, x, same_scale_viz = None, base_scale = None, scale = None):

        base_image_size = int(x.shape[-1]) 

        if scale > 1:
            image_scales_up = [np.ceil(base_image_size*scale), base_image_size]
            image_scales_down = []
        else:
            image_scales_up = []
            image_scales_down = [base_image_size, np.ceil(base_image_size/scale)]

        image_scales = image_scales_down + image_scales_up
        index_sort = np.argsort(image_scales)
        index_sort = index_sort[::-1]
        self.image_scales = [image_scales[i_s] for i_s in index_sort]


        if len(self.image_scales) > 1:
            # print('Right Hereeeeeee: ', self.image_scales)
            image_pyramid = []
            for i_s in self.image_scales:
                i_s = int(i_s)
                # print('i_s : ',i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear').clamp(min=0, max=1)

                image_pyramid.append(interpolated_img)

            return image_pyramid

    def forward(self, x, batch_idx = None, contrastive_scale_loss = False, contrastive_2_bool = False, ip_scales = None, scale = None):

        if x.shape[1] == 3:
            # print('0 1 : ',torch.equal(x[:,0:1], x[:,1:2]))
            # print('1 2 : ',torch.equal(x[:,1:2], x[:,2:3]))
            x = x[:,0:1]

        correct_scale_loss = 0

        ###############################################
        if not self.make_ip_2_bool:
            x_pyramid = self.make_ip(x, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale, ip_scales = ip_scales, scale = scale) # Out 17 Scales x BxCxHxW --> C = 3
        else:
            x_pyramid = self.make_ip_2(x, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale) # Out 17 Scales x BxCxHxW --> C = 3

        # print('x_pyramid : ',len(x_pyramid))
        ###############################################
        s1_maps = self.s1(x_pyramid, self.MNIST_Scale, batch_idx, prj_name = self.prj_name, category = self.category, save_rdms = self.save_rdms, plt_filters = self.plt_filters) # Out 17 Scales x BxCxHxW --> C = 4
        c1_maps = self.c1(s1_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale, c1_sp_kernel_sizes = self.c1_sp_kernel_sizes, image_scales = self.image_scales, save_rdms = self.save_rdms, plt_filters = self.plt_filters)  # Out 16 Scales x BxCxHxW --> C = 4

        ###############################################
        s2b_bef_maps_1 = self.s2b_before_1(c1_maps, MNIST_Scale = self.MNIST_Scale, prj_name = self.prj_name, category = self.category, x_input = x_pyramid, save_rdms = self.save_rdms, plt_filters = self.plt_filters) # Out 15 Scales x BxCxHxW --> C = 2000
        s2b_bef_maps_2 = self.s2b_before_2(s2b_bef_maps_1, MNIST_Scale = self.MNIST_Scale, prj_name = self.prj_name, category = self.category, x_input = x_pyramid, save_rdms = self.save_rdms, plt_filters = self.plt_filters) # Out 15 Scales x BxCxHxW --> C = 2000
        s2b_bef_maps_3 = self.s2b_before_3(s2b_bef_maps_2, MNIST_Scale = self.MNIST_Scale, prj_name = self.prj_name, category = self.category, x_input = x_pyramid, save_rdms = self.save_rdms, plt_filters = self.plt_filters) # Out 15 Scales x BxCxHxW --> C = 2000

        # c2b_bef_maps = []
        # for s2b_bef_i in range(len(s2b_bef_maps_2)):
        #     c2b_bef_maps.append(self.c2b_before(s2b_bef_maps_2[s2b_bef_i]))

        ###############################################
        # ByPass Route
        s2b_maps = self.s2b(s2b_bef_maps_3, MNIST_Scale = self.MNIST_Scale, prj_name = self.prj_name, category = self.category, x_input = x_pyramid, save_rdms = self.save_rdms, plt_filters = self.plt_filters) # Out 15 Scales x BxCxHxW --> C = 2000
        c2b_maps, c2b_scale_maps, max_scale_index, correct_scale_loss = self.c2b(s2b_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, \
                                                                        base_scale = self.base_scale, image_scales = self.image_scales, save_rdms = self.save_rdms, plt_filters = self.plt_filters, \
                                                                        scale_loss = False, argmax_bool = self.argmax_bool) # Overall x BxCx1x1 --> C = 2000
        

        ###############################################
        c2b_maps_flatten = torch.flatten(c2b_maps[0], 1) # Shape --> 1 x B x 400 x 1 x 1 

        if contrastive_2_bool:
            output = c2b_maps_flatten
        else:
            # Classify
            output = self.classifier(c2b_maps_flatten)

        if contrastive_2_bool:
            return output, c2b_scale_maps, max_scale_index, correct_scale_loss
        else:
            return output, c2b_maps[0].squeeze(), max_scale_index, correct_scale_loss
        
        

class S2_VGG_Residual(nn.Module):
    """
    VGG-style S2 layer that replaces large kernels with multiple 3x3 convolutions
    and adds residual connections for better gradient flow
    """
    def __init__(self, channels_in, channels_out, kernel_size, stride, s2b_bool=False):
        super(S2_VGG_Residual, self).__init__()
        
        self.s2b_bool = s2b_bool
        self.channels_in = channels_in
        self.channels_out = channels_out
        
        if type(kernel_size) == int:
            self.kernel_size = [kernel_size]
        else:
            self.kernel_size = kernel_size
            self.kernel_size.sort()
        
        # Create VGG-style layers for each kernel size
        for i, ks in enumerate(self.kernel_size):
            # Calculate number of 3x3 layers needed to approximate the receptive field
            # of the original kernel size
            if ks <= 3:
                num_layers = 1
            elif ks <= 5:
                num_layers = 2
            elif ks <= 7:
                num_layers = 3
            else:
                # For larger kernels, use more 3x3 layers
                num_layers = max(3, (ks + 1) // 3)
            
            # Build VGG-style block
            layers = []
            current_channels = channels_in
            
            for j in range(num_layers):
                if j == num_layers - 1:
                    # Last layer outputs the desired number of channels
                    out_ch = channels_out
                else:
                    # Intermediate layers maintain or gradually increase channels
                    out_ch = min(channels_out, current_channels + 32)
                
                layers.extend([
                    nn.Conv2d(current_channels, out_ch, 3, stride if j == 0 else 1, padding=1),
                    nn.BatchNorm2d(out_ch, 1e-3),
                    nn.ReLU(True) if j < num_layers - 1 else nn.Identity()  # No ReLU after last conv
                ])
                current_channels = out_ch
            
            # Add final ReLU after residual connection
            vgg_block = nn.Sequential(*layers)
            setattr(self, f's_{i}', vgg_block)
            
            # Residual connection projection if needed
            if channels_in != channels_out:
                setattr(self, f'residual_proj_{i}', 
                       nn.Sequential(
                           nn.Conv2d(channels_in, channels_out, 1, stride),
                           nn.BatchNorm2d(channels_out, 1e-3)
                       ))
            else:
                setattr(self, f'residual_proj_{i}', nn.Identity())
        
        self.final_relu = nn.ReLU(True)
    
    def forward(self, x_pyramid, prj_name=None, MNIST_Scale=None, category=None, x_input=None, save_rdms=None, plt_filters=None):
        s_maps_per_k = []
        
        for k in range(len(self.kernel_size)):
            s_maps_per_i = []
            vgg_layer = getattr(self, f's_{k}')
            residual_proj = getattr(self, f'residual_proj_{k}')
            
            for i in range(len(x_pyramid)):
                x = x_pyramid[i]
                ori_size = x.shape[2:4]
                
                # VGG-style forward pass
                s_map = vgg_layer(x)
                
                # Residual connection
                if hasattr(self, f'residual_proj_{k}'):
                    identity = residual_proj(x)
                    # Ensure same spatial dimensions for residual addition
                    if s_map.shape[2:] != identity.shape[2:]:
                        identity = pad_to_size(identity, s_map.shape[2:])
                    s_map = s_map + identity
                
                # Apply final ReLU after residual connection
                s_map = self.final_relu(s_map)
                
                # Pad back to original size
                s_map = pad_to_size(s_map, ori_size)
                s_maps_per_i.append(s_map)
            
            s_maps_per_k.append(s_maps_per_i)
        
        if len(s_maps_per_k) == 1:
            s_maps = s_maps_per_k[0]
        else:
            s_maps = []
            for i in range(len(x_pyramid)):
                k_list = [s_maps_per_k[j][i] for j in range(len(s_maps_per_k))]
                temp_maps = torch.cat(k_list, dim=1)
                s_maps.append(temp_maps)
        
        return s_maps


class HMAX_IP_VGG_Residual(nn.Module):
    """
    HMAX network based on HMAX_IP_basic_single_band_deeper but with VGG tricks 
    (3x3 convolutions instead of large kernels) and residual connections
    """
    def __init__(self,
                 ip_scales=18,
                 s1_scale=13,
                 s1_la=6.8,
                 s1_si=5.4,
                 n_ori=4,
                 num_classes=1000,
                 s1_trainable_filters=False,
                 visualize_mode=False,
                 prj_name=None,
                 MNIST_Scale=None,
                 category=None,
                 single_scale_bool=True,
                 ):
        super(HMAX_IP_VGG_Residual, self).__init__()
        self.ip_scales = 1
        self.ip_scale_bands = ip_scales
        self.single_scale_bool = False
        self.make_ip_2_bool = False
        self.argmax_bool = False

        # A few settings
        self.s1_scale = s1_scale
        self.s1_la = s1_la
        self.s1_si = s1_si
        self.n_ori = n_ori
        self.num_classes = num_classes
        self.s1_trainable_filters = s1_trainable_filters
        self.MNIST_Scale = MNIST_Scale
        self.category = category
        self.prj_name = prj_name
        self.scale = 4

        self.same_scale_viz = None
        self.base_scale = None
        self.orcale_bool = None
        self.save_rdms = []
        self.plt_filters = []

        self.force_const_size_bool = False
        self.c1_sp_kernel_sizes = [12, 10]

        print('c1_sp_kernel_sizes : ', self.c1_sp_kernel_sizes)
    
        # Setting the scale stride and number of scales pooled at a time
        self.c_scale_stride = 1
        self.c_num_scales_pooled = 2

        self.c1_scale_stride = self.c_scale_stride
        self.c1_num_scales_pooled = self.c_num_scales_pooled
        
        # Global pooling (spatially)
        self.c2b_scale_stride = self.c_scale_stride
        self.c2b_num_scales_pooled = ip_scales - 1

        # Feature extractors (same S1 and C1 as original)
        self.s1 = S1(scale=self.s1_scale, n_ori=n_ori, padding='valid', trainable_filters=True,
                     la=self.s1_la, si=self.s1_si, visualize_mode=visualize_mode, 
                     prj_name=self.prj_name, MNIST_Scale=self.MNIST_Scale)
        self.c1 = C(global_pool=False, sp_kernel_size=self.c1_sp_kernel_sizes, sp_stride_factor=0.5, 
                    n_in_sbands=ip_scales, num_scales_pooled=self.c1_num_scales_pooled, 
                    scale_stride=self.c1_scale_stride, visualize_mode=visualize_mode,
                    c1_bool=True, prj_name=self.prj_name, MNIST_Scale=self.MNIST_Scale)

        # VGG-style layers with residual connections
        self.s2b_before_1 = S2_VGG_Residual(channels_in=n_ori, channels_out=128, kernel_size=3, stride=1)
        self.s2b_before_2 = S2_VGG_Residual(channels_in=128, channels_out=128, kernel_size=3, stride=1)
        self.s2b_before_3 = S2_VGG_Residual(channels_in=128, channels_out=128, kernel_size=3, stride=1)

        # Multi-scale VGG-style layer (replaces large kernels with multiple 3x3)
        self.s2b = S2_VGG_Residual(channels_in=128, channels_out=128, kernel_size=[4, 8, 12, 16], stride=1, s2b_bool=True)
        self.c2b = C(global_pool=True, sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=ip_scales-1,
                     num_scales_pooled=self.c2b_num_scales_pooled, scale_stride=self.c2b_scale_stride, 
                     c2b_bool=True, prj_name=self.prj_name)
    
        # Classifier with deeper architecture and residual-like connections
        self.classifier = nn.Sequential(
            nn.Linear(self.get_s4_in_channels(), 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def get_s4_in_channels(self):
        c2b_out = len(self.s2b.kernel_size) * 128  # channels_out = 128
        s4_in = c2b_out
        return s4_in

    def make_ip(self, x, same_scale_viz=None, base_scale=None, ip_scales=None, scale=None):
        # Same implementation as original
        if ip_scales and scale:
            ip_scales = ip_scales
            scale = scale
            const_size_bool = True or self.force_const_size_bool
        else:
            ip_scales = self.ip_scales
            scale = self.scale
            const_size_bool = False or self.force_const_size_bool

        base_image_size = int(x.shape[-1]) 
        
        if ip_scales == 1:
            image_scales_down = [base_image_size]
            image_scales_up = []
        elif ip_scales == 2:
            image_scales_up = []
            image_scales_down = [np.ceil(base_image_size/(2**(1/scale))), base_image_size]
        else:
            image_scales_down = [np.ceil(base_image_size/(2**(i/scale))) for i in range(int(np.ceil(ip_scales/2)))]
            image_scales_up = [np.ceil(base_image_size*(2**(i/scale))) for i in range(1, int(np.ceil(ip_scales/2)))]
        
        image_scales = image_scales_down + image_scales_up
        index_sort = np.argsort(image_scales)
        index_sort = index_sort[::-1]
        self.image_scales = [image_scales[i_s] for i_s in index_sort]

        if const_size_bool:
            base_image_size = 112
        else:
            base_image_size = int(x.shape[-1]) 

        if len(self.image_scales) > 1:
            image_pyramid = []
            for i_s in self.image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size=(i_s, i_s), mode='bilinear').clamp(min=0, max=1)

                if const_size_bool:
                    if i_s <= base_image_size:
                        interpolated_img = pad_to_size(interpolated_img, (base_image_size, base_image_size))
                    elif i_s > base_image_size:
                        center_crop = torchvision.transforms.CenterCrop(base_image_size)
                        interpolated_img = center_crop(interpolated_img)
                
                image_pyramid.append(interpolated_img)

            return image_pyramid
        else:
            if self.orcale_bool:
                if x.shape[-1] > 224:
                    center_crop = torchvision.transforms.CenterCrop(224)
                    x = center_crop(x)
                elif x.shape[-1] < 224:
                    x = pad_to_size(x, (224, 224))
            return [x]

    def make_ip_2(self, x, same_scale_viz=None, base_scale=None, scale=None):
        # Same implementation as original
        base_image_size = int(x.shape[-1]) 

        if scale > 1:
            image_scales_up = [np.ceil(base_image_size*scale), base_image_size]
            image_scales_down = []
        else:
            image_scales_up = []
            image_scales_down = [base_image_size, np.ceil(base_image_size/scale)]

        image_scales = image_scales_down + image_scales_up
        index_sort = np.argsort(image_scales)
        index_sort = index_sort[::-1]
        self.image_scales = [image_scales[i_s] for i_s in index_sort]

        if len(self.image_scales) > 1:
            image_pyramid = []
            for i_s in self.image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size=(i_s, i_s), mode='bilinear').clamp(min=0, max=1)
                image_pyramid.append(interpolated_img)
            return image_pyramid

    def forward(self, x, batch_idx=None, contrastive_scale_loss=False, contrastive_2_bool=False, ip_scales=None, scale=None):
        if x.shape[1] == 3:
            x = x[:, 0:1]

        correct_scale_loss = 0

        # Image pyramid generation
        if not self.make_ip_2_bool:
            x_pyramid = self.make_ip(x, same_scale_viz=self.same_scale_viz, base_scale=self.base_scale, 
                                   ip_scales=ip_scales, scale=scale)
        else:
            x_pyramid = self.make_ip_2(x, same_scale_viz=self.same_scale_viz, base_scale=self.base_scale)

        # S1 and C1 layers (same as original)
        s1_maps = self.s1(x_pyramid, self.MNIST_Scale, batch_idx, prj_name=self.prj_name, 
                         category=self.category, save_rdms=self.save_rdms, plt_filters=self.plt_filters)
        c1_maps = self.c1(s1_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, 
                         same_scale_viz=self.same_scale_viz, base_scale=self.base_scale, 
                         c1_sp_kernel_sizes=self.c1_sp_kernel_sizes, image_scales=self.image_scales, 
                         save_rdms=self.save_rdms, plt_filters=self.plt_filters)

        # VGG-style layers with residual connections
        s2b_bef_maps_1 = self.s2b_before_1(c1_maps, MNIST_Scale=self.MNIST_Scale, prj_name=self.prj_name, 
                                          category=self.category, x_input=x_pyramid, save_rdms=self.save_rdms, 
                                          plt_filters=self.plt_filters)
        s2b_bef_maps_2 = self.s2b_before_2(s2b_bef_maps_1, MNIST_Scale=self.MNIST_Scale, prj_name=self.prj_name, 
                                          category=self.category, x_input=x_pyramid, save_rdms=self.save_rdms, 
                                          plt_filters=self.plt_filters)
        s2b_bef_maps_3 = self.s2b_before_3(s2b_bef_maps_2, MNIST_Scale=self.MNIST_Scale, prj_name=self.prj_name, 
                                          category=self.category, x_input=x_pyramid, save_rdms=self.save_rdms, 
                                          plt_filters=self.plt_filters)

        # Multi-scale VGG-style processing
        s2b_maps = self.s2b(s2b_bef_maps_3, MNIST_Scale=self.MNIST_Scale, prj_name=self.prj_name, 
                           category=self.category, x_input=x_pyramid, save_rdms=self.save_rdms, 
                           plt_filters=self.plt_filters)
        c2b_maps, c2b_scale_maps, max_scale_index, correct_scale_loss = self.c2b(
            s2b_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, 
            same_scale_viz=self.same_scale_viz, base_scale=self.base_scale, image_scales=self.image_scales, 
            save_rdms=self.save_rdms, plt_filters=self.plt_filters, scale_loss=False, argmax_bool=self.argmax_bool)

        # Classification
        c2b_maps_flatten = torch.flatten(c2b_maps[0], 1)

        if contrastive_2_bool:
            output = c2b_maps_flatten
        else:
            output = self.classifier(c2b_maps_flatten)

        if contrastive_2_bool:
            return output, c2b_scale_maps, max_scale_index, correct_scale_loss
        else:
            return output, c2b_maps[0].squeeze(), max_scale_index, correct_scale_loss




#########################################################################################################
class HMAX_2_streams(nn.Module):
    def __init__(self,
                 num_classes=10,
                 prj_name = None,
                 model_pre = None,
                 ):
        super(HMAX_2_streams, self).__init__()
#########################################################################################################

        self.num_classes = num_classes

        self.model_pre = model_pre
        # No Image Scale Pyramid
        self.model_pre.ip_scales = 1
        self.ip_scale_bands = 1
        self.contrastive_loss = True

        self.model_pre.single_scale_bool = False

        self.model_pre.force_const_size_bool = True

        self.stream_1_big = False
        
        self.stream_2_bool = True
        if self.stream_2_bool:
            self.stream_2_ip_scales = 5
            self.stream_2_scale = 4



    def forward(self, x, batch_idx = None):

        if x.shape[1] == 3:
            # print('0 1 : ',torch.equal(x[:,0:1], x[:,1:2]))
            # print('1 2 : ',torch.equal(x[:,1:2], x[:,2:3]))
            x = x[:,0:1]

        correct_scale_loss = 0.

        if self.stream_1_big:
            scale_factor_list = [0.707, 0.841, 1, 1.189, 1.414]
            # scale_factor_list = [0.841, 1, 1.189]
            scale_factor = random.choice(scale_factor_list)
            # print('scale_factor 1 : ', scale_factor)
            img_hw = x.shape[-1]
            new_hw = int(img_hw*scale_factor)
            x_rescaled = F.interpolate(x, size = (new_hw, new_hw), mode = 'bilinear').clamp(min=0, max=1)
            # print('x_rescaled : ',x_rescaled.shape)
            if new_hw <= img_hw:
                x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
            elif new_hw > img_hw:
                center_crop = torchvision.transforms.CenterCrop(img_hw)
                x_rescaled = center_crop(x_rescaled)
            # print('x_rescaled : ',x_rescaled.shape)
            
            stream_1_output, stram_1_c2b_feats, max_scale_index, _ = self.model_pre(x_rescaled, batch_idx, ip_scales = self.stream_2_ip_scales, scale = self.stream_2_scale)
            
        else:
            print('Hereeeeeeeee')
            stream_1_output, stram_1_c2b_feats, max_scale_index, _ = self.model_pre(x, batch_idx) #, ip_scales = 2, scale = 4)

        if self.stream_2_bool and self.training: #if not eval mode
            # print('Wrongggggg')
            scale_factor_list = [0.707, 0.841, 1, 1.189, 1.414]
            # scale_factor_list = [0.841, 1, 1.189]
            scale_factor = random.choice(scale_factor_list)
            # print('scale_factor 2 : ', scale_factor)
            img_hw = x.shape[-1]
            new_hw = int(img_hw*scale_factor)
            x_rescaled = F.interpolate(x, size = (new_hw, new_hw), mode = 'bilinear').clamp(min=0, max=1)
            # print('x_rescaled : ',x_rescaled.shape)
            if new_hw <= img_hw:
                x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
            elif new_hw > img_hw:
                center_crop = torchvision.transforms.CenterCrop(img_hw)
                x_rescaled = center_crop(x_rescaled)
            # print('x_rescaled : ',x_rescaled.shape)
            
            stream_2_output, stram_2_c2b_feats, _, _ = self.model_pre(x_rescaled, batch_idx, ip_scales = self.stream_2_ip_scales, scale = self.stream_2_scale)

            correct_scale_loss = torch.mean(torch.abs(stram_1_c2b_feats - stram_2_c2b_feats))
            return stream_2_output, correct_scale_loss

        

        return stream_1_output, correct_scale_loss


@register_model
def hmax_old(pretrained=False, **kwargs):
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    ip_scales = 18
    n_ori = 4
    n_classes=10
    visualize_mode = False 
    prj_name = "This isn't being used"
    MNIST_Scale = 24
    backbone = HMAX_IP_basic_single_band_deeper(ip_scales = ip_scales, n_ori=n_ori,num_classes=n_classes, visualize_mode = visualize_mode, prj_name = prj_name, MNIST_Scale = MNIST_Scale)
    model = HMAX_2_streams(num_classes=n_classes, prj_name = prj_name, model_pre = backbone)
    
    if pretrained:
        checkpoint_path = "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_18_hmax_old_gpu_1_cl_0.5_ip_3_224_224_0000_c1[_6,3,1_]_bypass_1/model_best.pth.tar"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = hmax_old().to(device).eval()
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.model_pre.base_scale = 224
        ip_scales = 18
        model.ip_scales = ip_scales
        model.scale = 2

        model.model_pre.ip_scales = ip_scales
        model.stream_2_bool = False
        
    return model


@register_model
def hmax_old_original(pretrained=False, **kwargs):
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    ip_scales = 18
    n_ori = 4
    n_classes=10
    visualize_mode = False
    prj_name = "This isn't being used"
    MNIST_Scale = 24
    backbone = HMAX_IP_basic_single_band_deeper(ip_scales = ip_scales, n_ori=n_ori,num_classes=n_classes,
                                visualize_mode = visualize_mode, prj_name = prj_name, MNIST_Scale = MNIST_Scale)
    model = HMAX_2_streams(num_classes=n_classes, prj_name = prj_name, model_pre = backbone)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def hmax_2_5(pretrained=False, **kwargs):
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    ip_scales = 18
    n_ori = 4
    visualize_mode = False
    prj_name = "This isn't being used"
    MNIST_Scale = 24
    n_classes = kwargs.get('num_classes', 1000)
    backbone = HMAX_IP_basic_single_band_deeper(ip_scales = ip_scales, n_ori=n_ori,num_classes=n_classes,
                                visualize_mode = visualize_mode, prj_name = prj_name, MNIST_Scale = MNIST_Scale)
    model = HMAX_2_streams(num_classes=n_classes, prj_name = prj_name, model_pre = backbone)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def hmax_2_1(pretrained=False, **kwargs):
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    ip_scales = 18
    n_ori = 4
    visualize_mode = False
    prj_name = "This isn't being used"
    MNIST_Scale = 24
    n_classes = kwargs.get('num_classes', 1000)
    model = HMAX_IP_basic_single_band_deeper(ip_scales = ip_scales, n_ori=n_ori,num_classes=n_classes,
                                visualize_mode = visualize_mode, prj_name = prj_name, MNIST_Scale = MNIST_Scale)
    if pretrained:
        raise NotImplementedError
    return model


@register_model
def hmax_old_deep(pretrained=False, **kwargs):
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    ip_scales = 18
    n_ori = 96
    n_classes=10
    visualize_mode = False 
    prj_name = "This isn't being used"
    MNIST_Scale = 24
    backbone = HMAX_IP_basic_single_band_alex_deep(ip_scales = ip_scales, n_ori=n_ori,num_classes=n_classes, visualize_mode = visualize_mode, prj_name = prj_name, MNIST_Scale = MNIST_Scale)
    model = HMAX_2_streams(num_classes=n_classes, prj_name = prj_name, model_pre = backbone)
        
    if pretrained:
        model.model_pre.base_scale = 224
        model.ip_scales = 18
        model.scale = 2

        model.model_pre.ip_scales = 18
        model.stream_2_bool = False
        
    return model


@register_model
def hmax_new_tricks(pretrained=False, **kwargs):
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    ip_scales = 18
    n_ori = 4
    n_classes=10
    visualize_mode = False 
    prj_name = "This isn't being used"
    MNIST_Scale = 24
    backbone = HMAX_IP_VGG_Residual(ip_scales = ip_scales, n_ori=n_ori,num_classes=n_classes, visualize_mode = visualize_mode, prj_name = prj_name, MNIST_Scale = MNIST_Scale)
    model = HMAX_2_streams(num_classes=n_classes, prj_name = prj_name, model_pre = backbone)
    
    if pretrained:
        model.model_pre.base_scale = 224
        ip_scales = 18
        model.ip_scales = ip_scales
        model.scale = 2

        model.model_pre.ip_scales = ip_scales
        model.stream_2_bool = False
        
    return model
        

@register_model
def hmax_old_s1_c_scoring(pretrained=False, **kwargs):
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    ip_scales = 18
    n_ori = 4
    n_classes=10
    visualize_mode = False 
    prj_name = "This isn't being used"
    MNIST_Scale = 24
    backbone = HMAX_IP_basic_single_band_deeper(ip_scales = ip_scales, n_ori=n_ori,num_classes=n_classes, visualize_mode = visualize_mode, prj_name = prj_name, MNIST_Scale = MNIST_Scale, use_c_scoring2_optimized = True)
    model = HMAX_2_streams(num_classes=n_classes, prj_name = prj_name, model_pre = backbone)
    
    if pretrained:
        model.model_pre.base_scale = 224
        ip_scales = 18
        model.ip_scales = ip_scales
        model.scale = 2

        model.model_pre.ip_scales = ip_scales
        model.stream_2_bool = False
        
    return model
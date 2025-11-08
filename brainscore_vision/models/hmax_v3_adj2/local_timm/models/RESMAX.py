"""
An implementation of RESMAX:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision

import numpy as np
import scipy as sp
import os
import time
import pdb
import random


from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs
from .ALEXMAX import C_scoring, C
from .ALEXMAX3_optimized import C_scoring2_optimized, C_scoring2_optimized_debug
from .HMAX import get_ip_scales

# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"


def pad_to_size(a, size, mode='constant'):
    current_size = (a.shape[-2], a.shape[-1])
    total_pad_h = size[0] - current_size[0]
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top

    total_pad_w = size[1] - current_size[1]
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    a = nn.functional.pad(a, (pad_left, pad_right, pad_top, pad_bottom), mode=mode, value=0)

    return a


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=1)
        
        if strides > 1 or in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    
class Residual1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size, padding=1, stride=strides, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=1, bias=False)
        
        if strides > 1 or in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.conv3 = None
            self.bn3 = None

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None and self.bn3 is not None:
            X = self.bn3(self.conv3(X))
        Y += X
        return F.relu(Y)
    
class BrainscoreSafe(nn.Module):
    """
    Wrapper for modules that return lists (multi-scale pyramids).

    BrainScore's hooks see a flattened tensor, but the actual forward
    pass still uses the proper list format internally.
    """
    def __init__(self, module):
        super().__init__()
        self.module = module
        self._last_list_output = None

    def forward(self, x):
        out = self.module(x)
        if isinstance(out, list):
            # Cache the real list for the next layer
            self._last_list_output = out
            # BrainScore's hooks see a flattened tensor
            # Use adaptive pooling to ensure all scales have same spatial size
            if len(out) > 0:
                target_size = out[0].shape[-2:]
                pooled = [F.adaptive_avg_pool2d(o, target_size) for o in out]
                # Stack and average across scales
                return torch.stack(pooled, dim=0).mean(dim=0)
            else:
                return out
        else:
            self._last_list_output = None
            return out

class S1_Res(nn.Module):
    def __init__(self):
        super(S1_Res, self).__init__()
        self.layer1 = nn.Sequential(
            Residual(3, 48, strides=2),
            Residual(48, 48),
            Residual(48, 96, strides=2)
        )

    def forward(self, x_pyramid):
        if type(x_pyramid) == list:
            return [self.layer1(x) for x in x_pyramid]
        else:
            return self.layer1(x_pyramid)

class S2_Res(nn.Module):
    def __init__(self):
        super(S2_Res, self).__init__()
        self.layer = nn.Sequential(
            Residual(96, 128),
            Residual(128, 256)
        )

    def forward(self, x_pyramid):
        return [self.layer(x) for x in x_pyramid]
    
class S2b_Res(nn.Module):
    def __init__(self):
        super(S2b_Res, self).__init__()
        # Each Residual block has 2 3x3 convs, so we need fewer blocks
        # to achieve same receptive field
        # 4x4 -> 1 residual block (2 3x3 convs)
        # 8x8 -> 2 residual blocks (4 3x3 convs)
        # 12x12 -> 3 residual blocks (6 3x3 convs)
        # 16x16 -> 4 residual blocks (8 3x3 convs)
        self.kernel_to_blocks = {4: 1, 8: 2, 12: 3, 16: 4}
        
        self.s2b_seqs = nn.ModuleList()
        for kernel_size, num_blocks in self.kernel_to_blocks.items():
            blocks = []
            # Initial projection to higher dimensions
            blocks.append(nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=1),  # 1x1 conv for dimension matching
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            ))
            # Stack of residual blocks
            for _ in range(num_blocks):
                blocks.append(Residual(256, 256))
                
            self.s2b_seqs.append(nn.Sequential(*blocks))

    def forward(self, x_pyramid):
        bypass = [torch.cat([seq(out) for seq in self.s2b_seqs], dim=1) for out in x_pyramid]
        return bypass
    
class S2b_Res1(nn.Module):
    def __init__(self):
        super(S2b_Res1, self).__init__()
        # Each Residual block has 2 3x3 convs, so we need fewer blocks
        # to achieve same receptive field
        # 4x4 -> 1 residual block (2 3x3 convs)
        # 8x8 -> 2 residual blocks (4 3x3 convs)
        # 12x12 -> 3 residual blocks (6 3x3 convs)
        # 16x16 -> 4 residual blocks (8 3x3 convs)
        self.kernel_to_blocks = {4: 1, 8: 2, 12: 3, 16: 4}
        
        self.s2b_seqs = nn.ModuleList()
        for kernel_size, num_blocks in self.kernel_to_blocks.items():
            blocks = []
            # Initial projection to higher dimensions
            blocks.append(nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=1),  # 1x1 conv for dimension matching
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            ))
            # Stack of residual blocks
            for _ in range(num_blocks):
                blocks.append(Residual1(256, 256))

            self.s2b_seqs.append(nn.Sequential(*blocks))

    def forward(self, x_pyramid):
        bypass = [torch.cat([seq(out) for seq in self.s2b_seqs], dim=1) for out in x_pyramid]
        return bypass

class C_adp(nn.Module):
    # Spatial then Scale
    def __init__(self,
                 pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                 pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                 global_scale_pool=None):
        super(C_adp, self).__init__()
        self.pool1 = pool_func1
        self.pool2 = pool_func2
        self.global_scale_pool = global_scale_pool

    def forward(self, x_pyramid):
        out = []

        if self.global_scale_pool is not None:
            pooled = [self.global_scale_pool(x) for x in x_pyramid]
            out = pooled[0]
            for p in pooled[1:]:
                out = torch.max(out, p)

        else:
            if len(x_pyramid) == 1:
                return [self.pool1(x_pyramid[0])]

            for i in range(len(x_pyramid) - 1):
                x_1 = self.pool1(x_pyramid[i])
                x_2 = self.pool2(x_pyramid[i + 1])

                # Interpolate to match spatial sizes
                if x_1.shape[-1] > x_2.shape[-1]:
                    x_2 = F.interpolate(x_2, size=x_1.shape[-2:], mode='bilinear')
                else:
                    x_1 = F.interpolate(x_1, size=x_2.shape[-2:], mode='bilinear')

                stacked = torch.stack([x_1, x_2], dim=-1)
                to_append, _ = torch.max(stacked, dim=-1)
                out.append(to_append)

        return out

class S3_Res(nn.Module):
    def __init__(self, input_channels=256):
        super(S3_Res, self).__init__()
        self.layer = nn.Sequential(
            Residual(input_channels, 256),
            Residual(256, 384),
            Residual(384, 384),
            Residual(384, 256)
        )

    def forward(self, x_pyramid):
        return [self.layer(x) for x in x_pyramid]

class RESMAX_V2(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False,
                 bypass=False, main_route=False,
                 c_scoring='v2',
                 **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        self.bypass = bypass
        self.c_scoring = c_scoring
        self.main_route = main_route
        super(RESMAX_V2, self).__init__()

        self.s1 = S1_Res()

        # C1 using optimized layer
        self.c1 = C_scoring2_optimized_debug(
            num_channels=96,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            skip=1,
            global_scale_pool=False
        )
        
        self.s2 = S2_Res()
        # C2 using optimized layer
        self.c2 = C_scoring2_optimized(
            num_channels=256,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            pool_func2=nn.MaxPool2d(kernel_size=6, stride=2),
            resize_kernel_1=3,
            resize_kernel_2=1,
            skip=2,
            global_scale_pool=False
        )
        
        if self.bypass:
            self.s2b = S2b_Res()
            self.c2b_seq = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

        self.s3 = S3_Res()
        if self.ip_scale_bands > 4:
            self.global_pool = C_scoring2_optimized(
                num_channels=256,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=6, stride=3, padding=1),
                resize_kernel_1=3,
                resize_kernel_2=1,
                skip=2,
                global_scale_pool=False
            )
        else:
            self.global_pool = C(global_scale_pool=True)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_size, 4096),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

        self.print_param_stats()

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

    def forward(self, x, pyramid=False):
        if self.main_route:
            out = self.make_ip(x, 2)
        else:
            out = self.make_ip(x, self.ip_scale_bands)
        
        out = self.s1(out)
        out_c1 = self.c1(out)
        out = self.s2(out_c1)
        out_c2 = self.c2(out)
        
        if self.bypass:
            bypass = self.s2b(out_c1)
            bypass = self.c2b_seq(bypass[0])
            bypass = bypass.reshape(bypass.size(0), -1)
        
        out = self.s3(out_c2)
        out = self.global_pool(out)
        if isinstance(out, list):
            out = out[0]
        out = out.reshape(out.size(0), -1)

        if self.bypass:
            out = torch.cat([out, bypass], dim=1)
        
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            if self.bypass:
                return out, out_c1, out_c2, bypass
            else:
                return out, out_c1, out_c2

        return out
    
    def print_param_stats(self):
        print(f"\nParameter breakdown for {self.__class__.__name__}:\n")
        total_params = 0
        stats = []
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += n_params
            stats.append((name, n_params))

        stats.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Module':30s} | {'# Params':>10s} | {'% of Total':>10s}")
        print("-" * 60)
        for name, count in stats:
            pct = 100 * count / total_params
            print(f"{name:30s} | {count:10,d} | {pct:10.2f}%")
        print("-" * 60)
        print(f"{'Total':30s} | {total_params:10,d} | {100.00:10.2f}%\n")

class RESMAX_V2_1(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False,
                 bypass=False, main_route=False,
                 c_scoring='v2',
                 **kwargs):
        """
        choose biggest band in bypass
        """
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        self.bypass = bypass
        self.c_scoring = c_scoring
        self.main_route = main_route
        super(RESMAX_V2_1, self).__init__()

        self.s1 = S1_Res()

        # C1 using optimized layer
        self.c1 = C_scoring2_optimized_debug(
            num_channels=96,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            skip=1,
            global_scale_pool=False
        )
        
        self.s2 = S2_Res()
        # C2 using optimized layer
        self.c2 = C_scoring2_optimized(
            num_channels=256,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            pool_func2=nn.MaxPool2d(kernel_size=6, stride=2),
            resize_kernel_1=3,
            resize_kernel_2=1,
            skip=2,
            global_scale_pool=False
        )
        
        if self.bypass:
            self.s2b = S2b_Res()
            self.c2b_seq = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

        self.s3 = S3_Res()
        if self.ip_scale_bands > 4:
            self.global_pool = C_scoring2_optimized(
                num_channels=256,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=6, stride=3, padding=1),
                resize_kernel_1=3,
                resize_kernel_2=1,
                skip=2,
                global_scale_pool=False
            )
        else:
            self.global_pool = C(global_scale_pool=True)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_size, 4096),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

        self.print_param_stats()

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

    def forward(self, x, pyramid=False):
        if self.main_route:
            out = self.make_ip(x, 2)
        else:
            out = self.make_ip(x, self.ip_scale_bands)
        
        out = self.s1(out)
        out_c1 = self.c1(out)
        out = self.s2(out_c1)
        out_c2 = self.c2(out)
        
        if self.bypass:
            bypass = self.s2b(out_c1)
            # bypass_processed = [self.c2b_seq(b) for b in bypass]
            # bypass = torch.stack(bypass_processed, dim=0)
            # bypass, _ = torch.max(bypass, dim=0)
            bypass = self.c2b_seq(bypass[-1])
            bypass = bypass.reshape(bypass.size(0), -1)
        
        out = self.s3(out_c2)
        out = self.global_pool(out)
        if isinstance(out, list):
            out = out[0]
        out = out.reshape(out.size(0), -1)

        if self.bypass:
            out = torch.cat([out, bypass], dim=1)
        
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            if self.bypass:
                return out, out_c1, out_c2, bypass
            else:
                return out, out_c1, out_c2

        return out
    
    def print_param_stats(self):
        print(f"\nParameter breakdown for {self.__class__.__name__}:\n")
        total_params = 0
        stats = []
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += n_params
            stats.append((name, n_params))

        stats.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Module':30s} | {'# Params':>10s} | {'% of Total':>10s}")
        print("-" * 60)
        for name, count in stats:
            pct = 100 * count / total_params
            print(f"{name:30s} | {count:10,d} | {pct:10.2f}%")
        print("-" * 60)
        print(f"{'Total':30s} | {total_params:10,d} | {100.00:10.2f}%\n")
        
"""smart choose bands after c2b"""
class RESMAX_V2_2(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=3, classifier_input_size=18432, contrastive_loss=False, pyramid=False,
                 bypass=True, main_route=False,
                 c_scoring='v2',
                 **kwargs):
        """
        smartly choose band in bypass use c score
        """
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        self.bypass = bypass
        self.c_scoring = c_scoring
        self.main_route = main_route
        super(RESMAX_V2_2, self).__init__()

        # Wrap list-returning layers with BrainscoreSafe for BrainScore compatibility
        self.s1 = BrainscoreSafe(S1_Res())

        # C1 using optimized layer - also returns list, so wrap it
        self.c1 = BrainscoreSafe(C_scoring2_optimized_debug(
            num_channels=96,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            skip=1,
            global_scale_pool=False
        ))

        self.s2 = BrainscoreSafe(S2_Res())
        # C2 using optimized layer - also returns list, so wrap it
        self.c2 = BrainscoreSafe(C_scoring2_optimized(
            num_channels=256,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            pool_func2=nn.MaxPool2d(kernel_size=6, stride=2),
            resize_kernel_1=3,
            resize_kernel_2=1,
            skip=2,
            global_scale_pool=False
        ))

        if self.bypass:
            self.s2b = BrainscoreSafe(S2b_Res())
            self.c2b_seq = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((6, 6))
            )
            # c2b_score returns tensor (global_scale_pool=True), so wrap it too
            self.c2b_score = BrainscoreSafe(C_scoring2_optimized_debug(
                num_channels=1024,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                global_scale_pool=True
            ))

        self.s3 = BrainscoreSafe(S3_Res())
        if self.ip_scale_bands > 4:
            self.global_pool = C_scoring2_optimized(
                num_channels=256,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=6, stride=3, padding=1),
                resize_kernel_1=3,
                resize_kernel_2=1,
                skip=2,
                global_scale_pool=False
            )
        else:
            self.global_pool = C(global_scale_pool=True)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_size, 4096),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

        self.print_param_stats()

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

    def forward(self, x, pyramid=False):
        if self.main_route:
            out = self.make_ip(x, 2)
        else:
            out = self.make_ip(x, self.ip_scale_bands)

        # s1 is wrapped with BrainscoreSafe - restore list for c1
        out = self.s1(out)
        if hasattr(self.s1, "_last_list_output") and self.s1._last_list_output is not None:
            out = self.s1._last_list_output

        out_c1 = self.c1(out)
        # Restore list from c1 for s2
        if hasattr(self.c1, "_last_list_output") and self.c1._last_list_output is not None:
            out_c1 = self.c1._last_list_output

        # s2 is wrapped with BrainscoreSafe - restore list for c2
        out = self.s2(out_c1)
        if hasattr(self.s2, "_last_list_output") and self.s2._last_list_output is not None:
            out = self.s2._last_list_output

        out_c2 = self.c2(out)
        # Restore list from c2
        if hasattr(self.c2, "_last_list_output") and self.c2._last_list_output is not None:
            out_c2 = self.c2._last_list_output

        if self.bypass:
            # s2b is wrapped with BrainscoreSafe - restore list for c2b_score
            bypass = self.s2b(out_c1)
            if hasattr(self.s2b, "_last_list_output") and self.s2b._last_list_output is not None:
                bypass = self.s2b._last_list_output

            bypass = self.c2b_score(bypass)
            # Restore list from c2b_score for c2b_seq
            if hasattr(self.c2b_score, "_last_list_output") and self.c2b_score._last_list_output is not None:
                bypass = self.c2b_score._last_list_output
            bypass = self.c2b_seq(bypass)
            bypass = bypass.reshape(bypass.size(0), -1)

        # s3 is wrapped with BrainscoreSafe - restore list for global_pool
        out = self.s3(out_c2)
        if hasattr(self.s3, "_last_list_output") and self.s3._last_list_output is not None:
            out = self.s3._last_list_output

        out = self.global_pool(out)
        if isinstance(out, list):
            out = out[0]
        out = out.reshape(out.size(0), -1)

        if self.bypass:
            out = torch.cat([out, bypass], dim=1)
        
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            if self.bypass:
                return out, out_c1, out_c2, bypass
            else:
                return out, out_c1, out_c2

        return out
    
    def print_param_stats(self):
        print(f"\nParameter breakdown for {self.__class__.__name__}:\n")
        total_params = 0
        stats = []
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += n_params
            stats.append((name, n_params))

        stats.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Module':30s} | {'# Params':>10s} | {'% of Total':>10s}")
        print("-" * 60)
        for name, count in stats:
            pct = 100 * count / total_params
            print(f"{name:30s} | {count:10,d} | {pct:10.2f}%")
        print("-" * 60)
        print(f"{'Total':30s} | {total_params:10,d} | {100.00:10.2f}%\n")
        
class RESMAX_V2_3(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False,
                 bypass=False, main_route=False,
                 c_scoring='v2',
                 **kwargs):
        """
        Use additional one bn after conv3 in residual block, check Residual1 plz
        choose smallest band in bypass
        """
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        self.bypass = bypass
        self.c_scoring = c_scoring
        self.main_route = main_route
        super(RESMAX_V2_3, self).__init__()

        self.s1 = nn.Sequential(
            Residual1(3, 48, strides=2),
            Residual1(48, 48),
            Residual1(48, 96, strides=2)
        )

        # C1 using optimized layer
        self.c1 = C_scoring2_optimized_debug(
            num_channels=96,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            skip=1,
            global_scale_pool=False
        )
        
        self.s2 = nn.Sequential(
            Residual1(96, 128),
            Residual1(128, 256)
        )
        # C2 using optimized layer
        self.c2 = C_scoring2_optimized(
            num_channels=256,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            pool_func2=nn.MaxPool2d(kernel_size=6, stride=2),
            resize_kernel_1=3,
            resize_kernel_2=1,
            skip=2,
            global_scale_pool=False
        )
        
        if self.bypass:
            self.s2b = S2b_Res()
            self.c2b_seq = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

        self.s3 = nn.Sequential(
            Residual1(256, 256),
            Residual1(256, 384),
            Residual1(384, 384),
            Residual1(384, 256)
        )
        if self.ip_scale_bands > 4:
            self.global_pool = C_scoring2_optimized(
                num_channels=256,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=6, stride=3, padding=1),
                resize_kernel_1=3,
                resize_kernel_2=1,
                skip=2,
                global_scale_pool=False
            )
        else:
            self.global_pool = C(global_scale_pool=True)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_size, 4096),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

        self.print_param_stats()

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

    def forward(self, x, pyramid=False):
        def apply(module, x):
            return [module(xi) for xi in x] if isinstance(x, list) else module(x)
        
        out = self.make_ip(x, self.ip_scale_bands)
        
        out = apply(self.s1, out)
        out_c1 = self.c1(out)
        out = apply(self.s2, out_c1)
        out_c2 = self.c2(out)
        
        if self.bypass:
            bypass = self.s2b(out_c1)
            bypass = self.c2b_seq(bypass[0])
            bypass = bypass.reshape(bypass.size(0), -1)
        
        out = apply(self.s3, out_c2)
        out = self.global_pool(out)
        if isinstance(out, list):
            out = out[0]
        out = out.reshape(out.size(0), -1)

        if self.bypass:
            out = torch.cat([out, bypass], dim=1)
        
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            if self.bypass:
                return out, out_c1, out_c2, bypass
            else:
                return out, out_c1, out_c2

        return out
    
    def print_param_stats(self):
        print(f"\nParameter breakdown for {self.__class__.__name__}:\n")
        total_params = 0
        stats = []
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += n_params
            stats.append((name, n_params))

        stats.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Module':30s} | {'# Params':>10s} | {'% of Total':>10s}")
        print("-" * 60)
        for name, count in stats:
            pct = 100 * count / total_params
            print(f"{name:30s} | {count:10,d} | {pct:10.2f}%")
        print("-" * 60)
        print(f"{'Total':30s} | {total_params:10,d} | {100.00:10.2f}%\n")
        
"""trying add adptive pooling at the end of bypass, not tested yet"""
class RESMAX_V4(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=9216, contrastive_loss=False,
                 bypass=False,
                 **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        # self.big_size = big_size
        self.bypass = bypass

        super(RESMAX_V4, self).__init__()

        self.s1 = nn.Sequential(
            Residual(3, 48, strides=2),
            Residual(48, 48),
            Residual(48, 96, strides=2)
        )

        # C1 using optimized layer
        self.c1 = C_scoring2_optimized_debug(
            num_channels=96,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            skip=1,
            global_scale_pool=False
        )
        
        self.s2 = nn.Sequential(
            Residual(96, 128),
            Residual(128, 256)
        )
        # C2 using optimized layer
        self.c2 = C_scoring2_optimized(
            num_channels=256,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            pool_func2=nn.MaxPool2d(kernel_size=6, stride=2),
            resize_kernel_1=3,
            resize_kernel_2=1,
            skip=2,
            global_scale_pool=False
        )
        
        if self.bypass:
            self.s2b = S2b_Res()
            self.c2b_score = C_scoring2_optimized_debug(
                num_channels=1024,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                global_scale_pool=True
            )
            self.c2b_seq = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((6, 6))
            )

        self.s3 = nn.Sequential(
            Residual(256, 256),
            Residual(256, 384),
            Residual(384, 384),
            Residual(384, 256),
        )
        
        if self.ip_scale_bands > 4:
            self.global_pool = C_scoring2_optimized(
                num_channels=256,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=6, stride=3, padding=1),
                resize_kernel_1=3,
                resize_kernel_2=1,
                skip=2,
                global_scale_pool=False
            )
        else:
            self.global_pool = C(global_scale_pool=True)

        self.main_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_size, 4096),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

        self.print_param_stats()

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
        def apply(module, x):
            return [module(xi) for xi in x] if isinstance(x, list) else module(x)
        
        out = self.make_ip(x, self.ip_scale_bands)

        out = apply(self.s1, out)
        out_c1 = self.c1(out)
        out = apply(self.s2, out_c1)
        out_c2 = self.c2(out)
        
        if self.bypass:
            bypass = self.s2b(out_c1)
            bypass = self.c2b_score(bypass)
            bypass = self.c2b_seq(bypass)
            bypass = bypass.reshape(bypass.size(0), -1)
        
        out = apply(self.s3, out_c2)
        out = self.global_pool(out)
        if isinstance(out, list):
            out = out[0]
        out = self.main_pool(out)
        out = out.reshape(out.size(0), -1)

        if self.bypass:
            out = torch.cat([out, bypass], dim=1)
        
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            if self.bypass:
                return out, out_c1, out_c2, bypass
            else:
                return out, out_c1, out_c2

        return out
    
    def print_param_stats(self):
        print(f"\nParameter breakdown for {self.__class__.__name__}:\n")
        total_params = 0
        stats = []
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += n_params
            stats.append((name, n_params))

        stats.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Module':30s} | {'# Params':>10s} | {'% of Total':>10s}")
        print("-" * 60)
        for name, count in stats:
            pct = 100 * count / total_params
            print(f"{name:30s} | {count:10,d} | {pct:10.2f}%")
        print("-" * 60)
        print(f"{'Total':30s} | {total_params:10,d} | {100.00:10.2f}%\n")

class AbsLikeReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) + F.relu(-x)

"""adding absolute S1"""
class RESMAX_abs(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=9216, contrastive_loss=False,
                 bypass=False,
                 **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        # self.big_size = big_size
        self.bypass = bypass

        super(RESMAX_abs, self).__init__()

        self.s1 = nn.Sequential(
            Residual(3, 48, strides=2),
            Residual(48, 48),
            Residual(48, 96, strides=2)
        )
        
        self.abs_layer = nn.Sequential(
            AbsLikeReLU()
        )

        # C1 using optimized layer
        self.c1 = C_scoring2_optimized_debug(
            num_channels=96,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            skip=1,
            global_scale_pool=False
        )
        
        self.s2 = nn.Sequential(
            Residual(96, 128),
            Residual(128, 256)
        )
        # C2 using optimized layer
        self.c2 = C_scoring2_optimized(
            num_channels=256,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            pool_func2=nn.MaxPool2d(kernel_size=6, stride=2),
            resize_kernel_1=3,
            resize_kernel_2=1,
            skip=2,
            global_scale_pool=False
        )
        
        if self.bypass:
            self.s2b = S2b_Res()
            self.c2b_score = C_scoring2_optimized_debug(
                num_channels=1024,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                global_scale_pool=True
            )
            self.c2b_seq = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((6, 6))
            )

        self.s3 = nn.Sequential(
            Residual(256, 256),
            Residual(256, 384),
            Residual(384, 384),
            Residual(384, 256),
        )
        
        if self.ip_scale_bands > 4:
            self.global_pool = C_scoring2_optimized(
                num_channels=256,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=6, stride=3, padding=1),
                resize_kernel_1=3,
                resize_kernel_2=1,
                skip=2,
                global_scale_pool=False
            )
        else:
            self.global_pool = C(global_scale_pool=True)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_size, 4096),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

        self.print_param_stats()

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
        def apply(module, x):
            return [module(xi) for xi in x] if isinstance(x, list) else module(x)
        
        out = self.make_ip(x, self.ip_scale_bands)

        out = apply(self.s1, out)
        out = apply(self.abs_layer, out)

        out_c1 = self.c1(out)
        out = apply(self.s2, out_c1)
        out_c2 = self.c2(out)
        
        if self.bypass:
            bypass = self.s2b(out_c1)
            bypass = self.c2b_score(bypass)
            bypass = self.c2b_seq(bypass)
            bypass = bypass.reshape(bypass.size(0), -1)
        
        out = apply(self.s3, out_c2)
        out = self.global_pool(out)
        if isinstance(out, list):
            out = out[0]
        out = out.reshape(out.size(0), -1)

        if self.bypass:
            out = torch.cat([out, bypass], dim=1)
        
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            if self.bypass:
                return out, out_c1, out_c2, bypass
            else:
                return out, out_c1, out_c2

        return out
    
    def print_param_stats(self):
        print(f"\nParameter breakdown for {self.__class__.__name__}:\n")
        total_params = 0
        stats = []
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += n_params
            stats.append((name, n_params))

        stats.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Module':30s} | {'# Params':>10s} | {'% of Total':>10s}")
        print("-" * 60)
        for name, count in stats:
            pct = 100 * count / total_params
            print(f"{name:30s} | {count:10,d} | {pct:10.2f}%")
        print("-" * 60)
        print(f"{'Total':30s} | {total_params:10,d} | {100.00:10.2f}%\n")

# """
# choose smartly in bypass
# """
# class CHRESMAX_V3_2_abs(nn.Module):
#     """
#     Example student-teacher style model with scale-consistency loss,
#     using RESMAX_V2 as the backbone.

#     In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
#     the returned features are [0] for C1 and C2.
#     """
#     def __init__(self, 
#                  num_classes=1000,
#                  in_chans=3,
#                  ip_scale_bands=1,
#                  classifier_input_size=13312,
#                  contrastive_loss=True,
#                  bypass=False,
#                  **kwargs):
#         super().__init__()
#         self.contrastive_loss = contrastive_loss
#         self.num_classes = num_classes
#         self.in_chans = in_chans
#         self.ip_scale_bands = ip_scale_bands
#         self.bypass = bypass
        
#         # Use the optimized backbone
#         self.model_backbone = RESMAX_abs(
#             num_classes=num_classes,
#             in_chans=in_chans,
#             ip_scale_bands=self.ip_scale_bands,
#             classifier_input_size=classifier_input_size,
#             contrastive_loss=self.contrastive_loss,
#             bypass=bypass,
#         )

#     def forward(self, x):
#         """
#         Creates two streams (original + random-scaled) for scale-consistency training.
#         Returns:
#             (output_of_stream1, correct_scale_loss)
#         """
#         # stream 1 (original scale)
#         result = self.model_backbone(x)
#         if self.bypass:
#             stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result
#         else:
#             stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result

#         # stream 2 (random scale)
#         scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
#         scale_factor = random.choice(scale_factor_list)
#         img_hw = x.shape[-1]
#         new_hw = int(img_hw * scale_factor)
#         x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

#         if new_hw <= img_hw:
#             # pad if smaller
#             x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
#         else:
#             # center-crop if bigger
#             center_crop = torchvision.transforms.CenterCrop(img_hw)
#             x_rescaled = center_crop(x_rescaled)

#         # forward pass on the scaled input
#         result = self.model_backbone(x_rescaled)
#         if self.bypass:
#             stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result
#         else:
#             stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result

#         # Compute scale-consistency loss between the two streams, list ver
#         c1_correct_scale_loss = 0
#         for i in range(len(stream_1_c1_feats)):
#             c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
#         c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
        
#         c2_correct_scale_loss = 0
#         for i in range(len(stream_1_c2_feats)):
#             c2_correct_scale_loss += torch.mean(torch.abs(stream_1_c2_feats[i] - stream_2_c2_feats[i]))
#         c2_correct_scale_loss /= len(stream_1_c2_feats)  # Average over all feature maps
        
#         out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

#         if self.bypass:
#             bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
#         else:
#             bypass_correct_scale_loss = 0

#         correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

#         return stream_1_output, correct_scale_loss

class RESMAX_rand(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=9216, contrastive_loss=False,
                 bypass=False,
                 **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        # self.big_size = big_size
        self.bypass = bypass

        super(RESMAX_rand, self).__init__()

        self.s1 = nn.Sequential(
            Residual(3, 48, strides=2),
            Residual(48, 48),
            Residual(48, 96, strides=2)
        )
        
        self.abs_layer = nn.Sequential(
            AbsLikeReLU()
        )

        # C1 using optimized layer
        self.c1 = C_scoring2_optimized_debug(
            num_channels=96,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            skip=1,
            global_scale_pool=False
        )
        
        self.s2 = nn.Sequential(
            Residual(96, 128),
            Residual(128, 256)
        )
        # C2 using optimized layer
        self.c2 = C_scoring2_optimized(
            num_channels=256,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            pool_func2=nn.MaxPool2d(kernel_size=6, stride=2),
            resize_kernel_1=3,
            resize_kernel_2=1,
            skip=2,
            global_scale_pool=False
        )
        
        if self.bypass:
            self.s2b = S2b_Res()
            self.c2b_score = C_scoring2_optimized_debug(
                num_channels=1024,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                global_scale_pool=True
            )
            self.c2b_seq = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((6, 6))
            )

        self.s3 = nn.Sequential(
            Residual(256, 256),
            Residual(256, 384),
            Residual(384, 384),
            Residual(384, 256),
        )
        
        # self.global_pool = C_scoring2_optimized(
        #     num_channels=256,
        #     pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
        #     pool_func2=nn.MaxPool2d(kernel_size=6, stride=3, padding=1),
        #     resize_kernel_1=3,
        #     resize_kernel_2=1,
        #     skip=2,
        #     global_scale_pool=False
        # )
        self.global_pool = C(global_scale_pool=True)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_size, 4096),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

        self.print_param_stats()

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
        def apply(module, x):
            return [module(xi) for xi in x] if isinstance(x, list) else module(x)
        
        out = self.make_ip(x, self.ip_scale_bands)
        
        if self.training and len(out) > 4:
            mid_idx = len(out) // 2

            low_band_indices = list(range(0, mid_idx))
            high_band_indices = list(range(mid_idx + 1, len(out)))

            low_selected = random.sample(low_band_indices, k=2) if len(low_band_indices) >= 2 else low_band_indices

            high_selected = [random.choice(high_band_indices)] if high_band_indices else []

            selected_indices = low_selected + [mid_idx] + high_selected

            out = [out[i] for i in selected_indices]
            
        
        out = apply(self.s1, out)
        out = apply(self.abs_layer, out)

        out_c1 = self.c1(out)
        out = apply(self.s2, out_c1)
        out_c2 = self.c2(out)
        
        if self.bypass:
            bypass = self.s2b(out_c1)
            bypass = self.c2b_score(bypass)
            bypass = self.c2b_seq(bypass)
            bypass = bypass.reshape(bypass.size(0), -1)
        
        out = apply(self.s3, out_c2)
        out = self.global_pool(out)
        if isinstance(out, list):
            out = out[0]
        out = out.reshape(out.size(0), -1)
        
        if self.bypass:
            out = torch.cat([out, bypass], dim=1)
        
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            if self.bypass:
                return out, out_c1, out_c2, bypass
            else:
                return out, out_c1, out_c2

        return out
    
    def print_param_stats(self):
        print(f"\nParameter breakdown for {self.__class__.__name__}:\n")
        total_params = 0
        stats = []
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += n_params
            stats.append((name, n_params))

        stats.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Module':30s} | {'# Params':>10s} | {'% of Total':>10s}")
        print("-" * 60)
        for name, count in stats:
            pct = 100 * count / total_params
            print(f"{name:30s} | {count:10,d} | {pct:10.2f}%")
        print("-" * 60)
        print(f"{'Total':30s} | {total_params:10,d} | {100.00:10.2f}%\n")

"""
choose randomly in forward
"""
class CHRESMAX_V3_2_rand(nn.Module):
    """
    Example student-teacher style model with scale-consistency loss,
    using RESMAX_V2 as the backbone.

    In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
    the returned features are [0] for C1 and C2.
    """
    def __init__(self, 
                 num_classes=1000,
                 in_chans=3,
                 ip_scale_bands=1,
                 classifier_input_size=13312,
                 contrastive_loss=True,
                 bypass=False,
                 **kwargs):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.ip_scale_bands = ip_scale_bands
        self.bypass = bypass
        
        # Use the optimized backbone
        self.model_backbone = RESMAX_rand(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )

    def forward(self, x):
        """
        Creates two streams (original + random-scaled) for scale-consistency training.
        Returns:
            (output_of_stream1, correct_scale_loss)
        """
        # stream 1 (original scale)
        result = self.model_backbone(x)
        if self.bypass:
            stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result
        else:
            stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result

        # stream 2 (random scale)
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            # pad if smaller
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
        else:
            # center-crop if bigger
            center_crop = torchvision.transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)

        # forward pass on the scaled input
        result = self.model_backbone(x_rescaled)
        if self.bypass:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result
        else:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result

        # Compute scale-consistency loss between the two streams, list ver
        c1_correct_scale_loss = 0
        for i in range(len(stream_1_c1_feats)):
            c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
        c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
        
        c2_correct_scale_loss = 0
        for i in range(len(stream_1_c2_feats)):
            c2_correct_scale_loss += torch.mean(torch.abs(stream_1_c2_feats[i] - stream_2_c2_feats[i]))
        c2_correct_scale_loss /= len(stream_1_c2_feats)  # Average over all feature maps
        
        out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

        if self.bypass:
            bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
        else:
            bypass_correct_scale_loss = 0

        correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

        return stream_1_output, correct_scale_loss

"""thinner but deeper version or resmax, following resnet18"""
class RESMAX_V3(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=512, contrastive_loss=False, pyramid=False,
                 bypass=False, main_route=False,
                 c_scoring='v2',
                 **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.bypass = bypass
        self.c_scoring = c_scoring
        self.main_route = main_route
        self.classifier_input_size = classifier_input_size
        
        super(RESMAX_V3, self).__init__()

        self.s1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # C1 using optimized layer
        self.c1 = C_scoring2_optimized_debug(
            num_channels=64,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            skip=1,
            global_scale_pool=False
        )
        
        self.s2 = nn.Sequential(
            Residual1(64, 64),
            Residual1(64, 64)
        )

        # C2 using optimized layer
        self.c2 = C_scoring2_optimized(
            num_channels=64,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            pool_func2=nn.MaxPool2d(kernel_size=6, stride=2),
            resize_kernel_1=3,
            resize_kernel_2=1,
            skip=2,
            global_scale_pool=False
        )
        
        if self.bypass:
            self.s2b = S2b_Res1()
            self.c2b_seq = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        self.s3 = nn.Sequential(
            Residual1(64, 128, strides=2),
            Residual1(128, 128),
            Residual1(128, 256, strides=2),
            Residual1(256, 256),
            Residual1(256, 512, strides=2),
            Residual1(512, 512)
        )

        if self.ip_scale_bands > 4:
            self.global_pool = C_scoring2_optimized(
                num_channels=256,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=6, stride=3, padding=1),
                resize_kernel_1=3,
                resize_kernel_2=1,
                skip=2,
                global_scale_pool=False
            )
        else:
            self.global_pool = C_adp(global_scale_pool=nn.AdaptiveAvgPool2d((1, 1)))

        if self.bypass:
            self.classifier_input_size = classifier_input_size * 2

        self.fc = nn.Sequential(
            nn.Linear(self.classifier_input_size, num_classes)
        )

        self.print_param_stats()

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
        def apply(module, x):
            return [module(xi) for xi in x] if isinstance(x, list) else module(x)

        if self.main_route:
            out = self.make_ip(x, 2)
        else:
            out = self.make_ip(x, self.ip_scale_bands)
        
        out = apply(self.s1, out)
        out_c1 = self.c1(out)
        out = apply(self.s2, out_c1)
        out_c2 = self.c2(out)
        
        if self.bypass:
            bypass = self.s2b(out_c1)
            bypass = self.c2b_seq(bypass[0]) ### FIXME
            bypass = bypass.reshape(bypass.size(0), -1)
        
        out = apply(self.s3, out_c2)
        out = self.global_pool(out)
        if isinstance(out, list):
            out = out[0]
        out = out.reshape(out.size(0), -1)

        if self.bypass:
            out = torch.cat([out, bypass], dim=1)
        
        out = self.fc(out)

        if self.contrastive_loss:
            if self.bypass:
                return out, out_c1, out_c2, bypass
            else:
                return out, out_c1, out_c2

        return out
    
    def print_param_stats(self):
        print(f"\nParameter breakdown for {self.__class__.__name__}:\n")
        total_params = 0
        stats = []
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += n_params
            stats.append((name, n_params))

        stats.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Module':30s} | {'# Params':>10s} | {'% of Total':>10s}")
        print("-" * 60)
        for name, count in stats:
            pct = 100 * count / total_params
            print(f"{name:30s} | {count:10,d} | {pct:10.2f}%")
        print("-" * 60)
        print(f"{'Total':30s} | {total_params:10,d} | {100.00:10.2f}%\n")

class RESMAX_V3_1(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=512, contrastive_loss=False, pyramid=False,
                 bypass=False, main_route=False,
                 c_scoring='v2',
                 **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.bypass = bypass
        self.c_scoring = c_scoring
        self.main_route = main_route
        super(RESMAX_V3, self).__init__()

        self.s1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # C1 using optimized layer
        self.c1 = C_scoring2_optimized(
            num_channels=64,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            skip=1,
            global_scale_pool=False
        )
        
        self.s2 = nn.Sequential(
            Residual1(64, 64),
            Residual1(64, 64)
        )

        # C2 using optimized layer
        self.c2 = C_scoring2_optimized(
            num_channels=64,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            pool_func2=nn.MaxPool2d(kernel_size=6, stride=2),
            resize_kernel_1=3,
            resize_kernel_2=1,
            skip=2,
            global_scale_pool=False
        )
        
        if self.bypass:
            self.s2b = S2b_Res1()
            self.c2b_seq = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

        self.s3 = nn.Sequential(
            Residual1(64, 128, strides=2),
            Residual1(128, 128),
            Residual1(128, 256, strides=2),
            Residual1(256, 256),
            Residual1(256, 512, strides=2),
            Residual1(512, 512)
        )

        if self.ip_scale_bands > 4:
            self.global_pool = C_scoring2_optimized(
                num_channels=256,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=6, stride=3, padding=1),
                resize_kernel_1=3,
                resize_kernel_2=1,
                skip=2,
                global_scale_pool=False
            )
        else:
            self.global_pool = C_adp(global_scale_pool=nn.AdaptiveAvgPool2d((1, 1)))

        self.fc= nn.Sequential(
            nn.Linear(classifier_input_size, num_classes)
        )

        self.print_param_stats()

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

    def forward(self, x, pyramid=False):
        def apply(module, x):
            return [module(xi) for xi in x] if isinstance(x, list) else module(x)

        if self.main_route:
            out = self.make_ip(x, 2)
        else:
            out = self.make_ip(x, self.ip_scale_bands)
        
        out = apply(self.s1, out)
        out_c1 = self.c1(out)
        out = apply(self.s2, out_c1)
        out_c2 = self.c2(out)
        
        if self.bypass:
            bypass = self.s2b(out_c1)
            bypass = self.c2b_seq(bypass[0])
            bypass = bypass.reshape(bypass.size(0), -1)
        
        out = apply(self.s3, out_c2)
        out = self.global_pool(out)
        if isinstance(out, list):
            out = out[0]
        out = out.reshape(out.size(0), -1)

        if self.bypass:
            out = torch.cat([out, bypass], dim=1)
        
        out = self.fc(out)

        if self.contrastive_loss:
            if self.bypass:
                return out, out_c1, out_c2, bypass
            else:
                return out, out_c1, out_c2

        return out
    
    def print_param_stats(self):
        print(f"\nParameter breakdown for {self.__class__.__name__}:\n")
        total_params = 0
        stats = []
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += n_params
            stats.append((name, n_params))

        stats.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Module':30s} | {'# Params':>10s} | {'% of Total':>10s}")
        print("-" * 60)
        for name, count in stats:
            pct = 100 * count / total_params
            print(f"{name:30s} | {count:10,d} | {pct:10.2f}%")
        print("-" * 60)
        print(f"{'Total':30s} | {total_params:10,d} | {100.00:10.2f}%\n")

"""
In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
the returned features are [0] for C1 and C2.

We get good result under 0.1 lambda, bypass True
"""
class CHRESMAX_V3(nn.Module):
    """
    Example student-teacher style model with scale-consistency loss,
    using RESMAX_V2 as the backbone.

    In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
    the returned features are [0] for C1 and C2.
    """
    def __init__(self, 
                 num_classes=1000,
                 in_chans=3,
                 ip_scale_bands=1,
                 classifier_input_size=13312,
                 contrastive_loss=True,
                 bypass=False,
                 **kwargs):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.ip_scale_bands = ip_scale_bands
        self.bypass = bypass
        
        # Use the optimized backbone
        self.model_backbone = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )

    def forward(self, x):
        """
        Creates two streams (original + random-scaled) for scale-consistency training.
        Returns:
            (output_of_stream1, correct_scale_loss)
        """
        # stream 1 (original scale)
        result = self.model_backbone(x)
        if self.bypass:
            stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result
        else:
            stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result

        # stream 2 (random scale)
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            # pad if smaller
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
        else:
            # center-crop if bigger
            center_crop = torchvision.transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)

        # forward pass on the scaled input
        result = self.model_backbone(x_rescaled)
        if self.bypass:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result
        else:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result

        # Compute scale-consistency loss between the two streams, list ver
        c1_correct_scale_loss = 0
        for i in range(len(stream_1_c1_feats)):
            c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
        c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
        
        c2_correct_scale_loss = 0
        for i in range(len(stream_1_c2_feats)):
            c2_correct_scale_loss += torch.mean(torch.abs(stream_1_c2_feats[i] - stream_2_c2_feats[i]))
        c2_correct_scale_loss /= len(stream_1_c2_feats)  # Average over all feature maps
        
        out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

        if self.bypass:
            bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
        else:
            bypass_correct_scale_loss = 0

        correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

        return stream_1_output, correct_scale_loss

class RESMAX_V2_bypass_only(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False,
                 bypass=False, main_route=False,c_debug=False,
                 c_scoring='v2',
                 **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        self.bypass = bypass
        self.c_scoring = c_scoring
        self.main_route = main_route
        super(RESMAX_V2_bypass_only, self).__init__()

        self.s1 = S1_Res()

        # C1 using optimized layer
        if c_debug:
            self.c1 = C_scoring2_optimized_debug(
                num_channels=96,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                skip=1,
                global_scale_pool=False
            )
        else:
            self.c1 = C_scoring2_optimized(
                num_channels=96,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                skip=1,
                global_scale_pool=False
            )
        
        if self.bypass:
            self.s2b = S2b_Res()
            self.c2b_seq = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_size, 4096),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )


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

    def forward(self, x, pyramid=False):
        if self.main_route:
            out = self.make_ip(x, 2)
        else:
            out = self.make_ip(x, self.ip_scale_bands)
        
        out = self.s1(out)
        out_c1 = self.c1(out)
        
        if self.bypass:
            bypass = self.s2b(out_c1)
            bypass = self.c2b_seq(bypass[0])
            bypass = bypass.reshape(bypass.size(0), -1)
        
        out = self.fc(bypass)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            if self.bypass:
                return out, out_c1, bypass
            else:
                return out, out_c1

        return out
    
    

class RESMAX_V2_bypass_only_c2b(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False,
                 bypass=False, main_route=False,c_debug=False,
                 c_scoring='v2',
                 **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        self.bypass = bypass
        self.c_scoring = c_scoring
        self.main_route = main_route
        super(RESMAX_V2_bypass_only_c2b, self).__init__()

        self.s1 = S1_Res()

        # C1 using optimized layer
        if c_debug:
            self.c1 = C_scoring2_optimized_debug(
                num_channels=96,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                skip=1,
                global_scale_pool=False
            )
        else:
            self.c1 = C_scoring2_optimized(
                num_channels=96,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                skip=1,
                global_scale_pool=False
            )
        
        if self.bypass:
            self.s2b = S2b_Res()
            # make channels smaller for c2b computing
            self.c2b_seq = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            # after c2b_seq the ref should be (B, 256, 4, 4)
            # self.c2b = C_scoring2_optimized(
            #     num_channels=256,
            #     pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            #     pool_func2=nn.MaxPool2d(kernel_size=3, stride=2),
            #     global_scale_pool=True
            # )
            self.c2b = C(global_scale_pool=True)


        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_size, 1024),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, num_classes)
        )


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

    def forward(self, x, pyramid=False):
        if self.main_route:
            out = self.make_ip(x, 2)
        else:
            out = self.make_ip(x, self.ip_scale_bands)
        
        out = self.s1(out)
        out_c1 = self.c1(out)
        
        if self.bypass:
            bypass = self.s2b(out_c1)
            # print(f"s2b {len(bypass)} outputs: ", [b.shape for b in bypass])
            for i in range(len(bypass)):
                bypass[i] = self.c2b_seq(bypass[i])
            bypass = self.c2b(bypass)
            bypass = bypass.reshape(bypass.size(0), -1)
        
        out = self.fc(bypass)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            if self.bypass:
                return out, out_c1, bypass
            else:
                return out, out_c1

        return out
    

class S2b_Res_tiny(nn.Module):
    def __init__(self):
        super(S2b_Res_tiny, self).__init__()
        # Each Residual block has 2 3x3 convs, so we need fewer blocks
        # to achieve same receptive field
        # 4x4 -> 1 residual block (2 3x3 convs)
        # 8x8 -> 2 residual blocks (4 3x3 convs)
        # 12x12 -> 3 residual blocks (6 3x3 convs)
        # 16x16 -> 4 residual blocks (8 3x3 convs)
        self.kernel_to_blocks = {4: 1, 8: 2, 12: 3, 16: 4}
        
        self.s2b_seqs = nn.ModuleList()
        for kernel_size, num_blocks in self.kernel_to_blocks.items():
            blocks = []
            # Initial projection to higher dimensions
            blocks.append(nn.Sequential(
                nn.Conv2d(4, 96, kernel_size=1),  # 1x1 conv for dimension matching
                nn.BatchNorm2d(96),
                nn.ReLU(True)
            ))
            # Stack of residual blocks
            for _ in range(num_blocks):
                blocks.append(Residual(96, 96))
                
            self.s2b_seqs.append(nn.Sequential(*blocks))

    def forward(self, x_pyramid):
        bypass = [torch.cat([seq(out) for seq in self.s2b_seqs], dim=1) for out in x_pyramid]
        return bypass
    
    
class RESMAX_V2_bypass_only_tiny(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=True, pyramid=False,
                 bypass=False, main_route=False,c_debug=False,
                 c_scoring='v2',
                 **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        self.bypass = bypass
        self.c_scoring = c_scoring
        self.main_route = main_route
        super(RESMAX_V2_bypass_only_tiny, self).__init__()

        self.s1 = nn.Sequential(
            Residual(3, 4, strides=2),
        )
        
        # self.abs_layer = nn.Sequential(
        #     AbsLikeReLU()
        # )

        # C1 using optimized layer
        if c_debug:
            self.c1 = C_scoring2_optimized_debug(
                num_channels=4,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                skip=1,
                global_scale_pool=False
            )
        else:
            self.c1 = C_scoring2_optimized(
                num_channels=4,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                skip=1,
                global_scale_pool=False
            )
        
        if self.bypass:
            self.s2b = S2b_Res_tiny()
            # self.c2b_score = C_scoring2_optimized_debug(
            #     num_channels=1024,
            #     pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            #     pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            #     global_scale_pool=True
            # )
            # self.c2b_seq = nn.Sequential(
            #     nn.Conv2d(1024, 256, kernel_size=1),
            #     nn.BatchNorm2d(256),
            #     nn.ReLU(inplace=True),
            #     nn.AdaptiveAvgPool2d((6, 6))
            # )
            self.c2b_seq = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(384, 96, kernel_size=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True)
            )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_size, 256),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, num_classes)
        )


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
        def apply(module, x):
            return [module(xi) for xi in x] if isinstance(x, list) else module(x)
        
        out = self.make_ip(x, self.ip_scale_bands)

        out = apply(self.s1, out)
        # out = apply(self.abs_layer, out)

        out_c1 = self.c1(out)

        
        bypass = self.s2b(out_c1)
        # bypass = self.c2b_score(bypass)
        bypass = self.c2b_seq(bypass[0])
        bypass = bypass.reshape(bypass.size(0), -1)
        
        out = self.fc(bypass)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1, bypass

        return out


class RESMAX_abs_bypass_only(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False,
                 bypass=False, main_route=False,c_debug=False,
                 c_scoring='v2',
                 **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        self.bypass = bypass
        self.c_scoring = c_scoring
        self.main_route = main_route
        super(RESMAX_abs_bypass_only, self).__init__()

        self.s1 = nn.Sequential(
            Residual(3, 48, strides=2),
            Residual(48, 48),
            Residual(48, 96, strides=2)
        )
        
        self.abs_layer = nn.Sequential(
            AbsLikeReLU()
        )

        # C1 using optimized layer
        if c_debug:
            self.c1 = C_scoring2_optimized_debug(
                num_channels=96,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                skip=1,
                global_scale_pool=False
            )
        else:
            self.c1 = C_scoring2_optimized(
                num_channels=96,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                skip=1,
                global_scale_pool=False
            )
        
        if self.bypass:
            self.s2b = S2b_Res()
            self.c2b_score = C_scoring2_optimized_debug(
                num_channels=1024,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                global_scale_pool=True
            )
            self.c2b_seq = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((6, 6))
            )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_size, 4096),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

        self.print_param_stats()

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
        def apply(module, x):
            return [module(xi) for xi in x] if isinstance(x, list) else module(x)
        
        out = self.make_ip(x, self.ip_scale_bands)

        out = apply(self.s1, out)
        out = apply(self.abs_layer, out)

        out_c1 = self.c1(out)

        
        if self.bypass:
            bypass = self.s2b(out_c1)
            bypass = self.c2b_score(bypass)
            bypass = self.c2b_seq(bypass)
            bypass = bypass.reshape(bypass.size(0), -1)
        
        out = self.fc(bypass)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            if self.bypass:
                return out, out_c1, bypass
            else:
                return out, out_c1

        return out
    
    def print_param_stats(self):
        print(f"\nParameter breakdown for {self.__class__.__name__}:\n")
        total_params = 0
        stats = []
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += n_params
            stats.append((name, n_params))

        stats.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Module':30s} | {'# Params':>10s} | {'% of Total':>10s}")
        print("-" * 60)
        for name, count in stats:
            pct = 100 * count / total_params
            print(f"{name:30s} | {count:10,d} | {pct:10.2f}%")
        print("-" * 60)
        print(f"{'Total':30s} | {total_params:10,d} | {100.00:10.2f}%\n")


class CH_2_streams_adjacent_scales(nn.Module):
    """
    Contrastive learning model with adjacent scale pairs instead of original vs random scale.
    Performs contrastive alignment between adjacent scale pairs like (0.49, 0.59), (0.59, 0.707), etc.
    """
    def __init__(self, 
                 contrastive_loss=True,
                 bypass=True,
                 model_backbone=None,
                 bypass_only_model_bool=False,
                 **kwargs):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.bypass = bypass
        self.model_backbone = model_backbone
        self.bypass_only_model_bool = bypass_only_model_bool
        
        self.model_backbone.contrastive_loss = contrastive_loss
        self.num_classes = self.model_backbone.num_classes
        
        # Define scale factor list and adjacent pairs
        self.scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        self.adjacent_pairs = [(self.scale_factor_list[i], self.scale_factor_list[i+1]) 
                              for i in range(len(self.scale_factor_list)-1)]
        
        self.print_param_stats(model_backbone)
        
    def _apply_scale_and_resize(self, x, scale_factor):
        """Apply scale factor and resize back to original dimensions"""
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_scaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            # pad if smaller
            x_scaled = pad_to_size(x_scaled, (img_hw, img_hw))
        else:
            # center-crop if bigger
            center_crop = torchvision.transforms.CenterCrop(img_hw)
            x_scaled = center_crop(x_scaled)
            
        return x_scaled
        
    def forward(self, x):
        """
        Creates two streams with adjacent scale pairs for scale-consistency training.
        During training: uses adjacent scaled pairs for contrastive learning
        During validation: uses original unscaled image
        Returns:
            (output, correct_scale_loss)
        """
        # During validation/evaluation, use original unscaled image
        if not self.training:
            result = self.model_backbone(x)
            correct_scale_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            
            if self.bypass_only_model_bool:
                if self.bypass:
                    output, _, _ = result
                else:
                    output, _ = result
            else:
                if self.bypass:
                    output, _, _, _ = result
                else:
                    output, _, _ = result
            
            return output, correct_scale_loss
        
        # Training mode: use adjacent scale pairs for contrastive learning
        # Randomly select an adjacent pair
        scale_1, scale_2 = random.choice(self.adjacent_pairs)
        
        # Create two scaled versions
        x_scale_1 = self._apply_scale_and_resize(x, scale_1)
        x_scale_2 = self._apply_scale_and_resize(x, scale_2)
        
        # Forward pass on first scaled input
        result_1 = self.model_backbone(x_scale_1)
        if self.bypass_only_model_bool:
            if self.bypass:
                stream_1_output, stream_1_c1_feats, stream_1_bypass = result_1
            else:
                stream_1_output, stream_1_c1_feats = result_1
        else:
            if self.bypass:
                stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result_1
            else:
                stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result_1

        # Forward pass on second scaled input
        result_2 = self.model_backbone(x_scale_2)
        if self.bypass_only_model_bool:
            if self.bypass:
                stream_2_output, stream_2_c1_feats, stream_2_bypass = result_2
            else:
                stream_2_output, stream_2_c1_feats = result_2
        else:
            if self.bypass:
                stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result_2
            else:
                stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result_2

        # Compute scale-consistency loss between the two adjacent scale streams
        c1_correct_scale_loss = 0
        for i in range(len(stream_1_c1_feats)):
            c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
        c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
        
        # Add C2 loss calculation for non-bypass-only models
        if not self.bypass_only_model_bool:
            c2_correct_scale_loss = 0
            for i in range(len(stream_1_c2_feats)):
                c2_correct_scale_loss += torch.mean(torch.abs(stream_1_c2_feats[i] - stream_2_c2_feats[i]))
            c2_correct_scale_loss /= len(stream_1_c2_feats)  # Average over all feature maps
        else:
            c2_correct_scale_loss = 0
        
        out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

        if self.bypass:
            bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
        else:
            bypass_correct_scale_loss = 0

        correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

        # Return the output from the first stream (could also average both streams)
        return stream_1_output, correct_scale_loss

    def print_param_stats(self, backbone):
        print(f"\nParameter breakdown for {backbone.__class__.__name__} (Adjacent Scales):\n")
        total_params = 0
        stats = []
        for name, module in backbone.named_children():
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += n_params
            stats.append((name, n_params))

        stats.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Module':30s} | {'# Params':>10s} | {'% of Total':>10s}")
        print("-" * 60)
        for name, count in stats:
            pct = 100 * count / total_params
            print(f"{name:30s} | {count:10,d} | {pct:10.2f}%")
        print("-" * 60)
        print(f"{'Total':30s} | {total_params:10,d} | {100.00:10.2f}%\n")


# class RESMAX_bypass_only_o(nn.Module):
#     def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
#                  ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False,
#                  bypass=False, main_route=False,c_debug=False,
#                  c_scoring='v2',
#                  **kwargs):
#         self.num_classes = num_classes
#         self.in_chans = in_chans
#         self.contrastive_loss = contrastive_loss
#         self.ip_scale_bands = ip_scale_bands
#         self.pyramid = pyramid
#         self.big_size = big_size
#         self.small_size = small_size
#         self.bypass = bypass
#         self.c_scoring = c_scoring
#         self.main_route = main_route
#         super(RESMAX_bypass_only_o, self).__init__()

#         self.s1 = nn.Sequential(
#             Residual(3, 48, strides=2),
#             Residual(48, 48),
#             Residual(48, 96, strides=2)
#         )
        
#         # self.abs_layer = nn.Sequential(
#         #     AbsLikeReLU()
#         # )

#         # C1 using optimized layer
#         if c_debug:
#             self.c1 = C_scoring2_optimized_debug(
#                 num_channels=96,
#                 pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
#                 pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
#                 skip=1,
#                 global_scale_pool=False
#             )
#         else:
#             self.c1 = C_scoring2_optimized(
#                 num_channels=96,
#                 pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
#                 pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
#                 skip=1,
#                 global_scale_pool=False
#             )
        
#         if self.bypass:
#             self.s2b = S2b_Res()
#             self.c2b_score = C_scoring2_optimized_debug(
#                 num_channels=1024,
#                 pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
#                 pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
#                 global_scale_pool=True
#             )
#             self.c2b_seq = nn.Sequential(
#                 nn.Conv2d(1024, 256, kernel_size=1),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(inplace=True),
#                 nn.AdaptiveAvgPool2d((6, 6))
#             )
#             # self.c2b_seq = nn.Sequential(
#             #     nn.MaxPool2d(kernel_size=3, stride=2),
#             #     nn.MaxPool2d(kernel_size=3, stride=2),
#             #     nn.Conv2d(1024, 256, kernel_size=1),
#             #     nn.BatchNorm2d(256),
#             #     nn.ReLU(inplace=True)
#             # )

#         self.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(classifier_input_size, 4096),
#             nn.ReLU()
#         )
#         self.fc1 = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU()
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(4096, num_classes)
#         )

#         self.print_param_stats()

#     def make_ip(self, x, num_scale_bands):
#         """
#         Build an image pyramid.
#         num_scale_bands = number of images in the pyramid - 1
#         """
#         base_image_size = int(x.shape[-1])
#         scale_factor = 4  # exponent factor for scaling
#         image_scales = get_ip_scales(num_scale_bands, base_image_size, scale_factor)
        
#         if len(image_scales) > 1:
#             image_pyramid = []
#             for i_s in image_scales:
#                 i_s = int(i_s)
#                 interp_img = F.interpolate(x, size=(i_s, i_s), mode='bilinear', align_corners=False)
#                 image_pyramid.append(interp_img)
#             return image_pyramid
#         else:
#             return [x]
    
#     def forward(self, x):
#         def apply(module, x):
#             return [module(xi) for xi in x] if isinstance(x, list) else module(x)
        
#         out = self.make_ip(x, self.ip_scale_bands)

#         out = apply(self.s1, out)
#         # out = apply(self.abs_layer, out)

#         out_c1 = self.c1(out)

#         if self.bypass:
#             bypass = self.s2b(out_c1)
#             bypass = self.c2b_score(bypass)
#             bypass = self.c2b_seq(bypass)
#             bypass = bypass.reshape(bypass.size(0), -1)
        
#         out = self.fc(bypass)
#         out = self.fc1(out)
#         out = self.fc2(out)

#         if self.contrastive_loss:
#             if self.bypass:
#                 return out, out_c1, bypass
#             else:
#                 return out, out_c1

#         return out
    
#     def print_param_stats(self):
#         print(f"\nParameter breakdown for {self.__class__.__name__}:\n")
#         total_params = 0
#         stats = []
#         for name, module in self.named_children():
#             n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
#             total_params += n_params
#             stats.append((name, n_params))

#         stats.sort(key=lambda x: x[1], reverse=True)
#         print(f"{'Module':30s} | {'# Params':>10s} | {'% of Total':>10s}")
#         print("-" * 60)
#         for name, count in stats:
#             pct = 100 * count / total_params
#             print(f"{name:30s} | {count:10,d} | {pct:10.2f}%")
#         print("-" * 60)
#         print(f"{'Total':30s} | {total_params:10,d} | {100.00:10.2f}%\n")

        
# class CHRESMAX_V3_bypass_only(nn.Module):
#     """
#     Example student-teacher style model with scale-consistency loss,
#     using RESMAX_V2 as the backbone.

#     In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
#     the returned features are [0] for C1 and C2.
#     """
#     def __init__(self, 
#                  num_classes=1000,
#                  in_chans=3,
#                  ip_scale_bands=1,
#                  classifier_input_size=13312,
#                  contrastive_loss=True,
#                  bypass=False,
#                  c_debug=False,
#                  **kwargs):
#         super().__init__()
#         self.contrastive_loss = contrastive_loss
#         self.num_classes = num_classes
#         self.in_chans = in_chans
#         self.ip_scale_bands = ip_scale_bands
#         self.bypass = bypass
        
#         # Use the optimized backbone
#         self.model_backbone = RESMAX_V2_bypass_only(
#             num_classes=num_classes,
#             in_chans=in_chans,
#             ip_scale_bands=self.ip_scale_bands,
#             classifier_input_size=classifier_input_size,
#             contrastive_loss=self.contrastive_loss,
#             bypass=bypass,
#             c_debug=c_debug,
#         )

#     def forward(self, x):
#         """
#         Creates two streams (original + random-scaled) for scale-consistency training.
#         During training: returns stream_2_output (scaled/augmented) for backpropagation
#         During evaluation: returns stream_1_output (original) for clean evaluation
#         Returns:
#             (output, correct_scale_loss)
#         """
#         # stream 1 (original scale)
#         result = self.model_backbone(x)
#         if self.bypass:
#             print("Stream 1: Bypass enabled, returning bypass features.")
#             stream_1_output, stream_1_c1_feats, stream_1_bypass = result
#         else:
#             stream_1_output, stream_1_c1_feats = result

#         # If in evaluation mode, return stream 1 output without scale augmentation
#         if not self.training:
#             print("Evaluation mode: returning original stream output without scale augmentation.")
#             correct_scale_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
#             return stream_1_output, correct_scale_loss

#         # stream 2 (random scale) - only during training
#         scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
#         scale_factor = random.choice(scale_factor_list)
#         img_hw = x.shape[-1]
#         new_hw = int(img_hw * scale_factor)
#         x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

#         if new_hw <= img_hw:
#             # pad if smaller
#             x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
#         else:
#             # center-crop if bigger
#             center_crop = torchvision.transforms.CenterCrop(img_hw)
#             x_rescaled = center_crop(x_rescaled)

#         # forward pass on the scaled input
#         result = self.model_backbone(x_rescaled)
#         if self.bypass:
#             stream_2_output, stream_2_c1_feats, stream_2_bypass = result
#         else:
#             stream_2_output, stream_2_c1_feats = result

#         # Compute scale-consistency loss between the two streams, list ver
#         c1_correct_scale_loss = 0
#         for i in range(len(stream_1_c1_feats)):
#             c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
#         c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
        
#         out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

#         if self.bypass:
#             bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
#         else:
#             bypass_correct_scale_loss = 0

#         correct_scale_loss = c1_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

#         # Return stream 2 output (scaled/augmented) for training to learn scale invariance
#         return stream_2_output, correct_scale_loss

#         # ======================== ORIGINAL CODE (COMMENTED OUT) ========================
#         # # ORIGINAL: Always returned stream_1_output regardless of training/eval mode
#         # return stream_1_output, correct_scale_loss
#         # ============================================================================

# class CHRESMAX_V3_bypass_only_tiny(nn.Module):
#     """
#     Example student-teacher style model with scale-consistency loss,
#     using RESMAX_V2 as the backbone.

#     In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
#     the returned features are [0] for C1 and C2.
#     """
#     def __init__(self, 
#                  num_classes=1000,
#                  in_chans=3,
#                  ip_scale_bands=1,
#                  classifier_input_size=13312,
#                  contrastive_loss=True,
#                  bypass=False,
#                  c_debug=False,
#                  **kwargs):
#         super().__init__()
#         self.contrastive_loss = contrastive_loss
#         self.num_classes = num_classes
#         self.in_chans = in_chans
#         self.ip_scale_bands = ip_scale_bands
#         self.bypass = bypass
        
#         # Use the optimized backbone
#         self.model_backbone = RESMAX_V2_bypass_only_tiny(
#             num_classes=num_classes,
#             in_chans=in_chans,
#             ip_scale_bands=self.ip_scale_bands,
#             classifier_input_size=classifier_input_size,
#             contrastive_loss=self.contrastive_loss,
#             bypass=bypass,
#             c_debug=c_debug,
#         )

#     def forward(self, x):
#         """
#         Creates two streams (original + random-scaled) for scale-consistency training.
#         Returns:
#             (output_of_stream1, correct_scale_loss)
#         """
#         # stream 1 (original scale)
#         result = self.model_backbone(x)
#         if self.bypass:
#             stream_1_output, stream_1_c1_feats, stream_1_bypass = result
#         else:
#             stream_1_output, stream_1_c1_feats = result

#         # stream 2 (random scale)
#         scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
#         scale_factor = random.choice(scale_factor_list)
#         img_hw = x.shape[-1]
#         new_hw = int(img_hw * scale_factor)
#         x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

#         if new_hw <= img_hw:
#             # pad if smaller
#             x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
#         else:
#             # center-crop if bigger
#             center_crop = torchvision.transforms.CenterCrop(img_hw)
#             x_rescaled = center_crop(x_rescaled)

#         # forward pass on the scaled input
#         result = self.model_backbone(x_rescaled)
#         if self.bypass:
#             stream_2_output, stream_2_c1_feats, stream_2_bypass = result
#         else:
#             stream_2_output, stream_2_c1_feats = result

#         # Compute scale-consistency loss between the two streams, list ver
#         c1_correct_scale_loss = 0
#         for i in range(len(stream_1_c1_feats)):
#             c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
#         c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
        
#         out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

#         if self.bypass:
#             bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
#         else:
#             bypass_correct_scale_loss = 0

#         correct_scale_loss = c1_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

#         return stream_1_output, correct_scale_loss

class CH_2_streams_training_eval_sep(nn.Module):
    """
    Generic 2-stream contrastive learning model with bypass architecture.
    Uses configurable backbone for different model variants.
    During training: returns stream_2_output (scaled/augmented) for backpropagation
    During evaluation: returns stream_1_output (original) for clean evaluation
    """
    def __init__(self, 
                 contrastive_loss=True,
                 bypass=True,
                 model_backbone=None,
                 bypass_only_model_bool=False,
                 stream_1_bool=False,
                 **kwargs):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.bypass = bypass
        self.model_backbone = model_backbone
        self.bypass_only_model_bool = bypass_only_model_bool
        
        self.model_backbone.contrastive_loss = contrastive_loss
        self.num_classes = self.model_backbone.num_classes
        self.stream_1_bool = stream_1_bool
        
        self.print_param_stats(model_backbone)
        
    def forward(self, x):
        """
        Creates two streams (original + random-scaled) for scale-consistency training.

        Returns:
            (output, correct_scale_loss)
        """
        # stream 1 (original scale)
        result = self.model_backbone(x)
        if self.bypass_only_model_bool:
            if self.bypass:
                stream_1_output, stream_1_c1_feats, stream_1_bypass = result
            else:
                stream_1_output, stream_1_c1_feats = result
        else:
            if self.bypass:
                stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result
            else:
                stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result

        # If in evaluation mode, return stream 1 output without scale augmentation
        # or when set stream_1_bool to True
        if not self.training or self.stream_1_bool:
            # debugging prints, do not remove
            # print("self.training is ", self.training, "stream_1_bool is ", self.stream_1_bool)
            # print("HERRRRRE, Korean exp goes to right place")
            correct_scale_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            return stream_1_output, correct_scale_loss

        # stream 2 (random scale) - only during training
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            # pad if smaller
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
        else:
            # center-crop if bigger
            center_crop = torchvision.transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)

        # forward pass on the scaled input
        result = self.model_backbone(x_rescaled)
        if self.bypass_only_model_bool:
            if self.bypass:
                stream_2_output, stream_2_c1_feats, stream_2_bypass = result
            else:
                stream_2_output, stream_2_c1_feats = result
        else:
            if self.bypass:
                stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result
            else:
                stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result

        # Compute scale-consistency loss between the two streams, list ver
        c1_correct_scale_loss = 0
        for i in range(len(stream_1_c1_feats)):
            c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
        c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
        
        # Add C2 loss calculation for non-bypass-only models
        if not self.bypass_only_model_bool:
            c2_correct_scale_loss = 0
            for i in range(len(stream_1_c2_feats)):
                c2_correct_scale_loss += torch.mean(torch.abs(stream_1_c2_feats[i] - stream_2_c2_feats[i]))
            c2_correct_scale_loss /= len(stream_1_c2_feats)  # Average over all feature maps
        else:
            c2_correct_scale_loss = 0
        
        out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

        if self.bypass:
            bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
        else:
            bypass_correct_scale_loss = 0

        correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

        # Return stream 2 output (scaled/augmented) for training to learn scale invariance
        return stream_2_output, correct_scale_loss

    def print_param_stats(self, backbone):
        print(f"\nParameter breakdown for {backbone.__class__.__name__}:\n")
        total_params = 0
        stats = []
        for name, module in backbone.named_children():
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += n_params
            stats.append((name, n_params))

        stats.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Module':30s} | {'# Params':>10s} | {'% of Total':>10s}")
        print("-" * 60)
        for name, count in stats:
            print(f"{name:30s} | {count:10,d} | {count/total_params*100:9.1f}%")
        print(f"{'Total':30s} | {total_params:10,d} | {100:9.1f}%")
        
        
class CH_2_streams(nn.Module):
    """
    Example student-teacher style model with scale-consistency loss,
    using RESMAX_V2 as the backbone.

    In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
    the returned features are [0] for C1 and C2.
    """
    def __init__(self, 
                 contrastive_loss=True,
                 bypass=False,
                 model_backbone=None,
                 bypass_only_model_bool=False,
                 **kwargs):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.bypass = bypass
        self.model_backbone = model_backbone
        self.bypass_only_model_bool = bypass_only_model_bool
        self.pad_color_background = None
        
        self.model_backbone.contrastive_loss = contrastive_loss
        self.num_classes = self.model_backbone.num_classes
        
        self.print_param_stats(model_backbone)
        
    def forward(self, x):
        """
        Creates two streams (original + random-scaled) for scale-consistency training.
        Returns:
            (output_of_stream1, correct_scale_loss)
        """
        # stream 1 (original scale)
        result = self.model_backbone(x)
        if self.bypass_only_model_bool:
            if self.bypass:
                stream_1_output, stream_1_c1_feats, stream_1_bypass = result
            else:
                stream_1_output, stream_1_c1_feats = result
        else:
            if self.bypass:
                stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result
            else:
                stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result 
            
        

        # stream 2 (random scale)
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            # pad if smaller
            if self.pad_color_background == 'blue':
                x_rescaled = pad_to_size_blue(x_rescaled, (img_hw, img_hw))
            elif self.pad_color_background == 'gray':
                x_rescaled = pad_to_size_gray(x_rescaled, (img_hw, img_hw))
            elif self.pad_color_background == 'noise':
                x_rescaled = pad_to_size_noise(x_rescaled, (img_hw, img_hw))
            else:
                x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
        else:
            # center-crop if bigger
            center_crop = torchvision.transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)

        # forward pass on the scaled input
        result = self.model_backbone(x_rescaled)
        
        if self.bypass_only_model_bool:
            if self.bypass:
                stream_2_output, stream_2_c1_feats, stream_2_bypass = result
            else:
                stream_2_output, stream_2_c1_feats = result
        else:
            if self.bypass:
                stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result
            else:
                stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result

        # Compute scale-consistency loss between the two streams, list ver
        c1_correct_scale_loss = 0
        for i in range(len(stream_1_c1_feats)):
            c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
        c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
        
        # Add C2 loss calculation for non-bypass-only models
        if not self.bypass_only_model_bool:
            c2_correct_scale_loss = 0
            for i in range(len(stream_1_c2_feats)):
                c2_correct_scale_loss += torch.mean(torch.abs(stream_1_c2_feats[i] - stream_2_c2_feats[i]))
            c2_correct_scale_loss /= len(stream_1_c2_feats)  # Average over all feature maps
        else:
            c2_correct_scale_loss = 0
        
        out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

        if self.bypass:
            bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
        else:
            bypass_correct_scale_loss = 0

        correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

        return stream_1_output, correct_scale_loss

    def print_param_stats(self, backbone):
        print(f"\nParameter breakdown for {backbone.__class__.__name__}:\n")
        total_params = 0
        stats = []
        for name, module in backbone.named_children():
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += n_params
            stats.append((name, n_params))

        stats.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Module':30s} | {'# Params':>10s} | {'% of Total':>10s}")
        print("-" * 60)
        for name, count in stats:
            pct = 100 * count / total_params
            print(f"{name:30s} | {count:10,d} | {pct:10.2f}%")
        print("-" * 60)
        print(f"{'Total':30s} | {total_params:10,d} | {100.00:10.2f}%\n")

# Wrong implementation messes streams, DO NOT USE

# class CHRESMAX_V3_bypass_only_1(nn.Module):
#     """
#     Example student-teacher style model with scale-consistency loss,
#     using RESMAX_V2 as the backbone.

#     In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
#     the returned features are [0] for C1 and C2.
    
#     change stream 2 list, add clamp, make active only when training
#     """
#     def __init__(self, 
#                  num_classes=1000,
#                  in_chans=3,
#                  ip_scale_bands=1,
#                  classifier_input_size=13312,
#                  contrastive_loss=True,
#                  bypass=False,
#                  c_debug=False,
#                  **kwargs):
#         super().__init__()
#         self.contrastive_loss = contrastive_loss
#         self.num_classes = num_classes
#         self.in_chans = in_chans
#         self.ip_scale_bands = ip_scale_bands
#         self.bypass = bypass
        
#         # Use the optimized backbone
#         self.model_backbone = RESMAX_V2_bypass_only(
#             num_classes=num_classes,
#             in_chans=in_chans,
#             ip_scale_bands=self.ip_scale_bands,
#             classifier_input_size=classifier_input_size,
#             contrastive_loss=self.contrastive_loss,
#             bypass=bypass,
#             c_debug=c_debug,
#         )

#     def forward(self, x):
#         """
#         Creates two streams (original + random-scaled) for scale-consistency training.
#         Returns:
#             (output_of_stream1, correct_scale_loss)
#         """
#         # stream 1 (original scale)
#         result = self.model_backbone(x)
#         if self.bypass:
#             stream_1_output, stream_1_c1_feats, stream_1_bypass = result
#         else:
#             stream_1_output, stream_1_c1_feats = result
        
#         correct_scale_loss = 0.
        
#         if self.training:
#             # stream 2 (random scale)
#             # scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
#             scale_factor_list = [0.707, 0.841, 1, 1.189, 1.414]
#             scale_factor = random.choice(scale_factor_list)
#             img_hw = x.shape[-1]
#             new_hw = int(img_hw * scale_factor)
#             x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False).clamp(min=0, max=1)

#             if new_hw <= img_hw:
#                 # pad if smaller
#                 x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
#             else:
#                 # center-crop if bigger
#                 center_crop = torchvision.transforms.CenterCrop(img_hw)
#                 x_rescaled = center_crop(x_rescaled)

#             # forward pass on the scaled input
#             result = self.model_backbone(x_rescaled)
#             if self.bypass:
#                 stream_2_output, stream_2_c1_feats, stream_2_bypass = result
#             else:
#                 stream_2_output, stream_2_c1_feats = result

#             # Compute scale-consistency loss between the two streams, list ver
#             c1_correct_scale_loss = 0
#             for i in range(len(stream_1_c1_feats)):
#                 c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
#             c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
            
#             out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

#             if self.bypass:
#                 bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
#             else:
#                 bypass_correct_scale_loss = 0

#             correct_scale_loss = c1_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss
            
#             return stream_1_output, correct_scale_loss

#         return stream_1_output, correct_scale_loss
    
# class CHRESMAX_V3_bypass_only_o(nn.Module):
#     """
#     Example student-teacher style model with scale-consistency loss,
#     using RESMAX_V2 as the backbone.

#     In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
#     the returned features are [0] for C1 and C2.
    
#     change stream 2 list, add clamp, make active only when training
#     only use c2b features for scale-consistency loss
#     """
#     def __init__(self, 
#                  num_classes=1000,
#                  in_chans=3,
#                  ip_scale_bands=1,
#                  classifier_input_size=13312,
#                  contrastive_loss=True,
#                  bypass=False,
#                  c_debug=False,
#                  **kwargs):
#         super().__init__()
#         self.contrastive_loss = contrastive_loss
#         self.num_classes = num_classes
#         self.in_chans = in_chans
#         self.ip_scale_bands = ip_scale_bands
#         self.bypass = bypass
        
#         # Use the optimized backbone
#         self.model_backbone = RESMAX_bypass_only_o(
#             num_classes=num_classes,
#             in_chans=in_chans,
#             ip_scale_bands=self.ip_scale_bands,
#             classifier_input_size=classifier_input_size,
#             contrastive_loss=self.contrastive_loss,
#             bypass=bypass,
#             c_debug=c_debug,
#         )

#     def forward(self, x):
#         """
#         Creates two streams (original + random-scaled) for scale-consistency training.
#         Returns:
#             (output_of_stream1, correct_scale_loss)
#         """
#         # stream 1 (original scale)
#         result = self.model_backbone(x)
#         if self.bypass:
#             stream_1_output, stream_1_c1_feats, stream_1_bypass = result
#         else:
#             stream_1_output, stream_1_c1_feats = result
        
#         correct_scale_loss = 0.
        
#         if self.training:
#             # stream 2 (random scale)
#             # scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
#             scale_factor_list = [0.707, 0.841, 1, 1.189, 1.414]
#             scale_factor = random.choice(scale_factor_list)
#             img_hw = x.shape[-1]
#             new_hw = int(img_hw * scale_factor)
#             x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False).clamp(min=0, max=1)

#             if new_hw <= img_hw:
#                 # pad if smaller
#                 x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
#             else:
#                 # center-crop if bigger
#                 center_crop = torchvision.transforms.CenterCrop(img_hw)
#                 x_rescaled = center_crop(x_rescaled)

#             # forward pass on the scaled input
#             result = self.model_backbone(x_rescaled)
#             if self.bypass:
#                 stream_2_output, stream_2_c1_feats, stream_2_bypass = result
#             else:
#                 stream_2_output, stream_2_c1_feats = result

#             # Compute scale-consistency loss between the two streams, list ver

#             # for i in range(len(stream_1_c1_feats)):
#             #     c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
#             # c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
            
#             # out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

#             if self.bypass:
#                 bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
#             else:
#                 bypass_correct_scale_loss = 0

#             # correct_scale_loss = c1_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss
            
#             correct_scale_loss = bypass_correct_scale_loss
            
#             return stream_2_output, correct_scale_loss

#         return stream_1_output, correct_scale_loss

def pad_to_size_blue(a, size):
    """
    Pads tensor `a` (B, C, H, W) to the given `size` (H_out, W_out) with blue color.
    """
    current_size = a.shape[-2:]  # (H, W)
    pad_h = size[0] - current_size[0]
    pad_w = size[1] - current_size[1]

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Create a blue canvas (R=0, G=0, B=1 if input is float; B=255 if uint8)
    dtype = a.dtype
    device = a.device
    B, C = a.shape[:2]
    blue_val = 1.0 if dtype == torch.float32 else 255
    canvas = torch.zeros((B, C, size[0], size[1]), dtype=dtype, device=device)
    if C == 3:
        canvas[:, 2, :, :] = blue_val  # Blue channel

    # Paste `a` in the center
    canvas[:, :, pad_top:pad_top + current_size[0], pad_left:pad_left + current_size[1]] = a
    return canvas


def pad_to_size_noise(a, size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Pads tensor `a` (B, C, H, W) to the given `size` (H_out, W_out) with Gaussian noise.
    Noise is generated per channel with the given mean and std (e.g., ImageNet stats).
    """
    current_size = a.shape[-2:]  # (H, W)
    pad_h = size[0] - current_size[0]
    pad_w = size[1] - current_size[1]

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    dtype = a.dtype
    device = a.device
    B, C = a.shape[:2]

    # Create noise background
    canvas = torch.zeros((B, C, size[0], size[1]), dtype=dtype, device=device)
    for c in range(C):
        noise = torch.randn((B, 1, size[0], size[1]), dtype=dtype, device=device) * std[c] + mean[c]
        canvas[:, c:c+1, :, :] = noise

    # Paste input tensor in the center
    canvas[:, :, pad_top:pad_top + current_size[0], pad_left:pad_left + current_size[1]] = a
    return canvas


def pad_to_size_gray(a, size, gray_val_float=0.5, gray_val_uint8=128):
    """
    Pads tensor `a` (B, C, H, W) to `size` with uniform gray background using F.pad.
    """
    current_size = a.shape[-2:]  # (H, W)
    pad_h = size[0] - current_size[0]
    pad_w = size[1] - current_size[1]

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if torch.is_floating_point(a):
        pad_val = gray_val_float
    else:
        pad_val = gray_val_uint8

    # Note: F.pad pads in (left, right, top, bottom) order
    a_padded = F.pad(a, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=pad_val)
    return a_padded

# class CHRESMAX_V3_blue(nn.Module):
#     """
#     Example student-teacher style model with scale-consistency loss,
#     using RESMAX_V2 as the backbone.

#     In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
#     the returned features are [0] for C1 and C2.
#     """
#     def __init__(self, 
#                  num_classes=1000,
#                  in_chans=3,
#                  ip_scale_bands=1,
#                  classifier_input_size=13312,
#                  contrastive_loss=True,
#                  bypass=False,
#                  **kwargs):
#         super().__init__()
#         self.contrastive_loss = contrastive_loss
#         self.num_classes = num_classes
#         self.in_chans = in_chans
#         self.ip_scale_bands = ip_scale_bands
#         self.bypass = bypass
        
#         # Use the optimized backbone
#         self.model_backbone = RESMAX_V2(
#             num_classes=num_classes,
#             in_chans=in_chans,
#             ip_scale_bands=self.ip_scale_bands,
#             classifier_input_size=classifier_input_size,
#             contrastive_loss=self.contrastive_loss,
#             bypass=bypass,
#         )

#     def forward(self, x):
#         """
#         Creates two streams (original + random-scaled) for scale-consistency training.
#         Returns:
#             (output_of_stream1, correct_scale_loss)
#         """
#         # stream 1 (original scale)
#         result = self.model_backbone(x)
#         if self.bypass:
#             stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result
#         else:
#             stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result

#         # stream 2 (random scale)
#         scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
#         scale_factor = random.choice(scale_factor_list)
#         img_hw = x.shape[-1]
#         new_hw = int(img_hw * scale_factor)
#         x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

#         if new_hw <= img_hw:
#             # pad if smaller
#             x_rescaled = pad_to_size_blue(x_rescaled, (img_hw, img_hw))
#         else:
#             # center-crop if bigger
#             center_crop = torchvision.transforms.CenterCrop(img_hw)
#             x_rescaled = center_crop(x_rescaled)

#         # forward pass on the scaled input
#         result = self.model_backbone(x_rescaled)
#         if self.bypass:
#             stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result
#         else:
#             stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result

#         # Compute scale-consistency loss between the two streams, list ver
#         c1_correct_scale_loss = 0
#         for i in range(len(stream_1_c1_feats)):
#             c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
#         c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
        
#         c2_correct_scale_loss = 0
#         for i in range(len(stream_1_c2_feats)):
#             c2_correct_scale_loss += torch.mean(torch.abs(stream_1_c2_feats[i] - stream_2_c2_feats[i]))
#         c2_correct_scale_loss /= len(stream_1_c2_feats)  # Average over all feature maps
        
#         out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

#         if self.bypass:
#             bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
#         else:
#             bypass_correct_scale_loss = 0

#         correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

#         return stream_1_output, correct_scale_loss
    


# class CHRESMAX_V3_noise(nn.Module):
#     """
#     Example student-teacher style model with scale-consistency loss,
#     using RESMAX_V2 as the backbone.

#     In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
#     the returned features are [0] for C1 and C2.
#     """
#     def __init__(self, 
#                  num_classes=1000,
#                  in_chans=3,
#                  ip_scale_bands=1,
#                  classifier_input_size=13312,
#                  contrastive_loss=True,
#                  bypass=False,
#                  **kwargs):
#         super().__init__()
#         self.contrastive_loss = contrastive_loss
#         self.num_classes = num_classes
#         self.in_chans = in_chans
#         self.ip_scale_bands = ip_scale_bands
#         self.bypass = bypass
        
#         # Use the optimized backbone
#         self.model_backbone = RESMAX_V2(
#             num_classes=num_classes,
#             in_chans=in_chans,
#             ip_scale_bands=self.ip_scale_bands,
#             classifier_input_size=classifier_input_size,
#             contrastive_loss=self.contrastive_loss,
#             bypass=bypass,
#         )

#     def forward(self, x):
#         """
#         Creates two streams (original + random-scaled) for scale-consistency training.
#         Returns:
#             (output_of_stream1, correct_scale_loss)
#         """
#         # stream 1 (original scale)
#         result = self.model_backbone(x)
#         if self.bypass:
#             stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result
#         else:
#             stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result

#         # stream 2 (random scale)
#         scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
#         scale_factor = random.choice(scale_factor_list)
#         img_hw = x.shape[-1]
#         new_hw = int(img_hw * scale_factor)
#         x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

#         if new_hw <= img_hw:
#             # pad if smaller
#             x_rescaled = pad_to_size_noise(x_rescaled, (img_hw, img_hw))
#         else:
#             # center-crop if bigger
#             center_crop = torchvision.transforms.CenterCrop(img_hw)
#             x_rescaled = center_crop(x_rescaled)

#         # forward pass on the scaled input
#         result = self.model_backbone(x_rescaled)
#         if self.bypass:
#             stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result
#         else:
#             stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result

#         # Compute scale-consistency loss between the two streams, list ver
#         c1_correct_scale_loss = 0
#         for i in range(len(stream_1_c1_feats)):
#             c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
#         c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
        
#         c2_correct_scale_loss = 0
#         for i in range(len(stream_1_c2_feats)):
#             c2_correct_scale_loss += torch.mean(torch.abs(stream_1_c2_feats[i] - stream_2_c2_feats[i]))
#         c2_correct_scale_loss /= len(stream_1_c2_feats)  # Average over all feature maps
        
#         out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

#         if self.bypass:
#             bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
#         else:
#             bypass_correct_scale_loss = 0

#         correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

#         return stream_1_output, correct_scale_loss


# class CHRESMAX_V3_gray(nn.Module):
#     """
#     Example student-teacher style model with scale-consistency loss,
#     using RESMAX_V2 as the backbone.

#     In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
#     the returned features are [0] for C1 and C2.
#     """
#     def __init__(self, 
#                  num_classes=1000,
#                  in_chans=3,
#                  ip_scale_bands=1,
#                  classifier_input_size=13312,
#                  contrastive_loss=True,
#                  bypass=False,
#                  **kwargs):
#         super().__init__()
#         self.contrastive_loss = contrastive_loss
#         self.num_classes = num_classes
#         self.in_chans = in_chans
#         self.ip_scale_bands = ip_scale_bands
#         self.bypass = bypass
        
#         # Use the optimized backbone
#         self.model_backbone = RESMAX_V2(
#             num_classes=num_classes,
#             in_chans=in_chans,
#             ip_scale_bands=self.ip_scale_bands,
#             classifier_input_size=classifier_input_size,
#             contrastive_loss=self.contrastive_loss,
#             bypass=bypass,
#         )

#     def forward(self, x):
#         """
#         Creates two streams (original + random-scaled) for scale-consistency training.
#         Returns:
#             (output_of_stream1, correct_scale_loss)
#         """
#         # stream 1 (original scale)
#         result = self.model_backbone(x)
#         if self.bypass:
#             stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result
#         else:
#             stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result

#         # stream 2 (random scale)
#         scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
#         scale_factor = random.choice(scale_factor_list)
#         img_hw = x.shape[-1]
#         new_hw = int(img_hw * scale_factor)
#         x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

#         if new_hw <= img_hw:
#             # pad if smaller
#             x_rescaled = pad_to_size_gray(x_rescaled, (img_hw, img_hw))
#         else:
#             # center-crop if bigger
#             center_crop = torchvision.transforms.CenterCrop(img_hw)
#             x_rescaled = center_crop(x_rescaled)

#         # forward pass on the scaled input
#         result = self.model_backbone(x_rescaled)
#         if self.bypass:
#             stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result
#         else:
#             stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result

#         # Compute scale-consistency loss between the two streams, list ver
#         c1_correct_scale_loss = 0
#         for i in range(len(stream_1_c1_feats)):
#             c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
#         c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
        
#         c2_correct_scale_loss = 0
#         for i in range(len(stream_1_c2_feats)):
#             c2_correct_scale_loss += torch.mean(torch.abs(stream_1_c2_feats[i] - stream_2_c2_feats[i]))
#         c2_correct_scale_loss /= len(stream_1_c2_feats)  # Average over all feature maps
        
#         out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

#         if self.bypass:
#             bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
#         else:
#             bypass_correct_scale_loss = 0

#         correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

#         return stream_1_output, correct_scale_loss
    
# """
# choose largest band in bypass
# """
# class CHRESMAX_V3_1(nn.Module):
#     """
#     Example student-teacher style model with scale-consistency loss,
#     using RESMAX_V2 as the backbone.

#     In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
#     the returned features are [0] for C1 and C2.
#     """
#     def __init__(self, 
#                  num_classes=1000,
#                  in_chans=3,
#                  ip_scale_bands=1,
#                  classifier_input_size=13312,
#                  contrastive_loss=True,
#                  bypass=False,
#                  **kwargs):
#         super().__init__()
#         self.contrastive_loss = contrastive_loss
#         self.num_classes = num_classes
#         self.in_chans = in_chans
#         self.ip_scale_bands = ip_scale_bands
#         self.bypass = bypass
        
#         # Use the optimized backbone
#         self.model_backbone = RESMAX_V2_1(
#             num_classes=num_classes,
#             in_chans=in_chans,
#             ip_scale_bands=self.ip_scale_bands,
#             classifier_input_size=classifier_input_size,
#             contrastive_loss=self.contrastive_loss,
#             bypass=bypass,
#         )

#     def forward(self, x):
#         """
#         Creates two streams (original + random-scaled) for scale-consistency training.
#         Returns:
#             (output_of_stream1, correct_scale_loss)
#         """
#         # stream 1 (original scale)
#         result = self.model_backbone(x)
#         if self.bypass:
#             stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result
#         else:
#             stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result

#         # stream 2 (random scale)
#         scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
#         scale_factor = random.choice(scale_factor_list)
#         img_hw = x.shape[-1]
#         new_hw = int(img_hw * scale_factor)
#         x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

#         if new_hw <= img_hw:
#             # pad if smaller
#             x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
#         else:
#             # center-crop if bigger
#             center_crop = torchvision.transforms.CenterCrop(img_hw)
#             x_rescaled = center_crop(x_rescaled)

#         # forward pass on the scaled input
#         result = self.model_backbone(x_rescaled)
#         if self.bypass:
#             stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result
#         else:
#             stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result

#         # Compute scale-consistency loss between the two streams, list ver
#         c1_correct_scale_loss = 0
#         for i in range(len(stream_1_c1_feats)):
#             c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
#         c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
        
#         c2_correct_scale_loss = 0
#         for i in range(len(stream_1_c2_feats)):
#             c2_correct_scale_loss += torch.mean(torch.abs(stream_1_c2_feats[i] - stream_2_c2_feats[i]))
#         c2_correct_scale_loss /= len(stream_1_c2_feats)  # Average over all feature maps
        
#         out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

#         if self.bypass:
#             bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
#         else:
#             bypass_correct_scale_loss = 0

#         correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

#         return stream_1_output, correct_scale_loss
    
# """
# choose smartly in bypass
# """
# class CHRESMAX_V3_2(nn.Module):
#     """
#     Example student-teacher style model with scale-consistency loss,
#     using RESMAX_V2 as the backbone.

#     In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
#     the returned features are [0] for C1 and C2.
#     """
#     def __init__(self, 
#                  num_classes=1000,
#                  in_chans=3,
#                  ip_scale_bands=1,
#                  classifier_input_size=13312,
#                  contrastive_loss=True,
#                  bypass=False,
#                  **kwargs):
#         super().__init__()
#         self.contrastive_loss = contrastive_loss
#         self.num_classes = num_classes
#         self.in_chans = in_chans
#         self.ip_scale_bands = ip_scale_bands
#         self.bypass = bypass
        
#         # Use the optimized backbone
#         self.model_backbone = RESMAX_V2_2(
#             num_classes=num_classes,
#             in_chans=in_chans,
#             ip_scale_bands=self.ip_scale_bands,
#             classifier_input_size=classifier_input_size,
#             contrastive_loss=self.contrastive_loss,
#             bypass=bypass,
#         )

#     def forward(self, x):
#         """
#         Creates two streams (original + random-scaled) for scale-consistency training.
#         Returns:
#             (output_of_stream1, correct_scale_loss)
#         """
#         # stream 1 (original scale)
#         result = self.model_backbone(x)
#         if self.bypass:
#             stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result
#         else:
#             stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result

#         # stream 2 (random scale)
#         scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
#         scale_factor = random.choice(scale_factor_list)
#         img_hw = x.shape[-1]
#         new_hw = int(img_hw * scale_factor)
#         x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

#         if new_hw <= img_hw:
#             # pad if smaller
#             x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
#         else:
#             # center-crop if bigger
#             center_crop = torchvision.transforms.CenterCrop(img_hw)
#             x_rescaled = center_crop(x_rescaled)

#         # forward pass on the scaled input
#         result = self.model_backbone(x_rescaled)
#         if self.bypass:
#             stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result
#         else:
#             stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result

#         # Compute scale-consistency loss between the two streams, list ver
#         c1_correct_scale_loss = 0
#         for i in range(len(stream_1_c1_feats)):
#             c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
#         c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
        
#         c2_correct_scale_loss = 0
#         for i in range(len(stream_1_c2_feats)):
#             c2_correct_scale_loss += torch.mean(torch.abs(stream_1_c2_feats[i] - stream_2_c2_feats[i]))
#         c2_correct_scale_loss /= len(stream_1_c2_feats)  # Average over all feature maps
        
#         out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

#         if self.bypass:
#             bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
#         else:
#             bypass_correct_scale_loss = 0

#         correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

#         return stream_1_output, correct_scale_loss
    
"""
use additional bn, backbone resnet_v2_3
"""
class CHRESMAX_V3_3(nn.Module):
    """
    Example student-teacher style model with scale-consistency loss,
    using RESMAX_V2 as the backbone.

    In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
    the returned features are [0] for C1 and C2.
    """
    def __init__(self, 
                 num_classes=1000,
                 in_chans=3,
                 ip_scale_bands=1,
                 classifier_input_size=13312,
                 contrastive_loss=True,
                 bypass=False,
                 **kwargs):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.ip_scale_bands = ip_scale_bands
        self.bypass = bypass
        
        # Use the optimized backbone
        self.model_backbone = RESMAX_V2_3(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )

    def forward(self, x):
        """
        Creates two streams (original + random-scaled) for scale-consistency training.
        Returns:
            (output_of_stream1, correct_scale_loss)
        """
        # stream 1 (original scale)
        result = self.model_backbone(x)
        if self.bypass:
            stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result
        else:
            stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result

        # stream 2 (random scale)
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            # pad if smaller
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
        else:
            # center-crop if bigger
            center_crop = torchvision.transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)

        # forward pass on the scaled input
        result = self.model_backbone(x_rescaled)
        if self.bypass:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result
        else:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result

        # Compute scale-consistency loss between the two streams, list ver
        c1_correct_scale_loss = 0
        for i in range(len(stream_1_c1_feats)):
            c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
        c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
        
        c2_correct_scale_loss = 0
        for i in range(len(stream_1_c2_feats)):
            c2_correct_scale_loss += torch.mean(torch.abs(stream_1_c2_feats[i] - stream_2_c2_feats[i]))
        c2_correct_scale_loss /= len(stream_1_c2_feats)  # Average over all feature maps
        
        out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

        if self.bypass:
            bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
        else:
            bypass_correct_scale_loss = 0

        correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

        return stream_1_output, correct_scale_loss

class CHRESMAX_V3_A(nn.Module):
    """
    Example student-teacher style model with scale-consistency loss,
    using RESMAX_V2 as the backbone.

    In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
    the returned features are [0] for C1 and C2.
    """
    def __init__(self, 
                 num_classes=1000,
                 in_chans=3,
                 ip_scale_bands=1,
                 classifier_input_size=13312,
                 contrastive_loss=True,
                 bypass=False,
                 **kwargs):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.ip_scale_bands = ip_scale_bands
        self.bypass = bypass
        
        # Use the optimized backbone
        self.model_backbone = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )

    def forward(self, x):
        """
        Creates two streams (original + random-scaled) for scale-consistency training.
        Returns:
            (output_of_stream1, correct_scale_loss)
        """
        # stream 1 (original scale)
        result = self.model_backbone(x)
        if self.bypass:
            stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result
        else:
            stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result

        # stream 2 (random scale)
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
        else:
            center_crop = torchvision.transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)

        result = self.model_backbone(x_rescaled)
        if self.bypass:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result
        else:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result

        # Cosine similarity loss for C1
        c1_correct_scale_loss = 0
        for i in range(len(stream_1_c1_feats)):
            c1 = F.cosine_similarity(stream_1_c1_feats[i].flatten(1), stream_2_c1_feats[i].flatten(1), dim=1)
            c1_correct_scale_loss += torch.mean(1 - c1)
        c1_correct_scale_loss /= len(stream_1_c1_feats)

        # Cosine similarity loss for C2
        c2_correct_scale_loss = 0
        for i in range(len(stream_1_c2_feats)):
            c2 = F.cosine_similarity(stream_1_c2_feats[i].flatten(1), stream_2_c2_feats[i].flatten(1), dim=1)
            c2_correct_scale_loss += torch.mean(1 - c2)
        c2_correct_scale_loss /= len(stream_1_c2_feats)

        # Cosine similarity loss for output
        out_cosine_loss = 1 - F.cosine_similarity(stream_1_output, stream_2_output, dim=1).mean()

        # Optional bypass loss
        if self.bypass:
            bypass_cosine_loss = 1 - F.cosine_similarity(stream_1_bypass, stream_2_bypass, dim=1).mean()
        else:
            bypass_cosine_loss = 0

        correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_cosine_loss + bypass_cosine_loss

        return stream_1_output, correct_scale_loss

"""C layer acter c2b + cosine loss"""
class CHRESMAX_V3_A_2(nn.Module):
    """
    Example student-teacher style model with scale-consistency loss,
    using RESMAX_V2 as the backbone.

    In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
    the returned features are [0] for C1 and C2.
    """
    def __init__(self, 
                 num_classes=1000,
                 in_chans=3,
                 ip_scale_bands=1,
                 classifier_input_size=13312,
                 contrastive_loss=True,
                 bypass=False,
                 **kwargs):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.ip_scale_bands = ip_scale_bands
        self.bypass = bypass
        
        # Use the optimized backbone
        self.model_backbone = RESMAX_V2_2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )

    def forward(self, x):
        """
        Creates two streams (original + random-scaled) for scale-consistency training.
        Returns:
            (output_of_stream1, correct_scale_loss)
        """
        # stream 1 (original scale)
        result = self.model_backbone(x)
        if self.bypass:
            stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result
        else:
            stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result

        # stream 2 (random scale)
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
        else:
            center_crop = torchvision.transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)

        result = self.model_backbone(x_rescaled)
        if self.bypass:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result
        else:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result

        # Cosine similarity loss for C1
        c1_correct_scale_loss = 0
        for i in range(len(stream_1_c1_feats)):
            c1 = F.cosine_similarity(stream_1_c1_feats[i].flatten(1), stream_2_c1_feats[i].flatten(1), dim=1)
            c1_correct_scale_loss += torch.mean(1 - c1)
        c1_correct_scale_loss /= len(stream_1_c1_feats)

        # Cosine similarity loss for C2
        c2_correct_scale_loss = 0
        for i in range(len(stream_1_c2_feats)):
            c2 = F.cosine_similarity(stream_1_c2_feats[i].flatten(1), stream_2_c2_feats[i].flatten(1), dim=1)
            c2_correct_scale_loss += torch.mean(1 - c2)
        c2_correct_scale_loss /= len(stream_1_c2_feats)

        # Cosine similarity loss for output
        out_cosine_loss = 1 - F.cosine_similarity(stream_1_output, stream_2_output, dim=1).mean()

        # Optional bypass loss
        if self.bypass:
            bypass_cosine_loss = 1 - F.cosine_similarity(stream_1_bypass, stream_2_bypass, dim=1).mean()
        else:
            bypass_cosine_loss = 0

        correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_cosine_loss + bypass_cosine_loss

        return stream_1_output, correct_scale_loss


"""
In V4, we use new loss clip loss. also use reflect for padding
"""
class CHRESMAX_V4(nn.Module):

    def __init__(self, 
                 num_classes=1000,
                 in_chans=3,
                 ip_scale_bands=1,
                 classifier_input_size=13312,
                 contrastive_loss=True,
                 bypass=False,
                 temperature=0.1,
                 **kwargs):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.ip_scale_bands = ip_scale_bands
        self.bypass = bypass
        self.temperature = temperature
        
        # Use the optimized backbone
        self.backbone = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Original input - full forward pass
        if self.backbone.bypass:
            out1, c1_feats1, c2_feats1, bypass1 = self.backbone(x)
        else:
            out1, c1_feats1, c2_feats1 = self.backbone(x)
        
        # Randomly scaled input
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            # pad if smaller
            # 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw), mode='reflect')
        else:
            # center-crop if bigger
            center_crop = transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)
            
        # Forward pass on scaled input
        if self.backbone.bypass:
            out2, c1_feats2, c2_feats2, bypass2 = self.backbone(x_rescaled)
        else:
            out2, c1_feats2, c2_feats2 = self.backbone(x_rescaled)


        def clip_style_loss(z1, z2, temperature=self.temperature):
            """
            CLIP-style contrastive loss between original and scaled features.
            - z1: features from original images [B, D]
            - z2: features from rescaled images [B, D]
            """
            # Normalize
            z1 = F.normalize(z1, dim=1)  # [B, D]
            z2 = F.normalize(z2, dim=1)  # [B, D]

            # Compute logits
            logits_per_orig = torch.matmul(z1, z2.T) / temperature  # [B, B]
            logits_per_scaled = torch.matmul(z2, z1.T) / temperature  # [B, B]

            # Labels are indices [0, 1, ..., B-1]
            labels = torch.arange(z1.size(0), device=z1.device)

            loss_orig = F.cross_entropy(logits_per_orig, labels)
            loss_scaled = F.cross_entropy(logits_per_scaled, labels)

            return (loss_orig + loss_scaled) / 2
        
        # Compute CLIP-style contrastive loss between features
        if isinstance(c1_feats1, list) and isinstance(c1_feats2, list):
            c1_contrastive_loss = 0
            for i in range(len(c1_feats1)):
                f1 = c1_feats1[i].reshape(c1_feats1[i].size(0), -1)
                f2 = c1_feats2[i].reshape(c1_feats2[i].size(0), -1)
                c1_contrastive_loss += clip_style_loss(f1, f2)
            c1_contrastive_loss /= len(c1_feats1)
        else:
            f1 = c1_feats1.reshape(c1_feats1.size(0), -1)
            f2 = c1_feats2.reshape(c2_feats2.size(0), -1)
            c1_contrastive_loss = clip_style_loss(f1, f2)

        if isinstance(c2_feats1, list) and isinstance(c2_feats2, list):
            c2_contrastive_loss = 0
            for i in range(len(c2_feats1)):
                f1 = c2_feats1[i].reshape(c2_feats1[i].size(0), -1)
                f2 = c2_feats2[i].reshape(c2_feats2[i].size(0), -1)
                c2_contrastive_loss += clip_style_loss(f1, f2)
            c2_contrastive_loss /= len(c2_feats1)
        else:
            f1 = c2_feats1.reshape(c2_feats1.size(0), -1)
            f2 = c2_feats2.reshape(c2_feats2.size(0), -1)
            c2_contrastive_loss = clip_style_loss(f1, f2)

        # Top-level output loss
        out_contrastive_loss = clip_style_loss(out1, out2)

        # Optional: bypass contrastive
        if self.backbone.bypass:
            bypass_contrastive_loss = clip_style_loss(bypass1.reshape(bypass1.size(0), -1), bypass2.reshape(bypass2.size(0), -1))
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss + 0.1 * bypass_contrastive_loss
        else:
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss

                            
        return out1, total_loss

"""
In V5 uses V3 Resmax, performance not good
""" 
class CHRESMAX_V5(nn.Module):
    def __init__(self, 
                 num_classes=1000,
                 in_chans=3,
                 ip_scale_bands=1,
                 classifier_input_size=512,
                 contrastive_loss=True,
                 bypass=False,
                 temperature=0.1,
                 **kwargs):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.ip_scale_bands = ip_scale_bands
        self.bypass = bypass
        self.temperature = temperature
        
        # Use the optimized backbone
        self.backbone = RESMAX_V3(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Original input - full forward pass
        if self.backbone.bypass:
            out1, c1_feats1, c2_feats1, bypass1 = self.backbone(x)
        else:
            out1, c1_feats1, c2_feats1 = self.backbone(x)
        
        # Randomly scaled input
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            # pad if smaller
            # 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw), mode='reflect')
        else:
            # center-crop if bigger
            center_crop = transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)
            
        # Forward pass on scaled input
        if self.backbone.bypass:
            out2, c1_feats2, c2_feats2, bypass2 = self.backbone(x_rescaled)
        else:
            out2, c1_feats2, c2_feats2 = self.backbone(x_rescaled)


        def clip_style_loss(z1, z2, temperature=self.temperature):
            """
            CLIP-style contrastive loss between original and scaled features.
            - z1: features from original images [B, D]
            - z2: features from rescaled images [B, D]
            """
            # Normalize
            z1 = F.normalize(z1, dim=1)  # [B, D]
            z2 = F.normalize(z2, dim=1)  # [B, D]

            # Compute logits
            logits_per_orig = torch.matmul(z1, z2.T) / temperature  # [B, B]
            logits_per_scaled = torch.matmul(z2, z1.T) / temperature  # [B, B]

            # Labels are indices [0, 1, ..., B-1]
            labels = torch.arange(z1.size(0), device=z1.device)

            loss_orig = F.cross_entropy(logits_per_orig, labels)
            loss_scaled = F.cross_entropy(logits_per_scaled, labels)

            return (loss_orig + loss_scaled) / 2
        
        # Compute CLIP-style contrastive loss between features
        if isinstance(c1_feats1, list) and isinstance(c1_feats2, list):
            c1_contrastive_loss = 0
            for i in range(len(c1_feats1)):
                f1 = c1_feats1[i].reshape(c1_feats1[i].size(0), -1)
                f2 = c1_feats2[i].reshape(c1_feats2[i].size(0), -1)
                c1_contrastive_loss += clip_style_loss(f1, f2)
            c1_contrastive_loss /= len(c1_feats1)
        else:
            f1 = c1_feats1.reshape(c1_feats1.size(0), -1)
            f2 = c1_feats2.reshape(c2_feats2.size(0), -1)
            c1_contrastive_loss = clip_style_loss(f1, f2)

        if isinstance(c2_feats1, list) and isinstance(c2_feats2, list):
            c2_contrastive_loss = 0
            for i in range(len(c2_feats1)):
                f1 = c2_feats1[i].reshape(c2_feats1[i].size(0), -1)
                f2 = c2_feats2[i].reshape(c2_feats2[i].size(0), -1)
                c2_contrastive_loss += clip_style_loss(f1, f2)
            c2_contrastive_loss /= len(c2_feats1)
        else:
            f1 = c2_feats1.reshape(c2_feats1.size(0), -1)
            f2 = c2_feats2.reshape(c2_feats2.size(0), -1)
            c2_contrastive_loss = clip_style_loss(f1, f2)

        # Top-level output loss
        out_contrastive_loss = clip_style_loss(out1, out2)

        # Optional: bypass contrastive
        if self.backbone.bypass:
            bypass_contrastive_loss = clip_style_loss(bypass1.reshape(bypass1.size(0), -1), bypass2.reshape(bypass2.size(0), -1))
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss + 0.1 * bypass_contrastive_loss
        else:
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss


        return out1, total_loss

# Create a new model for contrastive fine-tuning
class ContrastiveRESMAX(nn.Module):
    def __init__(self,
                num_classes=1000,
                in_chans=3,
                ip_scale_bands=1,
                classifier_input_size=9216,
                contrastive_loss=True,
                bypass=False,
                pretrained_path=None,
                temperature=0.1,
                **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.contrastive_loss = contrastive_loss

        pretrained_path = f'/oscar/data/tserre/xyu110/pytorch-output/train/recent_results/ip_{ip_scale_bands}_resmax_v2_gpu_8_cl_0_ip_3_322_322_{classifier_input_size}_c1[_6,3,1_]/model_best.pth.tar'
        self.temperature = temperature
        self.ip_scale_bands = ip_scale_bands

        # Create the backbone model
        self.backbone = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )
        
        # Load pretrained weights from file path
        checkpoint = torch.load(pretrained_path, weights_only=False, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            # If checkpoint contains a state_dict key (common in training frameworks)
            self.backbone.load_state_dict(checkpoint['state_dict'], strict=True)
            print('Loaded state dict from checkpoint if')
        else:
            # Directly load if it's just the state dict
            self.backbone.load_state_dict(checkpoint, strict=True)
            print('Loaded state dict from checkpoint else')
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze layers strategically for contrastive fine-tuning
        
        # Always unfreeze the FC layers
        for param in self.backbone.fc2.parameters():
            param.requires_grad = True
        for param in self.backbone.fc1.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
            
        # # Unfreeze S3_Res (deeper convolutional layers)
        # # This is crucial since S3_Res processes features after C2 scoring,
        # # which directly impacts scale robustness
        for param in self.backbone.s3.parameters():
            param.requires_grad = True
        # for param in self.backbone.c1.parameters():
        #     param.requires_grad = True
        # for param in self.backbone.c2.parameters():
        #     param.requires_grad = True
        # if bypass:
        #     for param in self.backbone.c2b_seq.parameters():
        #         param.requires_grad = True
            
        # Unfreeze global pooling layer that aggregates features
        for param in self.backbone.global_pool.parameters():
            param.requires_grad = True
            
        # No additional projection head needed
        # We'll use the backbone's FC layers for feature extraction
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Original input - full forward pass
        if self.backbone.bypass:
            out1, c1_feats1, c2_feats1, bypass1 = self.backbone(x)
        else:
            out1, c1_feats1, c2_feats1 = self.backbone(x)
        
        # Randomly scaled input
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            # pad if smaller
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
        else:
            # center-crop if bigger
            center_crop = transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)
            
        # Forward pass on scaled input
        if self.backbone.bypass:
            out2, c1_feats2, c2_feats2, bypass2 = self.backbone(x_rescaled)
        else:
            out2, c1_feats2, c2_feats2 = self.backbone(x_rescaled)

        ##############################NT-Xent Loss#############################
        
        # Helper function to compute NT-Xent loss between two feature maps
        def nt_xent_loss(f1, f2, temperature=self.temperature):
            # Flatten spatial dimensions if needed
            if len(f1.shape) > 2:
                f1 = f1.reshape(f1.size(0), -1)
                f2 = f2.reshape(f2.size(0), -1)
            
            # Normalize features
            z1 = F.normalize(f1, dim=1)
            z2 = F.normalize(f2, dim=1)
            
            # Concatenate features from both scales
            features = torch.cat([z1, z2], dim=0)
            
            # Compute similarity matrix
            sim_matrix = torch.matmul(features, features.T) / temperature
            
            # Create mask for positive pairs
            pos_mask = torch.zeros_like(sim_matrix)
            pos_mask[:batch_size, batch_size:] = torch.eye(batch_size)
            pos_mask[batch_size:, :batch_size] = torch.eye(batch_size)
            
            # Create mask to exclude self-similarity
            self_mask = torch.eye(2 * batch_size, device=sim_matrix.device)
            logits_mask = torch.ones_like(sim_matrix) - self_mask
            
            # NT-Xent loss calculation
            exp_logits = torch.exp(sim_matrix) * logits_mask
            log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True))
            mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
            
            return -mean_log_prob_pos.mean()
        
        # Compute NT-Xent loss between features from different scales
        if isinstance(c1_feats1, list) and isinstance(c1_feats2, list):
            # Handle list of feature maps
            c1_contrastive_loss = 0
            for i in range(len(c1_feats1)):
                c1_contrastive_loss += nt_xent_loss(c1_feats1[i], c1_feats2[i])
            c1_contrastive_loss /= len(c1_feats1)  # Average loss across feature maps
        else:
            # Direct feature map
            c1_contrastive_loss = nt_xent_loss(c1_feats1, c1_feats2)

        if isinstance(c2_feats1, list) and isinstance(c2_feats2, list):
            # Handle list of feature maps
            c2_contrastive_loss = 0
            for i in range(len(c2_feats1)):
                c2_contrastive_loss += nt_xent_loss(c2_feats1[i], c2_feats2[i])
            c2_contrastive_loss /= len(c2_feats1)  # Average loss across feature maps
        else:
            # Direct feature map
            c2_contrastive_loss = nt_xent_loss(c2_feats1, c2_feats2)

        # Compute output contrastive loss (this should be a single tensor, not a list)
        out_contrastive_loss = nt_xent_loss(out1, out2)

        # Compute bypass contrastive loss if needed
        if self.backbone.bypass:
            bypass_contrastive_loss = nt_xent_loss(bypass1, bypass2)
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss + 0.1 * bypass_contrastive_loss
        else:
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss 

        # print(f"c1_contrastive_loss: {c1_contrastive_loss}, c2_contrastive_loss: {c2_contrastive_loss}, out_contrastive_loss: {out_contrastive_loss}")
                
        return out1, total_loss

# Create a new model for contrastive fine-tuning
class ContrastiveRESMAX_V1(nn.Module):
    def __init__(self,
                num_classes=1000,
                in_chans=3,
                ip_scale_bands=1,
                classifier_input_size=9216,
                contrastive_loss=True,
                bypass=False,
                pretrained_path=None,
                temperature=0.1,
                **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.contrastive_loss = contrastive_loss

        pretrained_path = f'/oscar/data/tserre/xyu110/pytorch-output/train/recent_results/ip_{ip_scale_bands}_resmax_v2_gpu_8_cl_0_ip_3_322_322_{classifier_input_size}_c1[_6,3,1_]/model_best.pth.tar'
        self.temperature = temperature
        self.ip_scale_bands = ip_scale_bands

        # Create the backbone model
        self.backbone = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )
        
        # Load pretrained weights from file path
        checkpoint = torch.load(pretrained_path, weights_only=False, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            # If checkpoint contains a state_dict key (common in training frameworks)
            self.backbone.load_state_dict(checkpoint['state_dict'], strict=True)
            print('Loaded state dict from checkpoint if')
        else:
            # Directly load if it's just the state dict
            self.backbone.load_state_dict(checkpoint, strict=True)
            print('Loaded state dict from checkpoint else')
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze layers strategically for contrastive fine-tuning
        
        # Always unfreeze the FC layers
        for param in self.backbone.fc2.parameters():
            param.requires_grad = True
        for param in self.backbone.fc1.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
            
        # # Unfreeze S3_Res (deeper convolutional layers)
        for param in self.backbone.s3.parameters():
            param.requires_grad = True

        # Unfreeze global pooling layer that aggregates features
        for param in self.backbone.global_pool.parameters():
            param.requires_grad = True
            
        # No additional projection head needed
        # We'll use the backbone's FC layers for feature extraction
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Original input - full forward pass
        if self.backbone.bypass:
            out1, c1_feats1, c2_feats1, bypass1 = self.backbone(x)
        else:
            out1, c1_feats1, c2_feats1 = self.backbone(x)
        
        # Randomly scaled input
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            # pad if smaller
            # 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw), mode='reflect')
        else:
            # center-crop if bigger
            center_crop = transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)
            
        # Forward pass on scaled input
        if self.backbone.bypass:
            out2, c1_feats2, c2_feats2, bypass2 = self.backbone(x_rescaled)
        else:
            out2, c1_feats2, c2_feats2 = self.backbone(x_rescaled)

        ##############################NT-Xent Loss#############################
        
        # Helper function to compute NT-Xent loss between two feature maps
        def clip_style_loss(z1, z2, temperature=self.temperature):
            """
            CLIP-style contrastive loss between original and scaled features.
            - z1: features from original images [B, D]
            - z2: features from rescaled images [B, D]
            """
            # Normalize
            z1 = F.normalize(z1, dim=1)  # [B, D]
            z2 = F.normalize(z2, dim=1)  # [B, D]

            # Compute logits
            logits_per_orig = torch.matmul(z1, z2.T) / temperature  # [B, B]
            logits_per_scaled = torch.matmul(z2, z1.T) / temperature  # [B, B]

            # Labels are indices [0, 1, ..., B-1]
            labels = torch.arange(z1.size(0), device=z1.device)

            loss_orig = F.cross_entropy(logits_per_orig, labels)
            loss_scaled = F.cross_entropy(logits_per_scaled, labels)

            return (loss_orig + loss_scaled) / 2
        
        # Compute CLIP-style contrastive loss between features
        if isinstance(c1_feats1, list) and isinstance(c1_feats2, list):
            c1_contrastive_loss = 0
            for i in range(len(c1_feats1)):
                f1 = c1_feats1[i].reshape(c1_feats1[i].size(0), -1)
                f2 = c1_feats2[i].reshape(c1_feats2[i].size(0), -1)
                c1_contrastive_loss += clip_style_loss(f1, f2)
            c1_contrastive_loss /= len(c1_feats1)
        else:
            f1 = c1_feats1.reshape(c1_feats1.size(0), -1)
            f2 = c1_feats2.reshape(c2_feats2.size(0), -1)
            c1_contrastive_loss = clip_style_loss(f1, f2)

        if isinstance(c2_feats1, list) and isinstance(c2_feats2, list):
            c2_contrastive_loss = 0
            for i in range(len(c2_feats1)):
                f1 = c2_feats1[i].reshape(c2_feats1[i].size(0), -1)
                f2 = c2_feats2[i].reshape(c2_feats2[i].size(0), -1)
                c2_contrastive_loss += clip_style_loss(f1, f2)
            c2_contrastive_loss /= len(c2_feats1)
        else:
            f1 = c2_feats1.reshape(c2_feats1.size(0), -1)
            f2 = c2_feats2.reshape(c2_feats2.size(0), -1)
            c2_contrastive_loss = clip_style_loss(f1, f2)

        # Top-level output loss
        out_contrastive_loss = clip_style_loss(out1, out2)

        # Optional: bypass contrastive
        if self.backbone.bypass:
            bypass_contrastive_loss = clip_style_loss(bypass1.reshape(bypass1.size(0), -1), bypass2.reshape(bypass2.size(0), -1))
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss + 0.1 * bypass_contrastive_loss
        else:
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss

                            
        return out1, total_loss

"""Distillation, KL loss"""
class ContrastiveRESMAX_V2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 in_chans=3,
                 ip_scale_bands=1,
                 classifier_input_size=9216,
                 contrastive_loss=True,
                 bypass=False,
                 pretrained_path=None,
                 temperature=0.1,
                 use_kl_loss=True,
                 kl_loss_weight=0.5,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.contrastive_loss = contrastive_loss
        self.temperature = temperature
        self.use_kl_loss = use_kl_loss
        self.kl_loss_weight = kl_loss_weight
        self.ip_scale_bands = ip_scale_bands

        pretrained_path = f'/oscar/data/tserre/xyu110/pytorch-output/train/0/models_w_aug/ip_3_resmax_v2_gpu_8_cl_0_ip_3_322_322_18432_c1[_6,3,1_]_bypass_scale_0.08/model_best.pth.tar'

        # Backbone model
        self.backbone = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )

        checkpoint = torch.load(pretrained_path, weights_only=False, map_location='cpu')
        if 'state_dict' in checkpoint:
            self.backbone.load_state_dict(checkpoint['state_dict'], strict=True)
            print('Loaded state dict from checkpoint if')
        else:
            self.backbone.load_state_dict(checkpoint, strict=True)
            print('Loaded state dict from checkpoint else')

        # Freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Selective fine-tuning
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        for param in self.backbone.fc1.parameters():
            param.requires_grad = True
        for param in self.backbone.fc2.parameters():
            param.requires_grad = True
        for param in self.backbone.s3.parameters():
            param.requires_grad = True
        for param in self.backbone.global_pool.parameters():
            param.requires_grad = True

    def forward(self, x):
        batch_size = x.shape[0]

        # ======== Get frozen output for KL divergence ========
        with torch.no_grad():
            teacher_out, *_ = self.backbone(x)
            teacher_log_probs = F.log_softmax(teacher_out / self.temperature, dim=1)

        # ======== Forward pass through current model ========
        if self.backbone.bypass:
            out1, c1_feats1, c2_feats1, bypass1 = self.backbone(x)
        else:
            out1, c1_feats1, c2_feats1 = self.backbone(x)

        # ======== Scaled input ========
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
        else:
            center_crop = transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)

        if self.backbone.bypass:
            out2, c1_feats2, c2_feats2, bypass2 = self.backbone(x_rescaled)
        else:
            out2, c1_feats2, c2_feats2 = self.backbone(x_rescaled)

        # ======== NT-Xent (InfoNCE) Loss Function ========
        def nt_xent_loss(f1, f2, temperature=self.temperature):
            if len(f1.shape) > 2:
                f1 = f1.reshape(f1.size(0), -1)
                f2 = f2.reshape(f2.size(0), -1)
            z1 = F.normalize(f1, dim=1)
            z2 = F.normalize(f2, dim=1)
            features = torch.cat([z1, z2], dim=0)
            sim_matrix = torch.matmul(features, features.T) / temperature
            pos_mask = torch.zeros_like(sim_matrix)
            pos_mask[:batch_size, batch_size:] = torch.eye(batch_size)
            pos_mask[batch_size:, :batch_size] = torch.eye(batch_size)
            self_mask = torch.eye(2 * batch_size, device=sim_matrix.device)
            logits_mask = torch.ones_like(sim_matrix) - self_mask
            exp_logits = torch.exp(sim_matrix) * logits_mask
            log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True))
            mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
            return -mean_log_prob_pos.mean()

        # Contrastive losses from features
        def calc_contrastive_loss(f1_list, f2_list):
            if isinstance(f1_list, list):
                total = sum(nt_xent_loss(f1, f2) for f1, f2 in zip(f1_list, f2_list))
                return total / len(f1_list)
            else:
                return nt_xent_loss(f1_list, f2_list)

        c1_contrastive_loss = calc_contrastive_loss(c1_feats1, c1_feats2)
        c2_contrastive_loss = calc_contrastive_loss(c2_feats1, c2_feats2)
        out_contrastive_loss = nt_xent_loss(out1, out2)

        if self.backbone.bypass:
            bypass_contrastive_loss = nt_xent_loss(bypass1, bypass2)
            contrastive_total = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss + 0.1 * bypass_contrastive_loss
        else:
            contrastive_total = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss

        # ======== KL Divergence Loss ========
        if self.use_kl_loss:
            student_log_probs = F.log_softmax(out1 / self.temperature, dim=1)
            teacher_probs = F.softmax(teacher_out / self.temperature, dim=1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
            total_loss = contrastive_total + self.kl_loss_weight * kl_loss
        else:
            total_loss = contrastive_total

        return out1, total_loss


@register_model
def resmax_v2(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = RESMAX_V2(**kwargs)

    return model

@register_model
def resmax_v3(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = RESMAX_V3(**kwargs)

    return model

@register_model
def resmax_v4(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = RESMAX_V4(**kwargs)

    return model

@register_model
def chresmax_v3(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    model = CHRESMAX_V3(**kwargs)
        
    return model

@register_model
def chresmax_v3_1(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    # model = CHRESMAX_V3_1(**kwargs)
    model_backbone = RESMAX_V2_1(contrastive_loss=True, **kwargs)
    model = CH_2_streams(model_backbone=model_backbone, bypass_only_model_bool=False, **kwargs)
    return model

@register_model
def chresmax_v3_bypass_only(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    # model = CHRESMAX_V3_bypass_only(**kwargs)
    model_backbone = RESMAX_V2_bypass_only(contrastive_loss=True, **kwargs)
    model = CH_2_streams_training_eval_sep(model_backbone=model_backbone, bypass_only_model_bool=True, **kwargs)
    
    
    # if pretrained:
    #     model.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
        
    return model

@register_model
def chresmax_v3_bypass_only_c2b(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    # model = CHRESMAX_V3_bypass_only(**kwargs)
    model_backbone = RESMAX_V2_bypass_only_c2b(contrastive_loss=True, **kwargs)
    model = CH_2_streams_training_eval_sep(model_backbone=model_backbone, bypass_only_model_bool=True, **kwargs)
    
    
    # if pretrained:
    #     model.load_state_dict(torch.load(pretrained, map_location='cpu')['state_dict'], strict=True)
        
    return model

@register_model
def chresmax_v3_bypass_only_tiny(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)
        
    # print(kwargs)
    model_backbone = RESMAX_V2_bypass_only_tiny(contrastive_loss=True, **kwargs)
    model = CH_2_streams(model_backbone=model_backbone, bypass_only_model_bool=True, **kwargs)
    
    # if pretrained:
    #     model.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
        
    return model


# @register_model
# def chresmax_v3_bypass_only_1(pretrained=False, **kwargs):
#     """
#     Registry function to create a CHALEXMAX_V3_3_optimized model
#     via timm's create_model API.
#     """
#     for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
#         kwargs.pop(key, None)

#     for key, val in kwargs.items():
#         print(key, val)

#     model = CHRESMAX_V3_bypass_only_1(**kwargs)
    
#     # if pretrained:
#     #     model.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
        
#     return model


@register_model
def chresmax_abs_bypass_only(pretrained=False, **kwargs):
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)
    
    model_backbone = RESMAX_abs_bypass_only(contrastive_loss=True, **kwargs)
    model = CH_2_streams(model_backbone=model_backbone, bypass_only_model_bool=True, **kwargs)
    return model


# @register_model
# def chresmax_v3_bypass_only_o(pretrained=False, **kwargs):
#     """
#     Registry function to create a CHALEXMAX_V3_3_optimized model
#     via timm's create_model API.
#     """
#     for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
#         kwargs.pop(key, None)

#     for key, val in kwargs.items():
#         print(key, val)

#     model = CHRESMAX_V3_bypass_only_o(**kwargs)

        
#     return model




@register_model
def chresmax_v3_blue(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    # model = CHRESMAX_V3_blue(**kwargs)
    
    backbone = RESMAX_V2(contrastive_loss=True, **kwargs)
    model = CH_2_streams(model_backbone=backbone, bypass_only_model_bool=False, **kwargs)
    model.pad_color_background = 'blue'
    
    return model

@register_model
def chresmax_v3_noise(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    # model = CHRESMAX_V3_noise(**kwargs)
    
    backbone = RESMAX_V2(contrastive_loss=True, **kwargs)
    model = CH_2_streams(model_backbone=backbone, bypass_only_model_bool=False, **kwargs)
    model.pad_color_background = 'noise'
    
    return model

@register_model
def chresmax_v3_gray(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    # model = CHRESMAX_V3_gray(**kwargs)
    
    backbone = RESMAX_V2(contrastive_loss=True, **kwargs)
    model = CH_2_streams(model_backbone=backbone, bypass_only_model_bool=False, **kwargs)
    model.pad_color_background = 'gray'
    
    return model

@register_model
def chresmax_v3_a(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    model = CHRESMAX_V3_A(**kwargs)
    return model

@register_model
def chresmax_v3_a_2(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    model = CHRESMAX_V3_A(**kwargs)
    return model

@register_model
def chresmax_v3_2(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    # model = CHRESMAX_V3_2(**kwargs)
    model_backbone = RESMAX_V2_2(contrastive_loss=True, **kwargs)
    # model = CH_2_streams(model_backbone=model_backbone, bypass_only_model_bool=False, **kwargs)
    model = CH_2_streams_training_eval_sep(model_backbone=model_backbone, bypass_only_model_bool=False, **kwargs)
    return model


@register_model
def hmax_v3_adj(pretrained=False, **kwargs):
    """
    Registry function to create a CH_2_streams_adjacent_scales model
    via timm's create_model API.
    This model uses adjacent scale pairs for contrastive learning.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)
    
    model_backbone = RESMAX_V2_2(contrastive_loss=True, **kwargs)
    
    # Create the adjacent scales contrastive model
    model = CH_2_streams_adjacent_scales(
        model_backbone=model_backbone, 
        bypass_only_model_bool=False, 
        **kwargs
    )
    
    return model


@register_model
def chresmax_v3_2_abs(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    model_backbone = RESMAX_abs(contrastive_loss=True, **kwargs)
    # model = CH_2_streams(model_backbone=model_backbone, bypass_only_model_bool=False, **kwargs)
    model = CH_2_streams_training_eval_sep(model_backbone=model_backbone, bypass_only_model_bool=False, **kwargs)
    return model

@register_model
def chresmax_v3_2_rand(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    model = CHRESMAX_V3_2_rand(**kwargs)
    return model

@register_model
def chresmax_v4(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    model = CHRESMAX_V4(**kwargs)
    return model

@register_model
def chresmax_v5(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    model = CHRESMAX_V5(**kwargs)
    return model

@register_model
def contrastive_resmax(pretrained=False, **kwargs):
    """
    Registry function to create a ContrastiveRESMAX model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    if pretrained:
        pass
    
    model = ContrastiveRESMAX(**kwargs)
    return model

@register_model
def contrastive_resmaxv1(pretrained=False, **kwargs):
    """
    Registry function to create a ContrastiveRESMAX model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    if pretrained:
        pass
    
    model = ContrastiveRESMAX_V1(**kwargs)
    return model

@register_model
def ft_resmax_v2(pretrained=False, **kwargs):
    """
    Registry function to create a ContrastiveRESMAX model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    if pretrained:
        pass
    
    model = ContrastiveRESMAX_V2(**kwargs)
    return model


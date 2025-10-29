
"""
An implementation of HMAX:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy as sp
import time
import pdb

from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs
from .HMAX import HMAX_from_Alexnet, HMAX_from_Alexnet_bypass,get_ip_scales,pad_to_size




def check_for_nans(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaNs found in {name}")

class S1_old(nn.Module):
    def __init__(self, kernel_size=11, stride=4, padding=0):
        super(S1_old, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(96),
            nn.ReLU())

    def forward(self, x_pyramid):
        # get dimensions
        return [self.layer1(x) for x in x_pyramid]

class S1(nn.Module):
    def __init__(self,kernel_size=11, stride=4, padding=0):
        super(S1, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(96),
            nn.ReLU())

    def forward(self, x_pyramid):
        # get dimensions
        if type(x_pyramid) == list:
            return [self.layer1(x) for x in x_pyramid]
        else:
            return self.layer1(x_pyramid)

class S2b(nn.Module):
    def __init__(self):
        super(S2b, self).__init__()
        ## bypass layers
        self.s2b_kernel_size=[4,8,12,16]
        self.s2b_seqs = nn.ModuleList()
        for size in self.s2b_kernel_size:
            self.s2b_seqs.append(nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=size, stride=1, padding=size//2),
                nn.BatchNorm2d(256, 1e-3),
                nn.ReLU(True)
            ))

    def forward(self, x_pyramid):
        # get dimensions
        bypass = [torch.cat([seq(out) for seq in self.s2b_seqs], dim=1) for out in x_pyramid]
        return bypass

# lots of repeated code
class S2(nn.Module):
    def __init__(self,kernel_size=5, stride=1, padding=2):
        super(S2, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=kernel_size, stride=stride, padding=padding),
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

class C_mid(nn.Module):
    #Spatial then Scale
    def __init__(self,
                  pool_func1 = nn.MaxPool2d(kernel_size = 3, stride = 2),
                  pool_func2 = nn.MaxPool2d(kernel_size = 3, stride = 2),
                  global_scale_pool=False):
        super(C_mid, self).__init__()
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


            out_middle = self.pool1(x_pyramid[len(x_pyramid)//2])
            final_size = out_middle.shape[-2:]


            for i in range(0, len(x_pyramid) - 1):
                x_1 = x_pyramid[i]
                x_2 = x_pyramid[i+1]

                #spatial pooling
                x_1 = self.pool1(x_1)
                x_2 = self.pool2(x_2)

                ## Lets resize to middle size of the pyramid all the time.
                x_2 = F.interpolate(x_2, size =final_size, mode = 'bicubic')
                x_1 = F.interpolate(x_1, size = final_size, mode = 'bicubic')

                x = torch.stack([x_1, x_2], dim=4)

                #get index
                #index = torch.argmax(x, dim=4)
                #smoothing index selection so patches have the same size selected
                to_append = torch.max(x, dim=4)[0]
                out.append(to_append)
        return out


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
    
class ConvScoring(nn.Module):
    def __init__(self, num_channels):
        super(ConvScoring, self).__init__()
        # Use adaptive average pooling to reduce H and W to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Linear layer to map from num_channels to 1
        # make it not trainable
        self.fc = nn.Linear(num_channels, 1, bias=False)
        self.fc.weight.requires_grad = False

    def forward(self, x):
        # x: [B, K, H, W]
        B, K, H, W = x.shape
        # Apply adaptive pooling
        x_pooled = self.pool(x)  # Shape: [B, K, 1, 1]
        # todo : try with squeeze as oppose of view 
        x_pooled = x_pooled.view(B, K)  # Shape: [B, K]
        # Apply linear layer
        scores = self.fc(x_pooled)  # Shape: [B, 1]
        scores = scores.squeeze(1)  # Shape: [B]
        return scores

import torch.nn.functional as F

def soft_selection(scores, feature_maps):
    """
    Perform soft selection with global scores.
    Args:
        scores: Tensor of shape [batch_size, 2] (global scores for each map).
        feature_maps: Tensor of shape [batch_size, 2, C, W, H] (feature maps).
    Returns:
        Soft-selected feature map of shape [batch_size, C, W, H].
    """
    # Normalize scores across the scale dimension (dim=1)
    weights = F.softmax(scores, dim=1)  # Shape: [batch_size, 2]

    # Reshape weights to match feature maps: [batch_size, 2, 1, 1, 1]
    weights = weights.view(weights.size(0), weights.size(1), 1, 1, 1)

    # Perform weighted sum of feature maps
    weighted_maps = weights * feature_maps  # Broadcast: [batch_size, 2, C, W, H]
    return weighted_maps.sum(dim=1)  # Shape: [batch_size, C, W, H]

class C_scoring(nn.Module):
    # Spatial then Scale
    def __init__(self,
                 num_channels,
                 pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                 pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                 global_scale_pool=False):
        super(C_scoring, self).__init__()
        self.pool1 = pool_func1
        self.pool2 = pool_func2
        self.global_scale_pool = global_scale_pool
        self.num_channels = num_channels
        self.scoring_conv = ConvScoring(num_channels)

    def forward(self, x_pyramid):
        out = []
        if self.global_scale_pool:
            if len(x_pyramid) == 1:
                pooled = self.pool1(x_pyramid[0])
                _ = self.scoring_conv(pooled)  # Shape: [N]
                return pooled

            out = [self.pool1(x) for x in x_pyramid]
            final_size = out[len(out) // 2].shape[-2:]
            out_1 = F.interpolate(out[0], final_size, mode='bilinear')
            for x in x_pyramid[1:]:
                temp = F.interpolate(x, final_size, mode='bilinear')
                score_out = self.scoring_conv(out_1)  # Shape: [N, H, W]
                score_temp = self.scoring_conv(temp)  # Shape: [N, H, W]
                scores = torch.stack([score_out, score_temp], dim=1).unsqueeze(2)  # Shape: [N, 2, 1, H, W]
                x = torch.stack([out_1, temp], dim=1)  # Shape: [N, 2, C, H, W]
                out_1 = soft_selection(scores, x)  # Differentiable selection
                del temp

        else:  # Not global pool
            if len(x_pyramid) == 1:
                pooled = self.pool1(x_pyramid[0])
                _ = self.scoring_conv(pooled)
                return [pooled]

            out_middle = self.pool1(x_pyramid[-1])
            final_size = out_middle.shape[-2:]

            for i in range(len(x_pyramid) - 1):
                x_1 = self.pool1(x_pyramid[i])
                x_2 = self.pool2(x_pyramid[i + 1])
                
                # Compute scores using the learnable scoring function
                s_1 = self.scoring_conv(x_1)  # Shape: [N, 1, H, W]
                s_2 = self.scoring_conv(x_2)  # Shape: [N, 1, H, W]
                
                scores = torch.stack([s_1, s_2], dim=1)  # Shape: [N, 2, 1, H, W]

                x = torch.stack([x_1, x_2], dim=1)  # Shape: [N, 2, C, H, W]
                to_append = soft_selection(scores, x)
                #assert  torch.allclose(to_append,x_1) == True or torch.allclose(to_append,x_2) == True
                out.append(to_append)

        return out


import torch
import torch.nn as nn

class ALEXMAX_v0(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3, ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False,pyramid=False):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        super(ALEXMAX_v0, self).__init__()


        self.s1 = S1()
        self.c1= C_mid(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2))
        #self.s2b = S2b()
        #self.c2b = C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s2 = S2()
        self.c2 = C_mid(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s3 = S3()
        self.global_pool =  C_mid(global_scale_pool=True)
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
        scale = 4   ## factor in exponenet

        image_scales = get_ip_scales(num_scale_bands, base_image_size, scale)

        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear')
                image_pyramid.append(interpolated_img)
            return image_pyramid
        else:
            return [x]

    def forward(self, x,pyramid=False):


        #resize image

        if pyramid or self.pyramid:
            out = self.make_ip(x)
        else:
            out = [x]
        
        ## should make SxBxCxHxW
        out = self.s1(out)


        out_c1 = self.c1(out)
        #bypass layers
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        out = self.s3(out_c2)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0],out_c2[0]

        return out

class ALEXMAX_v0_scoring(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3, ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False,pyramid=False):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        super(ALEXMAX_v0_scoring, self).__init__()


        self.s1 = S1()
        self.c1= C_scoring(96,nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2))
        #self.s2b = S2b()
        #self.c2b = C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s2 = S2()
        self.c2 = C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s3 = S3()
        self.global_pool =  C(global_scale_pool=True)
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
        scale = 4   ## factor in exponenet

        image_scales = get_ip_scales(num_scale_bands, base_image_size, scale)

        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear')
                image_pyramid.append(interpolated_img)
            return image_pyramid
        else:
            return [x]

    def forward(self, x,pyramid=False):


        #resize image
        import pdb; pdb.set_trace()
        if pyramid or self.pyramid:
            out = self.make_ip(x)
        else:
            out = [x]

        ## should make SxBxCxHxW
        out = self.s1(out)


        out_c1 = self.c1(out)
        #bypass layers
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        #if pyramid:
        #    return out_c1[0],out_c2[0]

        out = self.s3(out_c2)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0],out_c2[0]

        return out

class ALEXMAX_v1(nn.Module):
    # kernel size for c1 is 2, stride is 3 with padding of 1. To get features on the 27x27 shape. Similar to alexnet
    def __init__(self, num_classes=1000, in_chans=3, ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False,pyramid=False):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        super(ALEXMAX_v1, self).__init__()
        self.s1 = S1()
        self.c1= C_mid(nn.MaxPool2d(kernel_size = 2, stride = 3,padding=1), nn.MaxPool2d(kernel_size = 3, stride = 2))
        #self.s2b = S2b()
        #self.c2b = C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s2 = S2()
        self.c2 = C_mid(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s3 = S3()
        self.global_pool =  C_mid(global_scale_pool=True)
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
        scale = 4   ## factor in exponenet

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

    def forward(self, x,pyramid=False):


        #resize image

        if pyramid or self.pyramid:
            out = self.make_ip(x)
        else:
            out = [x]

        ## should make SxBxCxHxW
        out = self.s1(out)


        out_c1 = self.c1(out)
        #bypass layers
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        #if pyramid:
        #    return out_c1[0],out_c2[0]

        out = self.s3(out_c2)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0],out_c2[0]

        return out

class ALEXMAX(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3, ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False,pyramid=False):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        super(ALEXMAX, self).__init__()


        self.s1 = S1()
        self.c1= C_mid(nn.MaxPool2d(kernel_size = 6, stride = 3,padding=3), nn.MaxPool2d(kernel_size = 3, stride = 2))
        #self.s2b = S2b()
        #self.c2b = C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s2 = S2()
        self.c2 = C_mid(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s3 = S3()
        self.global_pool =  C_mid(global_scale_pool=True)
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
        scale = 4   ## factor in exponenet

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
    def forward(self, x,pyramid=False):


        #resize image

        if pyramid or self.pyramid:
            out = self.make_ip(x)
        else:
            out = [x]
        
        ## should make SxBxCxHxW
        out = self.s1(out)
        out_c1 = self.c1(out)
        #bypass layers
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        if pyramid:
            return out_c1[0],out_c2[0]

        out = self.s3(out_c2)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0],out_c2[0]

        return out
class ALEXMAX_v1_3(nn.Module):
    def __init__(self, num_classes=1000,big_size =322,small_size =227, in_chans=3, ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False,pyramid=False):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        super(ALEXMAX_v1_3, self).__init__()


        self.s1 = S1(kernel_size=11, stride=4, padding=0)
       
        self.c1= C_scoring(96,nn.MaxPool2d(kernel_size = 3, stride =2), nn.MaxPool2d(kernel_size = 3, stride = 2))
        #self.s2b = S2b()
        #self.c2b = C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s2 = S2(kernel_size=5, stride=1, padding=2)
        self.c2 = C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s3 = S3()
        self.global_pool =  C(global_scale_pool=True)
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
        #num_scale_bands = self.ip_scale_bands
        #base_image_size = int(x.shape[-1])
        #scale = 4   ## factor in exponenet

        image_scales = [self.big_size,self.small_size]

        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear').clamp(min=0, max=1)

                image_pyramid.append(interpolated_img)
            return image_pyramid
        else:
            return [x]
    def forward(self, x,pyramid=False):
        #resize image
        # always making pyramid to start with only two scales. 
        out = self.make_ip(x)
        
        ## should make SxBxCxHxW
        out= self.s1(out)
        out = self.c1(out)
        #bypass layers
        out = self.s2(out)
        out = self.c2(out)
        out = self.s3(out)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
class ALEXMAX_v2(nn.Module):
    def __init__(self, num_classes=1000,big_size =322,small_size =227, in_chans=3, ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False,pyramid=False, **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        super(ALEXMAX_v2, self).__init__()

        self.s1_big = S1(kernel_size=11, stride=4, padding=0)
        self.s1_small = S1(kernel_size=9, stride=4, padding=0)
        self.c1= C_scoring(96,nn.MaxPool2d(kernel_size =6, stride = 3,padding=3), nn.MaxPool2d(kernel_size = 3, stride = 2))
        #self.s2b = S2b()
        #self.c2b = C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s2 = S2(kernel_size=5, stride=1, padding=2)
        self.c2 = C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s3 = S3()
        self.global_pool =  C(global_scale_pool=True)
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
        #num_scale_bands = self.ip_scale_bands
        #base_image_size = int(x.shape[-1])
        #scale = 4   ## factor in exponenet

        image_scales = [self.big_size,self.small_size]

        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear')

                image_pyramid.append(interpolated_img)
            return image_pyramid
        else:
            return [x]
    def forward(self, x,pyramid=False):
        #resize image
        # always making pyramid to start with only two scales. 
        out = self.make_ip(x)
        
        ## should make SxBxCxHxW
        out_1 = self.s1_big([out[0]])
        out_2 = self.s1_small([out[1]])
        out_c1 = self.c1([out_1[0],out_2[0]])
        #bypass layers
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        if pyramid:
            return out_c1[0],out_c2[0]

        out = self.s3(out_c2)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0],out_c2[0]

        return out
    

class S1_VGG_Big(nn.Module):
    def __init__(self):
        super(S1_VGG_Big, self).__init__()
        # Replace kernel_size=11, stride=4 with multiple 3x3 convs
        self.layer1 = nn.Sequential(
            # First conv with stride 2
            nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # Second conv
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # Third conv with stride 2
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

    def forward(self, x_pyramid):
        return [self.layer1(x) for x in x_pyramid]

class S1_VGG_Small(nn.Module):
    def __init__(self):
        super(S1_VGG_Small, self).__init__()
        # Replace kernel_size=9, stride=4 with multiple 3x3 convs
        self.layer1 = nn.Sequential(
            # First conv with stride 2
            nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # Second conv with stride 2
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

    def forward(self, x_pyramid):
        return [self.layer1(x) for x in x_pyramid]

class S2_VGG(nn.Module):
    def __init__(self):
        super(S2_VGG, self).__init__()
        # Replace kernel_size=5 with two 3x3 convs
        self.layer = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x_pyramid):
        return [self.layer(x) for x in x_pyramid]

class S3_VGG(nn.Module):
    def __init__(self):
        super(S3_VGG, self).__init__()
        # Add more 3x3 convs for increased depth
        self.layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x_pyramid):
        return [self.layer(x) for x in x_pyramid]

class VGGMAX(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False, **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        super(VGGMAX, self).__init__()

        self.s1_big = S1_VGG_Big()
        self.s1_small = S1_VGG_Small()
        
        self.c1 = C_scoring(96,
            nn.MaxPool2d(kernel_size=6, stride=3, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.s2 = S2_VGG()
        self.c2 = C(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            global_scale_pool=False
        )
        
        self.s3 = S3_VGG()
        self.global_pool = C(global_scale_pool=True)
        
        # Keep classifier layers the same
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

    def make_ip(self, x):
        image_scales = [self.big_size, self.small_size]

        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size=(i_s, i_s), mode='bilinear')
                image_pyramid.append(interpolated_img)
            return image_pyramid
        else:
            return [x]

    def forward(self, x,pyramid=False):
        #resize image
        # always making pyramid to start with only two scales. 
        out = self.make_ip(x)
        
        ## should make SxBxCxHxW
        out_1 = self.s1_big([out[0]])
        out_2 = self.s1_small([out[1]])
        out_c1 = self.c1([out_1[0],out_2[0]])
        #bypass layers
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        if pyramid:
            return out_c1[0],out_c2[0]

        out = self.s3(out_c2)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0],out_c2[0]

        return out

from .ALEXMAX3 import C_scoring2

class VGGMAX_V1(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False, **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        super(VGGMAX_V1, self).__init__()

        self.s1 = S1_VGG_Big()
        
        self.c1 = C_scoring2(96,
            nn.MaxPool2d(kernel_size=6, stride=3, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.s2 = S2_VGG()
        self.c2 = C(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            global_scale_pool=False
        )
        
        self.s3 = S3()
        self.global_pool = C(global_scale_pool=True)
        
        # Keep classifier layers the same
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

    def make_ip(self, x):
        ## num_scale_bands = num images in IP - 1
        num_scale_bands = self.ip_scale_bands
        base_image_size = int(x.shape[-1])
        scale = 4   ## factor in exponenet

        image_scales = get_ip_scales(num_scale_bands, base_image_size, scale)
        
        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear')

                image_pyramid.append(interpolated_img)
            return image_pyramid
        else: 
            return [x]

    def forward(self, x,pyramid=False):
        #resize image
        # always making pyramid to start with only two scales. 
        out = self.make_ip(x)
        
        ## should make SxBxCxHxW
        out_1 = self.s1(out)
        out_c1 = self.c1(out_1)

        #bypass layers
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        if pyramid:
            return out_c1[0],out_c2[0]

        out = self.s3(out_c2)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0],out_c2[0]

        return out
    

class NINBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(NINBlock, self).__init__()
        self.layers = nn.Sequential(
            # Main convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # First mlpconv
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # Second mlpconv
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layers(x)

class S1_VGG_NIN_Big(nn.Module):
    def __init__(self):
        super(S1_VGG_NIN_Big, self).__init__()
        self.layer1 = nn.Sequential(
            NINBlock(3, 48, stride=2),
            NINBlock(48, 48),
            NINBlock(48, 96, stride=2)
        )

    def forward(self, x_pyramid):
        return [self.layer1(x) for x in x_pyramid]

class S1_VGG_NIN_Small(nn.Module):
    def __init__(self):
        super(S1_VGG_NIN_Small, self).__init__()
        self.layer1 = nn.Sequential(
            NINBlock(3, 48, stride=2),
            NINBlock(48, 96, stride=2)
        )

    def forward(self, x_pyramid):
        return [self.layer1(x) for x in x_pyramid]

class S2_VGG_NIN(nn.Module):
    def __init__(self):
        super(S2_VGG_NIN, self).__init__()
        self.layer = nn.Sequential(
            NINBlock(96, 128),
            NINBlock(128, 256)
        )

    def forward(self, x_pyramid):
        return [self.layer(x) for x in x_pyramid]

class S3_VGG_NIN(nn.Module):
    def __init__(self):
        super(S3_VGG_NIN, self).__init__()
        self.layer = nn.Sequential(
            NINBlock(256, 256),
            NINBlock(256, 384),
            NINBlock(384, 384),
            NINBlock(384, 256)
        )

    def forward(self, x_pyramid):
        return [self.layer(x) for x in x_pyramid]


class NINMAX(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False, **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        super(NINMAX, self).__init__()

        self.s1_big = S1_VGG_NIN_Big()
        self.s1_small = S1_VGG_NIN_Small()
        
        self.c1 = C_scoring(96,
            nn.MaxPool2d(kernel_size=6, stride=3, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.s2 = S2_VGG_NIN()
        self.c2 = C(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            global_scale_pool=False
        )
        
        self.s3 = S3_VGG_NIN()
        self.global_pool = C(global_scale_pool=True)
        
        # Modify classifier to use NIN-style MLPConv
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1024, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def make_ip(self, x):
        image_scales = [self.big_size, self.small_size]
        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size=(i_s, i_s), mode='bilinear')
                image_pyramid.append(interpolated_img)
            return image_pyramid
        return [x]

    def forward(self, x, pyramid=False):
        out = self.make_ip(x)
        
        out_1 = self.s1_big([out[0]])
        out_2 = self.s1_small([out[1]])
        out_c1 = self.c1([out_1[0], out_2[0]])
        
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        if pyramid:
            return out_c1[0], out_c2[0]

        out = self.s3(out_c2)
        out = self.global_pool(out)
        
        # Use NIN-style classifier
        out = self.classifier(out)
        out = out.view(out.size(0), -1)

        if self.contrastive_loss:
            return out, out_c1[0], out_c2[0]

        return out
    


class ALEXMAX_v2_0(nn.Module):
    def __init__(self, num_classes=1000,big_size =322,small_size =227, in_chans=3, ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False,pyramid=False):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        super(ALEXMAX_v2_0, self).__init__()
        self.s1 = S1(kernel_size=11, stride=4, padding=0)
        #self.s1_small = S1(kernel_size=11, stride=4, padding=0)
        self.c1= C_scoring(96,nn.MaxPool2d(kernel_size =6, stride = 3,padding=3), nn.MaxPool2d(kernel_size = 3, stride = 2))
        #self.s2b = S2b()
        #self.c2b = C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s2 = S2(kernel_size=5, stride=1, padding=2)
        self.c2 = C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s3 = S3()
        self.global_pool =  C(global_scale_pool=True)
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
        #num_scale_bands = self.ip_scale_bands
        #base_image_size = int(x.shape[-1])
        #scale = 4   ## factor in exponenet

        image_scales = [self.big_size,self.small_size]

        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear')

                image_pyramid.append(interpolated_img)
            return image_pyramid
        else:
            return [x]
    def forward(self, x,pyramid=False):
        #resize image
        out = self.make_ip(x)
        ## should make SxBxCxHxW
        out = self.s1(out)
        #out_2 = self.s1_small([out[1]])
        out_c1 = self.c1(out)
        #bypass layers
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        if pyramid:
            return out_c1[0],out_c2[0]

        out = self.s3(out_c2)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0],out_c2[0]

        return out
     
class ALEXMAX_v2_C(nn.Module):
    def __init__(self, num_classes=1000,big_size =322,small_size =227, in_chans=3, ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False,pyramid=False):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        super(ALEXMAX_v2_C, self).__init__()


        self.s1_big = S1(kernel_size=11, stride=4, padding=0)
        self.s1_small = S1(kernel_size=9, stride=4, padding=0)
        self.c1= C(nn.MaxPool2d(kernel_size = 6, stride = 3,padding=3), nn.MaxPool2d(kernel_size = 3, stride = 2))
        #self.s2b = S2b()
        #self.c2b = C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s2 = S2(kernel_size=5, stride=1, padding=2)
        self.c2 = C_mid(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s3 = S3()
        self.global_pool =  C_mid(global_scale_pool=True)
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
        #num_scale_bands = self.ip_scale_bands
        #base_image_size = int(x.shape[-1])
        #scale = 4   ## factor in exponenet

        image_scales = [self.big_size,self.small_size]

        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear').clamp(min=0, max=1)

                image_pyramid.append(interpolated_img)
            return image_pyramid
        else:
            return [x]
    def forward(self, x,pyramid=False):
        #resize image
        # always making pyramid to start with only two scales. 
        out = self.make_ip(x)
        
        ## should make SxBxCxHxW
        out_1 = self.s1_big([out[0]])
        out_2 = self.s1_small([out[1]])
        out_c1 = self.c1([out_1[0],out_2[0]])
        #bypass layers
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        if pyramid:
            return out_c1[0],out_c2[0]

        out = self.s3(out_c2)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0],out_c2[0]

        return out

class ALEXMAX_v2_Cmid(nn.Module):
    def __init__(self, num_classes=1000,big_size =322,small_size =227, in_chans=3, ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False,pyramid=False):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        super(ALEXMAX_v2_Cmid, self).__init__()


        self.s1_big = S1(kernel_size=11, stride=4, padding=0)
        self.s1_small = S1(kernel_size=9, stride=4, padding=0)
        self.c1= C_mid(nn.MaxPool2d(kernel_size = 6, stride = 3,padding=3), nn.MaxPool2d(kernel_size = 3, stride = 2))
        #self.s2b = S2b()
        #self.c2b = C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s2 = S2(kernel_size=5, stride=1, padding=2)
        self.c2 = C_mid(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s3 = S3()
        self.global_pool =  C_mid(global_scale_pool=True)
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
        #num_scale_bands = self.ip_scale_bands
        #base_image_size = int(x.shape[-1])
        #scale = 4   ## factor in exponenet

        image_scales = [self.small_size,self.big_size]

        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear').clamp(min=0, max=1)

                image_pyramid.append(interpolated_img)
            return image_pyramid
        else:
            return [x]
    def forward(self, x,pyramid=False):
        #resize image
        # always making pyramid to start with only two scales. 
        out = self.make_ip(x)
        
        ## should make SxBxCxHxW
        out_1 = self.s1_big([out[0]])
        out_2 = self.s1_small([out[1]])
        out_c1 = self.c1([out_1[0],out_2[0]])
        #bypass layers
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        if pyramid:
            return out_c1[0],out_c2[0]

        out = self.s3(out_c2)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0],out_c2[0]

        return out
    
class ALEXMAX_v2_1_pr(nn.Module):
    def __init__(self, num_classes=1000,big_size =322,small_size =227, in_chans=3, ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False,pyramid=False):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        super(ALEXMAX_v2_1_pr, self).__init__()


        self.s1_big = S1(kernel_size=11, stride=4, padding=0)
        self.s1_small = S1(kernel_size=11, stride=4, padding=0)
        self.c1= C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2))
        #self.s2b = S2b()
        #self.c2b = C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s2 = S2(kernel_size=5, stride=1, padding=2)
        self.c2 = C_mid(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s3 = S3()
        self.global_pool =  C_mid(global_scale_pool=True)
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
        #num_scale_bands = self.ip_scale_bands
        #base_image_size = int(x.shape[-1])
        #scale = 4   ## factor in exponenet

        image_scales = [self.big_size,self.small_size]

        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear').clamp(min=0, max=1)

                image_pyramid.append(interpolated_img)
            return image_pyramid
        else:
            return [x]
    def forward(self, x,pyramid=False):
        #resize image
        # always making pyramid to start with only two scales. 
        out = self.make_ip(x)
        import pdb; pdb.set_trace()
        ## should make SxBxCxHxW
        out_1 = self.s1_big([out[0]])
        out_2 = self.s1_small([out[1]])
        out_c1 = self.c1([out_1[0],out_2[0]])
        #bypass layers
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        if pyramid:
            return out_c1[0],out_c2[0]

        out = self.s3(out_c2)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0],out_c2[0]

        return out

class ALEXMAX_pyramid(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3, ip_scale_bands=1, classifier_input_size=13312,contrastive_loss=False):
        self.contrastive_loss = contrastive_loss
        self.num_classes = num_classes
        self.in_chans = in_chans
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands
        super(ALEXMAX_pyramid, self).__init__()

        self.s1_big = S1(kernel_size=11, stride=4, padding=0)
        self.s1_small = S1(kernel_size=9, stride=3, padding=0)
        self.c1= C_scoring(96,nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2))
        #self.s2b = S2b()
        #self.c2b = C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s2 = S2()
        self.c2= C_scoring(256,nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.s3 = S3()
        self.global_pool =C(global_scale_pool=True)
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
        scale = 4   ## factor in exponenet

        image_scales = get_ip_scales(num_scale_bands, base_image_size, scale)

        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear')
                #ensure is between 0 and 1 after interpolation
                interpolated_img = interpolated_img.clamp(min=0, max=1)


                image_pyramid.append(interpolated_img)
            return image_pyramid
        else:
            return [x]

    def forward(self, x):

        out = self.make_ip(x)
        if self.ip_scale_bands == 1:
            out = self.s1_big(out)
        ## should make SxBxCxHxW
        # apply small filter to even scales and big filter to odd scales
        out = [self.s1_big(out[i]) if i % 2 == 1 else self.s1_small(out[i]) for i in range(len(out)-1)]
        #out = self.s1(out)
        out = self.c1(out)

        #bypass layers
        #bypass = self.s2b(out)
        #bypass = self.c2b(bypass)

        # main
        out = self.s2(out)

        out = self.c2(out)
        out = self.s3(out)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)
        #bypass = bypass.reshape(bypass.size(0), -1)

        ## merge here
        #out = torch.cat([bypass, out], dim=1)
        #del bypass

        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out



import random
import torchvision

class CHALEXMAX(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3, ip_scale_bands=1, classifier_input_size=13312, hmax_type="full"):
        super(CHALEXMAX, self).__init__()

        # the below line is so that the training script calculates the loss correctly
        self.contrastive_loss = True

        if hmax_type == "alexmax":
            self.model_backbone = ALEXMAX(num_classes=num_classes,
                                                        in_chans=in_chans,
                                                        ip_scale_bands=ip_scale_bands,
                                                        classifier_input_size=classifier_input_size,
                                                        contrastive_loss=True)

        else:
            raise(NotImplementedError)

        self.num_classes = num_classes
        self.in_chans = in_chans
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands

    def forward(self, x):

        # stream 1
        stream_1_output, stream_1_c1_feats,stream_1_c2_feats = self.model_backbone(x)

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

        stream_2_c1_feats, stream_2_c2_feats = self.model_backbone(x_rescaled,pyramid=True)
        correct_scale_loss_c1 = torch.mean(torch.abs(stream_1_c1_feats - stream_2_c1_feats))
        correct_scale_loss_c2 = torch.mean(torch.abs(stream_1_c2_feats - stream_2_c2_feats))
        correct_scale_loss = (correct_scale_loss_c1 + correct_scale_loss_c2) / 2
        return stream_1_output, correct_scale_loss

class CHALEXMAX_v1(nn.Module):
    #Version of CHALXMAX that uses a pretrained alexmax for backbone model. And a trainable copy of alexmax with the full pyramid for the second stream.
    def __init__(self, num_classes=1000, in_chans=3, ip_scale_bands=1, classifier_input_size=13312,weights_path ='', hmax_type="alexmax_v0"):
        super(CHALEXMAX_v1, self).__init__()

        # the below line is so that the training script calculates the loss correctly
        self.contrastive_loss = True
        self.weights_path ="/users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/train/debug_alexmax_cl_0_ip_1_322_9216/model_best.pth.tar"

        if hmax_type == "alexmax":
            self.model_backbone = ALEXMAX(num_classes=num_classes,
                                                        in_chans=in_chans,
                                                        ip_scale_bands=ip_scale_bands,
                                                        classifier_input_size=classifier_input_size,
                                                        contrastive_loss=True)
        elif hmax_type == "alexmax_v0":
            self.model_backbone = ALEXMAX_v0(num_classes=num_classes,
                                                        in_chans=in_chans,
                                                        ip_scale_bands=1,
                                                        classifier_input_size=classifier_input_size,
                                                        contrastive_loss=True)


            self.model_backbone.load_state_dict(torch.load(self.weights_path)['state_dict'])
            # freeze the model backbone
            for param in self.model_backbone.parameters():
                param.requires_grad = False


            self.trainable_model = ALEXMAX_v0(num_classes=num_classes,
                                                        in_chans=in_chans,
                                                        ip_scale_bands=ip_scale_bands,
                                                        classifier_input_size=classifier_input_size,
                                                        contrastive_loss=True)
            # start the trainable model with the same weights as the pretrained model
            self.trainable_model.load_state_dict(self.model_backbone.state_dict())

        elif hmax_type == "alexmax_v1":
            self.model_backbone = ALEXMAX_v0(num_classes=num_classes,
                                                        in_chans=in_chans,
                                                        ip_scale_bands=1,
                                                        classifier_input_size=classifier_input_size,
                                                        contrastive_loss=True)

            self.model_backbone.load_state_dict(torch.load(self.weights_path)['state_dict'])
            # freeze the model backbone
            for param in self.model_backbone.parameters():
                param.requires_grad = False

            self.trainable_model = ALEXMAX_v0_scoring(num_classes=num_classes,
                                                        in_chans=in_chans,
                                                        ip_scale_bands=ip_scale_bands,
                                                        classifier_input_size=classifier_input_size,
                                                        contrastive_loss=True)

        else:
            raise(NotImplementedError)

        self.num_classes = num_classes
        self.in_chans = in_chans
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands

    def forward(self, x):

        # stream 1
        stream_1_output, stream_1_c1_feats,stream_1_c2_feats = self.model_backbone(x)

        # stream 2
        scale_factor_list = [0.707, 0.841, 1, 1.189, 1.414]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw*scale_factor)
        x_rescaled = F.interpolate(x, size = (new_hw, new_hw), mode = 'bilinear').clamp(min=0, max=1)
        #if new_hw <= img_hw:
        #    x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
        #elif new_hw > img_hw:
        #   center_crop = torchvision.transforms.CenterCrop(img_hw)
        #   x_rescaled = center_crop(x_rescaled)

        stream_2_output, stream_2_c1_feats, stream_2_c2_feats = self.trainable_model(x_rescaled,pyramid=True)
        #mean squared error loss
        correct_scale_mse_loss_c1= torch.mean((stream_1_c1_feats - stream_2_c1_feats)**2)
        correct_scale_mse_loss_c2 = torch.mean((stream_1_c2_feats - stream_2_c2_feats)**2)
        correct_scale_mse_loss = (correct_scale_mse_loss_c1 + correct_scale_mse_loss_c2) / 2
        output_mse = torch.mean((stream_1_output - stream_2_output)**2)
        correct_scale_mse_loss = (correct_scale_mse_loss + output_mse) / 2
        return stream_2_output, correct_scale_mse_loss

class CHALEXMAX_v1(nn.Module):
    #Version of CHALXMAX that uses a pretrained alexmax for backbone model. And a trainable copy of alexmax with the full pyramid for the second stream.
    def __init__(self, num_classes=1000, in_chans=3, ip_scale_bands=1, classifier_input_size=13312,weights_path ='', hmax_type="alexmax_v0"):
        super(CHALEXMAX_v1, self).__init__()

        # the below line is so that the training script calculates the loss correctly
        self.contrastive_loss = True
        self.weights_path ="/users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/train/debug_alexmax_cl_0_ip_1_322_9216/model_best.pth.tar"

        if hmax_type == "alexmax":
            self.model_backbone = ALEXMAX(num_classes=num_classes,
                                                        in_chans=in_chans,
                                                        ip_scale_bands=ip_scale_bands,
                                                        classifier_input_size=classifier_input_size,
                                                        contrastive_loss=True)
        elif hmax_type == "alexmax_v0":
            self.model_backbone = ALEXMAX_v0(num_classes=num_classes,
                                                        in_chans=in_chans,
                                                        ip_scale_bands=1,
                                                        classifier_input_size=classifier_input_size,
                                                        contrastive_loss=True)


            self.model_backbone.load_state_dict(torch.load(self.weights_path)['state_dict'])
            # freeze the model backbone
            for param in self.model_backbone.parameters():
                param.requires_grad = False


            self.trainable_model = ALEXMAX_v0(num_classes=num_classes,
                                                        in_chans=in_chans,
                                                        ip_scale_bands=ip_scale_bands,
                                                        classifier_input_size=classifier_input_size,
                                                        contrastive_loss=True)
            # start the trainable model with the same weights as the pretrained model
            self.trainable_model.load_state_dict(self.model_backbone.state_dict())

        elif hmax_type == "alexmax_v1":
            self.model_backbone = ALEXMAX_v0(num_classes=num_classes,
                                                        in_chans=in_chans,
                                                        ip_scale_bands=1,
                                                        classifier_input_size=classifier_input_size,
                                                        contrastive_loss=True)

            self.model_backbone.load_state_dict(torch.load(self.weights_path)['state_dict'])
            # freeze the model backbone
            for param in self.model_backbone.parameters():
                param.requires_grad = False

            self.trainable_model = ALEXMAX_v0_scoring(num_classes=num_classes,
                                                        in_chans=in_chans,
                                                        ip_scale_bands=ip_scale_bands,
                                                        classifier_input_size=classifier_input_size,
                                                        contrastive_loss=True)

        else:
            raise(NotImplementedError)

        self.num_classes = num_classes
        self.in_chans = in_chans
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands

    def forward(self, x):

        # stream 1
        stream_1_output, stream_1_c1_feats,stream_1_c2_feats = self.model_backbone(x)

        # stream 2
        scale_factor_list =   np.arange(-self.ip_scale_bands//2 + 1, self.ip_scale_bands//2 + 2) #[0.707, 0.841, 1, 1.189, 1.414]
        scale_factor_list = [2**(i/4) for i in scale_factor_list]
        
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw*scale_factor)
        x_rescaled = F.interpolate(x, size = (new_hw, new_hw), mode = 'bilinear').clamp(min=0, max=1)
        if new_hw <= img_hw:
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
        elif new_hw > img_hw:
            center_crop = torchvision.transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)

        stream_2_output, stream_2_c1_feats, stream_2_c2_feats = self.trainable_model(x_rescaled,pyramid=True)
        #mean squared error loss
        correct_scale_mse_loss_c1= torch.mean((stream_1_c1_feats - stream_2_c1_feats)**2)
        correct_scale_mse_loss_c2 = torch.mean((stream_1_c2_feats - stream_2_c2_feats)**2)
        correct_scale_mse_loss = (correct_scale_mse_loss_c1 + correct_scale_mse_loss_c2) / 2
        output_mse = torch.mean((stream_1_output- stream_2_output)**2)
        correct_scale_mse_loss = (correct_scale_mse_loss + output_mse) / 2
        return stream_2_output, correct_scale_mse_loss

def checkpoint_filter_fn(state_dict, model: nn.Module):
    out_dict = {}
    for k, v in state_dict.items():
        out_dict[k] = v
    return out_dict


@register_model
def alexmax_v0(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = ALEXMAX_v0(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model
@register_model
def alexmax_v2_Cmid(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = ALEXMAX_v2_Cmid(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def alexmax(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = ALEXMAX(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def alexmax_v1_3(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = ALEXMAX_v1_3(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def alexmax_v2(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = ALEXMAX_v2(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def vggmax(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = VGGMAX(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model


@register_model
def vggmax_v1(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = VGGMAX_V1(**kwargs)
    if pretrained:
        pass
    return model

@register_model
def ninmax(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = NINMAX(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model


@register_model
def alexmax_v2_C(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = ALEXMAX_v2_C(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def alexmax_v2_0(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = ALEXMAX_v2_0(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model


@register_model
def alexmax_v2_1_pr(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = ALEXMAX_v2_1_pr(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model
@register_model
def alexmax_pyramid(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = ALEXMAX_pyramid(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def chalexmax(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = CHALEXMAX(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def chalexmax_v1(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = CHALEXMAX_v1(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model
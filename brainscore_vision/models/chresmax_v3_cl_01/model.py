from brainscore_vision.model_helpers.check_submission import check_models
import functools
import torch
import torch.nn as nn
import torchvision
import numpy as np
import random
from urllib.request import urlretrieve
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

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

# --------------------------------------------------
# Memory-Optimized C_scoring2
# --------------------------------------------------
class C_scoring2_optimized(nn.Module):
    """
    Memory-optimized version of the C_scoring2 layer:
      - Processes each scale (or scale-pair) in a loop rather than concatenating
        all scales at once.
      - Uses in-place ops when possible.
      - Explicitly deletes intermediate tensors to reduce peak memory usage.

    Args:
        num_channels (int): Number of input channels.
        pool_func1 (nn.Module): Pooling function for the first input (e.g. nn.MaxPool2d).
        pool_func2 (nn.Module): Pooling function for the second input.
        resize_kernel_1 (int): First resizing conv kernel size.
        resize_kernel_2 (int): Second resizing conv kernel size.
        skip (int): Skip step for iterating over scales.
        global_scale_pool (bool): Whether to use global scale pooling or not.
    """
    def __init__(
        self,
        num_channels,
        pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
        pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
        resize_kernel_1=1,
        resize_kernel_2=3,
        skip=1,
        global_scale_pool=False
    ):
        super().__init__()
        self.pool1 = pool_func1
        self.pool2 = pool_func2
        self.global_scale_pool = global_scale_pool
        self.num_channels = num_channels
        self.scoring_conv = ConvScoring(num_channels)
        self.skip = skip
        
        # Learnable resizing layers (with in-place ReLU)
        self.resizing_layers = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=resize_kernel_1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=resize_kernel_2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_pyramid):
        """
        x_pyramid: list of tensors, each [N, C, Hi, Wi].

        Returns:
            - If global_scale_pool=True, returns a single feature map [N, C, H', W'].
            - Otherwise, returns a list of feature maps [N, C, H', W'] (one per scale pair).
        """
        # ----------------------------------------------------------------
        # Global-scale pooling path
        # ----------------------------------------------------------------
        if self.global_scale_pool:
            # If only one scale, trivial path
            if len(x_pyramid) == 1:
                pooled = self.pool1(x_pyramid[0])
                # Just run scoring_conv so it’s in the graph
                _ = self.scoring_conv(pooled)
                return pooled

            # Gather a list of pooled outputs
            out_list = [self.pool1(x) for x in x_pyramid]

            # We'll iteratively soft-select from out_list[0] through out_list[-1]
            out_ref = out_list[0]
            final_size = out_list[len(out_list)//2].shape[-2:]  # pick a reference size

            for i in range(1, len(out_list)):
                tmp = F.interpolate(out_list[i], final_size, mode='bilinear', align_corners=False)
                tmp = self.resizing_layers(tmp)
                
                # Score each
                score_out_ref = self.scoring_conv(out_ref)
                score_tmp = self.scoring_conv(tmp)
                
                # Soft selection
                scores = torch.stack([score_out_ref, score_tmp], dim=1)  # [N, 2, 1, H, W]
                feats = torch.stack([out_ref, tmp], dim=1)               # [N, 2, C, H, W]

                del tmp, score_out_ref, score_tmp  # free memory

                out_ref = soft_selection(scores, feats)
                del scores, feats

            return out_ref

        # ----------------------------------------------------------------
        # Non-global path: pairwise scale merging
        # ----------------------------------------------------------------
        if len(x_pyramid) == 1:
            # Single scale path
            pooled = self.pool2(x_pyramid[0])
            _ = self.scoring_conv(pooled)
            return [pooled]

        # 1) Pool all scales
        pooled_1_list = [self.pool1(x) for x in x_pyramid]
        pooled_2_list = [self.pool2(x) for x in x_pyramid]

        # 2) Use the middle scale from pooled_2_list to define final_size
        mid_idx = len(x_pyramid) // 2
        final_size = pooled_2_list[mid_idx].shape[-2:]

        # 3) Build pairs: (pooled_1_list[i], pooled_2_list[i+1]) stepping by self.skip
        out_feats = []
        for i in range(0, len(x_pyramid) - 1, self.skip):
            a = F.interpolate(pooled_1_list[i], size=final_size, mode='bilinear', align_corners=False)
            b = F.interpolate(pooled_2_list[i + 1], size=final_size, mode='bilinear', align_corners=False)

            a = self.resizing_layers(a)
            b = self.resizing_layers(b)

            score_a = self.scoring_conv(a)
            score_b = self.scoring_conv(b)

            # Soft selection
            scores = torch.stack([score_a, score_b], dim=1)  # => [N, 2, 1, H', W']
            feats = torch.stack([a, b], dim=1)               # => [N, 2, C, H', W']

            del a, b, score_a, score_b
            out_feats.append(soft_selection(scores, feats))
            del scores, feats

        return out_feats
    

class C_scoring2_optimized_debug(nn.Module):
    """
    Memory-optimized version of the C_scoring2 layer:
      - Processes each scale (or scale-pair) in a loop rather than concatenating
        all scales at once.
      - Uses in-place ops when possible.
      - Explicitly deletes intermediate tensors to reduce peak memory usage.

    Args:
        num_channels (int): Number of input channels.
        pool_func1 (nn.Module): Pooling function for the first input (e.g. nn.MaxPool2d).
        pool_func2 (nn.Module): Pooling function for the second input.
        resize_kernel_1 (int): First resizing conv kernel size.
        resize_kernel_2 (int): Second resizing conv kernel size.
        skip (int): Skip step for iterating over scales.
        global_scale_pool (bool): Whether to use global scale pooling or not.
    """
    def __init__(
        self,
        num_channels,
        pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
        pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
        resize_kernel_1=1,
        resize_kernel_2=3,
        skip=1,
        global_scale_pool=False
    ):
        super().__init__()
        self.pool1 = pool_func1
        self.pool2 = pool_func2
        self.global_scale_pool = global_scale_pool
        self.num_channels = num_channels
        self.scoring_conv = ConvScoring(num_channels)
        self.skip = skip
        
        # Learnable resizing layers (with in-place ReLU)
        self.resizing_layers = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=resize_kernel_1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=resize_kernel_2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_pyramid):
        """
        x_pyramid: list of tensors, each [N, C, Hi, Wi].

        Returns:
            - If global_scale_pool=True, returns a single feature map [N, C, H', W'].
            - Otherwise, returns a list of feature maps [N, C, H', W'] (one per scale pair).
        """
        # ----------------------------------------------------------------
        # Global-scale pooling path
        # ----------------------------------------------------------------
        # import pdb; pdb.set_trace()
        if self.global_scale_pool:
            # If only one scale, trivial path
            if len(x_pyramid) == 1:
                pooled = self.pool1(x_pyramid[0])
                # Just run scoring_conv so it’s in the graph
                _ = self.scoring_conv(pooled)
                return pooled

            # Gather a list of pooled outputs
            out_list = [self.pool1(x) for x in x_pyramid]

            # We'll iteratively soft-select from out_list[0] through out_list[-1]
            final_size = out_list[len(out_list)//2].shape[-2:]  # pick a reference size
            out_ref = F.interpolate(out_list[0], final_size, mode='bilinear', align_corners=False)
            out_ref = self.resizing_layers(out_ref)

            for i in range(1, len(out_list)):
                tmp = F.interpolate(out_list[i], final_size, mode='bilinear', align_corners=False)
                tmp = self.resizing_layers(tmp)
                
                # Score each
                score_out_ref = self.scoring_conv(out_ref)
                score_tmp = self.scoring_conv(tmp)
                
                # Soft selection
                scores = torch.stack([score_out_ref, score_tmp], dim=1)  # [N, 2, 1, H, W]
                feats = torch.stack([out_ref, tmp], dim=1)               # [N, 2, C, H, W]

                del tmp, score_out_ref, score_tmp  # free memory

                out_ref = soft_selection(scores, feats)
                del scores, feats

            return out_ref

        # ----------------------------------------------------------------
        # Non-global path: pairwise scale merging
        # ----------------------------------------------------------------
        if len(x_pyramid) == 1:
            # Single scale path
            pooled = self.pool2(x_pyramid[0])
            _ = self.scoring_conv(pooled)
            return [pooled]

        # 1) Pool all scales
        pooled_1_list = [self.pool1(x) for x in x_pyramid]
        pooled_2_list = [self.pool2(x) for x in x_pyramid]

        # 2) Use the middle scale from pooled_2_list to define final_size
        mid_idx = len(x_pyramid) // 2
        final_size = pooled_2_list[mid_idx].shape[-2:]

        # 3) Build pairs: (pooled_1_list[i], pooled_2_list[i+1]) stepping by self.skip
        out_feats = []
        debug = [0] * len(x_pyramid)
        for i in range(0, len(x_pyramid) - 1, self.skip):
            a = F.interpolate(pooled_1_list[i], size=final_size, mode='bilinear', align_corners=False)
            b = F.interpolate(pooled_2_list[i + 1], size=final_size, mode='bilinear', align_corners=False)

            a = self.resizing_layers(a)
            b = self.resizing_layers(b)

            score_a = self.scoring_conv(a)
            score_b = self.scoring_conv(b)

            # Soft selection
            scores = torch.stack([score_a, score_b], dim=1)  # => [N, 2, 1, H', W']
            feats = torch.stack([a, b], dim=1)               # => [N, 2, C, H', W']

            score_softmax = F.softmax(scores, dim=1)
            selection = score_softmax.argmax(dim=1)
            selection_counts = torch.bincount(selection.flatten(), minlength=2)
            debug[i] += selection_counts[0].item()
            debug[i + 1] += selection_counts[1].item()

            del a, b, score_a, score_b

            out_feats.append(soft_selection(scores, feats))
            del scores, feats

        print(debug)

        return out_feats
    

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

class CHRESMAX_V3_blue(nn.Module):
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
            x_rescaled = pad_to_size_blue(x_rescaled, (img_hw, img_hw))
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

def chresmax_v3(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    model = CHRESMAX_V3_blue(**kwargs)
        
    return model

def load_model(model_name,
              root_dir='/oscar/data/tserre/xyu110/pytorch-output/train/0',
              device='cuda' if torch.cuda.is_available() else 'cpu'):
    model_class = model_name.split('.')[0]
    model_dir = '.'.join(model_name.split('.')[1:])
    checkpoint_url = "https://huggingface.co/cmulliken/chresmax_v3_cl_01/resolve/main/model_best.pth.tar"
    checkpoint = urlretrieve(checkpoint_url)[0]
    channel_size = int(model_name.split('ip_')[2].split('_')[1])
    ip = int(model_name.split('_ip_')[1].split('_')[0]) if '_ip_' in model_name else 3
    classifier_input_size = int(model_name.split('_c1[')[0].split('_')[-1])
    print(f"Loading model: {model_name} from {checkpoint_url} with channel size {channel_size}, ip {ip}, classifier input size {classifier_input_size}")
    if "chresmax" in model_name:
        checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)
        model = chresmax_v3(
            num_classes=1000, big_size=322, small_size=322, in_chans=3, 
            ip_scale_bands=ip, classifier_input_size=classifier_input_size, contrastive_loss=True, pyramid=False,
            bypass=True, main_route=False,
            c_scoring='v2'      
        ).to(device).eval()
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return model
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    

def get_model(name):
    assert name == 'chresmax_v3_cl_01'
    model = load_model('models_wo_aug.ip_3_chresmax_v3_blue_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6*3*1_]_bypass')
    print([name for name, _ in model.named_modules()])
    print([name for name, _ in model.named_modules()][7::7])
    preprocessing = functools.partial(load_preprocess_images, image_size=322)
    wrapper = PytorchWrapper(identifier='chresmax_v3_cl_01', model=model, preprocessing=preprocessing)
    wrapper.image_size = 322
    return wrapper

def get_layers(name):
    assert name == 'chresmax_v3_cl_01'
    return ["model_backbone.c1.resizing_layers.2","model_backbone.c2.scoring_conv.pool","model_backbone.c2b_seq.0","model_backbone.global_pool.pool1","model_backbone.s1.layer1.1","model_backbone.s1.layer1.2.bn2","model_backbone.s2.layer.1","model_backbone.s2b.s2b_seqs.0.0","model_backbone.s2b.s2b_seqs.1.0","model_backbone.s2b.s2b_seqs.1.2.conv1","model_backbone.s2b.s2b_seqs.2.1.conv1","model_backbone.s2b.s2b_seqs.2.3.conv1","model_backbone.s2b.s2b_seqs.3.1.conv1","model_backbone.s2b.s2b_seqs.3.3.conv1","model_backbone.s3.layer.0.bn1","model_backbone.s3.layer.2.conv2"]

def get_bibtex(model_identifier):
    return """"""

if __name__ == '__main__':
    check_models.check_base_models(__name__)

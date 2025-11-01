import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import random

from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

# --------------------------------------------------
# HMAX / ALEXMAX base imports
# --------------------------------------------------
from .HMAX import (
    HMAX_from_Alexnet, 
    HMAX_from_Alexnet_bypass,
    get_ip_scales,
    pad_to_size
)
from .ALEXMAX import (
    S1, S2, S3, C, C_scoring, ConvScoring, soft_selection
)

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
        if not global_scale_pool:
            self.resizing_layers = nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size=resize_kernel_1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_channels, num_channels, kernel_size=resize_kernel_2, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            # For global scale pooling, we don't need resizing layers
            self.resizing_layers = None

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
            
            # import pdb; pdb.set_trace()

            # We'll iteratively soft-select from out_list[0] through out_list[-1]
            out_ref = out_list[0]
            final_size = out_list[len(out_list)//2].shape[-2:]  # pick a reference size

            for i in range(1, len(out_list)):
                tmp = F.interpolate(out_list[i], size=final_size, mode='bilinear', align_corners=False)
                # tmp = self.resizing_layers(tmp)
                
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
            
        # this was for debugging purposes
        # print(debug)

        return out_feats

# --------------------------------------------------
# ALEXMAX_v3_3_optimized
# --------------------------------------------------
class ALEXMAX_v3_3_optimized(nn.Module):
    """
    Optimized version of ALEXMAX_v3_3 that uses C_scoring2_optimized
    for faster (and more memory-efficient) pyramid processing.
    """
    def __init__(self, 
                 num_classes=1000,
                 base_size=322, 
                 in_chans=3, 
                 ip_scale_bands=1,
                 classifier_input_size=13312, 
                 contrastive_loss=False,
                 pyramid=False, 
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.base_size = base_size

        # S1
        self.s1 = S1(kernel_size=11, stride=4, padding=0)

        # C1 using optimized layer
        self.c1 = C_scoring2_optimized(
            num_channels=96,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            skip=1,
            global_scale_pool=False
        )
        # S2
        self.s2 = S2(kernel_size=3, stride=1, padding=2)
        
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
        # S3
        self.s3 = S3()

        # If ip_scale_bands > 4, use a multi-scale final C_scoring2_optimized
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
            # Else use a single-scale global pool
            self.global_pool = C(global_scale_pool=True)

        # Fully-connected layers
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_size, 4096),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(4096, num_classes)

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

    def forward(self, x, pyramid=False, main_route=False):
        """
        x: [N, C, H, W]
        pyramid (bool): if True, return c1[0], c2[0] for external usage
        main_route (bool): if True, only build 2-scale pyramid; else ip_scale_bands scales
        """
        # Decide how many scales to build
        if main_route:
            # 2 total scales => effectively 1 band
            out = self.make_ip(x, 2)
        else:
            out = self.make_ip(x, self.ip_scale_bands)

        # S1
        out = self.s1(out)
        # C1
        out_c1 = self.c1(out)
        # S2
        out = self.s2(out_c1)
        # C2
        out_c2 = self.c2(out)

        if pyramid:
            # Return only the first scale's features
            return out_c1[0], out_c2[0]

        # S3
        out = self.s3(out_c2)

        # Global or multi-scale pooling
        out = self.global_pool(out)

        if isinstance(out, list):
            # If multi-scale, pick first
            out = out[0]

        # Flatten
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            # For contrastive usage, return (final out, c1[0], c2[0])
            return out, out_c1[0], out_c2[0]

        return out

# --------------------------------------------------
# Example of a "CHALEXMAX_V3_3" using the optimized backbone
# --------------------------------------------------
class CHALEXMAX_V3_3_optimized(nn.Module):
    """
    Example student-teacher style model with scale-consistency loss,
    using ALEXMAX_v3_3_optimized as the backbone.
    """
    def __init__(self, 
                 num_classes=1000,
                 in_chans=3,
                 ip_scale_bands=1,
                 classifier_input_size=13312,
                 contrastive_loss=True,
                 **kwargs):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.ip_scale_bands = ip_scale_bands
        
        # Use the optimized backbone
        self.model_backbone = ALEXMAX_v3_3_optimized(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss
        )

    def forward(self, x):
        """
        Creates two streams (original + random-scaled) for scale-consistency training.
        Returns:
            (output_of_stream1, correct_scale_loss)
        """
        # stream 1 (original scale)
        stream_1_output, stream_1_c1_feats, stream_1_c2_feats = self.model_backbone(x)

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
        stream_2_output, stream_2_c1_feats, stream_2_c2_feats = self.model_backbone(x_rescaled)

        # scale-consistency loss
        c1_correct_scale_loss = torch.mean(torch.abs(stream_1_c1_feats - stream_2_c1_feats))
        c2_correct_scale_loss = torch.mean(torch.abs(stream_1_c2_feats - stream_2_c2_feats))
        out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))
        correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_correct_scale_loss

        return stream_1_output, correct_scale_loss

# --------------------------------------------------
# Register models
# --------------------------------------------------
@register_model
def alexmax_v3_3_optimized(pretrained=False, **kwargs):
    """
    Registry function to create an ALEXMAX_v3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    if pretrained:
        raise ValueError("No pretrained weights available for ALEXMAX_v3_3_optimized.")
    model = ALEXMAX_v3_3_optimized(**kwargs)
    return model

@register_model
def chalexmax_v3_3_optimized(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    if pretrained:
        raise ValueError("No pretrained weights available for CHALEXMAX_V3_3_optimized.")
    model = CHALEXMAX_V3_3_optimized(**kwargs)
    return model

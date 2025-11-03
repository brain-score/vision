"""
HMAX3 Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import random

from ._registry import register_model

def pad_to_size(a, size, mode='constant'):
    """Pad tensor to specified size."""
    current_size = (a.shape[-2], a.shape[-1])
    total_pad_h = size[0] - current_size[0]
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top

    total_pad_w = size[1] - current_size[1]
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    a = nn.functional.pad(a, (pad_left, pad_right, pad_top, pad_bottom), mode=mode, value=0)
    return a


def get_ip_scales(num_scale_bands, base_image_size, scale=4):
    """
    General get ip scales function (cleaner code)
    scale: the denominator of the exponent. ie. if you want to scale by 2^(1/4), this should be 4
    """
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


class ConvScoring(nn.Module):
    """Convolutional scoring layer for feature map selection."""
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
        x_pooled = x_pooled.view(B, K)  # Shape: [B, K]
        # Apply linear layer
        scores = self.fc(x_pooled)  # Shape: [B, 1]
        scores = scores.squeeze(1)  # Shape: [B]
        return scores


class C(nn.Module):
    """
    Spatial then Scale pooling layer.
    Basic C layer implementation from ALEXMAX.
    """
    def __init__(self,
                 pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                 pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                 global_scale_pool=False):
        super(C, self).__init__()
        self.pool1 = pool_func1
        self.pool2 = pool_func2
        self.global_scale_pool = global_scale_pool

    def forward(self, x_pyramid):
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

        else:  # not global pool
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
                    x_2 = F.interpolate(x_2, size=x_1.shape[-2:], mode='bilinear')
                else:
                    x_1 = F.interpolate(x_1, size=x_2.shape[-2:], mode='bilinear')
                x = torch.stack([x_1, x_2], dim=4)

                to_append, _ = torch.max(x, dim=4)
                out.append(to_append)
        return out


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
                # Just run scoring_conv so it's in the graph
                _ = self.scoring_conv(pooled)
                return pooled

            # Gather a list of pooled outputs
            out_list = [self.pool1(x) for x in x_pyramid]

            # We'll iteratively soft-select from out_list[0] through out_list[-1]
            out_ref = out_list[0]
            final_size = out_list[len(out_list)//2].shape[-2:]  # pick a reference size

            for i in range(1, len(out_list)):
                tmp = F.interpolate(out_list[i], size=final_size, mode='bilinear', align_corners=False)
                
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


class Residual(nn.Module):
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




class S2b_Res(nn.Module):
    """S2b bypass layer with residual blocks."""
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


class Resmax(nn.Module):
    """
    Resmax, the backbone model used by HMAX3.
    """
    def __init__(self, num_classes=1000, in_chans=3, ip_scale_bands=1,
                 classifier_input_size=18432, contrastive_loss=False,
                 bypass=False, 
                 **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.bypass = bypass
        super(Resmax, self).__init__()

        self.s1 = nn.Sequential(
            Residual(3, 48, strides=2),
            Residual(48, 48),
            Residual(48, 96, strides=2)
        )

        # C1 using optimized layer
        self.c1 = C_scoring2_optimized(
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
            self.c2b_seq = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((6, 6))
            )
            self.c2b_score = C_scoring2_optimized(
                num_channels=1024,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                global_scale_pool=True
            )

        self.s3 = nn.Sequential(
            Residual(256, 256),
            Residual(256, 384),
            Residual(384, 384),
            Residual(384, 256)
        )
        
        
        if self.ip_scale_bands > 2:
            self.global_pool = C_scoring2_optimized(
                num_channels=256,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=6, stride=3, padding=1),
                resize_kernel_1=3,
                resize_kernel_2=1,
                skip=2,
                global_scale_pool=True
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
            # Use C scoring to smartly choose the best band
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


class CH_2_streams_training_eval_sep(nn.Module):
    """
    Generic 2-stream contrastive learning model with bypass architecture.
    Uses configurable backbone for different model variants.
    During training: returns stream_2_output (scaled/augmented) for backpropagation
    During evaluation: returns stream_1_output (original) for clean evaluation
    
    This is the wrapper class used by HMAX3.
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
            print("self.training is ", self.training, "stream_1_bool is ", self.stream_1_bool)
            print("HERRRRRE, Korean exp goes to right place")
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


@register_model
def hmax3(pretrained=False, **kwargs):
    """
    Creates a HMAX3 model instance.
    This function replicates the behavior from the original RESMAX.py file.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        **kwargs: Additional keyword arguments for model configuration
        
    Returns:
        A HMAX3 model instance
    """
    # Remove keys that might interfere with model creation
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    # Print configuration for debugging
    for key, val in kwargs.items():
        print(key, val)

    # Create the backbone model (Resmax)
    model_backbone = Resmax(contrastive_loss=True, **kwargs)
    
    # Wrap it with the 2-stream training/evaluation wrapper
    model = CH_2_streams_training_eval_sep(
        model_backbone=model_backbone, 
        bypass_only_model_bool=False, 
        **kwargs
    )
    
    return model


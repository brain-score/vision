""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer, original copyright/license info below

S3 (AutoFormerV2, https://arxiv.org/abs/2111.14725) Swin weights from
    - https://github.com/microsoft/Cream/tree/main/AutoFormerV2

Modifications and additions for timm hacked together by / Copyright 2021, Ross Wightman
"""
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import logging
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import PatchEmbed, Mlp, DropPath, ClassifierHead, to_2tuple, to_ntuple, trunc_normal_, \
    _assert, use_fused_attn, resize_rel_pos_bias_table, resample_patch_embed
from ._builder import build_model_with_cfg
from ._features_fx import register_notrace_function
from ._manipulate import checkpoint_seq, named_apply
from ._registry import generate_default_cfgs, register_model, register_model_deprecations
from .vision_transformer import get_init_weights_vit
from .vision_transformer_nca import NCA

__all__ = ['SwinTransformer']  # model_registry will add each entrypoint fn to this

_logger = logging.getLogger(__name__)

_int_or_tuple_2_t = Union[int, Tuple[int, int]]


def window_partition(
        x: torch.Tensor,
        window_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size: Tuple[int, int], H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def get_relative_position_index(win_h: int, win_w: int):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww


def sigmoid_temp(x, temp = 0.1):
    return torch.sigmoid(x / temp)

class BinaryizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports shifted and non-shifted windows.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            qkv_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            noisy_attn = False,
            binary_attn = False,
    ):
        """
        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            head_dim: Number of channels per head (dim // num_heads if not set)
            window_size: The height and width of the window.
            qkv_bias:  If True, add a learnable bias to query, key, value.
            attn_drop: Dropout ratio of attention weight.
            proj_drop: Dropout ratio of output.
        """
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn(experimental=True)  # NOTE not tested for prime-time yet

        self.noisy_attn = noisy_attn
        self.binary_attn = binary_attn
        if(self.binary_attn):
            self.binarize_func = BinaryizeSTE.apply
            self.binary_aggr = PerceptionAggr(dim, num_scales=2 * num_heads, multi_head_combine=False, head_num=1, scale_weight=True)
        

        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(win_h, win_w), persistent=False)

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) # 3, B, Nhead, N_tokens, C
        q, k, v = qkv.unbind(0)

        if False:
            attn_mask = self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                mask = mask.view(1, num_win, 1, N, N).expand(B_ // num_win, -1, self.num_heads, -1, -1)
                attn_mask = attn_mask + mask.reshape(-1, self.num_heads, N, N)
            if(self.noisy_attn):
                k = torch.mean(k, dim = -1, keepdim=True) * torch.randn_like(k) * 0.1 + k
                v = torch.mean(v, dim = -1, keepdim=True) * torch.randn_like(v) * 0.1 + v
            if(self.binary_attn):
                k_bin = F.normalize(self.binarize_func(k), dim = -1)
                v_bin = F.normalize(self.binarize_func(v), dim = -1)
                x_bin_attn = torch.nn.functional.scaled_dot_product_attention(
                    q, k_bin, v_bin,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
                H = int(N ** 0.5)
                W = N // H
                if(H * W != N):
                    print("In attention, Token Number Cannot Resize to an Image", H, W, N)
                    exit()
                x_weight_input = x.transpose(1, 2).reshape(B_, C, H, W)
                attn_combine_weight = self.binary_aggr(x_weight_input) # B, 1, H, W, 2
                attn_combine_weight = attn_combine_weight.reshape(B_, 1, H*W, 2).permute(0, 2, 1, 3).contiguous()


            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            if(self.binary_attn):
                attn_res = torch.stack([x, x_bin_attn], dim = -1)
                x = torch.sum(attn_res * attn_combine_weight, dim = -1)
        else:
            q = q * self.scale
            
            if(self.noisy_attn):
                k = torch.mean(k, dim = -1, keepdim=True) * torch.randn_like(k) * 0.1 + k
                v = torch.mean(v, dim = -1, keepdim=True) * torch.randn_like(v) * 0.1 + v
            if(self.binary_attn):
                k_bin = F.normalize(self.binarize_func(k), dim = -1)
                v_bin = F.normalize(self.binarize_func(v), dim = -1)
                
                attn_bin = q @ k_bin.transpose(-2, -1)
                attn_bin = attn_bin + self._get_rel_pos_bias()
                if mask is not None:
                    num_win = mask.shape[0]
                    attn_bin = attn_bin.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                    attn_bin = attn_bin.view(-1, self.num_heads, N, N)
                
                attn_bin = self.softmax(attn_bin)
                attn_bin = self.attn_drop(attn_bin)
                x_bin_attn = attn_bin @ v_bin
                
                H = int(N ** 0.5)
                W = N // H
                if(H * W != N):
                    print("In attention, Token Number Cannot Resize to an Image", H, W, N)
                    exit()
                x_weight_input = x.transpose(1, 2).reshape(B_, C, H, W)
                attn_combine_weight = self.binary_aggr(x_weight_input) # B, 1, H, W, 2*head
                attn_combine_weight = attn_combine_weight.reshape(B_, 1, H*W, 2*self.num_heads).reshape(B_, 1, H*W, self.num_heads, 2).permute(0, 3, 2, 1, 4).contiguous() # B, H, N, 1, 2
            
            attn = q @ k.transpose(-2, -1)
            attn = attn + self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = attn @ v
            
            if(self.binary_attn):
                attn_res = torch.stack([x, x_bin_attn], dim = -1) # B, H, N, C, 2
                x = torch.sum(attn_res * attn_combine_weight, dim = -1)

        x = x.transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PerceptionAggr(nn.Module):
    def __init__(self, dim, num_scales, multi_head_combine = False, head_num = 1, scale_weight = False):
        super(PerceptionAggr, self).__init__()
        
        self.multi_head_combine = multi_head_combine
        self.head_num = head_num
        self.scale_weight = scale_weight
        self.num_scales = num_scales
        if(multi_head_combine):
            assert head_num is not None
        self.proj = nn.Sequential(
            nn.Conv2d(dim, num_scales*head_num, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups = head_num, bias=False),
            nn.BatchNorm2d(num_scales*head_num),
            nn.Conv2d(num_scales*head_num, num_scales*head_num, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups = head_num, bias=False)
        )

    def forward(self, x):
        x = self.proj(x)
        if(self.multi_head_combine):
            # x: B, K*head, H, W
            B,K,H,W = x.shape
            x = x.reshape(B, self.head_num, K//self.head_num, H, W)
            x = x.permute(0, 1, 3, 4, 2) # B, head, H, W, K
            x = x.unsqueeze(2) # B, head, 1 H, W, K
            x = x.softmax(dim=-1)
        else:
            x = x.permute(0, 2, 3, 1) # B,H,W,K
            x = x.unsqueeze(1)
            x = x.softmax(dim=-1)
        if(self.scale_weight):
            x = x * self.num_scales
        return x
    
class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    """

    def __init__(
            self,
            dim: int,
            input_resolution: _int_or_tuple_2_t,
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            shift_size: int = 0,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            nca_local_perception_model = None,
            nca_local_perception_loop = 1,
            noisy_attn = False,
            binary_attn = False,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            window_size: Window size.
            num_heads: Number of attention heads.
            head_dim: Enforce the number of channels per head
            shift_size: Shift size for SW-MSA.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        ws, ss = self._calc_window_shift(window_size, shift_size)
        self.window_size: Tuple[int, int] = ws
        self.shift_size: Tuple[int, int] = ss
        self.window_area = self.window_size[0] * self.window_size[1]
        self.mlp_ratio = mlp_ratio
        
        self.nca_local_perception_model = nca_local_perception_model
        self.nca_local_perception_loop = nca_local_perception_loop
        if(self.nca_local_perception_model is not None):
            #print("additional flops: ", self.nca_local_perception_model.flops() / 1e9)
            self.perception_aggr_model = PerceptionAggr(dim, num_scales=2, multi_head_combine=False, head_num=1, scale_weight = True)
            
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            noisy_attn = noisy_attn,
            binary_attn = binary_attn,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if any(self.shift_size):
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            H = math.ceil(H / self.window_size[0]) * self.window_size[0]
            W = math.ceil(W / self.window_size[1]) * self.window_size[1]
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None)):
                for w in (
                        slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def _calc_window_shift(self, target_window_size, target_shift_size) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        target_window_size = to_2tuple(target_window_size)
        target_shift_size = to_2tuple(target_shift_size)
        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
        return tuple(window_size), tuple(shift_size)

    def _attn(self, x):
        B, H, W, C = x.shape

        # cyclic shift
        has_shift = any(self.shift_size)
        if has_shift:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # pad for resolution not divisible by window size
        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C
        shifted_x = shifted_x[:, :H, :W, :].contiguous()

        # reverse cyclic shift
        if has_shift:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        else:
            x = shifted_x
        return x

    def forward(self, x):
        B, H, W, C = x.shape
        mes_time = False
        if(mes_time):
            start = time.time()
        if(self.nca_local_perception_model is not None):
            x_norm = self.norm1(x)
            
            local_perception = x_norm.permute(0,3,1,2)
            if(isinstance(self.nca_local_perception_loop, list)):
                if(self.training):
                    step = torch.randint(low = self.nca_local_perception_loop[0], high = self.nca_local_perception_loop[1] + 1, size = (1,)).item()
                else:
                    step = self.nca_local_perception_loop[1] # (self.nca_local_perception_loop[1] + self.nca_local_perception_loop[0]) // 2
            else:
                step = self.nca_local_perception_loop
            for _ in range(step):
                local_perception = self.nca_local_perception_model.perceive_multiscale(local_perception.contiguous())
            if(mes_time):
                print("Local perception:", time.time() - start)
                start = time.time()
            
            global_perception = self._attn(x_norm)
            
            global_perception = global_perception.permute(0,3,1,2)#.reshape(B, self.num_heads, C // self.num_heads, H, W)
            perception_result = torch.stack([local_perception, global_perception], dim = -1)
            if(mes_time):
                print("Global perception:", time.time() - start)
                start = time.time()
            
            x_weight = x.permute(0,3,1,2)
            weight = self.perception_aggr_model(x_weight) # B head 1 H W K
            
            
            perception_local_global = torch.sum(perception_result * weight, dim=-1) # B, head, C, H, W
            perception_local_global = perception_local_global.permute(0,2,3,1)
            if(mes_time):
                print("Aggr perception:", time.time() - start)
                start = time.time()
            
            x = x + self.drop_path1(perception_local_global)
            
        elif(self.nca_local_perception_model is None):
            x = x + self.drop_path1(self._attn(self.norm1(x.contiguous())))
        x = x.reshape(B, -1, C)
        x = x + self.drop_path2(self.mlp(self.norm2(x.contiguous())))
        x = x.reshape(B, H, W, C)
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """

    def __init__(
            self,
            dim: int,
            out_dim: Optional[int] = None,
            norm_layer: Callable = nn.LayerNorm,
    ):
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels (or 2 * dim if None)
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        _assert(H % 2 == 0, f"x height ({H}) is not even.")
        _assert(W % 2 == 0, f"x width ({W}) is not even.")
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinTransformerStage(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    """

    def __init__(
            self,
            dim: int,
            out_dim: int,
            input_resolution: Tuple[int, int],
            depth: int,
            downsample: bool = True,
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: Union[List[float], float] = 0.,
            norm_layer: Callable = nn.LayerNorm,
            nca_model = False,
            before_nca_norm = False,
            stochastic_update = 0.0,
            times = 1,
            energy_minimization = 0,
            weighted_scale_combine = 0,
            nca_local_perception = 0,
            nca_local_perception_loop = [1,1,1,1],
            nca_delay = 0,
            energy_point_wise = 0,
            noisy_attn = False,
            binary_attn = False,
            nca_expand = 4,
            init_with_grad_kernel = False,
            perception_norm = "None",
            ablation_unroll = False,
            ablation_wsum = False,
            ablation_msp = False,
            ablation_aggrnorm = "bn"
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            downsample: Downsample layer at the end of the layer.
            num_heads: Number of attention heads.
            head_dim: Channels per head (dim // num_heads if not set)
            window_size: Local window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        self.depth = depth
        self.grad_checkpointing = False
        window_size = to_2tuple(window_size)
        shift_size = tuple([w // 2 for w in window_size])

        # patch merging layer
        if downsample:
            self.downsample = PatchMerging(
                dim=dim,
                out_dim=out_dim,
                norm_layer=norm_layer,
            )
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()
        
        self.nca_model = nca_model # if times > 0 else 0
        self.nca_delay = nca_delay
        self.ablation_unroll = ablation_unroll
        print(self.nca_delay)
        if(self.nca_model):
            #print(drop_path)
            self.nca_drop_path = DropPath(drop_path[self.nca_delay]) if drop_path[self.nca_delay] > 0. else nn.Identity()
#             if(drop_path[self.nca_delay] > 0. and drop_path[self.nca_delay] < 0.1):
#                 self.nca_drop_path = DropPath(0.1)
            if(self.nca_model >= 30 or self.nca_model == 5):
                self.nca_drop_path = nn.Identity()
            self.nca_norm = norm_layer(out_dim) if before_nca_norm else nn.Identity()
            if(not ablation_unroll):
                perception_scales = [0,1]
                if(ablation_msp == 1):
                    print("Ablation MSP [0]")
                    perception_scales = [0]
                elif(ablation_msp == 2):
                    print("Ablation MSP [0,1,2]")
                    perception_scales = [0,1,2]
                if(ablation_wsum):
                    print("Ablation Wsum")
                if(ablation_aggrnorm != "bn"):
                    print("Ablation Aggr Norm: ", ablation_aggrnorm)
                self.nca = NCA(
                    dim = out_dim,
                    num_heads = num_heads,
                    act_layer=nn.GELU,
                    norm_layer=None,
                    separate_norm = False,
                    stochastic_update = stochastic_update,
                    times = times,
                    alive_channel = 0,
                    alive_threshold = 0.0,
                    trainable_kernel = True,
                    normalize_filter = False,
                    padding_mode = "constant",
                    multi_head_perception = True,
                    perception_scales = perception_scales,
                    pos_emb = None,
                    perception_aggr = "wsum" if ablation_wsum == 0 else "sum",
                    sigmoid_alive = False,
                    energy_minimization = energy_minimization,
                    low_rank_approx = False,
                    multi_head_nca = True,
                    ablation_nca = False,
                    correct_alive = 0,
                    energy_multi_head = False,
                    energy_coeff_init = 0.001,
                    random_energy_coeff = 0,
                    weighted_scale_combine = weighted_scale_combine,
                    input_size = self.output_resolution,
                    energy_coeff_point_wise = energy_point_wise,
                    expand = nca_expand,
                    init_with_grad_kernel = init_with_grad_kernel,
                    perception_norm = perception_norm,
                    ablation_aggrnorm = ablation_aggrnorm,
                )
                print("additional FLOPs:", self.nca.flops(verbose = True) / 1e9, self.nca_drop_path, stochastic_update)
            elif(ablation_unroll):
                print("Create Unrolled NCA")
                nca_list = []
                for i in range(times):
                    nca_list += [
                            NCA(
                        dim = out_dim,
                        num_heads = num_heads,
                        act_layer=nn.GELU,
                        norm_layer=None,
                        separate_norm = False,
                        stochastic_update = stochastic_update,
                        times = 1,
                        alive_channel = 0,
                        alive_threshold = 0.0,
                        trainable_kernel = True,
                        normalize_filter = False,
                        padding_mode = "constant",
                        multi_head_perception = True,
                        perception_scales = [0, 1],
                        pos_emb = None,
                        perception_aggr = "wsum",
                        sigmoid_alive = False,
                        energy_minimization = energy_minimization,
                        low_rank_approx = False,
                        multi_head_nca = True,
                        ablation_nca = False,
                        correct_alive = 0,
                        energy_multi_head = False,
                        energy_coeff_init = 0.001,
                        random_energy_coeff = 0,
                        weighted_scale_combine = weighted_scale_combine,
                        input_size = self.output_resolution,
                        energy_coeff_point_wise = energy_point_wise,
                        expand = nca_expand,
                        init_with_grad_kernel = init_with_grad_kernel,
                        perception_norm = perception_norm,
                    )
                    ]
                    print("additional FLOPs:", nca_list[-1].flops(verbose = True) / 1e9)
                self.nca_list = nn.ModuleList(nca_list)
                
        if(nca_local_perception):
            if(depth < nca_local_perception):
                self.nca_list = [NCA(
                    dim = out_dim,
                    num_heads = num_heads,
                    act_layer=nn.GELU,
                    norm_layer=None,
                    separate_norm = False,
                    stochastic_update = stochastic_update,
                    times = times,
                    alive_channel = 0,
                    alive_threshold = 0.0,
                    trainable_kernel = True,
                    normalize_filter = False,
                    padding_mode = "constant",
                    multi_head_perception = True,
                    perception_scales = [0, 1],
                    pos_emb = None,
                    perception_aggr = "wsum",
                    sigmoid_alive = False,
                    energy_minimization = energy_minimization,
                    low_rank_approx = False,
                    multi_head_nca = True,
                    ablation_nca = False,
                    correct_alive = 0,
                    energy_multi_head = False,
                    energy_coeff_init = 0.001,
                    random_energy_coeff = 0,
                    weighted_scale_combine = weighted_scale_combine,
                    input_size = input_resolution,
                    local_perception_only = True,
                )]
            else:
                num_nca = depth // nca_local_perception
                self.nca_list = []
                for _ in range(num_nca):
                    self.nca_list += [
                        NCA(
                            dim = out_dim,
                            num_heads = num_heads,
                            act_layer=nn.GELU,
                            norm_layer=None,
                            separate_norm = False,
                            stochastic_update = stochastic_update,
                            times = times,
                            alive_channel = 0,
                            alive_threshold = 0.0,
                            trainable_kernel = True,
                            normalize_filter = False,
                            padding_mode = "constant",
                            multi_head_perception = True,
                            perception_scales = [0, 1],
                            pos_emb = None,
                            perception_aggr = "wsum",
                            sigmoid_alive = False,
                            energy_minimization = energy_minimization,
                            low_rank_approx = False,
                            multi_head_nca = True,
                            ablation_nca = False,
                            correct_alive = 0,
                            energy_multi_head = False,
                            energy_coeff_init = 0.001,
                            random_energy_coeff = 0,
                            weighted_scale_combine = weighted_scale_combine,
                            input_size = input_resolution,
                            local_perception_only = True,
                            
                    )]
            print("NCA block num: ", len(self.nca_list))

        # build blocks
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock(
                dim=out_dim,
                input_resolution=self.output_resolution,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                nca_local_perception_model = None if not nca_local_perception else self.nca_list[i // nca_local_perception],
                nca_local_perception_loop = nca_local_perception_loop,
                noisy_attn = noisy_attn,
                binary_attn = binary_attn,
            )
            for i in range(depth)])

    def forward(self, x, return_extra = False):
        x = self.downsample(x)
        
#         if(self.nca_model):
#             B,H,W,C = x.shape
#             x = x.reshape(B,H*W,C)
#             nca_output, _ = self.nca(self.nca_norm(x.contiguous()).contiguous())
#             nca_output = self.nca_drop_path(nca_output)
#             x = x + nca_output
#             x = x.reshape(B,H,W,C)

        if(return_extra):
            nca_middle_dict = {}
            nca_idx = 0
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x.contiguous())
        else:
            for i, blk in enumerate(self.blocks):
                if(self.nca_model and i == self.nca_delay):
                    B,H,W,C = x.shape
                    x = x.reshape(B,H*W,C)
                    if(not self.ablation_unroll):
                        nca_output, nca_middle_output = self.nca(self.nca_norm(x.contiguous()).contiguous())
                    else:
                        nca_output = self.nca_norm(x.contiguous())
                        for nca in self.nca_list:
                            nca_output, _ = nca(nca_output)
                    nca_output = self.nca_drop_path(nca_output)
                    x = x + nca_output
                    if(return_extra):
                        nca_middle_state = nca_middle_output["middle_state"]
                        for idx, state in enumerate(nca_middle_state):
                            nca_middle_dict[f"nca_{nca_idx}.{idx}"] = state.detach().clone()
                        nca_idx += 1
                    x = x.reshape(B,H,W,C)
                x = blk(x.contiguous())
            # x = self.blocks(x.contiguous())
        if(return_extra):
            return x, nca_middle_dict
        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer

    A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    """

    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 224,
            patch_size: int = 4,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 96,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            embed_layer: Callable = PatchEmbed,
            norm_layer: Union[str, Callable] = nn.LayerNorm,
            weight_init: str = '',
            nca_model = False,
            before_nca_norm = False,
            stochastic_update = 0.0,
            times = [1,1,1,1],
            energy_minimization = 0,
            weighted_scale_combine = 0,
            nca_local_perception = 0,
            nca_local_perception_loop = [1,1,1,1],
            energy_point_wise = 0,
            noisy_attn = False,
            binary_attn = False,
            nca_expand = 4,
            init_with_grad_kernel = 0,
            perception_norm = "None",
            ablation_unroll = False,
            ablation_wsum = 0,
            ablation_msp = 0,
            ablation_aggrnorm = "bn",
            **kwargs,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            embed_dim: Patch embedding dimension.
            depths: Depth of each Swin Transformer layer.
            num_heads: Number of attention heads in different layers.
            head_dim: Dimension of self-attention heads.
            window_size: Window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            drop_rate: Dropout rate.
            attn_drop_rate (float): Attention dropout rate.
            drop_path_rate (float): Stochastic depth rate.
            embed_layer: Patch embedding layer.
            norm_layer (nn.Module): Normalization layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.output_fmt = 'NHWC'

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feature_info = []

        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        # split image into non-overlapping patches
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            norm_layer=norm_layer,
            output_fmt='NHWC',
        )
        self.patch_grid = self.patch_embed.grid_size

        # build layers
        head_dim = to_ntuple(self.num_layers)(head_dim)
        if not isinstance(window_size, (list, tuple)):
            window_size = to_ntuple(self.num_layers)(window_size)
        elif len(window_size) == 2:
            window_size = (window_size,) * self.num_layers
        assert len(window_size) == self.num_layers
        mlp_ratio = to_ntuple(self.num_layers)(mlp_ratio)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        layers = []
        in_dim = embed_dim[0]
        scale = 1
        if(noisy_attn):
            print("****Noisy Attention Model****")
        if(binary_attn):
            print("****Binary Attention Branch Model****")
        
        for i in range(self.num_layers):
            nca_on = nca_model if i > 0 else 0
            cur_time = times[0] if nca_model >= 10 else times[i]
            cur_stochastic_update = stochastic_update
            if(nca_model):
                if(nca_model == 1):
                    nca_delay = 0
                elif(nca_model == 2):
                    nca_delay = 0 if i != 2 else 4
                    cur_stochastic_update = stochastic_update# / 2.0 * (i-1)
                    #print(cur_stochastic_update)
                elif(nca_model == 3):
                    if(i == 0):
                        nca_on = 0
                        nca_delay = 0
                    if(i == 1):
                        nca_on = 1
                        nca_delay = 1
                    if(i == 2):
                        nca_on = 1
                        nca_delay = 4
                    if(i == 3):
                        nca_on = 1
                        nca_delay = 0
                elif(nca_model == 4):
                    if(i == 0):
                        nca_on = 0
                        nca_delay = 0
                    if(i == 1):
                        nca_on = 1
                        nca_delay = 0
                    if(i == 2):
                        nca_on = 1
                        nca_delay = 4
                    if(i == 3):
                        nca_on = 1
                        nca_delay = 0
                    
                elif(nca_model == 5):
                    if(i == 0):
                        nca_on = 0
                        nca_delay = 0
                    if(i == 1):
                        nca_on = nca_model
                        nca_delay = 0
                    if(i == 2):
                        nca_on = nca_model
                        nca_delay = 4
                    if(i == 3):
                        nca_on = nca_model
                        nca_delay = 0
                    cur_stochastic_update = dpr[i][nca_delay]
                    if(cur_stochastic_update < 0.1):
                        cur_stochastic_update = 0.1
                elif(nca_model == 6):
                    if(i == 0):
                        nca_on = 0
                        nca_delay = 0
                    if(i == 1):
                        nca_on = 1
                        nca_delay = 0
                    if(i == 2):
                        nca_on = 1
                        nca_delay = 4
                    if(i == 3):
                        nca_on = 1
                        nca_delay = 0
                        nca_expand = 8
                        perception_norm = "ln"
                elif(nca_model >= 10 and nca_model < 30):
                    insert_pos = nca_model - 10
                    cur_basic_depth = 0 if i == 0 else sum(depths[:i])
                    cur_max_depth = cur_basic_depth + depths[i]
                    if(insert_pos >= cur_basic_depth and insert_pos < cur_max_depth):
                        nca_on = nca_model
                        nca_delay = insert_pos - cur_basic_depth
                    else:
                        nca_on = 0
                        nca_delay = 0
                elif(nca_model >= 30):
                    insert_pos = nca_model - 30
                    cur_basic_depth = 0 if i == 0 else sum(depths[:i])
                    cur_max_depth = cur_basic_depth + depths[i]
                    if(insert_pos >= cur_basic_depth and insert_pos < cur_max_depth):
                        nca_on = nca_model
                        nca_delay = insert_pos - cur_basic_depth
                    else:
                        nca_on = 0
                        nca_delay = 0
            else:
                nca_delay = 0
            out_dim = embed_dim[i]
            layers += [SwinTransformerStage(
                dim=in_dim,
                out_dim=out_dim,
                input_resolution=(
                    self.patch_grid[0] // scale,
                    self.patch_grid[1] // scale
                ),
                depth=depths[i],
                downsample=i > 0,
                num_heads=num_heads[i],
                head_dim=head_dim[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio[i],
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                nca_model = nca_on,
                before_nca_norm = before_nca_norm,
                stochastic_update = cur_stochastic_update,
                times = cur_time,
                energy_minimization = energy_minimization,
                weighted_scale_combine = weighted_scale_combine,
                nca_local_perception = nca_local_perception,
                nca_local_perception_loop = nca_local_perception_loop[i],
                nca_delay = nca_delay,
                energy_point_wise = energy_point_wise,
                noisy_attn = noisy_attn,
                binary_attn = binary_attn,
                nca_expand = nca_expand,
                init_with_grad_kernel = init_with_grad_kernel,
                perception_norm = perception_norm,
                ablation_unroll = ablation_unroll,
                ablation_wsum = ablation_wsum,
                ablation_msp = ablation_msp,
                ablation_aggrnorm = ablation_aggrnorm,
            )]
            in_dim = out_dim
            if i > 0:
                scale *= 2
            self.feature_info += [dict(num_chs=out_dim, reduction=4 * scale, module=f'layers.{i}')]
        self.layers = nn.Sequential(*layers)

        self.norm = norm_layer(self.num_features)
        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            input_fmt=self.output_fmt,
        )
        if weight_init != 'skip':
            if(init_with_grad_kernel):
                exclude_name = ["perception_conv"]
            else:
                exclude_name = []
            self.init_weights(weight_init, exclude_name)

    @torch.jit.ignore
    def init_weights(self, mode='', exclude_name = []):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        named_apply(get_init_weights_vit(mode, head_bias=head_bias), self, exclude_name = exclude_name)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
            if 'energy_coeff' in n:
                nwd.add(n)
        return nwd

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^patch_embed',  # stem and embed
            blocks=r'^layers\.(\d+)' if coarse else [
                (r'^layers\.(\d+).downsample', (0,)),
                (r'^layers\.(\d+)\.\w+\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_features(self, x, return_extra = False):
        x = self.patch_embed(x)
        if(return_extra):
            nca_middle_dict = {}
            for i, layer in enumerate(self.layers):
                x, layer_middle_dict = layer(x, return_extra = True)
                for key in layer_middle_dict:
                    nca_middle_dict[f"layer{i}.{key}"] = layer_middle_dict[key]
        else:
            x = self.layers(x)
        x = self.norm(x)
        if(return_extra):
            return x, nca_middle_dict
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x, return_extra = False):
        x = self.forward_features(x, return_extra = return_extra)
        if(return_extra):
            x, nca_middle_dict = x
        x = self.forward_head(x)
        if(return_extra):
            return x, nca_middle_dict
        return x


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    old_weights = True
    if 'head.fc.weight' in state_dict:
        old_weights = False
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    for k, v in state_dict.items():
        if any([n in k for n in ('relative_position_index', 'attn_mask')]):
            continue  # skip buffers that should not be persistent

        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-2] != H or v.shape[-1] != W:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation='bicubic',
                    antialias=True,
                    verbose=True,
                )

        if k.endswith('relative_position_bias_table'):
            m = model.get_submodule(k[:-29])
            if v.shape != m.relative_position_bias_table.shape or m.window_size[0] != m.window_size[1]:
                v = resize_rel_pos_bias_table(
                    v,
                    new_window_size=m.window_size,
                    new_bias_shape=m.relative_position_bias_table.shape,
                )

        if old_weights:
            k = re.sub(r'layers.(\d+).downsample', lambda x: f'layers.{int(x.group(1)) + 1}.downsample', k)
            k = k.replace('head.', 'head.fc.')

        out_dict[k] = v
    return out_dict


def _create_swin_transformer(variant, pretrained=False, **kwargs):
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get('depths', (1, 1, 3, 1))))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        SwinTransformer, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs)

    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head.fc',
        'license': 'mit', **kwargs
    }


default_cfgs = generate_default_cfgs({
    'swin_small_patch4_window7_224.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22kto1k_finetune.pth', ),
    'swin_base_patch4_window7_224.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth',),
    'swin_base_patch4_window12_384.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    'swin_large_patch4_window7_224.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth',),
    'swin_large_patch4_window12_384.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),

    'swin_tiny_patch4_window7_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',),
    'swin_small_patch4_window7_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',),
    'swin_base_patch4_window7_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth',),
    'swin_base_patch4_window12_384.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),

    # tiny 22k pretrain is worse than 1k, so moved after (untagged priority is based on order)
    'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22kto1k_finetune.pth',),

    'swin_tiny_patch4_window7_224.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth',
        num_classes=21841),
    'swin_small_patch4_window7_224.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth',
        num_classes=21841),
    'swin_base_patch4_window7_224.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
        num_classes=21841),
    'swin_base_patch4_window12_384.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, num_classes=21841),
    'swin_large_patch4_window7_224.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth',
        num_classes=21841),
    'swin_large_patch4_window12_384.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, num_classes=21841),

    'swin_s3_tiny_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/s3_t-1d53f6a8.pth'),
    'swin_s3_small_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/s3_s-3bb4c69d.pth'),
    'swin_s3_base_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/s3_b-a1e95db4.pth'),
})


@register_model
def swin_tiny_patch4_window7_224(pretrained=False, **kwargs) -> SwinTransformer:
    """ Swin-T @ 224x224, trained ImageNet-1k
    """
    model_args = dict(patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer(
        'swin_tiny_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swin_small_patch4_window7_224(pretrained=False, **kwargs) -> SwinTransformer:
    """ Swin-S @ 224x224
    """
    model_args = dict(patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), drop_path_rate = 0.3)
    return _create_swin_transformer(
        'swin_small_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swin_base_patch4_window7_224(pretrained=False, **kwargs) -> SwinTransformer:
    """ Swin-B @ 224x224
    """
    model_args = dict(patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), drop_path_rate = 0.5)
    return _create_swin_transformer(
        'swin_base_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swin_base_patch4_window12_384(pretrained=False, **kwargs) -> SwinTransformer:
    """ Swin-B @ 384x384
    """
    model_args = dict(patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    return _create_swin_transformer(
        'swin_base_patch4_window12_384', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swin_large_patch4_window7_224(pretrained=False, **kwargs) -> SwinTransformer:
    """ Swin-L @ 224x224
    """
    model_args = dict(patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    return _create_swin_transformer(
        'swin_large_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swin_large_patch4_window12_384(pretrained=False, **kwargs) -> SwinTransformer:
    """ Swin-L @ 384x384
    """
    model_args = dict(patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    return _create_swin_transformer(
        'swin_large_patch4_window12_384', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swin_s3_tiny_224(pretrained=False, **kwargs) -> SwinTransformer:
    """ Swin-S3-T @ 224x224, https://arxiv.org/abs/2111.14725
    """
    model_args = dict(
        patch_size=4, window_size=(7, 7, 14, 7), embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer('swin_s3_tiny_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swin_s3_small_224(pretrained=False, **kwargs) -> SwinTransformer:
    """ Swin-S3-S @ 224x224, https://arxiv.org/abs/2111.14725
    """
    model_args = dict(
        patch_size=4, window_size=(14, 14, 14, 7), embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer('swin_s3_small_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def swin_s3_base_224(pretrained=False, **kwargs) -> SwinTransformer:
    """ Swin-S3-B @ 224x224, https://arxiv.org/abs/2111.14725
    """
    model_args = dict(
        patch_size=4, window_size=(7, 7, 14, 7), embed_dim=96, depths=(2, 2, 30, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer('swin_s3_base_224', pretrained=pretrained, **dict(model_args, **kwargs))


register_model_deprecations(__name__, {
    'swin_base_patch4_window7_224_in22k': 'swin_base_patch4_window7_224.ms_in22k',
    'swin_base_patch4_window12_384_in22k': 'swin_base_patch4_window12_384.ms_in22k',
    'swin_large_patch4_window7_224_in22k': 'swin_large_patch4_window7_224.ms_in22k',
    'swin_large_patch4_window12_384_in22k': 'swin_large_patch4_window12_384.ms_in22k',
})

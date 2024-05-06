""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013

The official jax code is released and available at
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision

Acknowledgments:
  * The paper authors for releasing code and weights, thanks!
  * I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch
  * Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
  * Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""
import logging
import math
from collections import OrderedDict
from functools import partial
import time
from typing import Callable, List, Optional, Sequence, Tuple, Union
from scipy import signal
import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, PatchEmbedOverLap, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, _assert, to_2tuple, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn
from ._builder import build_model_with_cfg
from ._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from ._registry import generate_default_cfgs, register_model, register_model_deprecations
from ._features_fx import register_notrace_function
from .vonenet.vonenet import VOneNet

__all__ = ['NCAFormer']  # model_registry will add each entrypoint fn to this


_logger = logging.getLogger(__name__)

def neighbourhood_filters(neighbourhood_size, dilation, device):
    height, width = neighbourhood_size
    impulses = []
    for i in range(height):
        for j in range(width):
            if(i % dilation == 0 and j % dilation == 0):
                impulse = signal.unit_impulse((height, width), idx=(i,j), dtype=np.float32)
                impulses.append(impulse)
    filters = torch.tensor(np.stack(impulses), device=device)
    return filters

class LocalizeAttention(torch.nn.Module):
    def __init__(self, attn_neighbourhood_size = [3,3], dilation = 1, device="cuda") -> None:
        super().__init__()
        """
        Fetch the neighborhood for each input. 
        The input should be of shape b, h, n, d where h is head number and n is token number
        It will fetch the 2D neighbors with size specified by attn_neighbourhood_size, reshape it to 1D
        Therefore, the result will be of shape b, h, n, #neighbors, d
        """
        self.attn_neighbourhood_size = attn_neighbourhood_size
        self.device = device
        self.attn_filters = neighbourhood_filters(self.attn_neighbourhood_size, dilation, self.device)

    def forward(self, x, height, width):
        '''attn_filters: [filter_n, h, w]'''
        b, h, _, d = x.shape
        y = rearrange(x, 'b h (i j) d -> (b h d) 1 i j', i=height, j=width)
        y = F.conv2d(y, self.attn_filters[:, None], padding='same')
        _x = rearrange(y, '(b h d) filter_n i j -> b h (i j) filter_n d', b=b, h=h, d=d)
        return _x

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            alive_channel = 0,
            sigmoid_alive = False,
            alive_threshold = 0.1,
            localize = False,
            attn_neighbourhood_size = [3, 3],
            dilation = 1,
            correct_alive = 0,
            paas = 0,
            v2 = 0,
            cosine_attn = 0,
            global_filter = 0,
            relative_pos_emb = 0,
            img_size = (56, 56),
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.pytorch_version = torch.__version__.split("+")[0]

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.alive_mask = False
        self.correct_alive = correct_alive
        if(alive_threshold > 0): #  and correct_alive == 0
            self.alive_mask = True
            self.alive_channel = alive_channel
            self.alive_threshold = alive_threshold

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Sequential(*[
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
        ])
        
        #self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.alive_func = nn.Sigmoid() if sigmoid_alive else nn.Identity()

        self.localize = localize
        if(localize):
            self.localize_func = LocalizeAttention(attn_neighbourhood_size = attn_neighbourhood_size, dilation = dilation)
        
        self.paas = paas
        if(paas > 0):
            self.att_mask = nn.Parameter(torch.zeros(self.num_heads, paas, paas))
        
        self.v2 = v2
        self.cosine_attn = cosine_attn
        if(v2 or cosine_attn):
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
            self.softmax = nn.Softmax(dim=-1)
        
        self.relative_pos_emb = relative_pos_emb
        if(relative_pos_emb):
            height, width = img_size
            self.pos_enc = nn.Parameter(torch.Tensor(self.num_heads, (2 * height - 1) * (2 * width - 1)))
            trunc_normal_(self.pos_enc, std=.02)
            self.register_buffer("relative_indices", self.get_indices(height, width))
    
    @staticmethod
    def get_indices(h, w):
        y = torch.arange(h, dtype=torch.long)
        x = torch.arange(w, dtype=torch.long)
        
        y1, x1, y2, x2 = torch.meshgrid(y, x, y, x, indexing='ij')
        indices = (y1 - y2 + h - 1) * (2 * w - 1) + x1 - x2 + w - 1
        indices = indices.flatten()
        
        return indices
    
    def alive(self, x):
        b_x, n_tokens, c_x = x.shape
        H = int(n_tokens ** 0.5)
        W = n_tokens // H
        if(H * W != n_tokens):
            print("In attention alive mask compute, Token Number Cannot Resize to an Image", H, W, n_tokens)
            exit()
        x = x.transpose(1, 2).reshape(b_x, c_x, H, W)
        alive_mask = F.max_pool2d(self.alive_func(x[:, self.alive_channel:self.alive_channel + 1, :, :]), kernel_size=3, stride=1, padding=1) > self.alive_threshold
        alive_mask = alive_mask.permute(0, 2, 3, 1).reshape(b_x, n_tokens)
        return alive_mask

    def forward(self, x):
        B, N, C = x.shape
        self.input_resolution = (N, C)
        h = int(N ** 0.5)
        w = N // h
        if(h * w != N):
            print("In attention alive mask compute, Token Number Cannot Resize to an Image", h,w,N)
            exit()
        if(self.alive_mask):
            if(self.correct_alive):
                x_alive_mask = (self.alive_func(x[:, :, self.alive_channel]) > self.alive_threshold).float()
                # x_alive_mask = self.alive(x)
            else:
                x_alive_mask = (x[:, :, self.alive_channel] > self.alive_threshold).float()
            # x_alive_mask = self.alive(x)
            if(self.localize):
                x_alive_mask = x_alive_mask.view(B, 1, N, 1)
            else:
                x_alive_mask = x_alive_mask.view(B, 1, 1, N)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # 3, B, N_head, N_token, C // N_head
        q, k, v = qkv.unbind(0) # B, N_head, N_token, C // N_head
        q, k = self.q_norm(q), self.k_norm(k)

#         if(not self.localize and self.fused_attn and self.paas == 0 and not self.v2 and not self.cosine_attn): # self.fused_attn and 
# #             with torch.cuda.amp.autocast(enabled = False):
# #                 q = q.float()
# #                 k = k.float()
# #                 v = v.float()
#             x = F.scaled_dot_product_attention(
#                 q, k, v,
#                 dropout_p=self.attn_drop.p if self.training else 0.,
#                 attn_mask = x_alive_mask.repeat(1,1,N,1) if self.alive_mask else None,
#             )
#         else:
        if(not self.localize):
            if(not self.v2 and not self.cosine_attn):
                if(self.paas == 0):
                    q = q * self.scale
                    attn = q @ k.transpose(-2, -1) # B, N_head, N_token, N_token
                    if(self.alive_mask):
                        attn = attn + x_alive_mask * -10000.0
                    if(self.relative_pos_emb):
                        indices = self.relative_indices.expand(self.num_heads, -1)
                        rel_pos_enc = self.pos_enc.gather(-1, indices)
                        rel_pos_enc = rel_pos_enc.unflatten(-1, (h * w, h * w))

                        attn = attn + rel_pos_enc
                elif(self.paas > 0):
                    with torch.cuda.amp.autocast(enabled = False):
                        q = q.float()
                        k = k.float()
                        q = q * self.scale
                        attn = q @ k.transpose(-2, -1) # B, N_head, N_token, N_token
                        if(self.alive_mask):
                            attn = attn + x_alive_mask * -10000.0
                        attn = attn * torch.sigmoid(2.0*self.att_mask).expand(B, -1, -1, -1)
                with torch.cuda.amp.autocast(enabled = False):
                    attn = attn.softmax(dim=-1, dtype = torch.float32)
                attn = self.attn_drop(attn)

                x = attn @ v # B, N_head, N_token, C


            elif(self.v2 or self.cosine_attn):
                attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
                with torch.cuda.amp.autocast(enabled = False):
                    logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
                    attn = attn * logit_scale
                attn = self.softmax(attn)
                attn = self.attn_drop(attn)
                x = attn @ v # B, N_head, N_token, C
        else:
            # q,k,v : b,h,n,d
            q = rearrange(q, 'b h n d -> b h n 1 d')
            k = self.localize_func(k, h, w)  # b h n (neighbor_h neighbor_w) d
            v = self.localize_func(v, h, w)  # b h n (neighbor_h neighbor_w) d

            attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale # b h n 1 (neighbor_h neighbor_w)

            if(self.alive_mask):
                x_alive_mask_local = self.localize_func(x_alive_mask, h, w) # b, 1, n, (neighbor_h neighbor_w) 1
                attn = attn + x_alive_mask_local.transpose(3,4) * -10000.0

            attn = attn.softmax(dim=-1)
            x = torch.matmul(attn, v)  # b h n 1 d
            x = x.squeeze(3)
                
        x = x.transpose(1, 2).reshape(B, N, C)
        h = int(N ** 0.5)
        w = N // h
        x = x.reshape(B,h,w,C).permute(0,3,1,2).contiguous()
        x = self.proj(x)
        x = x.permute(0,2,3,1).reshape(B,N,C).contiguous()
        x = self.proj_drop(x)
        """B, N_token, C"""
        return x

    def flops(self, verbose = False):
        flops = 0
        N, C = self.input_resolution
        # qkv
        flops += N * C * C * 3
        if(not self.localize):
            # attn = (q @ k.transpose(-2, -1))
            flops += self.num_heads * N * (C // self.num_heads) * N
            #  x = (attn @ v)
            flops += self.num_heads * N * N * (C // self.num_heads)
            # x = self.proj(x)
            flops += N * C * C
        elif(self.localize):
            # attn = (q @ k.transpose(-2, -1))
            flops += self.num_heads * N * (C // self.num_heads) * 9
            #  x = (attn @ v)
            flops += self.num_heads * N * 9 * (C // self.num_heads)
            # x = self.proj(x)
            flops += N * C * C
        if(verbose):
            print("Attention: ", flops / 1e9)
        return flops

def window_partition(x, window_size: Tuple[int, int]):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]):
    """
    Args:
        windows: (num_windows * B, window_size[0], window_size[1], C)
        window_size (Tuple[int, int]): Window size
        img_size (Tuple[int, int]): Image size

    Returns:
        x: (B, H, W, C)
    """
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
            self,
            dim,
            window_size,
            num_heads,
            qkv_bias=True,
            attn_drop=0.,
            proj_drop=0.,
            pretrained_window_size=[0, 0],
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([
            relative_coords_h,
            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / math.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.register_buffer('k_bias', torch.zeros(dim), persistent=False)
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        self.input_resolution = (N, C)
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    def flops(self, N):
        # calculate flops for 1 window with token length of N
        _, C = self.input_resolution
        flops = 0
        # qkv = self.qkv(x)
        flops += N * C * 3 * C
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (C // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (C // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class StackedWindowAttention(nn.Module):
    """ .
    """

    def __init__(
            self,
            dim,
            input_resolution,
            num_heads,
            window_size=7,
            shift_size=0,
            qkv_bias=True,
            proj_drop=0.,
            attn_drop=0.,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            num_heads: Number of attention heads.
            window_size: Window size.
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
        self.input_resolution = to_2tuple(input_resolution)
        self.num_heads = num_heads
        ws, ss = self._calc_window_shift(window_size, shift_size)
        self.window_size: Tuple[int, int] = ws
        self.shift_size: Tuple[int, int] = ss
        self.window_area = self.window_size[0] * self.window_size[1]

        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        if any(self.shift_size):
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
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

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, self.input_resolution)  # B H' W' C

        # reverse cyclic shift
        if has_shift:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        else:
            x = shifted_x
        return x

    def forward(self, x):
        B,N,C = x.shape
        b_x, n_tokens, c_x = x.shape
        
        H = int(n_tokens ** 0.5)
        W = n_tokens // H
        self.input_resolution_flops = (H, W, c_x)
        x = x.reshape(b_x, H, W, c_x)
        B, H, W, C = x.shape
        x = self._attn(x)
        x = x.reshape(B, -1, C)
        return x

    def flops(self, verbose = False):
        flops = 0
        H, W, C = self.input_resolution_flops
        # W-MSA/SW-MSA
        nW = H * W / self.window_size[0] / self.window_size[1]
        flops += nW * self.attn.flops(self.window_size[0] * self.window_size[1])
        return flops
    

class CrossAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim_q,
            dim_kv,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            alive_channel = 0,
            sigmoid_alive = False,
            alive_threshold = 0.1,
            localize = False,
            attn_neighbourhood_size = [3, 3],
            dilation = 1,
            correct_alive = 0,
            paas = 0,
    ):
        super().__init__()
        assert dim_q % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim_q // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        # self.qkv = nn.Linear(dim_q, dim_q * 3, bias=qkv_bias)
        self.q_map = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.kv_map = nn.Linear(dim_q, dim_q * 2, bias=qkv_bias)
        self.linear_map_kv_to_q = nn.Linear(dim_kv, dim_q, bias = False)

        self.alive_mask = False
        if(alive_threshold > 0): #  and correct_alive == 0
            self.alive_mask = True
            self.alive_channel = alive_channel
            self.alive_threshold = alive_threshold
            self.alive_func = nn.Sigmoid() if sigmoid_alive else nn.Identity()

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.alive_func = nn.Sigmoid() if sigmoid_alive else nn.Identity()
    
    def alive(self, x):
        b_x, n_tokens, c_x = x.shape
        H = int(n_tokens ** 0.5)
        W = n_tokens // H
        if(H * W != n_tokens):
            print("In attention alive mask compute, Token Number Cannot Resize to an Image", H, W, n_tokens)
            exit()
        x = x.transpose(1, 2).reshape(b_x, c_x, H, W)
        alive_mask = F.max_pool2d(self.alive_func(x[:, self.alive_channel:self.alive_channel + 1, :, :]), kernel_size=3, stride=1, padding=1) > self.alive_threshold
        alive_mask = alive_mask.permute(0, 2, 3, 1).reshape(b_x, n_tokens)
        return alive_mask

    def forward(self, high_x, low_x):
        B, N, C = high_x.shape
        self.input_resolution_high = (N, C)
        B_low, N_low, C_low = low_x.shape
        assert B == B_low
        self.input_resolution_low = (N_low, C_low)

        if(self.alive_mask):
            high_x_alive_mask = (self.alive_func(high_x[:, :, self.alive_channel]) > self.alive_threshold)
            low_x_alive_mask = (self.alive_func(low_x[:, :, self.alive_channel]) > self.alive_threshold)
            # Expand the aliveness tensors
            alive_X_expanded = high_x_alive_mask.unsqueeze(-1)  # B, N, 1
            alive_Y_expanded = low_x_alive_mask.unsqueeze(1)   # B, 1, N_low

            # Create the attention mask
            x_alive_mask = (alive_X_expanded & alive_Y_expanded).float()  # B, N, N_low
            x_alive_mask = x_alive_mask.view(B, 1, N, N_low)

        low_x = self.linear_map_kv_to_q(low_x)

        q = self.q_map(high_x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        kv = self.kv_map(low_x).reshape(B, N_low, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0) # B, N_head, N_low, C // N_head

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # 3, B, N_head, N_token, C // N_head
        # q, k, v = qkv.unbind(0) # B, N_head, N_token, C // N_head
        q, k = self.q_norm(q), self.k_norm(k)

        if(self.fused_attn): # self.fused_attn and 
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask = x_alive_mask if self.alive_mask else None,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1) # B, N_head, N_token, N_low
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v # B, N_head, N_token, C // N_head
                
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        """B, N_token, C"""
        return x

    def flops(self, verbose = False):
        flops = 0
        N, C = self.input_resolution_high
        N_low, C_low = self.input_resolution_low


        flops += N * C * C_low
        # qkv
        flops += N * C * C * 3
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (C // self.num_heads) * N_low
        #  x = (attn @ v)
        flops += self.num_heads * N_low * N_low * (C // self.num_heads)
        # x = self.proj(x)
        flops += N * C * C

        if(verbose):
            print("Cross Attention: ", flops / 1e9)
        return flops

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class CPE2D(nn.Module):
    """
    Cartesian Positional Encoding 2D
    """

    def __init__(self):
        super(CPE2D, self).__init__()
        self.cached_penc = None
        self.last_tensor_shape = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, ch, x, y)
        :return: Positional Encoding Matrix of size (batch_size, 2, x, y)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.last_tensor_shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, orig_ch, h, w = tensor.shape
        xs = torch.arange(h, device=tensor.device) / h
        ys = torch.arange(w, device=tensor.device) / w
        xs = 2.0 * (xs - 0.5 + 0.5 / h)
        ys = 2.0 * (ys - 0.5 + 0.5 / w)
        xs = xs[None, :, None]
        ys = ys[None, None, :]
        emb = torch.zeros((2, h, w), device=tensor.device).type(tensor.type())
        emb[:1] = xs
        emb[1: 2] = ys

        self.cached_penc = emb.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        self.last_tensor_shape = tensor.shape

        return self.cached_penc

class PerceptionAggr(nn.Module):
    def __init__(self, dim, num_scales, multi_head_combine = False, head_num = 1, scale_weight = False, norm = "bn"):
        super(PerceptionAggr, self).__init__()
        
        self.multi_head_combine = multi_head_combine
        self.scale_weight = scale_weight
        self.head_num = head_num
        self.norm = norm
        self.num_scales = num_scales
        if(multi_head_combine):
            assert head_num is not None
        if(self.norm == "bn"):
            self.proj = nn.Sequential(
                nn.Conv2d(dim, num_scales*head_num, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups = head_num, bias=False),
                nn.BatchNorm2d(num_scales*head_num),
                nn.Conv2d(num_scales*head_num, num_scales*head_num, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups = head_num, bias=False)
            )
        elif(self.norm == "ln"):
            self.w1 = nn.Conv2d(dim, num_scales*head_num, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups = head_num)
            self.norm_layer = nn.LayerNorm(num_scales*head_num)
            self.w2 = nn.Conv2d(num_scales*head_num, num_scales*head_num, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups = head_num, bias=False)
        elif(self.norm == "ln1"):
            self.w1 = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups = dim)
            self.norm_layer = nn.LayerNorm(dim)
            self.w2 = nn.Conv2d(dim, num_scales*head_num, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups = head_num, bias=False)

    def forward(self, x):
        if(self.norm == "bn"):
            x = self.proj(x)
        elif("ln" in self.norm):
            x = self.w1(x)
            x = x.permute(0,2,3,1).contiguous()
            x = self.norm_layer(x)
            x = x.permute(0,3,1,2).contiguous()
            x = self.w2(x)
            
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

class NCA(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            act_layer=nn.GELU,
            norm_layer=None,
            separate_norm = False,
            stochastic_update = 0.0,
            times = 1,
            alive_channel = 0,
            alive_threshold = 0.1,
            trainable_kernel = False,
            normalize_filter = False,
            padding_mode = "constant",
            multi_head_perception = False,
            perception_scales = [0],
            pos_emb = None,
            perception_aggr = "concat",
            sigmoid_alive = False,
            energy_minimization = False,
            low_rank_approx = False,
            multi_head_nca = False,
            ablation_nca = False,
            correct_alive = 0,
            energy_multi_head = 0,
            energy_coeff_init = 0.01,
            random_energy_coeff = 0,
            weighted_scale_combine = 0,
            input_size = (56, 56),
            local_perception_only = 0,
            energy_coeff_point_wise = 0,
            init_with_grad_kernel = 0,
            expand = 4,
            perception_norm = "None",
            ablation_aggrnorm = "bn",
    ):
        """
        Temporally refining tokens via cell perception and update in NCA scheme

        Input: Tokens, shape: Batch, N_tokens, Token_dim

        About alive mask:
        # use additional channel, or use existing channel
        # transfer of the alive mask
        # additional channel: intialization? Transfer to next layer? Preserve of alive and death in patch merging? 
        # accumulating halting prob? A-ViT paper
        # masked attention

        About Perception:
        # multi-head perception? 
        # concat perception? Too computationally demanding. Add perception: Too much noise. Weighted combination (add)?
        # Multi scale perception
        # MobileNetV2, depthwise+pointwise, inverted Residual

        About stochastic update:
        # add noise like SDE?

        Multi Layer NCA communication, high level feed back
        # one NCA in different layer, but it knows the stage 


        ***NCA as local to global attention***
        Replace QKV by temporal gradual attention
        Local QKV operation, iteratively generating the final tokens. This is only for replacing the attention layer. 
        """
        super().__init__()
        is_use_cuda = torch.cuda.is_available()
        if(is_use_cuda):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.num_heads = num_heads
        
        self.alive_func = nn.Sigmoid() if sigmoid_alive else nn.Identity()
        
        self.input_size = (*input_size, dim)
        self.energy_minimization = energy_minimization
        self.energy_multi_head = energy_multi_head
        self.random_energy_coeff = random_energy_coeff
        self.weighted_scale_combine = weighted_scale_combine
        self.multi_head_nca = multi_head_nca
        self.energy_coeff_point_wise = energy_coeff_point_wise
        self.perception_norm = perception_norm
        
        if(weighted_scale_combine and len(perception_scales) > 1):
            self.perception_scale_aggr_model = PerceptionAggr(dim, num_scales=len(perception_scales), multi_head_combine=False, head_num=1)
        if(energy_minimization and not local_perception_only):
            #init_value = np.log2(num_heads/3) + 1
            if(self.energy_coeff_point_wise):
                self.energy_coeff_model = PerceptionAggr(dim, num_scales=2, multi_head_combine=False, head_num=1, scale_weight = True)
            else:
                if(not self.energy_multi_head):
                    self.energy_coeff = nn.Parameter(torch.zeros(1) + energy_coeff_init)
                else:
                    self.energy_coeff = nn.Parameter(torch.zeros(num_heads) + energy_coeff_init)
        
        self.separate_norm = separate_norm
        self.norm_layer = None
        if(norm_layer):
            if(separate_norm):
                if(isinstance(times, list)):
                    self.norm_layer = [norm_layer(dim) for _ in range(times[1])]
                elif(isinstance(times, int)):
                    self.norm_layer = [norm_layer(dim) for _ in range(times)]
                self.norm_layer = nn.Sequential(*self.norm_layer)
            else:
                self.norm_layer = norm_layer(dim)
        
        self.pos_emb = pos_emb if pos_emb != "None" else None
        self.pos_emb_2d = None
        self.c_cond = 0
        if self.pos_emb == 'CPE':
            self.pos_emb_2d = CPE2D()
            self.c_cond += 2
        
        self.times = times

        self.alive_channel = alive_channel
        self.alive_threshold = alive_threshold
        self.alive_mask = alive_threshold > 0.0

        self.trainable_kernel = trainable_kernel

        self.multi_head_perception = multi_head_perception

        self.expand = expand

        self.padding_mode = padding_mode

        self.perception_scales = perception_scales

        self.perception_aggr = perception_aggr

        self.stochastic_update = stochastic_update

        self.low_rank_approx = low_rank_approx

        self.c_in = dim
        
        # if self.trainable_kernel:
        #     if(self.multi_head_perception):
        #         """
        #         Input: B,C,H,W. 
        #         K groups, so each group is of shape B, C // K, H, W. Here group is num_heads
        #         Perception in each of the K groups, such that in each group, the convolution kernel is the same for every channel so it is a group convolution where the group number equals C//K. 
        #         Across different groups, the convolution kernels are different. 
        #         Therefore, in total we have K perception kernels, each of size out, 1, 3, 3 with group convolution C//K groups.
        #         """
        #         self.groups = num_heads
        #         self.group_convs = nn.ModuleList()

        #         # Create a convolution layer for each group
        #         for _ in range(self.groups):
        #             self.group_convs.append(
        #                 nn.Conv2d(self.c_in // self.groups, self.c_in * self.expand // self.groups, 3, padding=1, groups=self.c_in // self.groups, padding_mode=padding_mode)
        #             )
        #     else:
        #         self.conv_layer = torch.nn.Conv2d(self.c_in, self.c_in * self.expand, 3, padding=1, groups=self.c_in, padding_mode=padding_mode)
        #         torch.nn.init.xavier_normal_(self.conv_layer.weight, gain=1.0)
        # else:
        sobel_filter_x = torch.FloatTensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).to(self.device)
        if(normalize_filter):
            sobel_filter_x = F.normalize(sobel_filter_x, p = 1)
        if(init_with_grad_kernel):
            sobel_filter_x = F.normalize(sobel_filter_x.reshape(1,9), p = 1).reshape(3,3)
        sobel_filter_y = sobel_filter_x.T

        sobel_filter_x = sobel_filter_x.reshape(1, 1, 3, 3)
        sobel_filter_y = sobel_filter_y.reshape(1, 1, 3, 3)

        identity_filter = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).to(self.device)
        identity_filter = identity_filter.reshape(1, 1, 3, 3)

        laplacian_filter = torch.FloatTensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]]).to(self.device)
        if(normalize_filter):
            laplacian_filter = F.normalize(laplacian_filter, p = 1)
        if(init_with_grad_kernel):
            laplacian_filter = F.normalize(laplacian_filter.reshape(1,9), p = 1).reshape(3,3)
        
        laplacian_filter = laplacian_filter.reshape(1, 1, 3, 3)

        if(not multi_head_perception):
            sobel_filter_x = sobel_filter_x.repeat(self.c_in, 1, 1, 1)
            sobel_filter_y = sobel_filter_y.repeat(self.c_in, 1, 1, 1)
            identity_filter = identity_filter.repeat(self.c_in, 1, 1, 1)
            laplacian_filter = laplacian_filter.repeat(self.c_in, 1, 1, 1)

            sobel_xs = []
            sobel_ys = []
            laplacians = []
            for scale in self.perception_scales:
                sobel_xs.append(sobel_filter_x / ((scale + 1) ** 1.618))
                sobel_ys.append(sobel_filter_y / ((scale + 1) ** 1.618))
                laplacians.append(laplacian_filter / ((scale + 1) ** 1.618))

            if(self.trainable_kernel):
                self.sobel_filter_x = torch.stack(sobel_xs)
                self.sobel_filter_y = torch.stack(sobel_ys)
                self.laplacian_filter = torch.stack(laplacians)

                self.sobel_filter_x.requires_grad_(True)
                self.sobel_filter_y.requires_grad_(True)
                self.laplacian_filter.requires_grad_(True)
                
                self.sobel_filter_x = nn.Parameter(self.sobel_filter_x)
                self.sobel_filter_y = nn.Parameter(self.sobel_filter_y)
                self.laplacian_filter = nn.Parameter(self.laplacian_filter)
            else:
                sobel_filter_x = torch.stack(sobel_xs)
                sobel_filter_y = torch.stack(sobel_ys)
                laplacian_filter = torch.stack(laplacians)
                self.register_buffer('sobel_filter_x', sobel_filter_x)
                self.register_buffer('sobel_filter_y', sobel_filter_y)
                self.register_buffer('laplacian_filter', laplacian_filter)
        else:
            self.filters = torch.stack([identity_filter, sobel_filter_x, sobel_filter_y, laplacian_filter])
            self.filters.to(self.device)

            # Compute the interpolated weights for each kernel
            """Do we need to interpolate identity filter???"""
            self.groups = num_heads
            group_convs = []
            for k_idx, kernel in enumerate(self.filters):
                weight_group = []
                for g in range(self.groups):
                    if(k_idx == 0):
                        # weight = g / (self.groups - 1) * kernel
                        """Do we need to interpolate identity filter???"""
                        weight = kernel
                    else:
                        weight = (self.groups - g) / self.groups * kernel / (2.0 / ((1.0-(self.groups - g) / self.groups) * (2.0 - 1.0) + 1.0))
                    weight_group.append(weight)
                group_convs.append(torch.stack(weight_group)) # stack weight group: num_group, 1, 1, 3, 3

            # self.conv_weights: num_kernel, num_group, 3, 3
            # Repeat for each input channel and store on the device
            """For each kernel, we need to do the multi head perception. Inside each multihead perception, the kernel weights are different among groups. """
            group_convs = [w.repeat(1, self.c_in // self.groups, 1, 1, 1).to(self.device) for w in group_convs] # each thing in group conv is one kernel
            if(self.trainable_kernel):
                perception_conv_list = []
                if(init_with_grad_kernel):
                    grad_kernel = torch.cat([identity_filter.repeat(dim, 1, 1, 1), sobel_filter_x.repeat(dim, 1, 1, 1), sobel_filter_y.repeat(dim, 1, 1, 1), laplacian_filter.repeat(dim, 1, 1, 1)], dim=0)
                for i, scale in enumerate(self.perception_scales):
                    perception_conv_list += [
                        nn.Conv2d(dim, dim * self.expand, kernel_size=(3, 3), stride=(1, 1), padding=((3+2*i - 1) // 2, (3+2*i - 1) // 2), dilation = i + 1, groups = dim, bias=False)
                    ]
                    if(init_with_grad_kernel):
                        assert self.expand == 4
                        perception_conv_list[-1].weight.data = grad_kernel[...]
                        print("Perception Conv", i, perception_conv_list[-1].weight[0, 0], perception_conv_list[-1].weight[dim, 0], perception_conv_list[-1].weight[dim*2, 0], perception_conv_list[-1].weight[dim*3, 0])
                self.perception_conv = nn.ModuleList(perception_conv_list)
                if(self.perception_aggr == "wsum" and self.expand > 1):
                    self.perception_aggr_model = PerceptionAggr(dim, num_scales=self.expand, multi_head_combine=False, head_num=1, norm = ablation_aggrnorm)
                if(perception_norm != "None"):
                    print("NCA Perception Norm/Non-linear On", perception_norm)
                    self.perception_norm_act = nn.Identity()
                    if(perception_norm == "silu"):
                        self.perception_norm_layer = nn.ModuleList([nn.ModuleList([nn.SiLU() for _ in range(expand)]) for j in range(len(self.perception_scales))])
                    elif(perception_norm == "gelu"): 
                        self.perception_norm_layer = nn.ModuleList([nn.ModuleList([nn.GELU() for _ in range(expand)]) for j in range(len(self.perception_scales))])
                    elif(perception_norm == "ln"):
                        self.perception_norm_layer = nn.ModuleList([nn.ModuleList([nn.LayerNorm(dim) for _ in range(expand)]) for j in range(len(self.perception_scales))])
                    elif(perception_norm == "bn"):
                        self.perception_norm_layer = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(dim) for _ in range(expand)]) for j in range(len(self.perception_scales))])
                    elif(perception_norm == "lngelu"):
                        self.perception_norm_layer = nn.ModuleList([nn.ModuleList([nn.LayerNorm(dim) for _ in range(expand)]) for j in range(len(self.perception_scales))])
                        self.perception_norm_act = nn.GELU()
                    elif(perception_norm == "bngelu"):
                        self.perception_norm_layer = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(dim) for _ in range(expand)]) for j in range(len(self.perception_scales))])
                        self.perception_norm_act = nn.GELU()
                else:
                    self.perception_norm_act = nn.Identity()
                    self.perception_norm_layer = nn.Identity()
#                 self.group_convs = group_convs
#                 group_conv_train = []
#                 for scale in self.perception_scales:
#                     group_conv_train.append(torch.stack(group_convs[1:]) / ((scale + 1) ** 1.618)) 
#                 self.group_convs_train = torch.stack(group_conv_train) # #scales, num_kernel, num_groups (head), num_channels, 1, 3, 3
#                 # self.group_convs_train = torch.stack(group_convs[1:])
#                 self.group_convs_train.requires_grad_(True)
#                 self.group_convs_train = nn.Parameter(self.group_convs_train)
            else:
                group_convs_stack = []
                for scale in self.perception_scales:
                    group_convs_stack.append(torch.stack(group_convs) / ((scale + 1) ** 1.618)) 
                group_convs = torch.stack(group_convs_stack)
                self.register_buffer('group_convs', group_convs)
        self.local_perception_only = local_perception_only
        if(not local_perception_only):
#             if(self.multi_head_nca):
#                 c_in = self.c_in // self.num_heads
#             else:
            c_in = self.c_in
            if(self.perception_aggr == "concat"):
                in_channel = c_in * self.expand + self.c_cond
                out_channels = c_in
                self.fc_dim = in_channel * 2 if not multi_head_nca else in_channel
                if(ablation_nca):
                    self.fc_dim = int(in_channel * 1.5)
            elif(self.perception_aggr == "sum"):
                in_channel = c_in + self.c_cond
                out_channels = c_in
                self.fc_dim = c_in
            elif(self.perception_aggr == "wsum"):
                in_channel = c_in + self.c_cond
                out_channels = c_in
                self.fc_dim = c_in
            self.act = act_layer()

            if(self.perception_aggr == "concat"):
                self.w1 = torch.nn.Conv2d(in_channel, self.fc_dim, 1)
                #torch.nn.init.xavier_normal_(self.w1.weight, gain=0.2)
            elif(self.perception_aggr == "sum"):
                self.w1 = torch.nn.Conv2d(in_channel, self.fc_dim, 1)
                #torch.nn.init.xavier_normal_(self.w1.weight, gain=0.2)
            elif(self.perception_aggr == "wsum"):
                self.w1 = torch.nn.Conv2d(in_channel, self.fc_dim, 1)
                #torch.nn.init.xavier_normal_(self.w1.weight, gain=0.2)



            #if(not self.energy_minimization):
            self.w2 = torch.nn.Conv2d(self.fc_dim, out_channels, 1, bias=True)
    #         torch.nn.init.xavier_normal_(self.w2.weight, gain=0.1)
    #         torch.nn.init.zeros_(self.w2.bias)
#         elif(local_perception_only):
#             self.downsample_perception = torch.nn.Conv2d(self.c_in // self.num_heads * self.expand, self.c_in // self.num_heads, 1, bias=False)
        self.local_perception_only = local_perception_only
        if(self.low_rank_approx):
            self.build_low_rank_weight()
        
        self.correct_alive = correct_alive
        
    
    

    def alive(self, x):
        # use additional channel, or use existing channel
        # additional channel: intialization? Transfer to next layer? Preserve of alive and death in patch merging? 
        return F.max_pool2d(self.alive_func(x[:, self.alive_channel:self.alive_channel + 1, :, :]), kernel_size=3, stride=1, padding=1) > self.alive_threshold

    def perceive_torch(self, x, scale = 0):
        """x: B, C, H, W"""
        B,C,H,W = x.shape
        mes_time = False
        if(mes_time):
            start = time.time()
        if(self.trainable_kernel):
            y = self.perception_conv[scale](x)
            
            if(mes_time):
                print("Perception Conv", x.shape, y.shape, time.time() - start)
                start = time.time()
            if(self.perception_aggr == "wsum" and self.expand > 1):
                B, C_perception, H, W = y.shape
                y = y.reshape(B, self.expand, C_perception // self.expand, H, W).permute(0,2,3,4,1)
                if(self.perception_norm != "None"):
                    y_list = list(y.unbind(-1))
                    out_list = []
                    for p_idx, y_percep in enumerate(y_list):
                        if("ln" in self.perception_norm):
                            y_percep = y_percep.permute(0,2,3,1).contiguous()
                        y_percep = self.perception_norm_act(self.perception_norm_layer[scale][p_idx](y_percep))
                        if("ln" in self.perception_norm):
                            y_percep = y_percep.permute(0,3,1,2).contiguous()
                        out_list.append(y_percep)
                    y = torch.stack(out_list, dim = -1)
                weight = self.perception_aggr_model(x) # B, 1, H, W, K
                y = torch.sum(y * weight, dim = -1)
                if(mes_time):
                    print("Perception Aggr", time.time() - start)
                    start = time.time()
            elif(self.perception_aggr == "sum"):
                B, C_perception, H, W = y.shape
                y = y.reshape(B, self.expand, C_perception // self.expand, H, W)
                y = torch.sum(y, dim = 1)
            return y
        pad_num = 1
        dilation = 1
        if scale != 0:
            dilation += scale
            pad_num = (3+2*(dilation-1) - 1) // 2
            # _, _, h, w = x.shape
            # h_new = int(h // (2 ** scale))
            # w_new = int(w // (2 ** scale))
            # x = F.interpolate(x, size=(h_new, w_new), mode='bicubic', align_corners=False)

        # if self.trainable_kernel:
        #     if(self.multi_head_perception):
        #         # Split the input tensor along the channel dimension into K groups
        #         inputs = torch.split(x, x.shape[1] // self.groups, dim=1)

        #         # Apply the corresponding convolution operation to each group
        #         outputs = [conv(inp) for conv, inp in zip(self.group_convs, inputs)]

        #         # Concatenate the results along the channel dimension
        #         if(self.perception_aggr == "concat"):
        #             y = torch.cat(outputs, dim = 1)
        #         elif(self.perception_aggr == "sum"):
        #             y = torch.sum(torch.stack(outputs), dim=0)
        #     else:
        #         outputs = self.conv_layer(x)
        #         if(self.perception_aggr == "concat"):
        #             y = outputs
        #         elif(self.perception_aggr == "sum"):
        #             outputs_chunk = torch.split(outputs, outputs.shape[1] // self.expand, dim = 1)
        #             y = torch.sum(torch.stack(outputs_chunk), dim=0)
        # else:
        if(self.multi_head_perception):
            inputs = torch.split(x, self.c_in // self.groups , dim=1)
            input_pad = [F.pad(z, [pad_num, pad_num, pad_num, pad_num], self.padding_mode) for z in inputs]
            groups = self.c_in // self.groups 
            kernel_out = []
            if(not self.trainable_kernel):
                group_convs = self.group_convs[scale]
                for k_idx, kernel in enumerate(group_convs):
                    if(scale != 0 and k_idx == 0):
                        kernel = torch.zeros_like(kernel)
                    outputs = [F.conv2d(inp, conv_weights , dilation = dilation, groups=groups) for conv_weights, inp in zip(kernel, input_pad)]
                    kernel_out.append(torch.cat(outputs, dim=1))
            elif(self.trainable_kernel):
                if(scale != 0):
                    kernel = torch.zeros_like(self.group_convs[0])
                else:
                    kernel = self.group_convs[0]
                outputs = [F.conv2d(inp, conv_weights , dilation = dilation, groups=groups) for conv_weights, inp in zip(self.group_convs[0], input_pad)] # identity
                kernel_out.append(torch.cat(outputs, dim=1))

                group_convs_train = self.group_convs_train[scale]
                for kernel in group_convs_train:
                    outputs = [F.conv2d(inp, conv_weights , dilation = dilation, groups=groups) for conv_weights, inp in zip(kernel, input_pad)]
                    kernel_out.append(torch.cat(outputs, dim=1))
            if(self.perception_aggr == "concat"):
                y = torch.cat(kernel_out, dim = 1)
            elif(self.perception_aggr == "sum"):
                y = torch.sum(torch.stack(kernel_out), dim=0)
#                 direction_aware_feature = torch.cat([kernel_out[1], kernel_out[2]], dim = 1)
#                 direction_unaware_feature = torch.cat([kernel_out[0], kernel_out[3]], dim = 1)
#                 y = torch.sum(torch.stack([direction_aware_feature, direction_unaware_feature]), dim=0)
        else:
            z = F.pad(x, [pad_num, pad_num, pad_num, pad_num], self.padding_mode)
            
            sobel_filter_x = self.sobel_filter_x[scale]
            sobel_filter_y = self.sobel_filter_y[scale]
            laplacian_filter = self.laplacian_filter[scale]

            y1 = F.conv2d(z, sobel_filter_x , dilation = dilation,  groups=self.c_in)
            y2 = F.conv2d(z, sobel_filter_y , dilation = dilation, groups=self.c_in)
            y3 = F.conv2d(z, laplacian_filter , dilation = dilation, groups=self.c_in)
            if(self.perception_aggr == "concat"):
                if(scale != 0):
                    y = torch.cat((torch.zeros_like(x), y1, y2, y3), 1)
                else:
                    y = torch.cat((x , y1, y2, y3), 1)
            else:
                y = x  + y1 + y2 + y3
        
        # if scale != 0:
        #     y = F.interpolate(y, size=(h, w), mode='bicubic', align_corners=False)

        return y

    def perceive_multiscale(self, x, pos_emb_mat=None):
#         if(len(x.shape) != 4):
#             B,N_heads,C_head,H,W = x.shape
#             assert self.num_heads == N_heads
#             x = x.reshape(B,N_heads*C_head,H,W)
        if(len(self.perception_scales) == 1):
            y = self.perceive_torch(x)
        else:
            perceptions = []
            y = 0
            for scale in self.perception_scales:
                if(x.shape[2] <= 2**scale):
                    break
                z = self.perceive_torch(x, scale=scale) # B,C_percept,H,W
                
#                 if(self.multi_head_nca):
#                     B,C,H,W = z.shape
#                     z = z.reshape(B, self.num_heads, C//self.num_heads, H, W)#.reshape(B*self.num_heads, C//self.num_heads, H, W)
                perceptions.append(z)

            if(self.weighted_scale_combine):
                weight = self.perception_scale_aggr_model(x) # B,1,H,W,K, B head 1 H W K
                perceptions_cat = torch.stack(perceptions, dim = -1) # B,C,H,W,K; B head C H W K
                #print(weight.shape, perceptions_cat.shape)
                y = torch.sum(perceptions_cat * weight, dim=-1) # B,C,H,W; B head C H W
#                 if(self.local_perception_only):
#                     B,N_heads,C_head,H,W = y.shape
#                     y = self.downsample_perception(y.reshape(B*N_heads,C_head,H,W)).reshape(B, N_heads, self.c_in //self.num_heads, H, W)
#                 if(self.multi_head_nca):
#                     y = y.reshape(B*self.num_heads, C//self.num_heads, H, W)
            else:
                y = sum(perceptions)
                y = y / len(self.perception_scales)

        if pos_emb_mat is not None:
            y = torch.cat([y, pos_emb_mat], dim=1)

        return y

    def evolve(self, x, return_middle_state = False):
        b_x, c_x, H, W = x.shape
        return_dict = {}
        # if(self.energy_minimization):
        #     x.requires_grad_(True)
        
        # if(self.alive_mask):
        #     pre_life_mask = self.alive(x)

        # if self.pos_emb_2d:
        #     y_percept = self.perceive_multiscale(x, pos_emb_mat=self.pos_emb_2d(x))
        # else:
        #     y_percept = self.perceive_multiscale(x)
        
        # if(self.energy_minimization):
        #     with torch.cuda.amp.autocast(enabled = False):
        #         y_percept = y_percept.float()
        #         energy = self.act(self.w1(y_percept))
        #         energy_total = torch.sum(torch.sum(0.5 * torch.norm(energy, dim = 1, p = 2) ** 2, dim = [1, 2]))
        #         grad_energy = torch.autograd.grad(outputs=energy_total, inputs=x, retain_graph = True)
        #         y1 = -grad_energy[0]
        #     if(return_middle_state):
        #         return_dict["energy"] = torch.sum(torch.sum(0.5 * torch.norm(energy, dim = 1, p = 2) ** 2, dim = [1, 2])).item()
        #     y2 = self.w2(energy)
        #     y = y1 * self.energy_coeff + y2
        # else:
        #     if(self.low_rank_approx):
        #         y = self.low_rank_update(y_percept)
        #     else:
        #         y_p = self.act(self.w1(y_percept))
        #         if(return_middle_state):
        #             return_dict["energy"] = torch.sum(torch.sum(0.5 * torch.norm(y_p, dim = 1, p = 2) ** 2, dim = [1, 2])).item()
        #         y = self.w2(y_p)

        if(self.energy_minimization):
            with torch.enable_grad():
                x.requires_grad_(True)
                if(self.alive_mask):
                    pre_life_mask = self.alive(x)

                if self.pos_emb_2d:
                    y_percept = self.perceive_multiscale(x, pos_emb_mat=self.pos_emb_2d(x))
                else:
                    y_percept = self.perceive_multiscale(x)
                
#                 if(self.multi_head_nca):
#                     B,N,C,H,W = y_percept.shape
#                     y_percept = y_percept.reshape(B*N, C, H, W)

                with torch.cuda.amp.autocast(enabled = False):
                    y_percept = y_percept.float()
                    energy = self.act(self.w1(y_percept))
                    energy_total = torch.sum(torch.sum(0.5 * torch.norm(energy, dim = 1, p = 2) ** 2, dim = [1, 2]))
                    grad_energy = torch.autograd.grad(outputs=energy_total, inputs=x, retain_graph = True)
                    y1 = -grad_energy[0]
                if(return_middle_state):
                    return_dict["energy"] = torch.sum(torch.sum(0.5 * torch.norm(energy, dim = 1, p = 2) ** 2, dim = [1, 2])).item()
                y2 = self.w2(energy)
                #print(torch.mean(torch.norm(y1, dim = [1,2,3], p = 2)), torch.mean(torch.norm(y2, dim = [1,2,3], p = 2)))
                #print(self.energy_coeff)
#                 if(self.multi_head_nca):
#                     BN, C, H, W = y2.shape
#                     y2 = y2.reshape(BN//self.num_heads,self.num_heads,C,H,W).reshape(BN//self.num_heads, self.num_heads*C, H, W)
                if(self.energy_coeff_point_wise):
                    #print(torch.mean(torch.norm(y1, dim = [1,2,3], p = 2)), torch.mean(torch.norm(y2, dim = [1,2,3], p = 2)))
                    with torch.cuda.amp.autocast(enabled = False):
                        energy_input = torch.stack([y1, y2], dim = -1)
                        weight = self.energy_coeff_model(x)
                        y = torch.sum(energy_input * weight, dim = -1)
                    #y = energy_output + y2
                else:
                    if(self.energy_multi_head):
                        energy_coeff = torch.cat([self.energy_coeff[i:i+1].expand(b_x, c_x//self.num_heads, H, W) for i in range(self.num_heads)], dim = 1)
                    else:
                        energy_coeff = self.energy_coeff
                    with torch.cuda.amp.autocast(enabled = False):
                        if(self.random_energy_coeff and self.training):
                            noise_coeff = torch.randn(energy_coeff.shape, device = energy_coeff.device) * (energy_coeff / 5.0)
                            energy_coeff = energy_coeff + noise_coeff
                    y = y1 * energy_coeff + y2
        else:
            if(self.alive_mask):
                pre_life_mask = self.alive(x)

            if self.pos_emb_2d:
                y_percept = self.perceive_multiscale(x, pos_emb_mat=self.pos_emb_2d(x))
            else:
                y_percept = self.perceive_multiscale(x)
            
#             if(self.multi_head_nca):
#                 B,N,C,H,W = y_percept.shape
#                 y_percept = y_percept.reshape(B*N, C, H, W)
            
            if(self.low_rank_approx):
                y = self.low_rank_update(y_percept)
            else:
                y_p = self.act(self.w1(y_percept))
                if(return_middle_state):
                    return_dict["energy"] = torch.sum(torch.sum(0.5 * torch.norm(y_p, dim = 1, p = 2) ** 2, dim = [1, 2])).item()
                y = self.w2(y_p)
#                 if(self.multi_head_nca):
#                     BN, C, H, W = y.shape
#                     y = y.reshape(BN//self.num_heads,self.num_heads,C,H,W).reshape(BN//self.num_heads, self.num_heads*C, H, W)

        if(self.stochastic_update > 0 and self.training):
            b, c, h, w = y.shape
            update_mask = (torch.rand(b, 1, h, w, device=x.device) + (1.0 - self.stochastic_update)).floor()
            update_mask.div_(1.0 - self.stochastic_update)
            x = x + y * update_mask
        else:
            x = x + y
        
        if(self.alive_mask):
            post_life_mask = self.alive(x)
            life_mask = (pre_life_mask & post_life_mask).float()
            if(return_middle_state):
                return_dict["alive_mask"] = life_mask.detach()
                return_dict["pre_alive_mask"] = pre_life_mask.detach()
                return_dict["post_alive_mask"] = post_life_mask.detach()
            if(self.correct_alive == 2):
                x_alive = x[:, self.alive_channel:self.alive_channel+1]
                x = x * life_mask
                x_feature = x[:, self.alive_channel+1:]
                x = torch.cat([x_alive, x_feature], dim = 1)
            else:
                x = x * life_mask
        #if(self.energy_minimization):
            #x.requires_grad_(False)
        return x, return_dict
    
    def forward(self, x, return_middle_state = True):
        """x: B, N, C"""
        b_x, n_tokens, c_x = x.shape
        H = int(n_tokens ** 0.5)
        W = n_tokens // H
        self.input_size = (H, W, c_x)
        if(H * W != n_tokens):
            print("Token Number Cannot Resize to an Image", H, W, n_tokens)
            exit()
        x = x.transpose(1, 2).reshape(b_x, c_x, H, W)
        
        return_dict = {}
        if(return_middle_state):
            middle_state_list = []
        if(isinstance(self.times, list)):
            if(self.training):
                step = torch.randint(low = self.times[0], high = self.times[1] + 1, size = (1,)).item()
            else:
                #step = self.times[1]
                step = (self.times[1] + self.times[0]) // 2
                
#                 if(self.times[0] == 2 and self.times[1] == 4):
#                     step = 3
#                 if(self.times[0] == 3 and self.times[1] == 5):
#                     step = 3
                
#                 if(self.num_heads == 3):
#                     step = (self.times[1] + self.times[0]) // 2
#                 if(self.num_heads == 6):
#                     step = (self.times[1] + self.times[0]) // 2
#                 if(self.num_heads == 12):
#                     step = self.times[1]
#                 if(self.num_heads == 24):
#                     step = self.times[1]
        elif(isinstance(self.times, int)):
            step = self.times
        for time_step in range(step):
            if(self.norm_layer):
                if(self.separate_norm):
                    cur_norm_layer = self.norm_layer[time_step]
                else:
                    cur_norm_layer = self.norm_layer
                x = x.permute(0, 2, 3, 1)
                x = cur_norm_layer(x)
                x = x.permute(0, 3, 1, 2)
            x, return_dict_middle = self.evolve(x.contiguous(), return_middle_state)
            if(return_middle_state):
                middle_state_list.append(x)
                return_dict[f"energy.{time_step}"] = return_dict_middle["energy"]
                if("alive_mask" in return_dict_middle.keys()):
                    return_dict[f"alive_mask.{time_step}"] = return_dict_middle["alive_mask"]
                    return_dict[f"pre_alive_mask.{time_step}"] = return_dict_middle["pre_alive_mask"]
                    return_dict[f"post_alive_mask.{time_step}"] = return_dict_middle["post_alive_mask"]
                    
        x = x.reshape(b_x, c_x, n_tokens).transpose(1, 2)
        
        if(return_middle_state):
            return_dict["middle_state"] = middle_state_list
        return x, return_dict

    def flops(self, verbose = False):
        flops = 0
        H, W, c_x = self.input_size
        if(isinstance(self.times, list)):
            mean_time = (self.times[0] + self.times[1]) / 2.0
        elif(isinstance(self.times, int)):
            mean_time = self.times
        if(self.norm_layer):
            flops += (self.input_size[0] * self.input_size[1] * self.input_size[2]) * mean_time
        
        # Perception
        if(self.energy_minimization):
            energy_coeff = 2
        else:
            energy_coeff = 1
        # Perception
        if(self.local_perception_only):
            flops += self.input_size[0] * self.input_size[1] * 9 * self.input_size[2] * len(self.perception_scales)
            flops += self.input_size[0] * self.input_size[1] * 9 * self.expand * len(self.perception_scales)
            return flops
        flops += self.expand * self.input_size[0] * self.input_size[1] * 9 * self.input_size[2] * len(self.perception_scales) * energy_coeff
        if(verbose):
            flops_cur = flops
            print("Perception: ", flops_cur / 1e9)

        # MLP
        if(self.perception_aggr == "concat"):
            in_channel = c_x * self.expand + self.c_cond
        elif(self.perception_aggr == "sum"):
            in_channel = c_x
        elif(self.perception_aggr == "wsum"):
            in_channel = c_x
        out_channels = self.c_in
        if(self.low_rank_approx):
            flops += self.input_size[0] * self.input_size[1] * self.rank_num_w1 * (in_channel + self.fc_dim) * mean_time
            flops += self.input_size[0] * self.input_size[1] * self.rank_num_w2 * (self.fc_dim + out_channels) * mean_time
        else:
#             if(self.multi_head_nca):
#                 flops += self.input_size[0] * self.input_size[1] * (c_x * self.expand) * self.fc_dim * mean_time * energy_coeff
#             else:
            flops += self.input_size[0] * self.input_size[1] * in_channel * self.fc_dim * mean_time * energy_coeff
            if(verbose):
                print("W1: ", (flops - flops_cur) / 1e9)
                print(self.w1.weight.shape, self.input_size)
                flops_cur = flops
            flops += self.input_size[0] * self.input_size[1] * self.fc_dim * out_channels * mean_time
            if(verbose):
                print("W2: ", (flops - flops_cur) / 1e9)
        return flops
    def build_low_rank_weight(self):
        U1, S1, V1 = torch.linalg.svd(self.w1.weight.squeeze().squeeze().t().data, full_matrices=False) # in_c, out_c -> in_c, out_c; out_c; out_c, out_c
        U2, S2, V2 = torch.linalg.svd(self.w2.weight.squeeze().squeeze().t().data, full_matrices=False)
        bias1 = self.w1.bias.data
        bias2 = self.w2.bias.data

        low_rank_prop = 0.5
        rank_num_w1 = int(low_rank_prop * S1.shape[0])
        rank_num_w2 = int(low_rank_prop * S2.shape[0])
        self.rank_num_w1 = rank_num_w1
        self.rank_num_w2 = rank_num_w2

        # self.U1 = nn.Parameter(U1[:, :rank_num_w1])
        # self.S1 = nn.Parameter(S1[:rank_num_w1])
        self.US1 = nn.Parameter(U1[:, :rank_num_w1] @ torch.diag(S1[:rank_num_w1]))
        self.V1 = nn.Parameter(V1[:rank_num_w1, :])

        # self.U2 = nn.Parameter(U2[:, :rank_num_w2])
        # self.S2 = nn.Parameter(S2[:rank_num_w2])
        self.US2 = nn.Parameter(U2[:, :rank_num_w2] @ torch.diag(S2[:rank_num_w2])) # in_c, r
        self.V2 = nn.Parameter(V2[:rank_num_w2, :]) # r, outc

        self.US1.requires_grad_(True)
        # self.S1.requires_grad_(True)
        self.V1.requires_grad_(True)
        self.US2.requires_grad_(True)
        # self.S2.requires_grad_(True)
        self.V2.requires_grad_(True)

        self.bias1 = nn.Parameter(bias1)
        self.bias2 = nn.Parameter(bias2)
        self.bias1.requires_grad_(True)
        self.bias2.requires_grad_(True)
        
        del self.w1
        del self.w2
    
    def low_rank_update(self, x):
        """x: B, C, H, W"""
        x = x.permute(0, 2, 3, 1) # B, H, W, C
        x = torch.matmul(torch.matmul(x, self.US1), self.V1) + self.bias1 # HW in_c * r, HW r*out
        x = torch.matmul(torch.matmul(x, self.US2), self.V2) + self.bias2 
        x = x.permute(0, 3, 1, 2)
        return x

class AttnNCA(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            times = 1,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
            alive_channel = 0,
            sigmoid_alive = False,
            alive_threshold = 0.1,
            perception_scales = [0],
            block_type = "ode",
            linear_combine = False,
            correct_alive = 0,
            energy_minimization = False,
            stochastic_update = False,
            no_norm = False,
    ):
        super().__init__()
        self.no_norm = no_norm
        if(not no_norm):
            self.norm1 = norm_layer(dim)

        neighbor_size = [3+2*x for x in perception_scales]
        attn_layers = []
        self.times = times
        self.alive_channel = alive_channel
        self.alive_threshold = alive_threshold
        self.alive_mask = alive_threshold > 0.0
        self.energy_minimization = False
        self.stochastic_update = stochastic_update
        if(self.energy_minimization):
            self.energy_coeff = nn.Parameter(torch.zeros(1) + 0.01)
        
        self.alive_func = nn.Sigmoid() if sigmoid_alive else nn.Identity()
        
        self.correct_alive = correct_alive
        
        for scale in perception_scales:
            attn_layers += [Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                alive_channel = alive_channel,
                sigmoid_alive = sigmoid_alive,
                alive_threshold = 0.0, # we shouldn't do alive inside the attn now because this is an NCA and aliveness should be considered outside
                localize = True,
                attn_neighbourhood_size = [neighbor_size[scale], neighbor_size[scale]],
                dilation = scale + 1,
                correct_alive = correct_alive,
            )]
        self.attn = nn.Sequential(*attn_layers)
        # self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # if(block_type == "normal"):
        #     self.norm2 = norm_layer(dim)
        
        self.block_type = block_type

        self.mlp_in = dim
        self.mlp_hidden = int(dim * mlp_ratio)
        # self.mlp = mlp_layer(
        #     in_features=dim,
        #     hidden_features=int(dim * mlp_ratio),
        #     act_layer=act_layer,
        #     drop=proj_drop,
        # )
        self.w1 = nn.Linear(dim, self.mlp_hidden)
        self.act = act_layer()
        self.w2 = nn.Linear(self.mlp_hidden, dim)
        # self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.linear_combine = linear_combine # only valid with ODE block design
        if(linear_combine):
            self.combine_layer = nn.Sequential(
                nn.Linear(dim * 2, dim * 4),
                act_layer(),
                nn.Linear(dim * 4, dim)
            )

    def alive(self, x):
        b_x, n_tokens, c_x = x.shape
        H = int(n_tokens ** 0.5)
        W = n_tokens // H
        if(H * W != n_tokens):
            print("In attention NCA alive mask compute, Token Number Cannot Resize to an Image", H, W, n_tokens)
            exit()
        x = x.transpose(1, 2).reshape(b_x, c_x, H, W)
        alive_mask = F.max_pool2d(self.alive_func(x[:, self.alive_channel:self.alive_channel + 1, :, :]), kernel_size=3, stride=1, padding=1) > self.alive_threshold
        alive_mask = alive_mask.permute(0, 2, 3, 1).reshape(b_x, n_tokens, 1)
        return alive_mask

    def forward(self, x, return_middle_state = False):
        self.b_x, self.n_token, self.c_x = x.shape
        if(isinstance(self.times, list)):
            step = torch.randint(low = self.times[0], high = self.times[1] + 1, size = (1,)).item()
        elif(isinstance(self.times, int)):
            step = self.times
        
        return_dict = {}
        nca_middle_state_list = []
        for i in range(step):
            
            # x_norm1 = self.norm1(x)
            if(i == 0):
                if(not self.no_norm):
                    x_norm1 = self.norm1(x)
                else:
                    x_norm1 = x
            else:
                x_norm1 = x
            
            if(self.block_type == "normal"):
                if(self.energy_minimization):
                    with torch.enable_grad():
                        x_norm1.requires_grad_(True)
                        if(self.alive_mask):
                            pre_life_mask = self.alive(x_norm1)
                        """Multi scale perception"""
                        y_percept = self.attn[0](x_norm1)
                        if(len(self.attn) > 1):
                            for attn_layer in self.attn:
                                y_percept = y_percept + attn_layer(x_norm1)
                        return_dict["attn"] = y_percept.detach()
                        """Update"""
                        with torch.cuda.amp.autocast(enabled = False):
                            y_percept = y_percept.float()
                            energy = self.act(self.w1(y_percept))
                            energy_total = torch.sum(torch.sum(0.5 * torch.norm(energy, dim = 2, p = 2) ** 2, dim = 1))
                            grad_energy = torch.autograd.grad(outputs=energy_total, inputs=x_norm1, retain_graph = True)
                            y1 = -grad_energy[0]
                            if(return_middle_state):
                                return_dict["energy"] = energy_total.item()
                        y2 = self.w2(energy)
                        y = y1 * self.energy_coeff + y2
                else:
                    if(self.alive_mask):
                        pre_life_mask = self.alive(x_norm1)
                    """Multi scale perception"""
                    for attn_layer in self.attn:
                        x = x + attn_layer(x_norm1)
                    return_dict["attn"] = x.detach()
                    """Update"""
                    y = self.w2(self.act(self.w1(x)))
            elif(self.block_type == "ode"):
                if(self.energy_minimization):
                    with torch.enable_grad():
                        x_norm1.requires_grad_(True)
                        if(self.alive_mask):
                            pre_life_mask = self.alive(x_norm1)

                        F = self.attn[0](x_norm1)
                        if(len(self.attn) > 1):
                            for attn_layer in self.attn[1:]:
                                F = F + attn_layer(x_norm1)
                        return_dict["attn"] = F.detach()
                        with torch.cuda.amp.autocast(enabled = False):
                            x_norm1_full = x_norm1.float()
                            energy = self.act(self.w1(x_norm1_full))
                            energy_total = torch.sum(torch.sum(0.5 * torch.norm(energy, dim = 2, p = 2) ** 2, dim = 1))
                            grad_energy = torch.autograd.grad(outputs=energy_total, inputs=x_norm1, retain_graph = True)
                            y1 = -grad_energy[0]
                        if(return_middle_state):
                            return_dict["energy"] = energy_total.item()
                        y2 = self.w2(energy)
                        G = y1 * self.energy_coeff + y2
                else:
                    if(self.alive_mask):
                        pre_life_mask = self.alive(x_norm1)
                    F = self.attn[0](x_norm1)
                    if(len(self.attn) > 1):
                        for attn_layer in self.attn[1:]:
                            F = F + attn_layer(x_norm1)
                    return_dict["attn"] = F.detach()
                    G = self.w2(self.act(self.w1(x_norm1)))
                if(not self.linear_combine):
                    y = F + G
                else:
                    cell_interaction = torch.cat([F, G], dim = -1)
                    y = self.combine_layer(cell_interaction)
            
            if(self.stochastic_update):
                b, n, c = y.shape
                update_mask = (torch.rand(b, n, 1, device=y.device) + 0.5).floor()
                x = x_norm1 + y * update_mask
            else:
                x = x_norm1 + y

            if(self.alive_mask):
                post_life_mask = self.alive(x)
                life_mask = (pre_life_mask & post_life_mask).float()
                if(return_middle_state):
                    return_dict[f"alive_mask.{i}"] = life_mask.detach()
                    return_dict[f"pre_alive_mask.{i}"] = pre_life_mask.detach()
                    return_dict[f"post_alive_mask.{i}"] = post_life_mask.detach()
                """Correct implementation of aliveness when there is sigmoid!!!"""
                if(self.correct_alive == 2):
                    x_alive = x[:, :, self.alive_channel:self.alive_channel+1]
                    x = x * life_mask
                    x_feature = x[:, :, self.alive_channel+1:]
                    x = torch.cat([x_alive, x_feature], dim = -1)
                else:
                    x = x * life_mask
            if(return_middle_state):
                nca_middle_state_list.append(x)
        if(return_middle_state):
            return_dict["middle_state"] = nca_middle_state_list
            return_dict["nca_state"] = x
            return_dict["output"] = x
        return x, return_dict

    def flops(self, verbose = False):
        if(isinstance(self.times, list)):
            step = (self.times[0] + self.times[1]) / 2.0
        elif(isinstance(self.times, int)):
            step = self.times
        flops = 0
        for attn_layer in self.attn:
            flops += attn_layer.flops()  * step
            flops_cur = flops
        if(verbose):
            print("Attention NCA attn: ", flops / 1e9)
        flops += self.n_token * self.mlp_in * self.mlp_hidden * 2 * step
        if(verbose):
            print("Attention NCA MLP: ", (flops - flops_cur) / 1e9)
        flops = self.n_token * self.c_x * step * 2 if self.block_type == "normal" else self.n_token * self.c_x * step
        if(self.linear_combine):
            flops += (self.n_token * self.c_x * 2 * self.c_x * 4 + self.n_token * self.c_x * self.c_x * 4) * step
        return flops

class SparseQuery(nn.Module):

    def __init__(
            self,
            dim_q,
            dim_kv,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            alive_channel = 0,
            alive_threshold = 0.1,
            sigmoid_alive = False,
            method = "aliveness",
        ):
        super().__init__()
        
        self.alive_channel = alive_channel
        self.alive_threshold = alive_threshold
        self.alive_mask = alive_threshold > 0.0
        
        self.norm_high = norm_layer(dim_q)
        self.norm_low = norm_layer(dim_kv)

        self.alive_func = nn.Sigmoid() if sigmoid_alive else nn.Identity()

        if(method == "aliveness"):
            assert self.alive_mask
        self.method = method

        self.cross_attn = CrossAttention(
            dim_q = dim_q,
            dim_kv = dim_kv,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            alive_channel = alive_channel,
            sigmoid_alive = sigmoid_alive,
            alive_threshold = alive_threshold,
        )

    def select_top_cells(self, x, max_k=50):
        B, N, C = x.shape

        # Calculate k, which is the minimum of max_k and int(k_ratio * N)
        k = int(N ** 0.5)

        # Select the top-k indices based on the aliveness value (first channel)
        if(self.method == "aliveness"):
            x_judge = x[:, :, self.alive_channel]
        elif(self.method == "energy"):
            x_judge = torch.norm(x, dim = -1, p = 2)
        top_k_indices = torch.topk(x_judge, k, dim=1)[1]

        # Gather the top-k vectors based on the indices
        top_k_vectors = torch.gather(x, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, C))

        return top_k_vectors, top_k_indices

    def forward(self, high_cells, low_cells, return_middle_state = False):
        high_x, high_indices = self.select_top_cells(high_cells)
        low_x,_ = self.select_top_cells(low_cells)

        query_info = self.cross_attn(self.norm_high(high_x), self.norm_low(low_x))
        return query_info, high_indices

    def flops(self, verbose = False):
        flops = self.cross_attn.flops(verbose)
        if(verbose):
            print("Sparse Query: ", flops / 1e9)
        return flops

class Block_NCA(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            num_layer = 0,
            qkv_bias=False,
            qk_norm=False,
            drop_path=0.,
            proj_drop=0.,
            attn_drop=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            nca_norm_layer=None,
            separate_norm = False,
            stochastic_update = 0.0,
            times = 1,
            alive_channel = 0,
            alive_threshold = 0.1,
            trainable_kernel = False,
            normalize_filter = False,
            padding_mode = "replicate",
            multi_head_perception = False,
            perception_scales = [0],
            pos_emb = None,
            perception_aggr = "concat",
            block_type = "normal", 
            residual_nca = False,
            sigmoid_alive = False,
            energy_minimization = False,
            low_rank_approx = False,
            multi_head_nca = False,
            mlp_proj = False,
            ablation_nca = False,
            linear_downsample = False,
            linear_combine = False,
            correct_alive = 0,
            no_global = False,
            nca_type = "conv",
            recurrent_attention = 0,
            paas = 0,
            weighted_combine = 0,
            sparse_query = 0,
            sparse_query_method = "aliveness",
            group_norm = False,
            v2 = 0,
            recurrent_attention_norm = 0,
            layer_scale = 0,
            cosine_attn = 0,
            energy_multi_head = 0,
            energy_coeff_init = 0.01,
            relative_pos_emb = 0,
            window_attn = 0,
    ):
        super().__init__()
        self.dim = dim
        # if(multi_head_nca):
        #     norm1 = [norm_layer(dim // num_heads) for _ in range(num_heads)]
        #     self.norm1 = nn.ModuleList(norm1)
        # else:
        self.ablation_nca = ablation_nca
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.no_global = no_global
        self.group_norm = group_norm
        self.block_type = block_type
        self.energy_multi_head = energy_multi_head
        self.recurrent_attention = recurrent_attention
        self.num_layer = num_layer
        
        if(isinstance(times, list) and isinstance(times[0], list)):
            self.multi_stage = True
        else:
            self.multi_stage = False
        
        self.window_attn = window_attn
        if(not self.no_global and not (isinstance(times, list) and isinstance(times[0], list))):
            if(not window_attn):
                self.attn = Attention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    norm_layer=norm_layer,
                    alive_channel = alive_channel,
                    sigmoid_alive = sigmoid_alive,
                    alive_threshold = alive_threshold,
                    correct_alive = correct_alive,
                    paas = paas,
                    v2 = v2,
                    cosine_attn = cosine_attn,
                    relative_pos_emb = relative_pos_emb,
                    img_size = (56 // (2**num_layer), 56 // (2**num_layer)),
                )
            else:
                attn_list = []
                for _ in range(window_attn):
                    attn_list += [[
                        StackedWindowAttention(
                            dim = dim,
                            input_resolution = (56 // (2**num_layer), 56 // (2**num_layer)),
                            num_heads = num_heads,
                            window_size=7,
                            shift_size=0,
                            qkv_bias=True,
                            proj_drop=0.,
                            attn_drop=0.,
                        ),
                        StackedWindowAttention(
                            dim = dim,
                            input_resolution = (56 // (2**num_layer), 56 // (2**num_layer)),
                            num_heads = num_heads,
                            window_size=7,
                            shift_size=(3, 3),
                            qkv_bias=True,
                            proj_drop=0.,
                            attn_drop=0.,
                        ),
                    ]]
                self.attn = nn.Sequential(*[nn.Sequential(*x) for x in attn_list])
                    
            # self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.recurrent_attention = recurrent_attention
            if(isinstance(recurrent_attention, list)):
                min_recur_step = recurrent_attention[0]
            else:
                min_recur_step = recurrent_attention
            self.recurrent_attention_norm = recurrent_attention_norm
            if(recurrent_attention_norm and min_recur_step > 0):
                self.recur_attn_norm = norm_layer(dim)

        self.sparse_query = sparse_query
        if(self.sparse_query > 0):
            sparse_query_list = []
            for i in range(self.sparse_query):
                sparse_query_list.append(
                    SparseQuery(
                        dim_q = dim,
                        dim_kv = dim // (2**(self.sparse_query - i)),
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_norm,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        norm_layer=norm_layer,
                        alive_channel = alive_channel,
                        alive_threshold = alive_threshold,
                        sigmoid_alive = sigmoid_alive,
                        method = sparse_query_method,
                    )
                )
            self.sparse_query_list = nn.ModuleList(sparse_query_list)
            self.sparse_query_weight = nn.Parameter(-torch.flip(torch.arange(self.sparse_query), [0]).float())

        self.residual_nca = residual_nca
        if(v2 and v2 == 1):
            pass
        else:
            if(block_type != "ode"):
                if(residual_nca):
                    self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
                else:
                    self.drop_path2 = nn.Identity()

        if(not self.multi_stage):
            if(group_norm):
                self.norm1 = nn.GroupNorm(num_heads, dim)
            else:
                self.norm1 = norm_layer(dim)

        if(block_type != "ode"):
            self.norm2 = norm_layer(dim)
        
        if(block_type == "ode" and layer_scale > 0 and not self.multi_stage):
            self.ls1 = LayerScale(dim)
        else:
            self.ls1 = nn.Identity()
        
        self.multi_head_nca = multi_head_nca
        self.nca_type = nca_type

        self.linear_downsample = linear_downsample
        if(linear_downsample):
            downsample_dim = int(dim // 5 * 4)
            self.downsample_dim = downsample_dim
            self.fc_down = nn.Linear(dim // num_heads, downsample_dim // num_heads)
            self.fc_up = nn.Linear(downsample_dim // num_heads, dim // num_heads)

        if(not multi_head_nca):
            if(not ablation_nca):
                if(nca_type == "conv"):
                    
                    if(self.multi_stage): # stacked recurrent
                        num_ncas = len(times)
                        nca_list = []
                        attn_list = []
                        norm_list = []
                        drop_path_list = []
                        if(isinstance(drop_path, float)):
                            drop_path = [drop_path]
                        ls_list = []
                        if(weighted_combine):
                            weight_list = []
                        for stage_idx, time_step in enumerate(times):
                            if(time_step[0] == time_step[1]):
                                nca_time = time_step[0]
                            else:
                                nca_time = time_step
                            nca_list += [
                                        NCA(
                                            dim = dim if not linear_downsample else downsample_dim,
                                            num_heads = num_heads,
                                            act_layer = act_layer,
                                            norm_layer = nca_norm_layer,
                                            separate_norm = separate_norm,
                                            stochastic_update = stochastic_update,
                                            times = nca_time,
                                            alive_channel = alive_channel,
                                            alive_threshold = alive_threshold,
                                            trainable_kernel = trainable_kernel,
                                            normalize_filter = normalize_filter,
                                            padding_mode = padding_mode,
                                            multi_head_perception = multi_head_perception,
                                            perception_scales = perception_scales,
                                            pos_emb = pos_emb,
                                            perception_aggr = "sum",
                                            sigmoid_alive = sigmoid_alive,
                                            energy_minimization = energy_minimization,
                                            low_rank_approx = low_rank_approx,
                                            multi_head_nca = multi_head_nca,
                                            correct_alive = correct_alive,
                                            energy_multi_head = energy_multi_head,
                                            energy_coeff_init = energy_coeff_init,
                                        )
                            ]
                            attn_list += [
                                Attention(
                                    dim,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_norm=qk_norm,
                                    attn_drop=attn_drop,
                                    proj_drop=proj_drop,
                                    norm_layer=norm_layer,
                                    alive_channel = alive_channel,
                                    sigmoid_alive = sigmoid_alive,
                                    alive_threshold = alive_threshold,
                                    correct_alive = correct_alive,
                                    paas = paas,
                                    v2 = v2,
                                    cosine_attn = cosine_attn,
                                )
                            ]
                            norm_list += [
                                norm_layer(dim)
                            ]
                            weight_list += [
                                torch.zeros(2)
                            ]
                            drop_path_list += [
                                DropPath(drop_path[stage_idx]) if drop_path[stage_idx] > 0. else nn.Identity()
                            ]
                            ls_list += [
                                LayerScale(dim) if layer_scale > 0. else nn.Identity()
                            ]
                        self.nca_list = nn.ModuleList(nca_list)
                        self.attn_list = nn.ModuleList(attn_list)
                        self.norm_list = nn.ModuleList(norm_list)
                        self.weight_list = nn.Parameter(torch.stack(weight_list)) if weighted_combine else torch.stack(weight_list)
                        self.drop_path_list = nn.ModuleList(drop_path_list)
                        self.ls_list = nn.ModuleList(ls_list)
                            
                    else: # normal recurrent
                        self.nca = NCA(
                            dim = dim if not linear_downsample else downsample_dim,
                            num_heads = num_heads,
                            act_layer = act_layer,
                            norm_layer = nca_norm_layer,
                            separate_norm = separate_norm,
                            stochastic_update = stochastic_update,
                            times = times,
                            alive_channel = alive_channel,
                            alive_threshold = alive_threshold,
                            trainable_kernel = trainable_kernel,
                            normalize_filter = normalize_filter,
                            padding_mode = padding_mode,
                            multi_head_perception = multi_head_perception,
                            perception_scales = perception_scales,
                            pos_emb = pos_emb,
                            perception_aggr = perception_aggr,
                            sigmoid_alive = sigmoid_alive,
                            energy_minimization = energy_minimization,
                            low_rank_approx = low_rank_approx,
                            multi_head_nca = multi_head_nca,
                            correct_alive = correct_alive,
                            energy_multi_head = energy_multi_head,
                            energy_coeff_init = energy_coeff_init,
                        )
                # elif(nca_type == "attn"):
                #     self.nca = AttnNCA(
                #         dim = dim if not linear_downsample else downsample_dim, 
                #         num_heads=num_heads,
                #         mlp_ratio=4,
                #         qkv_bias=qkv_bias,
                #         qk_norm=qk_norm,
                #         times = times,
                #         act_layer=act_layer,
                #         norm_layer=norm_layer,
                #         alive_channel = alive_channel,
                #         sigmoid_alive = sigmoid_alive,
                #         alive_threshold = alive_threshold,
                #         perception_scales = perception_scales,
                #         block_type = "normal",
                #         linear_combine = False,
                #         correct_alive = correct_alive,
                #         energy_minimization = energy_minimization,
                #         stochastic_update = stochastic_update,
                #         no_norm = True,
                #     )
                # elif(nca_type == "convattn"):
                #     self.nca1 = NCA(
                #         dim = dim if not linear_downsample else downsample_dim,
                #         num_heads = num_heads,
                #         act_layer = act_layer,
                #         norm_layer = nca_norm_layer,
                #         separate_norm = separate_norm,
                #         stochastic_update = stochastic_update,
                #         times = times,
                #         alive_channel = alive_channel,
                #         alive_threshold = alive_threshold,
                #         trainable_kernel = trainable_kernel,
                #         normalize_filter = normalize_filter,
                #         padding_mode = padding_mode,
                #         multi_head_perception = multi_head_perception,
                #         perception_scales = perception_scales,
                #         pos_emb = pos_emb,
                #         perception_aggr = perception_aggr,
                #         sigmoid_alive = sigmoid_alive,
                #         energy_minimization = energy_minimization,
                #         low_rank_approx = low_rank_approx,
                #         multi_head_nca = multi_head_nca,
                #         correct_alive = correct_alive,
                #     )
                #     self.nca2 = AttnNCA(
                #         dim = dim if not linear_downsample else downsample_dim, 
                #         num_heads=num_heads,
                #         mlp_ratio=4,
                #         qkv_bias=qkv_bias,
                #         qk_norm=qk_norm,
                #         times = times,
                #         act_layer=act_layer,
                #         norm_layer=norm_layer,
                #         alive_channel = alive_channel,
                #         sigmoid_alive = sigmoid_alive,
                #         alive_threshold = alive_threshold,
                #         perception_scales = perception_scales,
                #         block_type = "normal",
                #         linear_combine = False,
                #         correct_alive = correct_alive,
                #         energy_minimization = energy_minimization,
                #         stochastic_update = stochastic_update,
                #         no_norm = True,
                #     )
            elif(ablation_nca):
                self.nca = NCA(
                    dim = dim if not linear_downsample else downsample_dim,
                    num_heads = num_heads,
                    act_layer = act_layer,
                    norm_layer = nca_norm_layer,
                    separate_norm = separate_norm,
                    stochastic_update = stochastic_update,
                    times = times,
                    alive_channel = alive_channel,
                    alive_threshold = alive_threshold,
                    trainable_kernel = trainable_kernel,
                    normalize_filter = normalize_filter,
                    padding_mode = padding_mode,
                    multi_head_perception = multi_head_perception,
                    perception_scales = perception_scales,
                    pos_emb = pos_emb,
                    perception_aggr = perception_aggr,
                    sigmoid_alive = sigmoid_alive,
                    energy_minimization = energy_minimization,
                    low_rank_approx = low_rank_approx,
                    multi_head_nca = multi_head_nca,
                    ablation_nca = ablation_nca,
                    correct_alive = correct_alive,
                )
                # nca_list = []
                # for i in range(times):
                #     nca_list += [NCA(
                #         dim = dim,
                #         num_heads = num_heads,
                #         act_layer = act_layer,
                #         norm_layer = nca_norm_layer,
                #         separate_norm = separate_norm,
                #         stochastic_update = stochastic_update,
                #         times = 1,
                #         alive_channel = alive_channel,
                #         alive_threshold = alive_threshold,
                #         trainable_kernel = trainable_kernel,
                #         normalize_filter = normalize_filter,
                #         padding_mode = padding_mode,
                #         multi_head_perception = multi_head_perception,
                #         perception_scales = perception_scales,
                #         pos_emb = pos_emb,
                #         perception_aggr = perception_aggr,
                #         sigmoid_alive = sigmoid_alive,
                #         energy_minimization = energy_minimization,
                #         low_rank_approx = low_rank_approx,
                #         multi_head_nca = multi_head_nca,
                #     )]
                #     self.nca = nn.ModuleList(nca_list)
        elif(multi_head_nca):
            
            self.mlp_proj = mlp_proj
            # self.nca = NCA(
            #     dim = dim // num_heads,
            #     num_heads = 1,
            #     act_layer = act_layer,
            #     norm_layer = nca_norm_layer,
            #     separate_norm = separate_norm,
            #     stochastic_update = stochastic_update,
            #     times = times,
            #     alive_channel = alive_channel,
            #     alive_threshold = alive_threshold,
            #     trainable_kernel = trainable_kernel,
            #     normalize_filter = normalize_filter,
            #     padding_mode = padding_mode,
            #     multi_head_perception = False,
            #     perception_scales = perception_scales,
            #     pos_emb = pos_emb,
            #     perception_aggr = perception_aggr,
            #     sigmoid_alive = sigmoid_alive,
            #     energy_minimization = energy_minimization,
            #     low_rank_approx = low_rank_approx,
            # )
            self.nca_list = []
            self.nca_num = 3
            for i in range(self.nca_num):
                self.nca_list += [
                        NCA(
                        dim = dim // num_heads,
                        num_heads = 1,
                        act_layer = act_layer,
                        norm_layer = nca_norm_layer,
                        separate_norm = separate_norm,
                        stochastic_update = stochastic_update,
                        times = times,
                        alive_channel = alive_channel,
                        alive_threshold = alive_threshold,
                        trainable_kernel = trainable_kernel,
                        normalize_filter = normalize_filter,
                        padding_mode = padding_mode,
                        multi_head_perception = False,
                        perception_scales = perception_scales,
                        pos_emb = pos_emb,
                        perception_aggr = perception_aggr,
                        sigmoid_alive = sigmoid_alive,
                        energy_minimization = energy_minimization,
                        low_rank_approx = low_rank_approx,
                        correct_alive = correct_alive,
                    )
                ]
            self.nca = nn.ModuleList(self.nca_list)
            if(mlp_proj):
                """To facilitate head communication???"""
                self.mlp = Mlp(
                    in_features=dim,
                    hidden_features=int(dim * 4),
                    act_layer=act_layer,
                    drop=proj_drop,
                )

        

        self.linear_combine = linear_combine # only valid with ODE block design
        if(linear_combine):
            self.combine_layer = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Tanh(),
            )
            self.ls = LayerScale(dim, init_values=1.0)
        
        self.weighted_combine = weighted_combine
        self.sigmoid = nn.Sigmoid()
        if(weighted_combine and not self.multi_stage):
            if(nca_type != "convattn"):
                self.perception_aggr = PerceptionAggr(dim, num_scales=2, multi_head_combine=False, head_num=1, scale_weight=True)
#                 self.weight = nn.Parameter(torch.zeros(2))
#                 if(num_layer == 0):
#                     pass
#                 elif(num_layer == 1):
#                     self.weight = nn.Parameter(torch.FloatTensor([1.0, 0.0]))
#                 elif(num_layer == 2):
#                     self.weight = nn.Parameter(torch.FloatTensor([1.5, -0.1]))
#                 elif(num_layer == 3):
#                     self.weight = nn.Parameter(torch.FloatTensor([2.0, -0.2]))
            else:
                self.weight = nn.Parameter(torch.zeros(3))
        elif(not weighted_combine):
            if(num_layer == 0):
                self.weight = torch.FloatTensor([0.0, 0.0])
            elif(num_layer == 1):
                self.weight = torch.FloatTensor([1.0, -0.25])
            elif(num_layer == 2):
                self.weight = torch.FloatTensor([2.0, -0.5])
            elif(num_layer == 3):
                self.weight = torch.FloatTensor([3.0, -0.75])
        self.v2 = v2
        if(v2 > 0):
            if(v2 == 1):
                pass
            elif(v2 == 2):
                if(group_norm):
                    self.norm2 = nn.GroupNorm(num_heads, dim)
                else:
                    self.norm2 = norm_layer(dim)
            self.init_respostnorm()

        
        # self.mlp = mlp_layer(
        #     in_features=dim,
        #     hidden_features=int(dim * mlp_ratio),
        #     act_layer=act_layer,
        #     drop=proj_drop,
        # )
        # self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x, return_middle_state = False, lower_layer_output_dict = None):
        """Here we might not need the residual after NCA evolution because in NCA it is already residual update"""
        b_x, n_tokens, c_x = x.shape
        self.input_resolution = (n_tokens, c_x)
        if(self.block_type == "normal"):
            x = x + self.drop_path1(self.attn(self.norm1(x)))
            nca_output, return_dict = self.nca(self.norm2(x), return_middle_state)
            
            if(self.residual_nca):
                x = x + self.drop_path2(nca_output)
            else:
                x = self.drop_path2(nca_output)
            # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        elif(self.block_type == "ode"):
            if(not self.multi_stage):
                if(not self.multi_head_nca):
                    if(not self.v2):
                        if(self.group_norm):
                            H = int(n_tokens ** 0.5)
                            W = n_tokens // H
                            self.input_size = (H, W, c_x)
                            if(H * W != n_tokens):
                                print("Token Number Cannot Resize to an Image", H, W, n_tokens)
                                exit()
                            x_gn = x.transpose(1, 2).reshape(b_x, c_x, H, W)
                            x_norm = self.norm1(x_gn)
                            x_norm = x_norm.reshape(b_x, c_x, H*W).transpose(1, 2)
                        else:
                            x_norm = self.norm1(x)
                    elif(self.v2):
                        x_norm = x.clone()
                    if(self.no_global):
                        G = torch.zeros_like(x)
                    else:
                        
                        if(self.recurrent_attention == 0):
                            if(self.v2 == 2):
                                if(not self.group_norm):
                                    G = self.norm1(self.attn(x_norm))
                                else:
                                    G = self.attn(x_norm)
                                    G = G.transpose(1, 2)
                                    G = self.norm1(G)
                                    G = G.transpose(1, 2)
                            else:
                                G = self.attn(x_norm)
                            if(self.v2 != 1):
                                # G = self.drop_path1(G)
                                pass
                        else:
                            if(not self.window_attn):
                                G = self.attn(x_norm)
                                if(isinstance(self.recurrent_attention, list)):
                                    if(self.training):
                                        recur_attn_step = torch.randint(low = self.recurrent_attention[0], high = self.recurrent_attention[1] + 1, size = (1,)).item()
                                    else:
                                        recur_attn_step = (self.recurrent_attention[0] + self.recurrent_attention[1]) // 2
                                else:
                                    recur_attn_step = self.recurrent_attention
                                for attn_step in range(recur_attn_step - 1):
                                    if(self.recurrent_attention_norm):
                                        G = self.attn(self.recur_attn_norm(G))
                                    else:
                                        G = self.attn(G)
                                if(self.v2 == 2):
                                    if(not self.group_norm):
                                        G = self.norm1(G)
                                    else:
                                        G = G.transpose(1, 2)
                                        G = self.norm1(G)
                                        G = G.transpose(1, 2)
                                if(self.v2 != 1):
                                    # G = self.drop_path1(G)
                                    pass
                            elif(self.window_attn):
                                G = x_norm
                                for attn_block in self.attn:
                                    for attn_step in range(self.recurrent_attention - 1):
                                        G = attn_block(G)

                    if(self.nca_type != "convattn"):
                        if(not self.linear_downsample):
                            if(not self.ablation_nca):
                                nca_output, return_dict = self.nca(x_norm, return_middle_state)
                            else:
                                nca_output, return_dict = self.nca(x_norm, return_middle_state)
                                # nca_output = x_norm
                                # for nca_model in self.nca:
                                #     nca_output, return_dict = nca_model(nca_output, return_middle_state)
                        elif(self.linear_downsample):
                            x_norm_head = x_norm.reshape(b_x, n_tokens, self.num_heads, self.head_dim)
                            x_norm_down = self.fc_down(x_norm_head).reshape(b_x, n_tokens, self.downsample_dim)
                            nca_output, return_dict = self.nca(x_norm_down, return_middle_state)
                            nca_output = self.fc_up(nca_output.reshape(b_x, n_tokens, self.num_heads, self.downsample_dim // self.num_heads))
                            nca_output = nca_output.reshape(b_x, n_tokens, c_x)
                        if(self.v2 == 2):
                            if(not self.group_norm):
                                nca_output = self.norm2(nca_output)
                            else:
                                nca_output = nca_output.transpose(1, 2)
                                nca_output = self.norm2(nca_output)
                                nca_output = nca_output.transpose(1, 2)
                        if(self.v2 != 1):
                            # F = self.drop_path2(nca_output)
                            F = nca_output
                        else:
                            F = nca_output
                    
                        if(return_middle_state):
                            return_dict["attn"] = G
                        if(self.residual_nca):
                            if(not self.linear_combine and not self.weighted_combine):
                                #x = x + self.drop_path1(self.ls1(F + G))
                                x = x + self.drop_path1(self.ls1(2.0 * self.sigmoid(self.weight[0]) * F + 2.0 * self.sigmoid(self.weight[1]) * G))
                            else:
                                if(self.linear_downsample):
                                    cell_interaction = torch.cat([F, G], dim = -1)
                                    x = x + self.ls(self.combine_layer(cell_interaction))
                                elif(self.weighted_combine):
                                    #print(2.0 * self.sigmoid(self.weight[0]), 2.0 * self.sigmoid(self.weight[1])) # 1,2 block: 1.5:1, 3,4 block: 2:1
                                    if(not self.v2):
                                        B,N,C = x.shape
                                        H = int(N ** 0.5)
                                        W = N // H
                                        x_weight = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
                                        weight = self.perception_aggr(x_weight) # B,1,H,W,K
                                        B,_,H,W,K = weight.shape
                                        weight = weight.reshape(B, 1, N, K).permute(0, 2, 1, 3)
                                        output_all = torch.stack([F,G], dim = -1) # B,N,C,2
                                        x = x + self.drop_path1(torch.sum(weight * output_all, dim = -1))

                                        #x = x + self.drop_path1(self.ls1(2.0 * self.sigmoid(self.weight[0]) * F + 2.0 * self.sigmoid(self.weight[1]) * G))
                                    else:
                                        if(self.v2 == 1):
                                            if(not self.group_norm):
                                                update_x = self.drop_path1(self.ls1(self.norm1(2.0 * self.sigmoid(self.weight[0]) * F + 2.0 * self.sigmoid(self.weight[1]) * G)))
                                            else:
                                                update_x = 2.0 * self.sigmoid(self.weight[0]) * F + 2.0 * self.sigmoid(self.weight[1]) * G
                                                update_x = update_x.transpose(1, 2)
                                                update_x = self.norm1(update_x)
                                                update_x = self.drop_path1(update_x.transpose(1, 2))
                                            x = x + update_x
                                        elif(self.v2 == 2):
                                            x = x + self.drop_path1(self.ls1(2.0 * self.sigmoid(self.weight[0]) * F + 2.0 * self.sigmoid(self.weight[1]) * G))
                        else:
                            x = x_norm + F + G
                    elif(self.nca_type == "convattn"):
                        nca_output, return_dict = self.nca1(x_norm, return_middle_state)
                        nca_output2, return_dict2 = self.nca2(x_norm, return_middle_state)
                        if(self.weighted_combine):
                            F = 2.0 * self.sigmoid(self.weight[0]) * nca_output + 2.0 * self.sigmoid(self.weight[1]) * nca_output2
                            x = x + F + G * 2.0 * self.sigmoid(self.weight[2])
                        else:
                            x = self.drop_path2(nca_output + nca_output2) + G
                        if(return_middle_state):
                            return_dict["attn"] = G
            elif(self.multi_stage):
                for i, (attn, nca, norm, weight, drop_path, ls) in enumerate(zip(self.attn_list, self.nca_list, self.norm_list, self.weight_list, self.drop_path_list, self.ls_list)):
                    identity = x
                    x_norm = norm(x)
                    attn_out = attn(x_norm)
                    if(self.recurrent_attention):
                        for attn_step in range(self.recurrent_attention - 1):
                            attn_out = attn(attn_out)
                    
                    nca_output, return_dict = nca(x_norm, return_middle_state)
                    if(return_middle_state):
                        return_dict["attn"] = attn_out
                    x = x + drop_path(ls(2.0 * self.sigmoid(weight[0]) * nca_output + 2.0 * self.sigmoid(weight[1]) * attn_out))

            # else:
            #     # x_head = x.reshape(b_x, n_tokens, self.num_heads, self.head_dim)
            #     # x_norm_head = torch.stack([self.norm1[head_idx](x_head[:, :, head_idx]) for head_idx in range(self.num_heads)], dim = 2)
            #     # x_norm = x_norm_head.reshape(b_x, n_tokens, c_x)
            #     x_norm = self.norm1(x)
            #     x_norm_head = x_norm.reshape(b_x, n_tokens, self.num_heads, self.head_dim)
            #     G = self.drop_path1(self.attn(x_norm))
                
            #     # output_list = [self.nca[head_idx](x_norm_head[:, :, head_idx], return_middle_state) for head_idx in range(self.num_heads)]
            #     nca_output = 0.0
            #     for nca_idx in range(self.nca_num):
            #         output_list = [self.nca[nca_idx](x_norm_head[:, :, head_idx, :], return_middle_state) for head_idx in range(self.num_heads)]

            #         nca_output_list, return_dict_list = zip(*output_list)
            #         return_dict = {}
            #         return_dict["middle_state"] = []
            #         min_length = min([len(return_dict_list[i]["middle_state"]) for i in range(self.num_heads)])
            #         for j in range(min_length):
            #             return_dict["middle_state"].append(torch.cat([return_dict_list[i]["middle_state"][j] for i in range(self.num_heads)], dim = 1))
            #         # for head_idx in range(self.num_heads):
            #         #     nca_output, return_dict = self.nca[head_idx](x_norm[:, :, head_idx], return_middle_state)
            #         #     nca_output_list.append(nca_output)
            #         nca_output += torch.cat(nca_output_list, dim = 2) # each output: b,n,head_dim
            #     if(self.mlp_proj):
            #         nca_output = self.mlp(nca_output)
            #     F = self.drop_path2(nca_output)
            #     if(self.residual_nca):
            #         x = x + F + G
            #     else:
            #         x = x_norm + F + G
        if(self.sparse_query > 0):
            low_layer_info_list = []
            for i, sparse_query_layer in enumerate(self.sparse_query_list):
                low_x = lower_layer_output_dict[f"block.{i}"]
                low_layer_info, indices = sparse_query_layer(x, low_x, return_middle_state)
                low_layer_info_list.append((low_layer_info, indices))
            for i, low_layer_info in enumerate(low_layer_info_list):
                low_layer_values = low_layer_info[0]
                update_indices = low_layer_info[1]
                B,N,C = x.shape
                x_selected = x[torch.arange(B).unsqueeze(1), update_indices]
                x_update = x_selected + low_layer_values * self.sigmoid(self.sparse_query_weight[i])
                new_x = torch.zeros_like(x)
                new_x[torch.arange(B).unsqueeze(1), update_indices] = x_update
                x = x + new_x
                # x[torch.arange(B).unsqueeze(1), update_indices] = x_update

        if(return_middle_state):
            """return_dict already has a middle_state"""
            return_dict["nca_state"] = nca_output
            return_dict["output"] = x
        return x, return_dict

    def init_respostnorm(self):
        nn.init.constant_(self.norm1.bias, 0)
        nn.init.constant_(self.norm1.weight, 0)
        if(self.v2 == 2):
            nn.init.constant_(self.norm2.bias, 0)
            nn.init.constant_(self.norm2.weight, 0)

    def flops(self, verbose = False):
        flops = 0
        if(self.block_type == "ode"):
            # norm1
            flops += self.input_resolution[0] * self.input_resolution[1]
            if(verbose):
                print("Block norm: ", flops / 1e9)
                flops_cur = flops
            
            if(not self.multi_stage):
                if(not self.multi_head_nca):
                    if(self.nca_type != "convattn"):
                        if(not self.ablation_nca):
                            flops += self.nca.flops(verbose)
                        else:
                            flops += self.nca.flops(verbose)
                            # for nca_model in self.nca:
                            #     flops += nca_model.flops(verbose)
                    else:
                        flops += self.nca1.flops(verbose)
                        flops += self.nca2.flops(verbose)
                else:
                    #flops += self.nca.flops(verbose) * self.num_heads * self.nca_num
                    for nca_model in self.nca:
                        flops += nca_model.flops(verbose) * self.num_heads
                if(verbose):
                    print("Block NCA: ", (flops - flops_cur) / 1e9)
                    flops_cur = flops
                
                if(not self.no_global):
                    if(isinstance(self.recurrent_attention, list)):
                        recur_attn_step = (self.recurrent_attention[0] + self.recurrent_attention[1]) // 2
                    else:
                        recur_attn_step = self.recurrent_attention
                    if(isinstance(self.attn, nn.Sequential)):
                        for attn_block in self.attn:
                            for idx in range(2):
                                flops += attn_block[idx].flops(verbose) * (recur_attn_step + 1)
                    else:
                        flops += self.attn.flops(verbose) * (recur_attn_step + 1)
                    if(verbose):
                        print("Block Attn: ", (flops - flops_cur) / 1e9)
                        flops_cur = flops
            elif(self.multi_stage):
                for nca_model in self.nca_list:
                    flops += nca_model.flops(verbose)
                if(verbose):
                    print("Block NCA: ", (flops - flops_cur) / 1e9)
                    flops_cur = flops

                for attn in self.attn_list:
                    flops += attn.flops(verbose) * (self.recurrent_attention + 1)
                if(verbose):
                    print("Block Attn: ", (flops - flops_cur) / 1e9)
                    flops_cur = flops

            if(self.sparse_query > 0):
                for i, sparse_query_layer in enumerate(self.sparse_query_list):
                    flops += sparse_query_layer.flops(verbose)
            if(self.linear_combine):
                flops += self.input_resolution[0] * self.input_resolution[1] * 2 * self.input_resolution[1] + self.input_resolution[0] * self.input_resolution[1] * self.input_resolution[1]

            return flops
        elif(self.block_type == "normal"):
            # norm1 + norm2
            flops += self.input_resolution[0] * self.input_resolution[1] * 2
            if(verbose):
                print("Block norm: ", flops)
                flops_cur = flops

            flops += self.nca.flops()
            if(verbose):
                print("Block NCA: ", flops - flops_cur)
                flops_cur = flops

            flops += self.attn.flops()
            if(verbose):
                print("Block Attn: ", flops - flops_cur)
                flops_cur = flops

            return flops


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """

    def __init__(
            self,
            dim: int,
            out_dim: Optional[int] = None,
            norm_layer: Callable = nn.LayerNorm,
            alive_channel = 0,
            alive_threshold = 0.1,
            learn = False,
            learn_alive_only = False,
            sigmoid_alive = False,
            correct_alive = 0,
    ):
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels (or 2 * dim if None)
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.learn = learn
        self.learn_alive_only = learn_alive_only
        
        self.alive_func = nn.Sigmoid() if sigmoid_alive else nn.Identity()
        
        self.correct_alive = correct_alive
        
        if(learn):
            if(learn_alive_only):
                # self.downsample = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0, bias = False)
                # init_mean = torch.ones(1,1,2,2) / 4.0
                # specific_weight_values = init_mean.as_strided(self.downsample.weight.size(), self.downsample.weight.stride())
                # self.downsample.weight.data = specific_weight_values.contiguous()
                # self.downsample.weight.data.fill_(0)
                # self.downsample.weight.data += 0.25
                self.downsample_weight = nn.Conv2d(in_channels=dim, out_channels=4, kernel_size=2, stride=2, padding=0, bias = False)
                # torch.nn.init.constant_(self.downsample.weight.data, 0.25)
            else:
                # self.downsample = nn.Conv2d(in_channels=dim, out_channels=dim * 4, kernel_size=3, stride=2, padding=2, padding_mode="zeros", dilation=2)
                self.downsample = nn.Conv2d(in_channels=dim, out_channels=dim * 4, kernel_size=2, stride=2, padding=0)
        self.out_dim = out_dim or 2 * dim
        
        self.alive_mask = alive_threshold > 0.0
        self.alive_channel = alive_channel
        self.alive_threshold = alive_threshold
        
        dim_alive_reduction = 0
        if(self.alive_mask):
            if(self.correct_alive):
                if(self.correct_alive >= 3):
                    dim_alive_reduction = 0
                else:
                    dim_alive_reduction = 3
            if(learn_alive_only and learn):
                dim_alive_reduction = 3
            
        
        self.norm = norm_layer(4 * dim - dim_alive_reduction)

        self.reduction = nn.Linear(4 * dim - dim_alive_reduction, self.out_dim, bias=False)

        

    def forward(self, x):
        b_x, n_tokens, c_x = x.shape
        H = int(n_tokens ** 0.5)
        W = n_tokens // H
        self.input_resolution = (H, W, c_x)
        if(H * W != n_tokens):
            print("Token Number Cannot Resize to an Image", H, W, n_tokens)
            exit()
        x = x.reshape(b_x, H, W, c_x)
        B, H, W, C = x.shape
        _assert(H % 2 == 0, f"x height ({H}) is not even.")
        _assert(W % 2 == 0, f"x width ({W}) is not even.")
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5) # B, H', W', 2, 2, C
        # x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2
        # x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2
        # x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2
        # x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2

        if(self.alive_mask and not self.learn):
            """There shouldn't be alive_func actually!!!"""
            if(self.correct_alive):
                if(self.correct_alive >= 3):
                    if(self.correct_alive == 3):
                        alive_values = x[..., self.alive_channel]
                    elif(self.correct_alive == 4):
                        alive_values = self.alive_func(x[..., self.alive_channel])
                    x_features = x[..., self.alive_channel+1:]
                    alive_value_fill = torch.mean(alive_values, dim = [-2, -1], keepdim = True) # B, H', W', 1, 1
                    expanded_mean = alive_value_fill.expand_as(alive_values).unsqueeze(-1)
                    x = torch.cat([expanded_mean, x_features], dim=-1)
                else:
                    alive_values = x[..., self.alive_channel]
                    x_features = x[..., self.alive_channel+1:] # B, H', W', 2, 2, C-1
                    alive_value_fill = torch.mean(alive_values, dim = [-2, -1]).unsqueeze(-1) # B, H', W', 1
                    # x = torch.cat([expanded_mean, x_features], dim=-1)
                    x_features = x_features.flatten(3) # B, H', W', 4C-4
                    x = torch.cat([alive_value_fill, x_features], dim=-1)
                
            else:
                alive_values = self.alive_func(x[..., self.alive_channel])
                x_features = x[..., self.alive_channel+1:]
                alive_value_fill = torch.mean(alive_values, dim = [-2, -1], keepdim = True) # B, H', W', 1, 1
                expanded_mean = alive_value_fill.expand_as(alive_values).unsqueeze(-1)
                x = torch.cat([expanded_mean, x_features], dim=-1)
        
        if(self.learn):
            x = x.permute(0, 1, 4, 2, 3, 5).reshape(B, H, W, C).permute(0, 3, 1, 2)
            if(self.learn_alive_only):
                """Assume 0 alive channel, or we need concat feature before and after alive channel"""
                x_feature = x[:, self.alive_channel+1:, ...].contiguous()
                x_alive = x[:, :self.alive_channel+1, ...].contiguous() # B, 1, H, W
                alive_combine_weight = self.downsample_weight(x) # B, 4, H/2, W/2
                alive_combine_weight = alive_combine_weight.reshape(B, 2, 2, H//2, W//2).permute(0,3,1,4,2).reshape(B,H,W,1).permute(0, 3, 1, 2)
                new_alive = F.avg_pool2d(x_alive*alive_combine_weight, kernel_size = 2, stride = 2) # B, 1, H', W'
                new_alive = new_alive.permute(0,2,3,1)
#                 new_alive = self.downsample(x_alive.contiguous()).permute(0, 2, 3, 1).contiguous()
                x_feature = x_feature.contiguous().permute(0, 2, 3, 1).contiguous().reshape(B, H // 2, 2, W // 2, 2, C-1).contiguous().permute(0, 1, 3, 4, 2, 5).contiguous().flatten(3).contiguous() # B, H', W', 2, 2, C-1
                x = torch.cat([new_alive, x_feature], dim = -1).contiguous()
                
            else:
                x = self.downsample(x)
                x = x.permute(0, 2, 3, 1) # B, H', W', C*4
        else:
            if(not self.correct_alive):
                x = x.flatten(3)
            elif(self.correct_alive >= 3):
                x = x.flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H*W, C)
        return x
    
    def flops(self, verbose = False):
        H, W, C = self.input_resolution
        flops = H * W * C
        flops += (H // 2) * (W // 2) * 4 * C * 2 * C
        if(verbose):
            print("Patch Merging: ", flops / 1e9)
        return flops

class NCAFormer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 4,
            num_heads: list = [8, 8, 8, 8],
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block_NCA,
            mlp_layer: Callable = Mlp,
            nca_norm = False,
            separate_norm = False,
            stochastic_update = 0.0,
            times : list = [2, 2, 6, 2],
            alive_channel = 0,
            alive_threshold = 0.1,
            trainable_kernel = False,
            normalize_filter = False,
            padding_mode = "replicate",
            multi_head_perception = False,
            perception_scales = [0],
            pos_emb = None,
            perception_aggr = "concat",
            block_type = "normal",
            learned_patch_merging = False,
            learn_patch_merging_alive_only = False,
            residual_nca = False,
            solver = "Euler",
            head_with_alive = False,
            sigmoid_alive = False,
            energy_minimization = False,
            low_rank_approx = False,
            multi_head_nca = False,
            mlp_proj = False,
            ablation_nca = False,
            linear_downsample = False,
            cnn_front_end = "None",
            local_attn_v2 = 0,
            local_attn_v2_block_type = "ode",
            middle_linear_supervision = 0,
            linear_combine = False,
            correct_alive = 0,
            no_global = 0,
            nca_str = "cccc",
            recurrent_attention = [0,0,0,0],
            paas = 0,
            weighted_combine = 0,
            sparse_query = 0,
            sparse_query_method = "aliveness",
            overlap_patch_embed = 0,
            group_norm = 0,
            v2 = 0,
            recurrent_attention_norm = 0,
            cosine_attn = 0,
            energy_multi_head = 0,
            energy_coeff_init = 0.01,
            relative_pos_emb = 0,
            window_attn_str = '0000',
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token', 'map')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        if(nca_norm):
            nca_norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            nca_norm_layer = None
        act_layer = act_layer or nn.GELU

        self.cnn_front_end = cnn_front_end
        self.local_attn_v2 = local_attn_v2
        self.local_attn_v2_block_type = local_attn_v2_block_type
        self.sparse_query = sparse_query
        if(cnn_front_end != "None"):
            if(cnn_front_end == "vone"):
                self.frontend, self.front_downsample = VOneNet(simple_channels=256, complex_channels=256, noise_mode = None, first_embed_dim = embed_dim) # "neuronal"
                # output: b, 512, 56, 56
                num_patches = 56*56
            elif(cnn_front_end == "resnet50"):
                from torchvision.models import resnet50, resnet152
                resnet_frontend = resnet152(weights="IMAGENET1K_V2")
                """Fetch layer 2"""
                self.frontend = nn.Sequential(*[resnet_frontend.conv1, 
                                                resnet_frontend.bn1, 
                                                resnet_frontend.relu, 
                                                resnet_frontend.maxpool, 
                                                resnet_frontend.layer1]) # layer1: 256*56*56, layer2: 512*28*28
                num_patches = 56*56
                self.front_downsample = nn.Conv2d(256, embed_dim, kernel_size=1, stride=1, bias=False)
                # output: b, 512, 28, 28
            for p in self.frontend.parameters():
                p.requires_grad = False
        self.head_with_alive = head_with_alive
        self.alive_channel = alive_channel
        self.alive_threshold = alive_threshold
        self.alive_mask = alive_threshold > 0.0
        self.alive_func = nn.Sigmoid() if sigmoid_alive else nn.Identity()
        if(head_with_alive):
            assert self.alive_mask

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        if(cnn_front_end == "None"):
            norm_layer_patch_emb = norm_layer
            if(overlap_patch_embed):
                embed_layer = PatchEmbedOverLap
                norm_layer_patch_emb = norm_layer if overlap_patch_embed == 1 else None
            self.patch_embed = embed_layer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
                dynamic_img_pad=dynamic_img_pad,
                norm_layer = norm_layer_patch_emb,
                **embed_args,
            )
            num_patches = self.patch_embed.num_patches
        else:
            pass

        self.solver = solver

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()
        
        dpr_depth = depth
        actual_depth_list = []
        for time in times:
            if(isinstance(time, list) and isinstance(time[0], list)):
                actual_depth_list.append(len(time))
            else:
                actual_depth_list.append(1)
        dpr_all = [x.item() for x in torch.linspace(0, drop_path_rate, sum(actual_depth_list))]  # stochastic depth decay rule
        dpr = []
        dd = 0
        for actual_depth in actual_depth_list:
            if(actual_depth == 1):
                dpr.append(dpr_all[dd])
            else:
                dpr.append(dpr_all[dd:dd+actual_depth])
            dd += actual_depth
        dpr = [drop_path_rate / depth * x for x in range(1, depth + 1)] # [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

#         spr = [x.item() for x in torch.linspace(0, stochastic_update, depth)][::-1] if stochastic_update > 0.0 else torch.zeros(depth)
        spr = [stochastic_update / depth * x for x in range(1, depth + 1)] if stochastic_update > 0.0 else torch.zeros(depth)

        self.depth = depth
        assert len(num_heads) == depth
        self.nca_blocks = []
        in_dim = embed_dim
        for i in range(depth):
            if(self.local_attn_v2 > 0 and i < self.local_attn_v2):
                self.nca_blocks += [
                    AttnNCA(
                        dim = in_dim, 
                        num_heads=num_heads[i],
                        mlp_ratio=4,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_norm,
                        proj_drop=proj_drop_rate,
                        attn_drop=attn_drop_rate,
                        times = times[i],
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        alive_channel = alive_channel,
                        sigmoid_alive = sigmoid_alive,
                        alive_threshold = alive_threshold,
                        perception_scales = [0],
                        block_type = self.local_attn_v2_block_type,
                        linear_combine = linear_combine,
                        correct_alive = correct_alive,
                        energy_minimization = energy_minimization,
                    )
                ]
            else:
                # if(correct_alive == 3):
                #     if(i >= depth - 2):
                #         ps = perception_scales
                #     else:
                #         ps = [0]
                # else:
                #     ps = perception_scales
                if(nca_str != "None"):
                    if(nca_str[i] == "c"):
                        nca_type = "conv"
                    elif(nca_str[i] == "a"):
                        nca_type = "attn"
                    elif(nca_str[i] == "s"):
                        nca_type = "convattn"
                else:
                    nca_type = "conv"
                if(i < no_global):
                    ng = True
                else:
                    ng = False
                if(energy_minimization > 1):
                    em = 1 if i >= energy_minimization else 0
                elif(energy_minimization == 1):
                    em = 1
                else:
                    em = 0
                if(paas and i>=2):
                    paas_size = (img_size // patch_size // (2**i)) ** 2
                else:
                    paas_size = 0

                if(sparse_query <= i and sparse_query > 0 and i >= 2): # only layer higher than <sparse_query> can query the info of low layer. 
                    sparse_query_layer = i
                else:
                    sparse_query_layer = 0
                
                window_attn = int(window_attn_str[i])
                print(f"NCA {i} spr {spr[i]} dpr {dpr[i]}")
                self.nca_blocks += [
                    Block_NCA(
                        dim = in_dim,
                        num_heads = num_heads[i],
                        num_layer = i,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_norm,
                        drop_path=dpr[i],
                        proj_drop=proj_drop_rate,
                        attn_drop=attn_drop_rate,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        nca_norm_layer = nca_norm_layer,
                        separate_norm = separate_norm,
                        stochastic_update = spr[i],
                        times = times[i],
                        alive_channel = alive_channel,
                        alive_threshold = alive_threshold,
                        trainable_kernel = trainable_kernel,
                        normalize_filter = normalize_filter,
                        padding_mode = padding_mode,
                        multi_head_perception = multi_head_perception,
                        perception_scales = perception_scales if i >= 2 else [0],
                        pos_emb = pos_emb, 
                        perception_aggr = perception_aggr,
                        block_type = block_type,
                        residual_nca = residual_nca,
                        sigmoid_alive = sigmoid_alive,
                        energy_minimization = em,
                        low_rank_approx = low_rank_approx,
                        multi_head_nca = multi_head_nca,
                        mlp_proj = multi_head_nca,
                        ablation_nca = ablation_nca,
                        linear_downsample = linear_downsample, 
                        linear_combine = linear_combine,
                        correct_alive = correct_alive,
                        no_global = ng,
                        nca_type = nca_type,
                        recurrent_attention = recurrent_attention[i],
                        paas = paas_size,
                        weighted_combine = weighted_combine,
                        sparse_query = sparse_query_layer,
                        sparse_query_method = sparse_query_method,
                        group_norm = group_norm,
                        v2 = v2,
                        recurrent_attention_norm = recurrent_attention_norm,
                        cosine_attn = cosine_attn,
                        energy_multi_head = energy_multi_head,
                        energy_coeff_init = energy_coeff_init,
                        relative_pos_emb = relative_pos_emb,
                        window_attn = window_attn,
                    )
            ]
            in_dim *= 2
        out_token_dim = self.nca_blocks[-1].dim
        self.nca_blocks = nn.Sequential(*self.nca_blocks)
        """Downsampling part"""
        in_dim = embed_dim
        self.downsample_layer = []
        for i in range(depth - 1):
            self.downsample_layer += [
                PatchMerging(
                    dim = in_dim,
                    norm_layer = norm_layer,
                    alive_channel = alive_channel,
                    alive_threshold = alive_threshold,
                    learn = learned_patch_merging,
                    learn_alive_only = learn_patch_merging_alive_only,
                    sigmoid_alive = sigmoid_alive,
                    correct_alive = correct_alive,
                )
            ]
            in_dim *= 2
        self.downsample_layer = nn.Sequential(*self.downsample_layer)
        self.norm = norm_layer(out_token_dim) if not use_fc_norm else nn.Identity()
        print("Former norm", self.norm)

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        if(not group_norm):
            self.fc_norm = norm_layer(out_token_dim) if use_fc_norm else nn.Identity()
        else:
            self.fc_norm = nn.GroupNorm(num_heads[-1], out_token_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.out_token_dim = out_token_dim
        self.num_classes = num_classes
        self.head = nn.Linear(out_token_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.middle_linear_supervision = middle_linear_supervision
        if(middle_linear_supervision):
            middle_linear_out = []
            if(self.num_classes >= 100):
                middle_layer_out_norm = []
            for i in range(middle_linear_supervision):
                middle_linear_out += [
                    nn.Linear(embed_dim * (2 ** (depth - i - 2)), self.num_classes)
                ]
                if(self.num_classes >= 100):
                    if(not group_norm):
                        middle_layer_out_norm += [
                            norm_layer(embed_dim * (2 ** (depth - i - 2)))
                        ]
                    elif(group_norm):
                        middle_layer_out_norm += [
                            nn.GroupNorm(num_heads[depth - i - 2], embed_dim * (2 ** (depth - i - 2)))
                        ]
            if(self.num_classes >= 100):
                self.middle_layer_out_norm = nn.Sequential(*middle_layer_out_norm)
            self.middle_linear_out = nn.Sequential(*middle_linear_out)
            self.middle_linear_out_layers = [depth - i - 2 for i in range(middle_linear_supervision)]
            


        if weight_init != 'skip':
            self.init_weights(weight_init)
#         if(self.num_classes == 1000):
#             self.head.weight.data.div_(10.0)
#             for layer in self.middle_linear_out:
#                 layer.weight.data.div_(10.0)

    def flops(self, verbose = False):
        flops = 0
        if(self.cnn_front_end == "None"):
            flops += self.patch_embed.flops()
            if(verbose):
                print("Patch Embed: ", flops)
        for i, nca_block in enumerate(self.nca_blocks):
            if(verbose):
                print("############")
                print(f"Block {i}")
            flops += nca_block.flops(verbose)
            if(i < self.depth - 1):
                flops += self.downsample_layer[i].flops(verbose)
        # flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers) # no norm
        flops += self.out_token_dim * self.num_classes
        return flops

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map ' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x):
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ):
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def forward_features(self, x, return_middle_state = False):
        return_dict_final = {}
        nca_middle_state_dict = {}
        middle_block_output = {}
        after_merge = {}
        middle_state_list = []
        if(self.cnn_front_end == "None"):
            x = self.patch_embed(x)
        else:
            x = self.frontend(x)
            return_dict_final["frontend"] = x.clone()
            x = self.front_downsample(x) # b, dim, h, w
            return_dict_final["frontend_down"] = x.clone()
            b, dim, h, w = x.shape
            x = x.reshape(b, dim, h*w).transpose(1,2)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.blocks, x)
        # else:
        #     x = self.blocks(x)
        lower_layer_output_dict = {}
        if(self.solver == "Euler"):
            for i, nca_block in enumerate(self.nca_blocks):
                x, return_dict = nca_block(x, return_middle_state, lower_layer_output_dict)
                if(self.sparse_query):
                    lower_layer_output_dict[f"block.{i}"] = return_dict["output"]
                if(self.middle_linear_supervision):
                    assert return_middle_state, "Need middle state to do middle supervision"
                if(return_middle_state):
                    middle_state_list.append(return_dict["nca_state"])
                    return_dict_final[f"block_nca.{i}"] = return_dict["nca_state"]
                    return_dict_final[f"block_attn.{i}"] = return_dict["attn"]
                    return_dict_final[f"block.{i}"] = return_dict["output"]
                    
                    if(self.middle_linear_supervision):
                        if(i in self.middle_linear_out_layers):
                            if(self.num_classes >= 100):
                                return_dict_final[f"block_class.{i}"] = self.middle_linear_out[self.depth - i - 2](self.middle_layer_out_norm[self.depth - i - 2](return_dict["output"].mean(dim = 1)))
                            else:
                                return_dict_final[f"block_class.{i}"] = self.middle_linear_out[self.depth - i - 2](return_dict["output"].mean(dim = 1))

                    for key in return_dict.keys():
                        if("alive_mask" in key or "energy" in key):
                            return_dict_final[f"block_nca.{i}.{key}"] = return_dict[key]
                    nca_middle_state_list = return_dict["middle_state"]
                    for m_length in range(len(nca_middle_state_list)):
                        return_dict_final[f"block_nca_middle.{i}.{m_length}"] = nca_middle_state_list[m_length]
                if(i < self.depth - 1):
                    x = self.downsample_layer[i](x)
                if(return_middle_state):
                    return_dict_final[f"block_merge.{i}"] = x
        elif(self.solver == "RK2"):
            for i, nca_block in enumerate(self.nca_blocks):
                runge_kutta_list = []
                residual = x
                # x = x + 0.5 * (F1 + F2)
                # F1 = F(x), F2 = F(x + F1)
                for step_size in range(2):
                    x, return_dict = nca_block(x, return_middle_state)
                    if(return_middle_state):
                        middle_state_list.append(return_dict["nca_state"])
                        return_dict_final[f"block_nca.{i}"] = return_dict["nca_state"]
                        return_dict_final[f"block.{i}"] = return_dict["output"]
                        nca_middle_state_list = return_dict["middle_state"]
                        for m_length in range(len(nca_middle_state_list)):
                            return_dict_final[f"block_nca_middle.{i}.{m_length}"] = nca_middle_state_list[m_length]
                    runge_kutta_list.append(x)
                    x = residual + x
                x = residual + 1/2 * (runge_kutta_list[0] + runge_kutta_list[1])
                if(i < self.depth - 1):
                    x = self.downsample_layer[i](x)
                if(return_middle_state):
                    return_dict_final[f"block_merge.{i}"] = x
        return_dict_final["nca_middle_state_list"] = middle_state_list
        x = self.norm(x)
        return x, return_dict_final

    def forward_head(self, x, pre_logits: bool = False):
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            if(self.head_with_alive):
                # instead of simple avg pool, we need to only mean over alive token
                B, N, C = x.shape
                x_alive_value = self.alive_func(x[:, :, self.alive_channel])
                x_alive = (x_alive_value > self.alive_threshold).float()
                
                batch_alive = torch.sum(x_alive, dim = 1) # B,
                no_alive_idx = torch.nonzero(batch_alive == 0, as_tuple = True)
                no_alive_data = x[no_alive_idx]
                no_alive = len(no_alive_data) != 0
                if(no_alive):
                    # select the one with highest alive value
                    alive_idx = torch.nonzero(batch_alive != 0, as_tuple = True) # all data points with at least one alive token
                    alive_data = x[alive_idx] # all data points with at least one alive token
                    alive_data = alive_data * x_alive[alive_idx].unsqueeze(-1)
                    alive_data = alive_data.sum(dim=1) / batch_alive[alive_idx].unsqueeze(1) # B', C, including mean over all token in data points with at least one alive token
                    
                    _, token_idx_max_alive = torch.max(x_alive_value, dim = 1) # find the token with the max alive value in the data points without any alive token
                    x_alive_max = torch.gather(x, 1, token_idx_max_alive.unsqueeze(1).unsqueeze(2).expand(-1, -1, C)) # the data point with token having the max alive value, ***including*** the data points with at least one alive token, B, 1, C
                    x_alive_max_no_alive = x_alive_max[no_alive_idx].squeeze(1) # the data point with token having the max alive value, ***excluding*** the data points with at least one alive token
                    x = torch.cat([alive_data, x_alive_max_no_alive], dim = 0)
                    x = x[torch.randperm(B)]
                else:
                    x = x * x_alive.unsqueeze(-1)
                    x = x.sum(dim=1) / batch_alive.unsqueeze(1)
            else:
                x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, return_middle_state = True):
        x, return_dict = self.forward_features(x, return_middle_state)
        x = self.forward_head(x)
        return x, return_dict


def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = ''):
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


def resize_pos_embed(
        posemb,
        posemb_new,
        num_prefix_tokens=1,
        gs_new=(),
        interpolation='bicubic',
        antialias=False,
):
    """ Rescale the grid of position embeddings when loading from state_dict.

    *DEPRECATED* This function is being deprecated in favour of resample_abs_pos_embed

    Adapted from:
        https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info(f'Resized position embedding: {posemb.shape} ({[gs_old, gs_old]}) to {posemb_new.shape} ({gs_new}).')
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=interpolation, antialias=antialias, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


@torch.no_grad()
def _load_weights(model: NCAFormer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    interpolation = 'bilinear'
    antialias = False
    big_vision = False
    if not prefix:
        if 'opt/target/embedding/kernel' in w:
            prefix = 'opt/target/'
        elif 'params/embedding/kernel' in w:
            prefix = 'params/'
            big_vision = True
        elif 'params/img/embedding/kernel' in w:
            prefix = 'params/img/'
            big_vision = True

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
        embed_conv_w = resample_patch_embed(
            embed_conv_w,
            model.patch_embed.proj.weight.shape[-2:],
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )

    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    if model.cls_token is not None:
        model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    if big_vision:
        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
    else:
        pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        old_shape = pos_embed_w.shape
        num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
        pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            new_size=model.patch_embed.grid_size,
            num_prefix_tokens=num_prefix_tokens,
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if (isinstance(model.head, nn.Linear) and
            f'{prefix}head/bias' in w and
            model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]):
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    if model.attn_pool is not None:
        block_prefix = f'{prefix}MAPHead_0/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
        model.attn_pool.latent.copy_(_n2p(w[f'{block_prefix}probe'], t=False))
        model.attn_pool.kv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('key', 'value')]))
        model.attn_pool.kv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('key', 'value')]))
        model.attn_pool.q.weight.copy_(_n2p(w[f'{mha_prefix}query/kernel'], t=False).flatten(1).T)
        model.attn_pool.q.bias.copy_(_n2p(w[f'{mha_prefix}query/bias'], t=False).reshape(-1))
        model.attn_pool.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        model.attn_pool.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        model.attn_pool.norm.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        model.attn_pool.norm.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        for r in range(2):
            getattr(model.attn_pool.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/kernel']))
            getattr(model.attn_pool.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/bias']))

    mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias']))


def _convert_openai_clip(state_dict, model, prefix='visual.'):
    out_dict = {}
    swaps = [
        ('conv1', 'patch_embed.proj'), ('positional_embedding', 'pos_embed'),
        ('transformer.resblocks.', 'blocks.'), ('ln_pre', 'norm_pre'), ('ln_post', 'norm'), ('ln_', 'norm'),
        ('in_proj_', 'qkv.'), ('out_proj', 'proj'), ('mlp.c_fc', 'mlp.fc1'), ('mlp.c_proj', 'mlp.fc2'),
    ]
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, '')
        for sp in swaps:
            k = k.replace(sp[0], sp[1])

        if k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'class_embedding':
            k = 'cls_token'
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == 'pos_embed':
            v = v.unsqueeze(0)
            if v.shape[1] != model.pos_embed.shape[1]:
                # To resize pos embedding when using model at different size from pretrained weights
                v = resize_pos_embed(
                    v,
                    model.pos_embed,
                    0 if getattr(model, 'no_embed_class') else getattr(model, 'num_prefix_tokens', 1),
                    model.patch_embed.grid_size
                )
        out_dict[k] = v
    return out_dict


def _convert_dinov2(state_dict, model):
    import re
    out_dict = {}
    state_dict.pop("mask_token", None)
    if 'register_tokens' in state_dict:
        # convert dinov2 w/ registers to no_embed_class timm model (neither cls or reg tokens overlap pos embed)
        out_dict['reg_token'] = state_dict.pop('register_tokens')
        out_dict['cls_token'] = state_dict.pop('cls_token') + state_dict['pos_embed'][:, 0]
        out_dict['pos_embed'] = state_dict.pop('pos_embed')[:, 1:]
    for k, v in state_dict.items():
        if re.match(r"blocks\.(\d+)\.mlp\.w12\.(?:weight|bias)", k):
            out_dict[k.replace("w12", "fc1")] = v
            continue
        elif re.match(r"blocks\.(\d+)\.mlp\.w3\.(?:weight|bias)", k):
            out_dict[k.replace("w3", "fc2")] = v
            continue
        out_dict[k] = v
    return out_dict


def checkpoint_filter_fn(
        state_dict,
        model,
        adapt_layer_scale=False,
        interpolation='bicubic',
        antialias=True,
):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    prefix = ''

    if 'visual.class_embedding' in state_dict:
        return _convert_openai_clip(state_dict, model)
    elif 'module.visual.class_embedding' in state_dict:
        return _convert_openai_clip(state_dict, model, prefix='module.visual.')

    if "mask_token" in state_dict:
        state_dict = _convert_dinov2(state_dict, model)

    if "encoder" in state_dict:
        state_dict = state_dict['encoder']
        prefix = 'module.'

    if 'visual.trunk.pos_embed' in state_dict:
        # convert an OpenCLIP model with timm vision encoder
        # FIXME remap final nn.Linear if it exists outside of the timm .trunk (ie in visual.head.proj)
        prefix = 'visual.trunk.'

    if prefix:
        # filter on & remove prefix string from keys
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = model.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif adapt_layer_scale and 'gamma_' in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = generate_default_cfgs({

    # re-finetuned augreg 21k FT on in1k weights
    'vit_base_patch16_224.augreg2_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/'),
    'vit_base_patch16_384.augreg2_in21k_ft_in1k': _cfg(),
    'vit_base_patch8_224.augreg2_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/'),

    # How to train your ViT (augreg) weights, pretrained on 21k FT on in1k
    'vit_tiny_patch16_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_tiny_patch16_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_small_patch32_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_small_patch16_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_base_patch32_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_base_patch16_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch8_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_large_patch16_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_large_patch16_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),

    # patch models (weights from official Google JAX impl) pretrained on in21k FT on in1k
    'vit_base_patch16_224.orig_in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        hf_hub_id='timm/'),
    'vit_base_patch16_384.orig_in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch32_384.orig_in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0),

    # How to train your ViT (augreg) weights trained on in1k only
    'vit_small_patch16_224.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_small_patch16_384.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_base_patch32_384.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_base_patch16_384.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),

    'vit_large_patch14_224.untrained': _cfg(url=''),
    'vit_huge_patch14_224.untrained': _cfg(url=''),
    'vit_giant_patch14_224.untrained': _cfg(url=''),
    'vit_gigantic_patch14_224.untrained': _cfg(url=''),

    # patch models, imagenet21k (weights from official Google JAX impl)
    'vit_large_patch32_224.orig_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        hf_hub_id='timm/',
        num_classes=21843),
    'vit_huge_patch14_224.orig_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),

    # How to train your ViT (augreg) weights, pretrained on in21k
    'vit_tiny_patch16_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_small_patch32_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_small_patch16_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_base_patch32_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_base_patch16_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_base_patch8_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_large_patch16_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),

    # SAM trained models (https://arxiv.org/abs/2106.01548)
    'vit_base_patch32_224.sam_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz', custom_load=True,
        hf_hub_id='timm/'),
    'vit_base_patch16_224.sam_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz', custom_load=True,
        hf_hub_id='timm/'),

    # DINO pretrained - https://arxiv.org/abs/2104.14294 (no classifier head, for fine-tune only)
    'vit_small_patch16_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_small_patch8_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_base_patch16_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_base_patch8_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),

    # DINOv2 pretrained - https://arxiv.org/abs/2304.07193 (no classifier head, for fine-tune/features only)
    'vit_small_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_base_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_large_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_giant_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),

    # DINOv2 pretrained w/ registers - https://arxiv.org/abs/2309.16588 (no classifier head, for fine-tune/features only)
    'vit_small_patch14_reg4_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_base_patch14_reg4_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_large_patch14_reg4_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_giant_patch14_reg4_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),

    # ViT ImageNet-21K-P pretraining by MILL
    'vit_base_patch16_224_miil.in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/vit_base_patch16_224_in21k_miil-887286df.pth',
        hf_hub_id='timm/',
        mean=(0., 0., 0.), std=(1., 1., 1.), crop_pct=0.875, interpolation='bilinear', num_classes=11221),
    'vit_base_patch16_224_miil.in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/vit_base_patch16_224_1k_miil_84_4-2deb18e3.pth',
        hf_hub_id='timm/',
        mean=(0., 0., 0.), std=(1., 1., 1.), crop_pct=0.875, interpolation='bilinear'),

    # Custom timm variants
    'vit_base_patch16_rpn_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_base_patch16_rpn_224-sw-3b07e89d.pth',
        hf_hub_id='timm/'),
    'vit_medium_patch16_gap_240.sw_in12k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=11821),
    'vit_medium_patch16_gap_256.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), crop_pct=0.95),
    'vit_medium_patch16_gap_384.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=0.95, crop_mode='squash'),
    'vit_base_patch16_gap_224': _cfg(),

    # CLIP pretrained image tower and related fine-tuned weights
    'vit_base_patch32_clip_224.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch32_clip_384.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, input_size=(3, 384, 384)),
    'vit_base_patch32_clip_448.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, input_size=(3, 448, 448)),
    'vit_base_patch16_clip_224.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=0.95),
    'vit_base_patch16_clip_384.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, crop_pct=1.0),
    'vit_large_patch14_clip_336.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),
    'vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),
    'vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),

    'vit_base_patch32_clip_224.openai_ft_in12k_in1k': _cfg(
        # hf_hub_id='timm/vit_base_patch32_clip_224.openai_ft_in12k_in1k',  # FIXME weight exists, need to push
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch32_clip_384.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=0.95, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_base_patch16_clip_224.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=0.95),
    'vit_base_patch16_clip_384.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=0.95, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_large_patch14_clip_224.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),
    'vit_large_patch14_clip_336.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),

    'vit_base_patch32_clip_224.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch16_clip_224.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),
    'vit_base_patch16_clip_384.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_large_patch14_clip_224.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, crop_pct=1.0),
    'vit_large_patch14_clip_336.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),
    'vit_huge_patch14_clip_224.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),
    'vit_huge_patch14_clip_336.laion2b_ft_in1k': _cfg(
        hf_hub_id='',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),

    'vit_base_patch32_clip_224.openai_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch16_clip_224.openai_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch16_clip_384.openai_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_large_patch14_clip_224.openai_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),

    'vit_base_patch32_clip_224.laion2b_ft_in12k': _cfg(
        #hf_hub_id='timm/vit_base_patch32_clip_224.laion2b_ft_in12k',  # FIXME weight exists, need to push
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821),
    'vit_base_patch16_clip_224.laion2b_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821),
    'vit_large_patch14_clip_224.laion2b_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, crop_pct=1.0, num_classes=11821),
    'vit_huge_patch14_clip_224.laion2b_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=11821),

    'vit_base_patch32_clip_224.openai_ft_in12k': _cfg(
        # hf_hub_id='timm/vit_base_patch32_clip_224.openai_ft_in12k',  # FIXME weight exists, need to push
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821),
    'vit_base_patch16_clip_224.openai_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821),
    'vit_large_patch14_clip_224.openai_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=11821),

    'vit_base_patch32_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=512),
    'vit_base_patch16_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-B-16-laion2B-s34B-b88K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=512),
    'vit_base_patch16_clip_224.datacompxl': _cfg(
        hf_hub_id='laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=512),
    'vit_base_patch16_clip_224.dfn2b': _cfg(
        hf_hub_id='apple/DFN2B-CLIP-ViT-B-16',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=512),
    'vit_large_patch14_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-L-14-laion2B-s32B-b82K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, crop_pct=1.0, num_classes=768),
    'vit_large_patch14_clip_224.datacompxl': _cfg(
        hf_hub_id='laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=768),
    'vit_large_patch14_clip_224.dfn2b': _cfg(
        hf_hub_id='apple/DFN2B-CLIP-ViT-L-14',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=768),
    'vit_huge_patch14_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1024),
    'vit_huge_patch14_clip_224.dfn5b': _cfg(
        hf_hub_id='apple/DFN5B-CLIP-ViT-H-14',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1024),
    'vit_huge_patch14_clip_378.dfn5b': _cfg(
        hf_hub_id='apple/DFN5B-CLIP-ViT-H-14-378',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1024),
    'vit_giant_patch14_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-g-14-laion2B-s12B-b42K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1024),
    'vit_gigantic_patch14_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1280),

    'vit_base_patch32_clip_224.openai': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=512),
    'vit_base_patch16_clip_224.openai': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=512),
    'vit_large_patch14_clip_224.openai': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=768),
    'vit_large_patch14_clip_336.openai': _cfg(
        hf_hub_id='timm/', hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 336, 336), num_classes=768),

    # experimental (may be removed)
    'vit_base_patch32_plus_256.untrained': _cfg(url='', input_size=(3, 256, 256), crop_pct=0.95),
    'vit_base_patch16_plus_240.untrained': _cfg(url='', input_size=(3, 240, 240), crop_pct=0.95),
    'vit_small_patch16_36x1_224.untrained': _cfg(url=''),
    'vit_small_patch16_18x2_224.untrained': _cfg(url=''),
    'vit_base_patch16_18x2_224.untrained': _cfg(url=''),

    # EVA fine-tuned weights from MAE style MIM - EVA-CLIP target pretrain
    # https://github.com/baaivision/EVA/blob/7ecf2c0a370d97967e86d047d7af9188f78d2df3/eva/README.md#eva-l-learning-better-mim-representations-from-eva-clip
    'eva_large_patch14_196.in22k_ft_in22k_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_l_psz14_196px_21k_to_1k_ft_88p6.pt',
        hf_hub_id='timm/', license='mit',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 196, 196), crop_pct=1.0),
    'eva_large_patch14_336.in22k_ft_in22k_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_l_psz14_336px_21k_to_1k_ft_89p2.pt',
        hf_hub_id='timm/', license='mit',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 336, 336), crop_pct=1.0, crop_mode='squash'),
    'eva_large_patch14_196.in22k_ft_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_l_psz14_196px_1k_ft_88p0.pt',
        hf_hub_id='timm/', license='mit',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 196, 196), crop_pct=1.0),
    'eva_large_patch14_336.in22k_ft_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_l_psz14_336px_1k_ft_88p65.pt',
        hf_hub_id='timm/', license='mit',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 336, 336), crop_pct=1.0, crop_mode='squash'),

    'flexivit_small.1200ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_s_i1k.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_small.600ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_s_i1k_600ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_small.300ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_s_i1k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),

    'flexivit_base.1200ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i1k.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_base.600ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i1k_600ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_base.300ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i1k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_base.1000ep_in21k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i21k_1000ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=21843),
    'flexivit_base.300ep_in21k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i21k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=21843),

    'flexivit_large.1200ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_l_i1k.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_large.600ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_l_i1k_600ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_large.300ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_l_i1k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),

    'flexivit_base.patch16_in21k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/vit_b16_i21k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=21843),
    'flexivit_base.patch30_in21k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/vit_b30_i21k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=21843),

    'vit_base_patch16_xp_224.untrained': _cfg(url=''),
    'vit_large_patch14_xp_224.untrained': _cfg(url=''),
    'vit_huge_patch14_xp_224.untrained': _cfg(url=''),

    'vit_base_patch16_224.mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_large_patch16_224.mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_huge_patch14_224.mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth',
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),

    'vit_huge_patch14_gap_224.in1k_ijepa': _cfg(
        url='https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar',
        # hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_huge_patch14_gap_224.in22k_ijepa': _cfg(
        url='https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar',
        # hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_huge_patch16_gap_448.in1k_ijepa': _cfg(
        url='https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar',
        # hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        input_size=(3, 448, 448), crop_pct=1.0,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_giant_patch16_gap_224.in22k_ijepa': _cfg(
        url='https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar',
        # hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),

    'vit_base_patch16_siglip_224.webli': _cfg(
        hf_hub_id='timm/ViT-B-16-SigLIP',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=0),
    'vit_base_patch16_siglip_256.webli': _cfg(
        hf_hub_id='timm/ViT-B-16-SigLIP-256',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 256, 256),
        num_classes=0),
    'vit_base_patch16_siglip_384.webli': _cfg(
        hf_hub_id='timm/ViT-B-16-SigLIP-384',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 384, 384),
        num_classes=0),
    'vit_base_patch16_siglip_512.webli': _cfg(
        hf_hub_id='timm/ViT-B-16-SigLIP-512',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 512, 512),
        num_classes=0),
    'vit_large_patch16_siglip_256.webli': _cfg(
        hf_hub_id='timm/ViT-L-16-SigLIP-256',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 256, 256),
        num_classes=0),
    'vit_large_patch16_siglip_384.webli': _cfg(
        hf_hub_id='timm/ViT-L-16-SigLIP-384',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 384, 384),
        num_classes=0),
    'vit_so400m_patch14_siglip_224.webli': _cfg(
        hf_hub_id='timm/ViT-SO400M-14-SigLIP',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=0),
    'vit_so400m_patch14_siglip_384.webli': _cfg(
        hf_hub_id='timm/ViT-SO400M-14-SigLIP-384',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 384, 384),
        num_classes=0),

    'vit_medium_patch16_reg4_256': _cfg(
        input_size=(3, 256, 256)),
    'vit_medium_patch16_reg4_gap_256': _cfg(
        input_size=(3, 256, 256)),
    'vit_base_patch16_reg8_gap_256': _cfg(input_size=(3, 256, 256)),
})


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    strict = True
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False

    return build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        **kwargs,
    )


# register_model_deprecations(__name__, {
#     'vit_tiny_patch16_224_in21k': 'vit_tiny_patch16_224.augreg_in21k',
#     'vit_small_patch32_224_in21k': 'vit_small_patch32_224.augreg_in21k',
#     'vit_small_patch16_224_in21k': 'vit_small_patch16_224.augreg_in21k',
#     'vit_base_patch32_224_in21k': 'vit_base_patch32_224.augreg_in21k',
#     'vit_base_patch16_224_in21k': 'vit_base_patch16_224.augreg_in21k',
#     'vit_base_patch8_224_in21k': 'vit_base_patch8_224.augreg_in21k',
#     'vit_large_patch32_224_in21k': 'vit_large_patch32_224.orig_in21k',
#     'vit_large_patch16_224_in21k': 'vit_large_patch16_224.augreg_in21k',
#     'vit_huge_patch14_224_in21k': 'vit_huge_patch14_224.orig_in21k',
#     'vit_base_patch32_224_sam': 'vit_base_patch32_224.sam',
#     'vit_base_patch16_224_sam': 'vit_base_patch16_224.sam',
#     'vit_small_patch16_224_dino': 'vit_small_patch16_224.dino',
#     'vit_small_patch8_224_dino': 'vit_small_patch8_224.dino',
#     'vit_base_patch16_224_dino': 'vit_base_patch16_224.dino',
#     'vit_base_patch8_224_dino': 'vit_base_patch8_224.dino',
#     'vit_base_patch16_224_miil_in21k': 'vit_base_patch16_224_miil.in21k',
#     'vit_base_patch32_224_clip_laion2b': 'vit_base_patch32_clip_224.laion2b',
#     'vit_large_patch14_224_clip_laion2b': 'vit_large_patch14_clip_224.laion2b',
#     'vit_huge_patch14_224_clip_laion2b': 'vit_huge_patch14_clip_224.laion2b',
#     'vit_giant_patch14_224_clip_laion2b': 'vit_giant_patch14_clip_224.laion2b',
# })

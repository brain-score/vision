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
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, _assert, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn
from ._builder import build_model_with_cfg
from ._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['NCAFormer']  # model_registry will add each entrypoint fn to this


_logger = logging.getLogger(__name__)


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
            alive_threshold = 0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.alive_mask = False
        if(alive_threshold > 0):
            self.alive_mask = True
            self.alive_channel = alive_channel
            self.alive_threshold = alive_threshold

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        if(self.alive_mask):
            x_alive_mask = (x[:, :, self.alive_channel] > self.alive_threshold).float()
            x_alive_mask = x_alive_mask.view(B, 1, 1, N)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # 3, B, N_head, N_token, C // N_head
        q, k, v = qkv.unbind(0) # B, N_head, N_token, C // N_head
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask = x_alive_mask.repeat(1,1,N,1) if self.alive_mask else None,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1) # B, N_head, N_token, N_token
            if(self.alive_mask):
                attn = attn + x_alive_mask * -10000.0
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v # B, N_head, N_token, C

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        """B, N_token, C"""
        return x


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

class NCA(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            act_layer=nn.GELU,
            norm_layer=None,
            separate_norm = False,
            stochastic_update = False,
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
            sigmoid_alive = False,
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

        
        self.alive_func = nn.Sigmoid() if sigmoid_alive else nn.Identity()
        
        self.separate_norm = separate_norm
        self.norm_layer = None
        if(norm_layer):
            if(separate_norm):
                if(isinstance(times, list)):
                    self.norm_layer = [norm_layer(dim) for _ in range(times[1])]
                elif(isinstance(times, int)):
                    self.norm_layer = [norm_layer(dim) for _ in range(times)]
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
        self.alive_mask = alive_threshold >= 0.0

        self.trainable_kernel = trainable_kernel

        self.multi_head_perception = multi_head_perception

        self.expand = 4

        self.padding_mode = padding_mode

        self.perception_scales = perception_scales

        self.perception_aggr = perception_aggr

        self.stochastic_update = stochastic_update

        self.c_in = dim
        if self.trainable_kernel:
            if(self.multi_head_perception):
                """
                Input: B,C,H,W. 
                K groups, so each group is of shape B, C // K, H, W. Here group is num_heads
                Perception in each of the K groups, such that in each group, the convolution kernel is the same for every channel so it is a group convolution where the group number equals C//K. 
                Across different groups, the convolution kernels are different. 
                Therefore, in total we have K perception kernels, each of size out, 1, 3, 3 with group convolution C//K groups.
                """
                self.groups = num_heads
                self.group_convs = nn.ModuleList()

                # Create a convolution layer for each group
                for _ in range(self.groups):
                    self.group_convs.append(
                        nn.Conv2d(self.c_in // self.groups, self.c_in * self.expand // self.groups, 3, padding=1, groups=self.c_in // self.groups, padding_mode=padding_mode)
                    )
            else:
                self.conv_layer = torch.nn.Conv2d(self.c_in, self.c_in * self.expand, 3, padding=1, groups=self.c_in, padding_mode=padding_mode)
                torch.nn.init.xavier_normal_(self.conv_layer.weight, gain=1.0)
        else:
            self.sobel_filter_x = torch.FloatTensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).to(self.device)
            if(normalize_filter):
                self.sobel_filter_x = F.normalize(self.sobel_filter_x, p = 1)
            self.sobel_filter_y = self.sobel_filter_x.T

            self.sobel_filter_x = self.sobel_filter_x.reshape(1, 1, 3, 3)
            self.sobel_filter_y = self.sobel_filter_y.reshape(1, 1, 3, 3)

            self.identity_filter = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).to(self.device)
            self.identity_filter = self.identity_filter.reshape(1, 1, 3, 3)

            self.laplacian_filter = torch.FloatTensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]]).to(self.device)
            if(normalize_filter):
                self.laplacian_filter = F.normalize(self.laplacian_filter, p = 1)
            
            self.laplacian_filter = self.laplacian_filter.reshape(1, 1, 3, 3)

            if(not multi_head_perception):
                self.sobel_filter_x = self.sobel_filter_x.repeat(self.c_in, 1, 1, 1)
                self.sobel_filter_y = self.sobel_filter_y.repeat(self.c_in, 1, 1, 1)
                self.identity_filter = self.identity_filter.repeat(self.c_in, 1, 1, 1)
                self.laplacian_filter = self.laplacian_filter.repeat(self.c_in, 1, 1, 1)

            self.filters = torch.stack([self.identity_filter, self.sobel_filter_x, self.sobel_filter_y, self.laplacian_filter])
            self.filters.to(self.device)

            # Compute the interpolated weights for each kernel
            """Do we need to interpolate identity filter???"""
            if(multi_head_perception):
                self.groups = num_heads
                self.group_convs = []
                for k_idx, kernel in enumerate(self.filters):
                    weight_group = []
                    for g in range(self.groups):
                        if(k_idx == 0):
                            # weight = g / (self.groups - 1) * kernel
                            """Do we need to interpolate identity filter???"""
                            weight = kernel
                        else:
                            weight = (self.groups - g) / self.groups * kernel
                        weight_group.append(weight)
                    self.group_convs.append(torch.stack(weight_group)) # stack weight group: num_group, 1, 1, 3, 3

                # self.conv_weights: num_kernel, num_group, 3, 3
                # Repeat for each input channel and store on the device
                """For each kernel, we need to do the multi head perception. Inside each multihead perception, the kernel weights are different among groups. """
                self.group_convs = [w.repeat(1, self.c_in // self.groups, 1, 1, 1).to(self.device) for w in self.group_convs] # each thing in group conv is one kernel

        in_channel = self.c_in * self.expand + self.c_cond
        out_channels = self.c_in
        self.fc_dim = in_channel * 2
        self.act = act_layer()

        self.w1 = torch.nn.Conv2d(in_channel, self.fc_dim, 1, device=self.device)
        torch.nn.init.xavier_normal_(self.w1.weight, gain=0.2)

        self.w2 = torch.nn.Conv2d(self.fc_dim, out_channels, 1, bias=True, device=self.device)
        torch.nn.init.xavier_normal_(self.w2.weight, gain=0.1)
        torch.nn.init.zeros_(self.w2.bias)
            
    def alive(self, x):
        # use additional channel, or use existing channel
        # additional channel: intialization? Transfer to next layer? Preserve of alive and death in patch merging? 
        return F.max_pool2d(self.alive_func(x[:, self.alive_channel:self.alive_channel + 1, :, :]), kernel_size=3, stride=1, padding=1) > self.alive_threshold

    def perceive_torch(self, x, scale = 0):
        """x: B, C, H, W"""
        if scale != 0:
            _, _, h, w = x.shape
            h_new = int(h // (2 ** scale))
            w_new = int(w // (2 ** scale))
            x = F.interpolate(x, size=(h_new, w_new), mode='bicubic', align_corners=False)

        if self.trainable_kernel:
            if(self.multi_head_perception):
                # Split the input tensor along the channel dimension into K groups
                inputs = torch.split(x, x.shape[1] // self.groups, dim=1)

                # Apply the corresponding convolution operation to each group
                outputs = [conv(inp) for conv, inp in zip(self.group_convs, inputs)]

                # Concatenate the results along the channel dimension
                if(self.perception_aggr == "concat"):
                    y = torch.cat(outputs, dim = 1)
                elif(self.perception_aggr == "sum"):
                    y = torch.sum(torch.stack(outputs), dim=0)
            else:
                outputs = self.conv_layer(x)
                if(self.perception_aggr == "concat"):
                    y = outputs
                elif(self.perception_aggr == "sum"):
                    outputs_chunk = torch.split(outputs, outputs.shape[1] // self.expand, dim = 1)
                    y = torch.sum(torch.stack(outputs_chunk), dim=0)
        else:
            if(self.multi_head_perception):
                inputs = torch.split(x, self.c_in // self.groups , dim=1)
                input_pad = [F.pad(z, [1, 1, 1, 1], self.padding_mode) for z in inputs]
                groups = self.c_in // self.groups 
                kernel_out = []
                for kernel in self.group_convs:
                    outputs = [F.conv2d(inp, conv_weights, groups=groups) for conv_weights, inp in zip(kernel, input_pad)]
                    kernel_out.append(torch.cat(outputs, dim=1))
                if(self.perception_aggr == "concat"):
                    y = torch.cat(kernel_out, dim = 1)
                elif(self.perception_aggr == "sum"):
                    y = torch.sum(torch.stack(kernel_out), dim=0)
            else:
                z = F.pad(x, [1, 1, 1, 1], self.padding_mode)
                y1 = F.conv2d(z, self.sobel_filter_x, groups=self.c_in)
                y2 = F.conv2d(z, self.sobel_filter_y, groups=self.c_in)
                y3 = F.conv2d(z, self.laplacian_filter, groups=self.c_in)
                if(self.perception_aggr == "concat"):
                    y = torch.cat((x, y1, y2, y3), 1)
                else:
                    y = x + y1 + y2 + y3
        
        if scale != 0:
            y = F.interpolate(y, size=(h, w), mode='bicubic', align_corners=False)

        return y

    def perceive_multiscale(self, x, pos_emb_mat=None):
        if(len(self.perception_scales) == 1):
            return self.perceive_torch(x)
        else:
            perceptions = []
            y = 0
            for scale in self.perception_scales:
                if(x.shape[2] <= 2**scale):
                    break
                z = self.perceive_torch(x, scale=scale)
                perceptions.append(z)

            y = sum(perceptions)
            y = y / len(self.perception_scales)

            if pos_emb_mat is not None:
                y = torch.cat([y, pos_emb_mat], dim=1)

            return y

    def evolve(self, x):
        b_x, c_x, H, W = x.shape
        if(self.alive_mask):
            pre_life_mask = self.alive(x)

        if self.pos_emb_2d:
            y_percept = self.perceive_multiscale(x, pos_emb_mat=self.pos_emb_2d(x))
        else:
            y_percept = self.perceive_multiscale(x)
        
        y = self.w2(self.act(self.w1(y_percept)))

        if(self.stochastic_update):
            b, c, h, w = y.shape
            update_mask = (torch.rand(b, 1, h, w, device=self.device) + 0.5).floor()
            x = x + y * update_mask
        else:
            x = x + y
        
        if(self.alive_mask):
            post_life_mask = self.alive(x)
            life_mask = (pre_life_mask & post_life_mask).float()
            x = x * life_mask

        return x
    
    def forward(self, x, return_middle_state = False):
        """x: B, N, C"""
        b_x, n_tokens, c_x = x.shape
        H = int(n_tokens ** 0.5)
        W = n_tokens // H
        if(H * W != n_tokens):
            print("Token Number Cannot Resize to an Image", H, W, n_tokens)
            exit()
        x = x.transpose(1, 2).reshape(b_x, c_x, H, W)

        if(return_middle_state):
            middle_state_list = []
        if(isinstance(self.times, list)):
            step = torch.randint(low = self.times[0], high = self.times[1], size = (1,)).item()
        elif(isinstance(self.times, int)):
            step = self.times
        for time_step in range(step):
            if(self.norm_layer):
                if(self.separate_norm):
                    cur_norm_layer = self.norm_layer[time_step]
                else:
                    cur_norm_layer = self.norm_layer
                x = cur_norm_layer(x)
            x = self.evolve(x)
            if(return_middle_state):
                middle_state_list.append(x)
        
        x = x.reshape(b_x, c_x, n_tokens).transpose(1, 2)
        return_dict = {}
        if(return_middle_state):
            return_dict["middle_state"] = middle_state_list
        return x, return_dict

class Block_NCA(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            qk_norm=False,
            drop_path=0.,
            proj_drop=0.,
            attn_drop=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            nca_norm_layer=None,
            separate_norm = False,
            stochastic_update = False,
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
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            alive_channel = alive_channel,
            alive_threshold = alive_threshold,
        )
        # self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.residual_nca = residual_nca
        if(residual_nca):
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        else:
            self.drop_path2 = nn.Identity()

        if(block_type != "ode"):
            self.norm2 = norm_layer(dim)
        self.nca = NCA(
            dim = dim,
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
        )

        self.block_type = block_type
        # self.mlp = mlp_layer(
        #     in_features=dim,
        #     hidden_features=int(dim * mlp_ratio),
        #     act_layer=act_layer,
        #     drop=proj_drop,
        # )
        # self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, return_middle_state = False):
        """Here we might not need the residual after NCA evolution because in NCA it is already residual update"""
        if(self.block_type == "normal"):
            x = x + self.drop_path1(self.attn(self.norm1(x)))
            nca_output, return_dict = self.nca(self.norm2(x), return_middle_state)
            
            if(self.residual_nca):
                x = x + self.drop_path2(nca_output)
            else:
                x = self.drop_path2(nca_output)
            # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        elif(self.block_type == "ode"):
            x_norm = self.norm1(x)
            G = self.drop_path1(self.attn(x_norm))
            nca_output, return_dict = self.nca(x_norm, return_middle_state)
            F = self.drop_path2(nca_output)
            if(self.residual_nca):
                x = x + F + G
            else:
                x = F + G
        if(return_middle_state):
            return_dict["nca_state"] = nca_output
        return x, return_dict

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
        
        if(learn):
            if(learn_alive_only):
                self.downsample = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0)
            else:
                # self.downsample = nn.Conv2d(in_channels=dim, out_channels=dim * 4, kernel_size=3, stride=2, padding=2, padding_mode="zeros", dilation=2)
                self.downsample = nn.Conv2d(in_channels=dim, out_channels=dim * 4, kernel_size=2, stride=2, padding=0)
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)

        dim_alive_reduction = 3 if learn_alive_only and learn else 0
        self.reduction = nn.Linear(4 * dim - dim_alive_reduction, self.out_dim, bias=False)

        self.alive_mask = alive_threshold >= 0.0
        self.alive_channel = alive_channel
        self.alive_threshold = alive_threshold

    def forward(self, x):
        b_x, n_tokens, c_x = x.shape
        H = int(n_tokens ** 0.5)
        W = n_tokens // H
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
            alive_values = self.alive_func(x[..., self.alive_channel])
            x_features = x[..., self.alive_channel+1:]
            alive_value_fill = torch.mean(alive_values, dim = [-2, -1], keepdim = True) # B, H', W', 1, 1
            expanded_mean = alive_value_fill.expand_as(alive_values).unsqueeze(-1)
            x = torch.cat([expanded_mean, x_features], dim=-1)
        
        if(self.learn):
            x = x.permute(0, 1, 4, 2, 3, 5).reshape(B, H, W, C).permute(0, 3, 1, 2)
            if(self.learn_alive_only):
                """Assume 0 alive channel, or we need concat feature before and after alive channel"""
                x_feature = x[:, self.alive_channel+1:, ...]
                x_alive = x[:, :self.alive_channel+1, ...]
                new_alive = self.downsample(x_alive).permute(0, 2, 3, 1)
                x_feature = x_feature.permute(0, 2, 3, 1).reshape(B, H // 2, 2, W // 2, 2, C-1).permute(0, 1, 3, 4, 2, 5).flatten(3) # B, H', W', 2, 2, C-1
                x = torch.cat([new_alive, x_feature], dim = -1)
            else:
                x = self.downsample(x)
                x = x.permute(0, 2, 3, 1) # B, H', W', C*4
        else:
            x = x.flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H*W, C)
        return x

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
            stochastic_update = False,
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
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            norm_layer = norm_layer,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

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

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # self.blocks = nn.Sequential(*[
        #     block_fn(
        #         dim=embed_dim,
        #         num_heads=num_heads,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         qk_norm=qk_norm,
        #         init_values=init_values,
        #         proj_drop=proj_drop_rate,
        #         attn_drop=attn_drop_rate,
        #         drop_path=dpr[i],
        #         norm_layer=norm_layer,
        #         act_layer=act_layer,
        #         mlp_layer=mlp_layer,
        #     )
        #     for i in range(depth)])
        self.depth = depth
        assert len(num_heads) == depth
        self.nca_blocks = []
        in_dim = embed_dim
        for i in range(depth):
            self.nca_blocks += [
                Block_NCA(
                    dim = in_dim,
                    num_heads = num_heads[i],
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    drop_path=dpr[i],
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    nca_norm_layer = nca_norm_layer,
                    separate_norm = separate_norm,
                    stochastic_update = stochastic_update,
                    times = times[i],
                    alive_channel = alive_channel,
                    alive_threshold = alive_threshold,
                    trainable_kernel = trainable_kernel,
                    normalize_filter = normalize_filter,
                    padding_mode = padding_mode,
                    multi_head_perception = multi_head_perception,
                    perception_scales = perception_scales,
                    pos_emb = pos_emb,
                    perception_aggr = perception_aggr,
                    block_type = block_type,
                    residual_nca = residual_nca,
                    sigmoid_alive = sigmoid_alive,
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
        self.fc_norm = norm_layer(out_token_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(out_token_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

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
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.blocks, x)
        # else:
        #     x = self.blocks(x)
        nca_middle_state_list = []
        if(self.solver == "Euler"):
            for i, nca_block in enumerate(self.nca_blocks):
                x, return_dict = nca_block(x, return_middle_state)
                if(return_middle_state):
                    nca_middle_state_list.append(return_dict["nca_state"])
                if(i < self.depth - 1):
                    x = self.downsample_layer[i](x)
        elif(self.solver == "RK2"):
            for i, nca_block in enumerate(self.nca_blocks):
                runge_kutta_list = []
                residual = x
                # x = x + 0.5 * (F1 + F2)
                # F1 = F(x), F2 = F(x + F1)
                for step_size in range(2):
                    x, return_dict = nca_block(x, return_middle_state)
                    if(return_middle_state):
                        nca_middle_state_list.append(return_dict["nca_state"])
                    runge_kutta_list.append(x)
                    x = residual + x
                x = residual + 1/2 * (runge_kutta_list[0] + runge_kutta_list[1])

        return_dict["nca_middle_state_list"] = nca_middle_state_list
        x = self.norm(x)
        return x, return_dict

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

    def forward(self, x, return_middle_state = False):
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



import torch
from torch import nn
import math
import time

from functools import partial
from timm.models.layers import trunc_normal_, DropPath
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.vision_transformer import _cfg
from einops import rearrange

from timm.models.swin_transformer import PerceptionAggr
from timm.models.vision_transformer_nca import NCA

from timm.models.registry import register_model


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        if in_features == 768:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)
        else:
            self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
            self.bn1 = nn.BatchNorm2d(hidden_features)
            self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
            self.bn2 = nn.BatchNorm2d(hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
            self.bn3 = nn.BatchNorm2d(out_features)
            self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.in_features == 768:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
        else:
            B,N,C = x.shape
            x = x.reshape(B, int(N**0.5), int(N**0.5), C).permute(0,3,1,2).contiguous()
            x = self.bn1(self.fc1(x))
            x = self.act(x)
            x = self.drop(x)
            x = self.act(self.bn2(self.dwconv(x)))
            x = self.bn3(self.fc2(x))
            x = self.drop(x)
            x = x.permute(0,2,3,1).reshape(B, -1, C)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., use_mask=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_mask = use_mask
        if use_mask:
            self.att_mask = nn.Parameter(torch.Tensor(self.num_heads, 196, 196))

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (q * self.scale @ k.transpose(-2, -1))
        if self.use_mask:
            attn = attn * torch.sigmoid(self.att_mask).expand(B, -1, -1, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_mask=False,
                 nca_local_perception_model = None,
                nca_local_perception_loop = [1,1,1,1]
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_mask=use_mask)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.nca_local_perception_model = nca_local_perception_model
        self.nca_local_perception_loop = nca_local_perception_loop
        if(self.nca_local_perception_model is not None):
            #print("additional flops: ", self.nca_local_perception_model.flops() / 1e9)
            self.perception_aggr_model = PerceptionAggr(dim, num_scales=2, multi_head_combine=False, head_num=1, scale_weight = True)

    def forward(self, x):
        mes_time = False
        if(self.nca_local_perception_model is not None):
            x_norm = self.norm1(x)
            b_x, n_tokens, c_x = x_norm.shape
            Hx = int(n_tokens ** 0.5)
            Wx = n_tokens // Hx
            self.input_size = (Hx, Wx, c_x)
            if(Hx * Wx != n_tokens):
                print("Token Number Cannot Resize to an Image", Hx, Wx, n_tokens)
                exit()
            local_perception = x_norm.transpose(1, 2).reshape(b_x, c_x, Hx, Wx)
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
            
            global_perception = self.attn(x_norm)
            global_perception = global_perception.transpose(1, 2).reshape(b_x, c_x, Hx, Wx)

            perception_result = torch.stack([local_perception, global_perception], dim = -1)
            if(mes_time):
                print("Global perception:", time.time() - start)
                start = time.time()
            
            x_weight = x.transpose(1, 2).reshape(b_x, c_x, Hx, Wx)
            weight = self.perception_aggr_model(x_weight) # B 1 H W K
            
            perception_local_global = torch.sum(perception_result * weight, dim=-1) # B, C, H, W
            perception_local_global = perception_local_global.reshape(b_x, c_x, Hx*Wx).transpose(1, 2)
            if(mes_time):
                print("Aggr perception:", time.time() - start)
                start = time.time()
            x = x + self.drop_path(perception_local_global)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, base_dim, depth, heads, mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None, use_mask=False, masked_block=None,
                 nca_model = False,
                before_nca_norm = False,
                stochastic_update = 0.0,
                times = [1,1,1,1],
                energy_minimization = 0,
                weighted_scale_combine = 0,
                nca_local_perception = 0,
                nca_local_perception_loop = [1,1,1,1],
                energy_point_wise = 0,
                nca_expand = 4,
                init_with_grad_kernel = False,):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]


        if(nca_local_perception):
            num_nca = depth // nca_local_perception
            self.nca = []
            for _ in range(num_nca):
                self.nca += [
                    NCA(
                        dim = embed_dim,
                        num_heads = heads,
                        act_layer=nn.GELU,
                        norm_layer=None,
                        separate_norm = False,
                        stochastic_update = stochastic_update,
                        times = times, # this times is not used
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
                        input_size = (14, 14),
                        local_perception_only = True,
                )]
            print("NCA block num: ", len(self.nca))

        if use_mask==True:
            assert masked_block is not None
            self.blocks = nn.ModuleList()
            for i in range(depth):
                if i < masked_block:
                    self.blocks.append(Block(
                        dim=embed_dim,
                        num_heads=heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path_prob[i],
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        use_mask=use_mask, 
                        nca_local_perception_model = None if not nca_local_perception else self.nca[i // nca_local_perception],
                        nca_local_perception_loop = None if not nca_local_perception else nca_local_perception_loop[i // nca_local_perception]
                    ))
                else:
                    self.blocks.append(Block(
                        dim=embed_dim,
                        num_heads=heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path_prob[i],
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        use_mask=False, 
                        nca_local_perception_model = None if not nca_local_perception else self.nca[i // nca_local_perception],
                        nca_local_perception_loop = None if not nca_local_perception else nca_local_perception_loop[i // nca_local_perception]
                    ))
        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_prob[i],
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    use_mask=use_mask, 
                    nca_local_perception_model = None if not nca_local_perception else self.nca[i // nca_local_perception],
                    nca_local_perception_loop = None if not nca_local_perception else nca_local_perception_loop[i // nca_local_perception]
                )
                for i in range(depth)])

        self.nca_model = nca_model # if times > 0 else 0
        self.nca_preprocess_apply_layer = []
        print(self.nca_model)
        if(self.nca_model):
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            # self.nca_drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
            nca_norm_preprocess = []
            nca_preprocess = []
            nca_drop_path = []
            num_nca_preprocess = nca_model # 3
            self.nca_preprocess_apply_layer = []
            if(nca_model == 2):
                self.nca_preprocess_apply_layer = [4, 8]
            elif(nca_model == 1):
                self.nca_preprocess_apply_layer = [4]
            elif(nca_model == 3):
                self.nca_preprocess_apply_layer = [4, 10]
            elif(nca_model >= 10):
                self.nca_preprocess_apply_layer = [nca_model - 10]
            #self.nca_preprocess_apply_layer = [depth // (nca_model + 1) * x for x in range(1, nca_model + 1)] # 3,6,9
            print(self.nca_preprocess_apply_layer)

            for i in range(len(self.nca_preprocess_apply_layer)):
                nca_norm_preprocess += [norm_layer(embed_dim) if before_nca_norm else nn.Identity()]
                if(self.nca_preprocess_apply_layer[0] >= self.depth):
                    nca_drop_path += [DropPath(drop_path_prob[-1]) if drop_path_prob[-1] > 0. else nn.Identity()]
                else:
                    nca_drop_path += [DropPath(drop_path_prob[self.nca_preprocess_apply_layer[i]]) if drop_path_prob[self.nca_preprocess_apply_layer[i]] > 0. else nn.Identity()]
                nca_preprocess += [
                    NCA(
                        dim = embed_dim,
                        num_heads = heads,
                        act_layer=nn.GELU,
                        norm_layer=None,
                        separate_norm = False,
                        stochastic_update = stochastic_update,
                        times = times[i],
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
                        input_size = (14,14),
                        energy_coeff_point_wise = energy_point_wise,
                        expand = nca_expand,
                        init_with_grad_kernel = init_with_grad_kernel,
                )]
                print("additional FLOPs:", nca_preprocess[-1].flops(verbose = True) / 1e9)
            self.nca_norm_preprocess = nn.ModuleList(nca_norm_preprocess)
            self.nca_preprocess = nn.ModuleList(nca_preprocess)
            self.nca_drop_path = nn.ModuleList(nca_drop_path)
            print(self.nca_drop_path)
        

    def forward(self, x):
        B,C,H,W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        # x = x.permute(0,2,3,1).reshape(B, H * W, C)
        nca_idx = 0
        for i in range(self.depth):
            if(self.nca_model):
                if(i in self.nca_preprocess_apply_layer):
                    nca_output, _ = self.nca_preprocess[nca_idx](self.nca_norm_preprocess[nca_idx](x.contiguous()).contiguous())
                    nca_output = self.nca_drop_path[nca_idx](nca_output)
                    x = x + nca_output
                    nca_idx += 1
            x = self.blocks[i](x)
        # if(self.depth in self.nca_preprocess_apply_layer):
        #     nca_output, _ = self.nca_preprocess[nca_idx](self.nca_norm_preprocess[nca_idx](x.contiguous()).contiguous())
        #     #nca_output = self.nca_drop_path[nca_idx](nca_output)
        #     x = x + nca_output
        # x = x.reshape(B, H, W, C).permute(0,3,1,2)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class conv_head_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)

    def forward(self, x):

        x = self.conv(x)

        return x

class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size,
                 stride, padding):
        super(conv_embedding, self).__init__()

        self.out_channels = out_channels

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(7, 7), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(32, out_channels, kernel_size=(4, 4), stride=(4, 4))
            
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)
        return x


class PoolingTransformer(nn.Module):
    def __init__(self, image_size, patch_size, stride, base_dims, depth, heads,
                 mlp_ratio, num_classes=1000, in_chans=3,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0, use_mask=False, masked_block=None,
                 nca_model = False,
                before_nca_norm = False,
                stochastic_update = 0.0,
                times = [1,1,1,1],
                energy_minimization = 0,
                weighted_scale_combine = 0,
                nca_local_perception = 0,
                nca_local_perception_loop = [1,1,1,1],
                energy_point_wise = 0,
                nca_expand = 4,
                init_with_grad_kernel = False,
                perception_norm = "None"):
        super(PoolingTransformer, self).__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = math.floor(
            (image_size / stride))

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.patch_size = patch_size
        self.patch_embed = conv_embedding(in_chans, base_dims[0] * heads[0],
                                          patch_size, stride, padding)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            if stage == 0:
                self.transformers.append(
                    Transformer(base_dims[stage], depth[stage], heads[stage],
                                mlp_ratio,
                                drop_rate, attn_drop_rate, drop_path_prob, use_mask=use_mask, masked_block=masked_block,
                                nca_model = nca_model,
                                before_nca_norm = before_nca_norm,
                                stochastic_update = stochastic_update,
                                times = times,
                                energy_minimization = energy_minimization,
                                weighted_scale_combine = weighted_scale_combine,
                                nca_local_perception = nca_local_perception,
                                nca_local_perception_loop = nca_local_perception_loop,
                                energy_point_wise = energy_point_wise,
                                nca_expand = nca_expand,
                                init_with_grad_kernel = init_with_grad_kernel,)
                )
            else:
                self.transformers.append(
                    Transformer(base_dims[stage], depth[stage], heads[stage],
                                mlp_ratio,
                                drop_rate, attn_drop_rate, drop_path_prob, 
                                nca_model = nca_model,
                                before_nca_norm = before_nca_norm,
                                stochastic_update = stochastic_update,
                                times = times,
                                energy_minimization = energy_minimization,
                                weighted_scale_combine = weighted_scale_combine,
                                nca_local_perception = nca_local_perception,
                                nca_local_perception_loop = nca_local_perception_loop,
                                energy_point_wise = energy_point_wise,
                                nca_expand = nca_expand,
                                init_with_grad_kernel = init_with_grad_kernel,)
                )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(base_dims[stage] * heads[stage],
                                      base_dims[stage + 1] * heads[stage + 1],
                                      stride=2
                                      )
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for stage in range(len(self.pools)):
            x = self.transformers[stage](x)
            x = self.pools[stage](x)
        x = self.transformers[-1](x)
        cls_features = self.norm(self.gap(x).squeeze())

        return cls_features

    def forward(self, x):
        cls_features = self.forward_features(x)
        output = self.head(cls_features)
        return output

@register_model
def rvt_tiny(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[32, 32],
        depth=[10, 2],
        heads=[6, 12],
        mlp_ratio=4,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = \
        torch.load('rvt_ti.pth', map_location='cpu')['model']
        model.load_state_dict(state_dict)
    return model

@register_model
def rvt_tiny_plus(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[32, 32],
        depth=[10, 2],
        heads=[6, 12],
        mlp_ratio=4,
        use_mask=True,
        masked_block=10,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = \
        torch.load('rvt_ti*.pth', map_location='cpu')['model']
        model.load_state_dict(state_dict)
    return model

@register_model
def rvt_small(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[64],
        depth=[12],
        heads=[6],
        mlp_ratio=4,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = \
        torch.load('pretrained/rvt_small.pth', map_location='cpu')['model']
        model.load_state_dict(state_dict)
        print("Successfully Load RVT Small Model")
    return model

@register_model
def rvt_small_plus(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[64],
        depth=[12],
        heads=[6],
        mlp_ratio=4,
        use_mask=True,
        masked_block=5,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = \
        torch.load('pretrained/rvt_small_plus.pth', map_location='cpu')['model']
        model.load_state_dict(state_dict)
        print("Successfully Load RVT Small Plus Model")
        
    return model

@register_model
def rvt_base(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[64],
        depth=[12],
        heads=[12],
        mlp_ratio=4,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = \
        torch.load('pretrained/rvt_base.pth', map_location='cpu')['model']
        model.load_state_dict(state_dict)
        print("Successfully Load RVT Base Model")
    return model

@register_model
def rvt_base_plus(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[64],
        depth=[12],
        heads=[12],
        mlp_ratio=4,
        use_mask=True,
        masked_block=5,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.load('/scratch/students/2022-fall-sp-yitao/Transformer/NCAFormer/pretrained/rvt_base_plus.pth', map_location='cpu')['model']
#         state_dict = torch.load('/scratch/students/2022-fall-sp-yitao/Transformer/Robust-Vision-Transformer/output/checkpoint.pth', map_location='cpu')['model']
#         state_dict = torch.load('/scratch/students/2022-fall-sp-yitao/Transformer/Robust-Vision-Transformer/backup_rasampler/model_best.pth', map_location='cpu')['model_ema']
        model.load_state_dict(state_dict)
        print("Successfully Load RVT Base Plus Model")
    return model
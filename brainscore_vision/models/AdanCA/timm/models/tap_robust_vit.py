

import torch
from torch import nn
import math

from functools import partial
from timm.models.layers import DropPath, drop_path
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.vision_transformer import _cfg
from einops import rearrange


from timm.models.registry import register_model
from torch.nn import functional as F
from torch.nn import Parameter
from timm.models.cnn_backbone import _create_hybrid_backbone, HybridEmbed
from timm.models import load_checkpoint


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class DilationPredictor(nn.Module):
    def __init__(self, dim, num_scales, temperature=1):
        super(DilationPredictor, self).__init__()

        self.temperature = temperature
        self.proj = nn.Sequential(
            nn.Conv2d(dim, num_scales, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_scales),
            nn.Conv2d(num_scales, num_scales, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = x.unsqueeze(1)
        x = x / self.temperature
        x = x.softmax(dim=-1)
        return x


class DilatedAvgPooling(nn.Module):
    def __init__(self, dilation=0):
        super(DilatedAvgPooling, self).__init__()
        self.dilation = dilation
        if dilation>0:
            self.unfold = nn.Unfold(3, dilation=dilation, padding=dilation, stride=1)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.dilation > 0:
            x = self.unfold(x)
            new_shape = (B, C, 9, H, W)
            x = x.reshape(new_shape).mean(dim=2)
        return x.unsqueeze(-1)


class MatchedDropout(nn.Module):
    """ definition of mode
        0: no mask, i.e., no dropout
        1: dropout
        2: use previous mask
    """
    def __init__(self, drop_p, inplace=False):
        super().__init__()
        self.masker = nn.Dropout(p=drop_p, inplace=inplace)
        self.mode = 0
        self.pre_mask = None

    def forward(self, input):
        self.masker.training = True
        if self.mode == 0:
            output = input
        elif self.mode == 1:
            mask = self.masker(torch.ones_like(input))
            self.pre_mask = mask.clone()
            output = input * mask
        elif self.mode == 2:
            assert self.pre_mask is not None
            mask = self.pre_mask
            if mask.size(0) != input.size(0):
                new_shape = (2,) + (-1,) * input.ndim
                mask = mask.expand(new_shape)
                mask = mask.reshape(input.shape)
            output = input * mask
        return output


class MatchedDropPath(nn.Module):
    """ definition of mode
        0: no mask, i.e., no droppath
        1: droppath
        2: use previous mask
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.mode = 0
        self.pre_mask = None

    def forward(self, input):
        if self.mode == 0:
            output = input
        elif self.mode == 1:
            mask = drop_path(torch.ones_like(input), self.drop_prob, True, self.scale_by_keep)
            self.pre_mask = mask.clone()
            output = input * mask
        elif self.mode == 2:
            assert self.pre_mask is not None
            mask = self.pre_mask
            if mask.size(0) != input.size(0):
                new_shape = (2,) + (-1,) * input.ndim
                mask = mask.expand(new_shape)
                mask = mask.reshape(input.shape)
            output = input * mask
        return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        if in_features == 1768:
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
        if self.in_features == 1768:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
        else:
            B,N,C = x.shape
            x = x.reshape(B, int(N**0.5), int(N**0.5), C).permute(0,3,1,2)
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
            trunc_normal_(self.att_mask, std=.02)

        self.vis_attn = None
        self.distraction_loss = None
        self.negattn_loss = None
        self.distraction_loss_type = 'None'
        self.clean_attention_noisy_feature = False
        self.noisy_attention_clean_feature = False

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.use_mask:
            attn = attn * torch.sigmoid(self.att_mask[:,:N,:N]).expand(B, -1, -1, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_mask=False, num_scales=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_mask=use_mask)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.token_pools = nn.ModuleList([])
        for i in range(num_scales):
            self.token_pools.append(DilatedAvgPooling(i))
        self.dilation_predictor = DilationPredictor(dim, len(self.token_pools))
        self.num_scales = num_scales

    def forward(self, x):
        x, attn_list = x

        B, N, C = x.shape
        x_attn = x.reshape(B, int(N**0.5), int(N**0.5), C).permute(0,3,1,2)
        mask = self.dilation_predictor(x_attn) # B 1 H W K
        self.pred_scale = mask
        mixed_scales = []
        for i in range(self.num_scales):
            mixed_scales.append(self.token_pools[i](x_attn))
        x_attn = torch.cat(mixed_scales, dim=-1)
        x_attn = torch.sum(x_attn * mask, dim=-1)
        x_attn = x_attn.permute(0,2,3,1).reshape(B, -1, C)

        x_new, attn_s = self.attn(self.norm1(x_attn))
        x = x + self.drop_path(x_new)
        attn_list.append(attn_s.clone())
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return [x, attn_list]


class Transformer(nn.Module):
    def __init__(self, base_dim, depth, heads, mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None, use_mask=False, masked_block=None, num_scales=1):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

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
                        num_scales=num_scales
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
                        num_scales=num_scales
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
                    num_scales=num_scales
                )
                for i in range(depth)])


    def forward(self, x, mask_matrix=None, mask_layer_index=None):
        x, attn_list = x
        B,C,H,W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = [x, attn_list]
        for i in range(self.depth):
            x = self.blocks[i](x)
        x, attn_list = x
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = [x, attn_list]
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
        self.patch_size = patch_size

        if patch_size==4:
            final_ks = 1
        else:
            final_ks = 4
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(7, 7), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(32, out_channels, kernel_size=(final_ks, final_ks), stride=(final_ks, final_ks))
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PoolingTransformer(nn.Module):
    def __init__(self, image_size, patch_size, stride, base_dims, depth, heads,
                 mlp_ratio, num_classes=1000, in_chans=3,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0, use_mask=False, masked_block=None, backbone=None, num_scales=4, temperature=1.0):
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
        self.num_scales = num_scales

        if backbone is not None:
            self.patch_embed = backbone
        else:
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
                                drop_rate, attn_drop_rate, drop_path_prob, use_mask=use_mask, masked_block=masked_block, num_scales=num_scales)
                )
            else:
                self.transformers.append(
                    Transformer(base_dims[stage], depth[stage], heads[stage],
                                mlp_ratio,
                                drop_rate, attn_drop_rate, drop_path_prob, num_scales=num_scales)
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
        x = [x, []]
        for stage in range(len(self.pools)):
            x = self.transformers[stage](x)
            x, attn_list = x
            x = self.pools[stage](x)
            x = [x, attn_list]
        x = self.transformers[-1](x)
        x, attn_list = x
        cls_features = self.norm(torch.flatten(self.gap(x), 1))

        return cls_features, attn_list

    def forward(self, x, x1=None, return_attn=False):
        cls_features, attn_list = self.forward_features(x)
        output = self.head(cls_features)
        if return_attn:
            return output, attn_list
        else:
            return output


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, student_width=1.0, drop_rate=0):
        super(BasicBlock, self).__init__()
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate, inplace=False)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, student_width=student_width)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.block_index = 0

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.drop_rate > 0:
            out = self.dropout(out)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0):
        super(Bottleneck, self).__init__()
        self.name = "resnet-bottleneck"
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate, inplace=False)
        self.conv1 = conv1x1(inplanes, planes)
        # nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        # nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * 4)
        # nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.block_index = 0

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.drop_rate > 0:
            out = self.dropout(out)

        out += residual
        out = self.relu(out)
        return out


class ResNet_IMAGENET(nn.Module):

    def __init__(self, depth, num_classes=1000, student_width=1.0, drop_rate=0):
        self.inplanes = 64
        super(ResNet_IMAGENET, self).__init__()
        self.num_classes = num_classes
        if depth < 50:
            block = BasicBlock
        else:
            block = Bottleneck

        if depth == 18:
            layers = [2, 2, 2, 2]
        elif depth == 34:
            layers = [3, 4, 6, 3]
        elif depth == 50:
            layers = [3, 4, 6, 3]
        elif depth == 101:
            layers = [3, 4, 23, 3]
        elif depth == 152:
            layers = [3, 8, 36, 3]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], student_width=student_width, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, student_width=student_width,
                                       drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, student_width=student_width,
                                       drop_rate=drop_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, student_width=student_width,
                                       drop_rate=drop_rate)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, student_width=1.0, drop_rate=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate=drop_rate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, drop_rate=drop_rate))

        return nn.Sequential(*layers)

    def forward(self, x, mask_matrix=None, mask_layer_index=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


@register_model
def tap_rvt_tiny(pretrained, **kwargs):
    _ = kwargs.pop('pretrained_cfg')
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
    return model


@register_model
def tap_rvt_tiny_plus(pretrained, **kwargs):
    _ = kwargs.pop('pretrained_cfg')
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
    return model


@register_model
def tap_rvt_small(pretrained, **kwargs):
    _ = kwargs.pop('pretrained_cfg')
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
    return model


@register_model
def tap_rvt_small_plus(pretrained, **kwargs):
    #_ = kwargs.pop('pretrained_cfg')
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
    return model


@register_model
def tap_rvt_base(pretrained, **kwargs):
    #_ = kwargs.pop('pretrained_cfg')
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
    return model

@register_model
def tap_rvt_base_plus(pretrained, **kwargs):
    #_ = kwargs.pop('pretrained_cfg')
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
    if(pretrained):
        load_checkpoint(model, 'pretrained/tapadl_rvt_base.pth.tar', use_ema=False)
        print("Successfully Load TAP-RVT Base Plus Model")
    return model
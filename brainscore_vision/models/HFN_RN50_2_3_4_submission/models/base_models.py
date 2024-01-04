# Custom Pytorch model from:
# https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb

from model_tools.check_submission import check_models

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import functools

from model_tools.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
from model_tools.activations.pytorch import load_preprocess_images
from brainscore import score_model

# -------------

# Default arguments

args_arch = "ResNet50"
args_name = "ResNet50"
args_num_classes = 1000
args_width_mult = 1.0
args_conv_type = "DenseConv"
args_freeze_weights = False
args_mode = "fan_in"
args_nonlinearity = "relu"
args_bn_type = "NonAffineBatchNorm"
args_init = "kaiming_normal"
args_no_bn_decay = False
args_scale_fan = False
args_prune_rate = 1.0
args_first_layer_dense = False
args_last_layer_dense = False
args_first_layer_type = None
args_score_init_constant = None
args_nonlinearity_2 = "relu"
args_init_2 = "kaiming_normal"
args_scale_fan_2 = "store_true"
args_prune_rate_2 = 1.0
args_mode_2 = "fan_in"
args_conv_type_2 = None
args_bn_type_2 = None
args_freeze_weights_2 = False

# -------------

# Model arguments

# Architecture
args_name = "HFN_RN50_2_3_4"
args_arch = "HFResNet50_2_3_4_ubn"

args_n_classes = 1000

# ===== Builder =========== #
args_conv_type = "SubnetConv"
args_bn_type = "NonAffineBatchNorm"
args_init = "signed_constant"
args_mode = "fan_in"
args_nonlinearity = "relu"
args_scale_fan = True
args_prune_rate = 0.3
args_freeze_weights = True

# ===== BuilderZero =========== #
args_conv_type_2 = "SubnetConv"
args_bn_type_2 = "NonAffineBatchNorm"
args_init_2 = "signed_constant"
args_mode_2 = "fan_in"
args_nonlinearity_2 = "relu"
args_scale_fan_2 = True
args_prune_rate_2 = 0.3
args_freeze_weights_2 = True

# -------------

Layers = ['conv1', 'bn1', 'relu', 
          'layer1_block2.bn3', 'layer1_block2.relu3', 'layer1_block2.conv3',
          'layer2_block1.bnorm3_2', 'layer2_block1.relu3', 'layer2_block1.conv3',
          'layer3_block1.bnorm3_4', 'layer3_block1.relu3', 'layer3_block1.conv3',
          'layer4_block1.bnorm3_1', 'layer4_block1.relu3', 'layer4_block1.conv3',
          'avgpool', 'fc']

# -------------

LearnedBatchNorm = nn.BatchNorm2d
class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

# -------------

# Not learning weights, finding subnet
class DenseConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.zero = False
        self.freeze = False        

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out #out is the supermask. scores are in SubnetConv

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

# Not learning weights, finding subnet
class SubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.zero = False
        self.freeze = False        

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

# -------------

class Builder(object):
    def __init__(self, conv_layer, bn_layer, first_layer=None):
        self.conv_layer = conv_layer
        self.bn_layer = bn_layer
        self.first_layer = first_layer or conv_layer
        self.zero = False
        self.freeze = args_freeze_weights

    def conv(self, kernel_size, in_planes, out_planes, stride=1, first_layer=False):
        conv_layer = self.first_layer if first_layer else self.conv_layer

        if first_layer:
            print(f"==> Building first layer with {str(self.first_layer)}")

        if kernel_size == 3:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,               
            )
        elif kernel_size == 1:
            conv = conv_layer(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=False,               
            )
        elif kernel_size == 5:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=False,               
            )
        elif kernel_size == 7:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=False,               
            )
        else:
            return None

        self._init_conv(conv)

        return conv

    def conv3x3(self, in_planes, out_planes, stride=1, first_layer=False):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1, first_layer=False):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1, first_layer=False):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1, first_layer=False):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def batchnorm(self, planes, last_bn=False, first_layer=False):
        return self.bn_layer(planes)

    def activation(self):
        if args_nonlinearity == "relu":
            return (lambda: nn.ReLU(inplace=True))()
        else:
            raise ValueError(f"{args_nonlinearity} is not an initialization option!")

    def _init_conv(self, conv):
        conv.zero = self.zero
        conv.freeze = self.freeze

        if args_init == "signed_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, args_mode)
            if args_scale_fan:
                #fan = fan * (1 - args_prune_rate)
                fan = fan * args_prune_rate
            gain = nn.init.calculate_gain(args_nonlinearity)

            std = gain / math.sqrt(fan)
            conv.weight.data = conv.weight.data.sign() * std

            # print(std)

        elif args_init == "unsigned_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, args_mode)
            if args_scale_fan:
                #fan = fan * (1 - args_prune_rate)
                fan = fan * args_prune_rate

            gain = nn.init.calculate_gain(args_nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = torch.ones_like(conv.weight.data) * std

        elif args_init == "kaiming_normal":

            if args_scale_fan:
                fan = nn.init._calculate_correct_fan(conv.weight, args_mode)
                #fan = fan * (1 - args_prune_rate)
                fan = fan * args_prune_rate
                gain = nn.init.calculate_gain(args_nonlinearity)
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    conv.weight.data.normal_(0, std)
            else:
                nn.init.kaiming_normal_(
                    conv.weight, mode=args_mode, nonlinearity=args_nonlinearity
                )

        elif args_init == "kaiming_uniform":
            nn.init.kaiming_uniform_(
                conv.weight, mode=args_mode, nonlinearity=args_nonlinearity
            )
        elif args_init == "xavier_normal":
            nn.init.xavier_normal_(conv.weight)
        elif args_init == "xavier_constant":

            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            conv.weight.data = conv.weight.data.sign() * std

        elif args_init == "standard":

            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))

        else:
            raise ValueError(f"{args_init} is not an initialization option!")

def get_builder():

    print("==> Conv Type: {}".format(args_conv_type))
    print("==> BN Type: {}".format(args_bn_type))

    conv_layer = globals()[args_conv_type] #getattr(utils_conv_type, args_conv_type)
    bn_layer = globals()[args_bn_type] #getattr(utils_bn_type, args_bn_type)

    if args_first_layer_type is not None:
        first_layer = getattr(utils_conv_type, args_first_layer_type)
        print(f"==> First Layer Type: {args_first_layer_type}")
    else:
        first_layer = None

    builder = Builder(conv_layer=conv_layer, bn_layer=bn_layer, first_layer=first_layer)

    return builder

class BuilderZero(object):
    def __init__(self, conv_layer, bn_layer):
        self.conv_layer = conv_layer
        self.bn_layer = bn_layer
        self.zero = True
        self.freeze = args_freeze_weights_2

    def conv(self, kernel_size, in_planes, out_planes, stride=1):
        conv_layer = self.conv_layer

        if kernel_size == 3:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,               
            )
        elif kernel_size == 1:
            conv = conv_layer(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=False,               
            )
        elif kernel_size == 5:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=False,               
            )
        elif kernel_size == 7:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=False,               
            )
        else:
            return None

        self._init_conv(conv)

        return conv

    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride)
        return c

    def batchnorm(self, planes, last_bn=False):
        return self.bn_layer(planes)

    def activation(self):
        if args_nonlinearity == "relu":
            return (lambda: nn.ReLU(inplace=True))()
        else:
            raise ValueError(f"{args_nonlinearity} is not an initialization option!")

    def _init_conv(self, conv):
        conv.zero = self.zero
        conv.freeze = self.freeze

        if args_init_2 == "signed_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, args_mode_2)
            if args_scale_fan_2:
                # fan = fan * (1 - args_prune_rate_2)
                fan = fan * args_prune_rate_2
            gain = nn.init.calculate_gain(args_nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = conv.weight.data.sign() * std

        elif args_init_2 == "unsigned_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, args_mode_2)
            if args_scale_fan_2:
                # fan = fan * (1 - args_prune_rate_2)
                fan = fan * args_prune_rate_2

            gain = nn.init.calculate_gain(args_nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = torch.ones_like(conv.weight.data) * std

        elif args_init_2 == "kaiming_normal":

            if args_scale_fan_2:
                fan = nn.init._calculate_correct_fan(conv.weight, args_mode_2)
                # fan = fan * (1 - args_prune_rate_2)
                fan = fan * args_prune_rate_2
                gain = nn.init.calculate_gain(args_nonlinearity)
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    conv.weight.data.normal_(0, std)
            else:
                nn.init.kaiming_normal_(
                    conv.weight, mode=args_mode_2, nonlinearity=args_nonlinearity
                )

        elif args_init_2 == "kaiming_uniform":
            nn.init.kaiming_uniform_(
                conv.weight, mode=args_mode_2, nonlinearity=args_nonlinearity
            )
        elif args_init_2 == "xavier_normal":
            nn.init.xavier_normal_(conv.weight)
        elif args_init_2 == "xavier_constant":

            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            conv.weight.data = conv.weight.data.sign() * std

        elif args_init_2 == "standard":

            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))

        else:
            raise ValueError(f"{args_init_2} is not an initialization option!")

def get_builder_zero():

    print("==> Conv Type: {}".format(args_conv_type_2))
    print("==> BN Type: {}".format(args_bn_type_2))

    conv_layer = globals()[args_conv_type_2] #getattr(utils_conv_type, args_conv_type_2)
    bn_layer = globals()[args_bn_type_2] #getattr(utils_bn_type, args_bn_type_2)

    builder = BuilderZero(conv_layer=conv_layer, bn_layer=bn_layer)

    return builder

# -------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, builder, in_planes, planes, wider=1, stride=1):
        super(BasicBlock, self).__init__()

        width = planes * wider

        self.conv1 = builder.conv3x3(in_planes, width, stride=stride)
        self.bn1 = builder.batchnorm(width)
        self.relu1 = builder.activation()
        self.conv2 = builder.conv3x3(width, self.expansion * planes, stride=1)
        self.bn2 = builder.batchnorm(self.expansion * planes)
        self.relu2 = builder.activation()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.batchnorm(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, builder, in_planes, planes, wider=1, stride=1):
        super(Bottleneck, self).__init__()

        width = planes * wider

        self.conv1 = builder.conv1x1(in_planes, width)
        self.bn1 = builder.batchnorm(width)
        self.relu1 = builder.activation()
        self.conv2 = builder.conv3x3(width, width, stride=stride)
        self.bn2 = builder.batchnorm(width)
        self.relu2 = builder.activation()
        self.conv3 = builder.conv1x1(width, self.expansion * planes)
        self.bn3 = builder.batchnorm(self.expansion * planes)
        self.relu3 = builder.activation()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.batchnorm(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)

        return out

class FoldBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, builder, in_planes, planes, wider=1, stride=1, iters=1):
        super(FoldBasicBlock, self).__init__()

        width = planes * wider

        self.iters = iters

        self.conv1 = builder.conv3x3(in_planes, width, stride=stride)
        self.bn1 = builder.batchnorm(width)
        self.relu1 = builder.activation()
        self.conv2 = builder.conv3x3(width, self.expansion * planes, stride=1)
        self.bn2 = builder.batchnorm(self.expansion * planes)
        self.relu2 = builder.activation()


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.batchnorm(self.expansion * planes),
            )

    def forward(self, x):
        for t in range(self.iters):
            x_i = x
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            x += self.shortcut(x_i)
            x = self.relu2(x)
        return x

class FoldBottleneck(nn.Module):
    expansion = 4

    def __init__(self, builder, in_planes, planes, wider=1, stride=1, iters=1):
        super(FoldBottleneck, self).__init__()

        width = planes * wider

        self.iters = iters

        self.conv1 = builder.conv1x1(in_planes, width)
        self.bn1 = builder.batchnorm(width)
        self.relu1 = builder.activation()
        self.conv2 = builder.conv3x3(width, width, stride=stride)
        self.bn2 = builder.batchnorm(width)
        self.relu2 = builder.activation()
        self.conv3 = builder.conv1x1(width, self.expansion * planes)
        self.bn3 = builder.batchnorm(self.expansion * planes)
        self.relu3 = builder.activation()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.batchnorm(self.expansion * planes),
            )
    def forward(self, x):
        for t in range(self.iters):
            x_i = x
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
            x += self.shortcut(x_i)
            x = self.relu3(x)
        return x

class FoldBasicBlockUBN(nn.Module):
    expansion = 1

    def __init__(self, builder, in_planes, planes, wider=1, stride=1, iters=1):
        super(FoldBasicBlockUBN, self).__init__()

        width = planes * wider

        self.iters = iters

        self.conv1 = builder.conv3x3(in_planes, width, stride=stride)
        self.relu1 = builder.activation()
        self.conv2 = builder.conv3x3(width, self.expansion * planes, stride=1)
        self.relu2 = builder.activation()

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = builder.conv1x1(in_planes, self.expansion * planes, stride=stride)

        for t in range(self.iters):
            setattr(self, f'bnorm1_{t}', nn.BatchNorm2d(width))
            setattr(self, f'bnorm2_{t}', nn.BatchNorm2d(width))
            if self.shortcut:
                setattr(self, f'bnorm3_{t}', nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        for t in range(self.iters):
            x_i = x

            x = self.conv1(x)
            x = getattr(self, f'bnorm1_{t}')(x)
            x = self.relu1(x)

            x = self.conv2(x)
            x = getattr(self, f'bnorm2_{t}')(x)

            if self.shortcut:
                y = self.shortcut(x_i)                
                x += getattr(self, f'bnorm3_{t}')(y)
            else:
                x += x_i

            x = self.relu2(x)

        return x

class FoldBottleneckUBN(nn.Module):
    expansion = 4

    def __init__(self, builder, in_planes, planes, wider=1, stride=1, iters=1):
        super(FoldBottleneckUBN, self).__init__()

        width = planes * wider

        self.iters = iters

        self.conv1 = builder.conv1x1(in_planes, width)
        self.relu1 = builder.activation()
        self.conv2 = builder.conv3x3(width, width, stride=stride)
        self.relu2 = builder.activation()
        self.conv3 = builder.conv1x1(width, self.expansion * planes)
        self.relu3 = builder.activation()


        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = builder.conv1x1(in_planes, self.expansion * planes, stride=stride)

        for t in range(self.iters):
            setattr(self, f'bnorm1_{t}', nn.BatchNorm2d(width))
            setattr(self, f'bnorm2_{t}', nn.BatchNorm2d(width))
            setattr(self, f'bnorm3_{t}', nn.BatchNorm2d(self.expansion * planes))
            if self.shortcut:
                setattr(self, f'bnorm4_{t}', nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        for t in range(self.iters):
            x_i = x

            x = self.conv1(x)
            x = getattr(self, f'bnorm1_{t}')(x)
            x = self.relu1(x)

            x = self.conv2(x)
            x = getattr(self, f'bnorm2_{t}')(x)
            x = self.relu2(x)

            x = self.conv3(x)
            x = getattr(self, f'bnorm3_{t}')(x)

            if self.shortcut:
                y = self.shortcut(x_i)                
                x += getattr(self, f'bnorm4_{t}')(y)
            else:
                x += x_i
            
            x = self.relu3(x)

        return x

class HFResNet(nn.Module):
    def __init__(self, builder, builder_zero, block, block_zero, zeros, num_blocks, wider=1):
        super(HFResNet, self).__init__()
        self.in_planes = 64
        self.builder = builder
        self.builder_zero = builder_zero
        self.block = block
        self.block_zero = block_zero
        self.zeros = zeros
        self.num_blocks = num_blocks

        #PRE-NET
        self.conv1 = builder.conv7x7(3, 64, stride=2, first_layer=True)
        self.bn1 = builder.batchnorm(64)
        self.relu = builder.activation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
      
        #CORE      
        self._make_layer(layer_n=0, planes=64, stride=1, wider=wider)
        self._make_layer(layer_n=1, planes=128, stride=2, wider=wider)
        self._make_layer(layer_n=2, planes=256, stride=2, wider=wider)
        self._make_layer(layer_n=3, planes=512, stride=2, wider=wider)

        #POST-NET
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if args_last_layer_dense:
            self.fc = nn.Conv2d(512 * block.expansion, args_n_classes, 1)
        else:
            self.fc = builder.conv1x1(512 * block.expansion, args_n_classes)

    def _make_layer(self, layer_n, planes, stride, wider):
        # projection block
        self.add_module(f"layer{layer_n+1}_block{0}", self.block(self.builder, self.in_planes, planes, wider, stride=stride))
        
        self.in_planes = planes * self.block.expansion

        # rest of blocks
        sel_block = self.block_zero if self.zeros[layer_n] else self.block
        sel_builder = self.builder_zero if self.zeros[layer_n] else self.builder
        
        if sel_block.__name__.startswith('Fold'):
            self.add_module(f"layer{layer_n+1}_block1", sel_block(sel_builder, self.in_planes, planes, wider, stride=1, iters=self.num_blocks[layer_n]-1))
        else:
            for i in range(1, self.num_blocks[layer_n]):
                self.add_module(f"layer{layer_n+1}_block{i}", sel_block(sel_builder, self.in_planes, planes, wider, stride=1))
        self.in_planes = planes * sel_block.expansion

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        for layer_n in range(4):
            sel_block = self.block_zero if self.zeros[layer_n] else self.block
            if sel_block.__name__.startswith('Fold'):
                out = getattr(self, f'layer{layer_n+1}_block0')(out)
                out = getattr(self, f'layer{layer_n+1}_block1')(out)
            else:
                for i in range(self.num_blocks[layer_n]):
                    out = getattr(self, f'layer{layer_n+1}_block{i}')(out)

        out = self.avgpool(out)
        out = self.fc(out)
        return out.flatten(1)

def HFResNet50():
    return HFResNet(get_builder(), None, Bottleneck, None, [False, False, False, False], [3, 4, 6, 3])

def HFResNet50_1_2_3_4_normal():
    return HFResNet(get_builder(), get_builder_zero(), Bottleneck, Bottleneck, [True, True, True, True], [3, 4, 6, 3])
def HFResNet50_1_2_3_4_fold():
    return HFResNet(get_builder(), get_builder_zero(), Bottleneck, FoldBottleneck, [True, True, True, True], [3, 4, 6, 3])
def HFResNet50_1_2_3_4_ubn():
    return HFResNet(get_builder(), get_builder_zero(), Bottleneck, FoldBottleneckUBN, [True, True, True, True], [3, 4, 6, 3])

def HFResNet50_2_3_4_normal():
    return HFResNet(get_builder(), get_builder_zero(), Bottleneck, Bottleneck, [False, True, True, True], [3, 4, 6, 3])
def HFResNet50_2_3_4_fold():
    return HFResNet(get_builder(), get_builder_zero(), Bottleneck, FoldBottleneck, [False, True, True, True], [3, 4, 6, 3])
def HFResNet50_2_3_4_ubn():
    return HFResNet(get_builder(), get_builder_zero(), Bottleneck, FoldBottleneckUBN, [False, True, True, True], [3, 4, 6, 3])

def HFResNet50_3_4_normal():
    return HFResNet(get_builder(), get_builder_zero(), Bottleneck, Bottleneck, [False, False, True, True], [3, 4, 6, 3])
def HFResNet50_3_4_fold():
    return HFResNet(get_builder(), get_builder_zero(), Bottleneck, FoldBottleneck, [False, False, True, True], [3, 4, 6, 3])
def HFResNet50_3_4_ubn():
    return HFResNet(get_builder(), get_builder_zero(), Bottleneck, FoldBottleneckUBN, [False, False, True, True], [3, 4, 6, 3])

def HFResNet50_4_normal():
    return HFResNet(get_builder(), get_builder_zero(), Bottleneck, Bottleneck, [False, False, False, True], [3, 4, 6, 3])
def HFResNet50_4_fold():
    return HFResNet(get_builder(), get_builder_zero(), Bottleneck, FoldBottleneck, [False, False, False, True], [3, 4, 6, 3])
def HFResNet50_4_ubn():
    return HFResNet(get_builder(), get_builder_zero(), Bottleneck, FoldBottleneckUBN, [False, False, False, True], [3, 4, 6, 3])

def HFResNet50_3_normal():
    return HFResNet(get_builder(), get_builder_zero(), Bottleneck, Bottleneck, [False, False, True, False], [3, 4, 6, 3])
def HFResNet50_3_fold():
    return HFResNet(get_builder(), get_builder_zero(), Bottleneck, FoldBottleneck, [False, False, True, False], [3, 4, 6, 3])
def HFResNet50_3_ubn():
    return HFResNet(get_builder(), get_builder_zero(), Bottleneck, FoldBottleneckUBN, [False, False, True, False], [3, 4, 6, 3])

def HFResNet34():
    return HFResNet(get_builder(), None, BasicBlock, None, [False, False, False, False], [3, 4, 6, 3])

# -------------

def freeze_model_weights(model):
    print("=> Freezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "freeze") and m.freeze == True:
            print(f"Freezing layer {n}")
            if hasattr(m, "weight") and m.weight is not None:
                print(f"==> No gradient to {n}.weight")
                m.weight.requires_grad = False
                if m.weight.grad is not None:
                    print(f"==> Setting gradient of {n}.weight to None")
                    m.weight.grad = None

                if hasattr(m, "bias") and m.bias is not None:
                    print(f"==> No gradient to {n}.bias")
                    m.bias.requires_grad = False

                    if m.bias.grad is not None:
                        print(f"==> Setting gradient of {n}.bias to None")
                        m.bias.grad = None
        elif hasattr(m, "freeze") and m.freeze == False:
            print(f"Not Freezing layer {n}")

def set_model_prune_rate(model, prune_rate, prune_rate_2):
    print(f"==> Setting prune rate of network to {prune_rate}")

    for n, m in model.named_modules():
        if hasattr(m, "zero") and m.zero == False:
            if hasattr(m, "set_prune_rate"):
                m.set_prune_rate(prune_rate)
                print(f"==> Setting prune rate of {n} to {prune_rate}")
        elif hasattr(m, "zero") and m.zero == True:
            if hasattr(m, "set_prune_rate"):
                m.set_prune_rate(prune_rate_2)
                print(f"==> Setting prune rate of {n} to {prune_rate_2}")

# -------------

Identifier = args_name

Model = locals()[args_arch]()

if (args_prune_rate > 0 or args_prune_rate_2 > 0):
    set_model_prune_rate(Model, prune_rate=args_prune_rate, prune_rate_2=args_prune_rate_2)
if args_freeze_weights or args_freeze_weights_2:
    freeze_model_weights(Model)

print(Identifier)
print(Model)
# exit()

# init the model and the preprocessing:
preprocessing = functools.partial(load_preprocess_images, image_size=224)

# get an activations model from the Pytorch Wrapper
activations_model = PytorchWrapper(identifier=Identifier, model=Model, preprocessing=preprocessing)

# actually make the model, with the layers you want to see specified:
model = ModelCommitment(identifier=Identifier, activations_model=activations_model,
                        # specify layers to consider
                        layers=Layers)


# The model names to consider. If you are making a custom model, then you most likley want to change
# the return value of this function.
def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """

    return [Identifier]


# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.
def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == Identifier

    # link the custom model to the wrapper object(activations_model above):
    wrapper = activations_model
    wrapper.image_size = 224
    return wrapper


# get_layers method to tell the code what layers to consider. If you are submitting a custom
# model, then you will most likley need to change this method's return values.
def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """

    # quick check to make sure the model is the correct one:
    assert name == Identifier

    # returns the layers you want to consider
    return  Layers

# Bibtex Method. For submitting a custom model, you can either put your own Bibtex if your
# model has been published, or leave the empty return value if there is no publication to refer to.
def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """

    # from pytorch.py:
    return ''

# Main Method: In submitting a custom model, you should not have to mess with this.
if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
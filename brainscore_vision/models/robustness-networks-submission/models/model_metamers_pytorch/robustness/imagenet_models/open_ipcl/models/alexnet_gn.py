from __future__ import print_function

import torch
import torch.nn as nn
from IPython.core.debugger import set_trace

import sys
# TODO: change to not be a direct path
sys.path.append('/om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/robustness/imagenet_models')
from custom_modules import FakeReLU, SequentialWithArgs

__all__ = ['alexnet_gn']

class ReluSequentialWrapper(nn.Module):
    def __init__(self):
        super(ReluSequentialWrapper, self).__init__()
        self.relu = nn.ReLU(inplace=False)
    def forward(self, x, fake_relu=False):
        if fake_relu:
            return FakeReLU.apply(x)
        else:
            return self.relu(x)
    
class alexnet_gn(nn.Module):
    def __init__(self, in_channel=3, out_dim=128, l2norm=True, layer_for_fc=None, out_dim_fc=1000, no_embedding=False):
        super(alexnet_gn, self).__init__()
        self._l2norm = l2norm
        self.no_embedding = no_embedding

        self.conv_block_1 = SequentialWithArgs(
            nn.Conv2d(in_channel, 96, 11, 4, 2, bias=False),
            nn.GroupNorm(32, 96),
            ReluSequentialWrapper(),
#             nn.MaxPool2d(3, 2), # move the maxpool out of sequential to align with other models
        )
        self.maxpool_block_1 = nn.MaxPool2d(3, 2)

        self.conv_block_2 = SequentialWithArgs(
            nn.Conv2d(96, 256, 5, 1, 2, bias=False),
            nn.GroupNorm(32, 256),
            ReluSequentialWrapper(),
#             nn.MaxPool2d(3, 2), # move the maxpool out of sequential to align with other models
        )
        self.maxpool_block_2 =  nn.MaxPool2d(3, 2)

        self.conv_block_3 = SequentialWithArgs(
            nn.Conv2d(256, 384, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 384),
            ReluSequentialWrapper(),
        )
        self.conv_block_4 = SequentialWithArgs(
            nn.Conv2d(384, 384, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 384),
            ReluSequentialWrapper(),
        )
        self.conv_block_5 = SequentialWithArgs(
            nn.Conv2d(384, 256, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 256),
            ReluSequentialWrapper(),
#             nn.MaxPool2d(3, 2),
        )
        self.maxpool_block_5 = nn.MaxPool2d(3, 2)
        self.ave_pool = nn.AdaptiveAvgPool2d((6,6))
        self.fc6 = SequentialWithArgs(
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            ReluSequentialWrapper(),
        )
        self.fc7 = SequentialWithArgs(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            ReluSequentialWrapper(),
        )
        self.fc8 = SequentialWithArgs(
            nn.Linear(4096, out_dim)
        )
        if self._l2norm: self.l2norm = Normalize(2)

        self.layer_for_fc = layer_for_fc
        if layer_for_fc is not None:
            if self.layer_for_fc == 'fc7':
                self.fc_final = nn.Linear(4096, out_dim_fc)
            else:
                self.fc_final = None
                raise ValueError('Linear eval not set up for layer %s'%layer_for_fc)

    def forward(self, x, with_latent=False, no_relu=False, fake_relu=False):
        del no_relu # not used for this architecture
        if with_latent:
            all_outputs = {}
            all_outputs['input_after_preproc'] = x

        # ReLU is the final part of the block
        if with_latent:
            all_outputs['conv_block_1'] = self.conv_block_1(x, fake_relu=fake_relu)
        x = self.conv_block_1(x)
        x = self.maxpool_block_1(x)

        # ReLU is the final part of the block
        if with_latent:
            all_outputs['conv_block_2'] = self.conv_block_2(x, fake_relu=fake_relu)
        x = self.conv_block_2(x)
        x = self.maxpool_block_2(x)

        # ReLU is the final part of the block
        if with_latent:
            all_outputs['conv_block_3'] = self.conv_block_3(x, fake_relu=fake_relu)
        x = self.conv_block_3(x)
       
        # ReLU is the final part of the block
        if with_latent:
            all_outputs['conv_block_4'] = self.conv_block_4(x, fake_relu=fake_relu)
        x = self.conv_block_4(x)

        if with_latent:
            all_outputs['conv_block_5'] = self.conv_block_5(x, fake_relu=fake_relu)
        x = self.conv_block_5(x)
        x = self.maxpool_block_5(x)

        x = self.ave_pool(x)
        if with_latent:
            all_outputs['avgpool'] = x

        x = x.view(x.shape[0], -1)

        # ReLU is the final part of the block
        if with_latent:
            all_outputs['fc6'] = self.fc6(x, fake_relu=fake_relu)
        x = self.fc6(x)

        # ReLU is the final part of the block
        if with_latent:
            all_outputs['fc7'] = self.fc7(x, fake_relu=fake_relu)
        x = self.fc7(x)

        # TODO: add in for the other layers that we need for linear evals. 
        if self.layer_for_fc == 'fc7':
            final_out = self.fc_final(x)
        else:
            final_out = None

        if self.no_embedding:
            if final_out is None:
                final_out = x
            if with_latent:
                return final_out, x, all_outputs
            else:
                return final_out

        else:
            x = self.fc8(x)
            if with_latent:
                all_outputs['fc8'] = x
    
            if self._l2norm: 
                x = self.l2norm(x)
                if with_latent:
                    all_outputs['l2norm'] = x
    
            if final_out is None:
                final_out = x
    
            if with_latent:
                all_outputs['final'] = final_out
                return final_out, x, all_outputs
            else:
                return final_out

    def compute_feat(self, x, layer):
        if layer <= 0:
            return x
        x = self.conv_block_1(x)
        x = self.maxpool_block_1(x) # moved max pooling out of this block for metamers project
        if layer == 1:
            return x
        x = self.conv_block_2(x)
        x = self.maxpool_block_2(x) # moved max pooling out of this block for metamers project
        if layer == 2:
            return x
        x = self.conv_block_3(x)
        if layer == 3:
            return x
        x = self.conv_block_4(x)
        if layer == 4:
            return x
        x = self.conv_block_5(x)
        x = self.maxpool_block_5(x) # moved max pooling out of this block for metamers project
        if layer == 5:
            return x
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)
        if layer == 6:
            return x
        x = self.fc7(x)
        if layer == 7:
            return x
        x = self.fc8(x)
        if self._l2norm: x = self.l2norm(x)
        return x


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


if __name__ == '__main__':
    import torch
    model = alexnet_gn().cuda()
    data = torch.rand(10, 3, 224, 224).cuda()
    out = model.compute_feat(data, 5)

    for i in range(10):
        out = model.compute_feat(data, i)
        print(i, out.shape)

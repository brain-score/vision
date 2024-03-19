# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from .. import models

__all__ = [
    'av_wrapper'
]


class Head(nn.Module):
    def __init__(self, input_dim, proj_dims):
        super(Head, self).__init__()
        if not isinstance(proj_dims, list):
            proj_dims = [proj_dims]

        projection = []
        for i, d in enumerate(proj_dims):
            projection += [nn.Linear(input_dim, d)]
            input_dim = d
            if i < len(proj_dims)-1:
                projection += [nn.ReLU(inplace=True)]
        self.projection = nn.Sequential(*projection)
        self.out_dim = proj_dims[-1]

    def forward(self, x):
        return self.projection(x)


class AV_Wrapper(nn.Module):
    def __init__(self, video_model, audio_model, proj_dim=128):
        super(AV_Wrapper, self).__init__()
        self.video_model = video_model
        self.audio_model = audio_model

        self.use_linear_proj = proj_dim is not None
        if proj_dim is not None:
            self.video_proj = Head(video_model.out_dim, proj_dim)
            self.audio_proj = Head(audio_model.out_dim, proj_dim)
            self.out_dim = self.video_proj.out_dim
        else:
            self.out_dim = video_model.out_dim

    def forward(self, video, audio):
        video_emb = self.video_model(video)
        video_emb = video_emb.view(video_emb.shape[0], video_emb.shape[1])
        if self.use_linear_proj:
            video_emb = self.video_proj(video_emb)

        audio_emb = self.audio_model(audio)
        audio_emb = audio_emb.view(audio_emb.shape[0], audio_emb.shape[1])
        if self.use_linear_proj:
            audio_emb = self.audio_proj(audio_emb)

        return video_emb, audio_emb


def av_wrapper(video_backbone, video_backbone_args, audio_backbone, audio_backbone_args, proj_dim=128, checkpoint=None):
    assert video_backbone in models.__dict__, 'Unknown model architecture'
    assert audio_backbone in models.__dict__, 'Unknown model architecture'
    video_model = models.__dict__[video_backbone](**video_backbone_args)
    audio_model = models.__dict__[audio_backbone](**audio_backbone_args)

    model = AV_Wrapper(video_model, audio_model, proj_dim=proj_dim)
    if checkpoint is not None:
        ckp = torch.load(checkpoint, map_location='cpu')
        print("Weights loaded.")
        nn.DataParallel(model).load_state_dict(ckp['model'])

    return model

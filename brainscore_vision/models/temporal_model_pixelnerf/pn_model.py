import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from pyhocon import ConfigFactory

class PN(nn.Module):
    def __init__(self, model_weights, config_path):
        super().__init__()
        from phys_extractors.models.pixelnerf.src.model import make_model
        conf = ConfigFactory.parse_file(config_path)
        self.net = make_model(conf["model"])
        self.net.load_state_dict(torch.load(model_weights, map_location='cpu'))
        self.latent_dim = 8192

    def get_encoder_feats(self, images):
        '''
        videos: [B, T, C, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, D] extracted features
        '''
        features = self.net.encoder(images)
        features = features.reshape(features.shape[0], -1)
        # Make sure the number of elements in the last dimension is at least 8192
        features = nn.AdaptiveAvgPool1d(8192)(features.float().cpu())
        return features

    def forward(self, videos):
        bs, num_frames, num_channels, h, w = videos.shape
        videos = videos.flatten(0, 1)
        input_states = self.get_encoder_feats(videos)
        input_states = input_states.reshape(bs, num_frames, -1).cuda()
        return input_states

# Given sequence of images, predicts next latent
class FrozenPretrainedEncoder(nn.Module):
    def __init__(self, model_weights, config_path):
        super().__init__()
        self.encoder = PN(model_weights, config_path)

    def forward(self, x):
        # set frozen pretrained encoder to eval mode
        self.encoder.eval()
        encoder_output = self.encoder(x)
        return encoder_output

def pfPN(model_weights, config_path, **kwargs):
    return FrozenPretrainedEncoder(model_weights,config_path,  **kwargs)

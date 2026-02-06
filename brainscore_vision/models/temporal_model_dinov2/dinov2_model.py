import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms
from transformers import AutoModel as automodel
    
class DINOV2(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = automodel.from_pretrained(model_name)
        self.latent_dim = 8192

    def extract_features(self, images):
        '''
        images: [B, C, H, W], Image is normalized with imagenet norm
        '''
        input_dict = {'pixel_values': images}
        decoder_outputs = self.model(**input_dict, output_hidden_states=True)
        features = decoder_outputs.last_hidden_state
        features_1 = features[:, 0]
        features_2 = nn.AdaptiveAvgPool1d((7168))(features[:, 1:].reshape(features.shape[0], -1).float())
        features = torch.cat((features_1, features_2), dim=1)
        return features

    def forward(self, videos):
        '''
        videos: [B, C, T, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, D] extracted features
        '''
        bs, num_frames, num_channels, h, w = videos.shape
        videos = videos.flatten(0, 1)
        features = self.extract_features(videos)
        features = features.reshape(bs, num_frames, -1)
        return features

# Given sequence of images, predicts next latent
class pfDINOV2(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = DINOV2(model_name)

    def forward(self, x):
        # set frozen pretrained encoder to eval mode
        self.encoder.eval()
        # x is (Bs, T, 3, H, W)
        return self.encoder(x)


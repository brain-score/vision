import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms
from transformers import ResNetModel

class ResNet(nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        self.model = ResNetModel.from_pretrained(weights_path)

    def fwd(self, images):
        '''
        images: [B, C, H, W], Image is normalized with imagenet norm
        '''
        input_dict = {'pixel_values': images}

        decoder_outputs = self.model(**input_dict, output_hidden_states=True)

        features = decoder_outputs.last_hidden_state

        return features


    def forward(self, videos):
        '''
        videos: [B, C, T, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, C, H, W] extracted features
        '''
        bs, num_frames, num_channels, h, w = videos.shape
        videos = videos.flatten(0, 1)
        features = self.fwd(videos)
        features = features.reshape(bs, num_frames, features.shape[1], features.shape[2], features.shape[3])
        return features

# Given sequence of images, predicts next latent
class pfResNet(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.encoder = ResNet(model_name)

    def forward(self, x):
        # set frozen pretrained encoder to eval mode
        self.encoder.eval()
        return self.encoder(x)


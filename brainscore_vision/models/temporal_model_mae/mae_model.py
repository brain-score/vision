import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms
from transformers import ViTMAEForPreTraining as automodel
    
class MAE(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = automodel.from_pretrained(model_name, mask_ratio=0.0)
        self.latent_dim = 100352#8192

    def fwd(self, images):
        '''
        images: [B, C, H, W], Image is normalized with imagenet norm
        '''
        input_dict = {'pixel_values': images}
        return_dict = self.model.config.use_return_dict
        outputs = self.model.vit(
            input_dict['pixel_values'],
            noise=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        decoder_outputs = self.model.decoder(latent, ids_restore, output_hidden_states=True)
        features = decoder_outputs.hidden_states[-4][:, 1:]
        features = features.reshape(features.shape[0], -1)
        #features = nn.AdaptiveAvgPool1d((8192))(features.float())
        return features

    def forward(self, videos):
        '''
        videos: [B, C, T, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, C, H, W] extracted features
        '''
        bs, num_frames, num_channels, h, w = videos.shape
        videos = videos.flatten(0, 1)
        features = self.fwd(videos)
        features = features.reshape(bs, num_frames, -1)
        return features

# Given sequence of images, predicts next latent
class pfMAE(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.encoder = MAE(model_name)

    def forward(self, x):
        # set frozen pretrained encoder to eval mode
        self.encoder.eval()
        # x is (Bs, T, 3, H, W)
        return self.encoder(x)

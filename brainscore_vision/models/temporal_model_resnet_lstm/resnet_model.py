import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from collections import OrderedDict
from transformers import ResNetModel


R3M_VAL_TRANSFORMS = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
]


def load_model(
    model, model_path, state_dict_key="state_dict"
):
    params = torch.load(model_path, map_location="cpu")
    sd = params
        
    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith("module.") and 'dynamics' in k:
            name = k[7:]
        elif k.startswith("module."):
            name = k[7:]
        else:
            name = k
        new_sd[name] = v
    model.load_state_dict(new_sd)
    print(f"Loaded parameters from {model_path}")

    return model

class ResNet50(nn.Module):
    def __init__(self):

        super().__init__()
        self.model = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.latent_dim = 8192


    def get_encoder_feats(self, images):
        '''
        images: [B, C, H, W], Image is normalized with imagenet norm
        '''
        input_dict = {'pixel_values': images}

        decoder_outputs = self.model(**input_dict, output_hidden_states=True)

        features = decoder_outputs.last_hidden_state
        
        features = features.reshape(features.shape[0], -1)
        
        features = nn.AdaptiveAvgPool1d(8192)(features.float())

        return features

    def forward(self, videos, n_past=None):
        bs, num_frames, num_channels, h, w = videos.shape
        videos = videos.flatten(0, 1)
        input_states = self.get_encoder_feats(videos)
        input_states = input_states.reshape(bs, num_frames, -1)
        output = {
            "input_states": input_states[:, : n_past],
            "observed_states": input_states,
        }
        return output

class LSTM(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(self.latent_dim, 1024, batch_first=True)
        self.regressor = nn.Linear(1024, self.latent_dim)

    def forward_step(self, feats):
        assert feats.ndim == 3
        # note: for lstms, hidden is the last timestep output
        _, hidden = self.lstm(feats)
        # assumes n_layers=1
        x = torch.squeeze(hidden[0].permute(1, 0, 2), dim=1)
        x = self.regressor(x)
        return x

    def forward(self, input_states, rollout_steps):
        simulated_states = []
        prev_states = input_states
        for step in range(rollout_steps):
            # dynamics model predicts next latent from past latents
            pred_state = self.forward_step(prev_states)
            simulated_states.append(pred_state)
            # add most recent pred and delete oldest (to maintain a temporal window of length n_past)
            
            prev_states = torch.cat([prev_states[:, 1:], pred_state.unsqueeze(1)], axis=1)

        output = {
            "simulated_states": torch.stack(simulated_states, axis=1),
            "rollout_states": torch.cat([input_states, torch.stack(simulated_states, axis=1)], axis=1),
        }
        return output

# Given sequence of images, predicts next latent
class FrozenPretrainedEncoder(nn.Module):
    def __init__(self, n_past=7):
        super().__init__()

        self.n_past = n_past
        self.encoder = ResNet50()

        dynamics_kwargs = {"latent_dim": self.encoder.latent_dim}
        self.dynamics = LSTM(**dynamics_kwargs)

    def forward(self, x, n_past=None):
        # set frozen pretrained encoder to eval mode
        self.encoder.eval()
        # x is (Bs, T, 3, H, W)
        assert len(x.shape) == 5
        if x.shape[1] <= self.n_past:
            self.n_past = -1
        
        rollout_steps = x[:, self.n_past :].shape[1]
        encoder_output = self.encoder(x, self.n_past)
        dynamics_output = self.dynamics(encoder_output['input_states'], rollout_steps)

        output = {
            "input_states": encoder_output['input_states'],
            "observed_states": encoder_output['observed_states'],
            "simulated_states": dynamics_output['simulated_states'],
            "rollout_states": dynamics_output['rollout_states'],
        }
        return output

def pfResNet_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(n_past=n_past, **kwargs)
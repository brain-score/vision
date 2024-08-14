import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from collections import OrderedDict
from pyhocon import ConfigFactory

from phys_readouts.models.pixelnerf.src.model import make_model

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

class PN(nn.Module):
    def __init__(self):

        super().__init__()

        conf = '../conf/exp/sn64.conf'

        conf = ConfigFactory.parse_file(conf)

        self.net = make_model(conf["model"])
        
        weights_path = '/ccn2/u/thekej/pixel_nerf/pixel_nerf_latest'
        self.net.load_state_dict(torch.load(weights_path, map_location='cpu'))

        self.latent_dim = 8192

    def forward(self, images):
        '''
        videos: [B, T, C, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, D] extracted features
        '''

        features = self.net.encoder(images)
        features = features.reshape(features.shape[0], -1)

        features = nn.AdaptiveAvgPool1d(8192)(features.float())
        return features

class LSTM(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(self.latent_dim, 1024)
        self.regressor = nn.Linear(1024, self.latent_dim)

    def forward_step(self, x):
        feats = torch.stack(x)  # (T, Bs, self.latent_dim)
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
            prev_states.append(pred_state)
            prev_states.pop(0)

        output = {
            "simulated_states": torch.stack(simulated_states, axis=1),
            "rollout_states": torch.cat([torch.stack(input_states, axis=1), torch.stack(simulated_states, axis=1)], axis=1),
        }
        return output

# Given sequence of images, predicts next latent
class FrozenPretrainedEncoder(nn.Module):
    def __init__(self, n_past=7):
        super().__init__()

        self.n_past = n_past
        self.encoder = PN()

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
            "input_states": torch.stack(encoder_output['input_states'], axis=1),
            "observed_states": encoder_output['observed_states'],
            "simulated_states": dynamics_output['simulated_states'],
            "rollout_states": dynamics_output['rollout_states'],
        }
        return output

def pfPN_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(n_past=n_past, **kwargs)

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from collections import OrderedDict
from pyhocon import ConfigFactory


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
    def __init__(self, config_path):
        super().__init__()
        from phys_extractors.models.pixelnerf.src.model import make_model
        conf = ConfigFactory.parse_file(config_path)
        self.net = make_model(conf["model"])
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

    def forward(self, videos, n_past):
        bs, num_frames, num_channels, h, w = videos.shape
        videos = videos.flatten(0, 1)
        input_states = self.get_encoder_feats(videos)
        input_states = input_states.reshape(bs, num_frames, -1)
        if torch.cuda.is_available():
            input_states = input_states.cuda()
        output = {
            "input_states": input_states[:, : n_past],
            "observed_encoder_states": input_states,
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
        
    def forward(self, input_states, rollout_steps, n_simulation, n_past):
        observed_dynamics_states, simulated_states = [], []
        prev_states = input_states["observed_encoder_states"]
        n_context = n_past
        for step in range(rollout_steps):
            # dynamics model predicts next latent from past latents
            prev_states_ = prev_states[:, step:step+n_context]
            pred_state = self.forward_step(prev_states_)
            observed_dynamics_states.append(pred_state)
            
        simulation_input = input_states["input_states"]
        for step in range(n_simulation):
            # dynamics model predicts next latent from past latents
            pred_state = self.forward_step(simulation_input)
            simulated_states.append(pred_state)
            # add most recent pred and delete oldest (to maintain a temporal window of length n_past)
            
            simulation_input = torch.cat([simulation_input[:, 1:], pred_state.unsqueeze(1)], axis=1)

        output = {
            "simulated_rollout_states": torch.cat([input_states["input_states"],
                                            torch.stack(simulated_states, axis=1)], 
                                            axis=1),
            "observed_dynamic_states": torch.cat([input_states["input_states"], 
                                                  torch.stack(observed_dynamics_states, axis=1)], 
                                                 axis=1),
        }
        return output

# Given sequence of images, predicts next latent
class FrozenPretrainedEncoder(nn.Module):
    def __init__(self, config_path, n_past=7, simulation_length=25):
        super().__init__()

        self.n_past = n_past
        self.n_simulation = simulation_length
        self.encoder = PN(config_path)

        dynamics_kwargs = {"latent_dim": self.encoder.latent_dim}
        self.dynamics = LSTM(**dynamics_kwargs)

    def forward(self, x, n_past=None):
        # set frozen pretrained encoder to eval mode
        self.encoder.eval()
        # x is (Bs, T, 3, H, W)
        #assert len(x.shape) == 5 and x.shape[1] >= self.n_past
        
        observed_rollout_steps = max(1, x[:, self.n_past :].shape[1]-self.n_past)
        encoder_output = self.encoder(x, self.n_past)
        dynamics_output = self.dynamics(encoder_output, 
                                        observed_rollout_steps, 
                                        self.n_simulation,
                                        self.n_past)

        output = {
            "observed_encoder_states": encoder_output['observed_encoder_states'],
            "observed_dynamic_states": dynamics_output['observed_dynamic_states'],
            "simulated_rollout_states": dynamics_output['simulated_rollout_states'],
        }
        return output

def pfPN_LSTM_physion(config_path, n_past=7, **kwargs):
    return FrozenPretrainedEncoder(config_path, n_past=n_past, **kwargs)

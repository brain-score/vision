import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from collections import OrderedDict
from r3m import load_r3m

def load_model(
    model, identifier, model_path, state_dict_key="state_dict"
):
    params = torch.load(model_path, map_location="cpu")
    sd = params

    if identifier == 'R3M-LSTM-ARAN':
        aran = True
        sd = sd['state_dict']
        
    new_sd = OrderedDict()
    for k, v in sd.items():
        if identifier == 'R3M-LSTM-ARAN':
            name = 'encoder.r3m.' + k[19:]
        elif k.startswith("module.") and 'r3m' in k and not identifier == 'R3M-LSTM-ARAN':
            name = 'encoder.r3m.module.' + k[19:]  # remove 'module.' of dataparallel/DDP
        elif k.startswith("module.") and 'dynamics' in k:
            name = k[7:]
        elif k.startswith("module."):
            name = k[7:]
        else:
            name = k
        new_sd[name] = v
    model.load_state_dict(new_sd)
    print(f"Loaded parameters from {model_path}")

    return model

class R3M_pretrained(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        self.r3m = load_r3m(model_name)
        self.latent_dim = 2048  # resnet50 final fc in_features

    def forward(self, x, n_past=None):
        input_states = self.get_encoder_feats(x)
        
        output = {
            "input_states": input_states[: n_past],
            "observed_encoder_states": torch.stack(input_states, axis=1),
        }
        return output

    def get_encoder_feats(self, x):
        # applies encoder to each image in x: (Bs, T, 3, H, W) or (Bs, 3, H, W)
        with torch.no_grad():  # TODO: best place to put this?
            if x.ndim == 4:  # (Bs, 3, H, W)
                feats = [self._extract_feats(x)]
            else:
                assert x.ndim == 5, "Expected input to be of shape (Bs, T, 3, H, W)"
                feats = []
                for _x in torch.split(x, 1, dim=1):
                    _x = torch.squeeze(
                        _x, dim=1
                    )  # _x is shape (Bs, 1, 3, H, W) => (Bs, 3, H, W) TODO: put this in _extract_feats?
                    feats.append(self._extract_feats(_x))
        return feats

    def _extract_feats(self, x):
        feats = self.r3m(x)
        feats = torch.flatten(feats, start_dim=1)  # (Bs, -1)
        return feats

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

    def forward(self, input_states, rollout_steps, n_simulation):
        observed_dynamics_states, simulated_states = [], []
        prev_states = input_states["observed_encoder_states"]
        n_context = prev_states.shape[1]
        for step in range(rollout_steps):
            # dynamics model predicts next latent from past latents
            prev_states = prev_states[:, step:step+n_context]
            pred_state = self.forward_step(prev_states)
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
    def __init__(self, n_past=7, simulation_length=25):
        super().__init__()

        self.n_past = n_past
        self.n_simulation = simulation_length
        self.encoder = R3M_pretrained()

        dynamics_kwargs = {"latent_dim": self.encoder.latent_dim}
        self.dynamics = LSTM(**dynamics_kwargs)

    def forward(self, x, n_past=None):
        # set frozen pretrained encoder to eval mode
        self.encoder.eval()
        # x is (Bs, T, 3, H, W)
        assert len(x.shape) == 5 and x.shape[1] >= self.n_past
        
        observed_rollout_steps = x[:, self.n_past :].shape[1]
        encoder_output = self.encoder(x, self.n_past)
        dynamics_output = self.dynamics(encoder_output, 
                                        observed_rollout_steps, 
                                        self.n_simulation)

        output = {
            "observed_encoder_states": encoder_output['observed_encoder_states'],
            "observed_dynamic_states": dynamics_output['observed_dynamic_states'],
            "simulated_rollout_states": dynamics_output['simulated_rollout_states'],
        }
        return output

def pfR3M_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(n_past=n_past, **kwargs)

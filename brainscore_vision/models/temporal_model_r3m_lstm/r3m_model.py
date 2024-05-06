import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from collections import OrderedDict

R3M_VAL_TRANSFORMS = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
]

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

class ID(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, x):
        assert isinstance(x, list)
        return x[-1]  # just return last embedding

class R3M_pretrained(nn.Module):
    def __init__(self):
        from .r3m import load_r3m
        super().__init__()
        self.r3m = load_r3m("resnet50")
        self.latent_dim = 2048  # resnet50 final fc in_features

    def forward(self, x):
        return self.r3m(x * 255.0)  # R3M expects image input to be [0-255]

class LSTM(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(self.latent_dim, 1024)
        self.regressor = nn.Linear(1024, self.latent_dim)

    def forward(self, x):
        feats = torch.stack(x)  # (T, Bs, self.latent_dim)
        assert feats.ndim == 3
        # note: for lstms, hidden is the last timestep output
        _, hidden = self.lstm(feats)
        # assumes n_layers=1
        x = torch.squeeze(hidden[0].permute(1, 0, 2), dim=1)
        x = self.regressor(x)
        return x

# Given sequence of images, predicts next latent
class FrozenPretrainedEncoder(nn.Module):
    def __init__(self, n_past=7, full_rollout=False):
        super().__init__()

        self.full_rollout = full_rollout
        self.n_past = n_past
        self.encoder = R3M_pretrained()

        dynamics_kwargs = {"latent_dim": self.encoder.latent_dim}
        if self.dynamics_name == "mlp":
            dynamics_kwargs["n_past"] = self.n_past
        self.dynamics = LSTM(**dynamics_kwargs)

    def forward(self, x, n_past=None):
        # set frozen pretrained encoder to eval mode
        self.encoder.eval()
        # x is (Bs, T, 3, H, W)
        assert len(x.shape) == 5
        if x.shape[1] <= self.n_past:
            self.n_past = -1
        
        inputs = x[:, : self.n_past]
        input_states = self.get_encoder_feats(inputs)

        if self.full_rollout:
            # roll out the entire trajectory
            label_images = x[:, self.n_past :]
            rollout_steps = label_images.shape[1]
        else:
            label_images = x[:, self.n_past]
            rollout_steps = 1

        observed_states = self.get_encoder_feats(label_images)
        simulated_states = []
        prev_states = input_states
        for step in range(rollout_steps):
            # dynamics model predicts next latent from past latents
            pred_state = self.dynamics(prev_states)
            simulated_states.append(pred_state)
            # add most recent pred and delete oldest (to maintain a temporal window of length n_past)
            prev_states.append(pred_state)
            prev_states.pop(0)

        input_states = torch.stack(input_states, axis=1)
        observed_states = torch.stack(observed_states, axis=1)
        simulated_states = torch.stack(simulated_states, axis=1)
        assert observed_states.shape == simulated_states.shape

        output = {
            "input_states": input_states,
            "observed_states": observed_states,
            "simulated_states": simulated_states,
        }
        if self.full_rollout:
            output["states"] = torch.cat([input_states, simulated_states], axis=1)
            # adding this one as a visualizable sanity check of feature extractor
            output["inputs_test"] = torch.cat([inputs, label_images], axis=1)
            assert output["inputs_test"].shape == x.shape
            assert np.array_equal(output["inputs_test"].cpu().numpy(), x.cpu().numpy())
            # should be matched in B and T dimensions
            assert output["states"].shape[0] == output["inputs_test"].shape[0]
            assert output["states"].shape[1] == output["inputs_test"].shape[1]
            assert output["states"].ndim >= 3
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
        feats = self.encoder(x)
        feats = torch.flatten(feats, start_dim=1)  # (Bs, -1)
        return feats

def pfR3M_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(n_past=n_past, **kwargs)
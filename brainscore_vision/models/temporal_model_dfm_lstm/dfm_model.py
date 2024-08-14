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

class DFM(nn.Module):
    def __init__(self, weights_path):

        super().__init__()

        render_settings = {
            "n_coarse": 64,
            "n_fine": 64,
            "n_coarse_coarse": 32,
            "n_coarse_fine": 0,
            "num_pixels": 64 ** 2,
            "n_feats_out": 64,
            "num_context": 1,
            "sampling": "patch",
            "cnn_refine": False,
            "self_condition": False,
            "lindisp": False,
            # "cnn_refine": True,
        }
        from PixelNeRF import PixelNeRFModelCond

        self.model = PixelNeRFModelCond(
            near=1.0,
            # dataset.z_near, we set this to be slightly larger than the one we used for training to avoid floaters
            far=2,
            model='dit',
            use_first_pool=False,
            mode='cond',
            feats_cond=True,
            use_high_res_feats=True,
            render_settings=render_settings,
            use_viewdir=False,
            image_size=128,
            use_abs_pose=False,
        )
        #weights_path = '/ccn2/u/rmvenkat/data/dfm_weights/re10k_model.pt'
        self.latent_dim = 8192

    def forward(self, videos):
        '''
        videos: [B, T, C, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, D] extracted features
        '''
        ctxt_rgb = videos.unsqueeze(1)

        b, num_context, c, h, w = ctxt_rgb.shape

        ctxt_rgb = rearrange(ctxt_rgb, "b t h w c -> (b t) h w c")

        t = torch.zeros((b, num_context), device=ctxt_rgb.device, dtype=torch.long)

        t_resnet = rearrange(t, "b t -> (b t)")
        ctxt_inp = ctxt_rgb
        feature_map = self.model.get_feats(ctxt_inp, t_resnet, abs_camera_poses=None)

        # To downscale it by a factor of 4, we are reducing the size of H and W
        # Calculate the new dimensions
        H_new = feature_map.shape[-1] // 4
        W_new = feature_map.shape[-2] // 4

        # Now, we will use the interpolate function from the torch.nn.functional module
        feature_map = F.interpolate(feature_map, size=(H_new, W_new), mode='bilinear', align_corners=False)

        features = feature_map.reshape(feature_map.shape[0], -1)

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
        self.encoder = DFM()

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

def pfDFM_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(n_past=n_past, **kwargs)

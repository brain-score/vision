import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from collections import OrderedDict
from einops import rearrange
from torch.functional import F

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
    def __init__(self):

        super().__init__()
        from phys_extractors.models.DFM_physion.PixelNeRF import PixelNeRFModelCond
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
        }

        self.model = PixelNeRFModelCond(
            near=1.0,
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
        self.latent_dim = 8192

    def forward(self, videos, n_past):
        '''
        videos: [B, T, C, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, D] extracted features
        '''
        ctxt_rgb = videos

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

        input_states = nn.AdaptiveAvgPool1d(8192)(features.float().cpu())
        input_states = input_states.reshape(b, num_context, -1)

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
    def __init__(self, n_past=7, simulation_length=25):
        super().__init__()

        self.n_past = n_past
        self.n_simulation = simulation_length
        self.encoder = DFM()

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

        if x.shape[1] >= self.n_past:
            output = {
                "observed_encoder_states": encoder_output['observed_encoder_states'],
                "observed_dynamic_states": dynamics_output['observed_dynamic_states'],
                "simulated_rollout_states": dynamics_output['simulated_rollout_states'],
            }
        else:
            output = {
                "observed_encoder_states": encoder_output['observed_encoder_states'],
                "observed_dynamic_states": dynamics_output['observed_dynamic_states'],
                "simulated_rollout_states": dynamics_output['simulated_rollout_states'],
            }
        return output

def pfDFM_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(n_past=n_past, **kwargs)

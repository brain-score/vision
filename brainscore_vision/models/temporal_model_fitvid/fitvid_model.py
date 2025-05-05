import torch
import torch.nn as nn
from collections import OrderedDict
from phys_extractors.models.FitVid import fitvid

def load_model(
    model, model_path, state_dict_key="state_dict"
):
    params = torch.load(model_path, map_location="cpu")
    new_sd = OrderedDict()
    for k, v in params.items():
        name = 'encoder.'+k[7:] if k.startswith("module.") else k
        new_sd[name] = v
    model.load_state_dict(new_sd)
    print(f"Loaded parameters from {model_path}")
    model.eval()
    return model

# Given sequence of images, predicts next latent
class FrozenPretrainedEncoder(nn.Module):
    def __init__(self, n_past=7):
        super().__init__()
        self.n_past = n_past
        self.encoder = fitvid.FitVid(n_past=n_past, train=False)

    def forward(self, x):
        # set frozen pretrained encoder to eval mode
        with torch.no_grad():
            output = self.encoder(x, n_past=x.shape[1])
        features = output['h_preds']
        return features

def FitVidEncoder(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(n_past=n_past, **kwargs)

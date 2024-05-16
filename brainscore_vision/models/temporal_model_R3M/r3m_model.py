import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms
from r3m import load_r3m

class R3M_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.r3m = load_r3m("resnet50")
        self.latent_dim = 2048  # resnet50 final fc in_features

    def forward(self, x):
        return self.get_encoder_feats(x)  # R3M expects image input to be [0-255]

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
        feats = self.r3m(x * 255.0)
        feats = torch.flatten(feats, start_dim=1)  # (Bs, -1)
        return feats

# Given sequence of images, predicts next latent
class pfR3M(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = R3M_pretrained()

    def forward(self, x):
        # set frozen pretrained encoder to eval mode
        self.encoder.eval()
        # x is (Bs, T, 3, H, W)
        assert len(x.shape) == 5
        return self.encoder(x)


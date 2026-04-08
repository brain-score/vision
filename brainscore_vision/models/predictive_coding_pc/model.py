"""
Brain-Score Model: Predictive Coding Visual Hierarchy
======================================================
Hierarchical Predictive Coding network (Rao & Ballard 1999).
Layers r0-r3 correspond to ResNet-50 layer1-layer4 features.

RSA against 7T fMRI (THINGS-fMRI, N=3 subjects) shows crossing
hierarchy gradient: r0 maximally correlated with V1 (rho=0.30),
r3 with IT (rho=0.16), interaction Δr0-Δr3 = +0.266, p=0.007.

Reference:
    Leutenegger, N. (2025).
    https://github.com/nilsleut/Predictive-Coding-and-the-Visual-Cortex
"""

import functools
import torch
import torch.nn as nn
import torchvision.models as models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.check_submission import check_models

MODEL_IDENTIFIER = 'predictive_coding_pc'


# ── PC Network ───────────────────────────────────────────────

class PCConfig:
    d_layer1 = 256;  d_layer2 = 512
    d_layer3 = 1024; d_layer4 = 2048
    T_infer  = 30;   lr_r     = 0.01


class PredictiveCodingNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W1 = nn.Parameter(torch.randn(cfg.d_layer1, cfg.d_layer2) * 0.01)
        self.W2 = nn.Parameter(torch.randn(cfg.d_layer2, cfg.d_layer3) * 0.01)
        self.W3 = nn.Parameter(torch.randn(cfg.d_layer3, cfg.d_layer4) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(cfg.d_layer1))
        self.b2 = nn.Parameter(torch.zeros(cfg.d_layer2))
        self.b3 = nn.Parameter(torch.zeros(cfg.d_layer3))

    def predict(self, r, W, b):
        return torch.tanh(r @ W.T + b)

    @torch.no_grad()
    def infer(self, inputs):
        cfg = self.cfg
        r0, r1 = inputs['layer1'].clone(), inputs['layer2'].clone()
        r2, r3 = inputs['layer3'].clone(), inputs['layer4'].clone()
        for _ in range(cfg.T_infer):
            eps0 = r0 - self.predict(r1, self.W1, self.b1)
            eps1 = r1 - self.predict(r2, self.W2, self.b2)
            eps2 = r2 - self.predict(r3, self.W3, self.b3)
            r0 = r0 + cfg.lr_r * 0.5 * (-eps0)
            r1 = r1 + cfg.lr_r * (-eps1 + eps0 @ self.W1)
            r2 = r2 + cfg.lr_r * (-eps2 + eps1 @ self.W2)
            r3 = r3 + cfg.lr_r * (eps2 @ self.W3)
        return r0, r1, r2, r3


# ── Brain-Score Wrapper ───────────────────────────────────────

class PCWrapper(nn.Module):
    """
    Wraps PC network for Brain-Score.
    Exposes r0-r3 as named Identity submodules for neural layer hooking.
    fc exposes ResNet-50 classification logits (1000-dim) for
    behavior benchmarks (ImageNet-C, etc.).
    """
    def __init__(self):
        super().__init__()
        cfg = PCConfig()

        # ResNet-50 feature extractor
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        resnet.eval()
        self._cache = {}
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            def make_hook(n):
                def hook(m, i, o):
                    self._cache[n] = o.mean(dim=[2, 3])
                return hook
            getattr(resnet, name).register_forward_hook(make_hook(name))
        self.resnet = resnet

        # PC network — random init (inference dynamics drive hierarchy)
        torch.manual_seed(42)
        self.pc = PredictiveCodingNet(cfg)

        # Named hook targets for Brain-Score (neural benchmarks)
        self.r0 = nn.Identity()
        self.r1 = nn.Identity()
        self.r2 = nn.Identity()
        self.r3 = nn.Identity()

        # Classification head for behavior benchmarks (1000-dim logits)
        # Reuses pretrained ResNet-50 avgpool + fc weights
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc  # Linear(2048, 1000)

    def forward(self, x):
        self._cache.clear()
        with torch.no_grad():
            self.resnet(x)
        feats = {k: self._cache[k] for k in
                 ['layer1', 'layer2', 'layer3', 'layer4']}

        # PC inference — updates representations hierarchically
        r0, r1, r2, r3 = self.pc.infer(feats)

        # Expose PC layers for neural benchmark hooks
        self.r0(r0)
        self.r1(r1)
        self.r2(r2)
        self.r3(r3)

        # Classification logits for behavior benchmarks
        # layer4 cache is [B, 2048] after spatial mean — restore to [B, 2048, 1, 1] for avgpool
        pooled = self.avgpool(
            self._cache['layer4'].unsqueeze(-1).unsqueeze(-1)
        ).flatten(1)
        logits = self.fc(pooled)
        return logits


# ── Brain-Score API ───────────────────────────────────────────

def get_model(identifier):
    assert identifier == MODEL_IDENTIFIER
    model = PCWrapper()
    model.eval()

    preprocessing = functools.partial(load_preprocess_images, image_size=224)

    return PytorchWrapper(
        identifier=identifier,
        model=model,
        preprocessing=preprocessing,
        batch_size=32,
    )


def get_layers(identifier):
    return ['r0', 'r1', 'r2', 'r3', 'fc']


def get_bibtex(identifier):
    return """
@misc{leutenegger2025pc,
  title={Predictive Coding and the Visual Cortex},
  author={Leutenegger, Nils},
  year={2025},
  url={https://github.com/nilsleut/Predictive-Coding-and-the-Visual-Cortex}
}
"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)

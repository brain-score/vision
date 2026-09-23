import functools
import hashlib

import torch
import torch.nn as nn
from diffusers import AutoencoderKL

from brainscore_core.supported_data_standards.brainio.s3 import load_file
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

MODEL_IDENTIFIER = 'jem_imagenet_p0p1_s1'
WEIGHTS = dict(bucket='brainscore-storage', folder_name='brainscore-vision/models/user_843', relative_path='jem_imagenet_p0p1_s1.pt', version_id='InSv3o18jA2Q.JXt.CVlb9aYvSaIqrU7')
WEIGHTS_SHA1 = '286d68809c6ae4cdf756753154ff606bea5d3eae'
VAE_REPO = 'stabilityai/sd-vae-ft-mse'
VAE_REVISION = '31f26fdeee1355a5c34592e401dd41e45d25a493'
LAYERS = ['jem.f.conv1', 'jem.f.layer1', 'jem.f.layer2', 'jem.f.layer3', 'jem.f.bn1']
BIBTEX = """@misc{jem_imagenet_p0p1_s1,
  title = {Latent JEM: Wide-ResNet-22-8 energy-based classifier on a frozen Stable-Diffusion VAE latent (ImageNet, penalty p=0.1, seed 1)},
  note = {Unpublished. Derived from the Joint Energy-based Model of Grathwohl et al., Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One, ICLR 2020, arXiv:1912.03263},
  year = {2026}
}"""


# JEM architecture from the project's JEM.py, trimmed to what these checkpoints use (Wide-ResNet-22-8,
# norm=None, dropout 0, no f.proj). Module names are unchanged so the state_dict loads strictly.
class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1, leak=.2):
        super().__init__()
        self.lrelu = nn.LeakyReLU(leak)
        self.bn1 = nn.Identity()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Identity()
        self.bn2 = nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True))

    def forward(self, x):
        out = self.dropout(self.conv1(self.lrelu(self.bn1(x))))
        out = self.conv2(self.lrelu(self.bn2(out)))
        return out + self.shortcut(x)


class Wide_ResNet(nn.Module):
    def __init__(self, depth=22, widen_factor=8, input_channels=4, leak=.2):
        super().__init__()
        self.in_planes = 16
        self.lrelu = nn.LeakyReLU(leak)
        n = (depth - 4) // 6
        stages = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.conv1 = nn.Conv2d(input_channels, stages[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self._wide_layer(stages[1], n, stride=1)
        self.layer2 = self._wide_layer(stages[2], n, stride=2)
        self.layer3 = self._wide_layer(stages[3], n, stride=2)
        self.bn1 = nn.Identity()

    def _wide_layer(self, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(wide_basic(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer3(self.layer2(self.layer1(self.conv1(x))))
        out = self.lrelu(self.bn1(out))
        return out.reshape(out.size(0), -1)


class CCF(nn.Module):
    def __init__(self, n_classes=1000, latent_dim=25088):
        super().__init__()
        self.f = Wide_ResNet()
        self.energy_output = nn.Linear(latent_dim, 1)   # not on the classify path; kept so weights load strictly
        self.class_output = nn.Linear(latent_dim, n_classes)

    def classify(self, z):
        return self.class_output(self.f(z.float()))


class LatentJEM(nn.Module):
    """Frozen SD-VAE encoder -> JEM classifier (pure feed-forward, no MCMC).

    `vae` is registered BEFORE `jem` on purpose: Brain-Score reads the 'logits' layer from the last
    registered leaf module (PytorchWrapper._output_layer), which must be jem.class_output.
    """
    def __init__(self, vae, jem):
        super().__init__()
        self.vae = vae
        self.jem = jem

    def forward(self, x):
        z = self.vae.encode(x).latent_dist.mean * self.vae.config.scaling_factor
        return self.jem.classify(z)


def get_model(name=MODEL_IDENTIFIER):
    assert name == MODEL_IDENTIFIER
    path = load_file(**WEIGHTS)
    with open(path, 'rb') as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()
    if sha1 != WEIGHTS_SHA1:
        raise RuntimeError(f"{path}: sha1 {sha1} != {WEIGHTS_SHA1} (wrong file or version_id?)")
    jem = CCF()
    jem.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model_state_dict'])
    vae = AutoencoderKL.from_pretrained(VAE_REPO, revision=VAE_REVISION, torch_dtype=torch.float32)
    vae.requires_grad_(False)
    model = LatentJEM(vae, jem).eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=224,
                                      normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5))
    wrapper = PytorchWrapper(identifier=MODEL_IDENTIFIER, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name=MODEL_IDENTIFIER):
    assert name == MODEL_IDENTIFIER
    return LAYERS


def get_bibtex(name=MODEL_IDENTIFIER):
    assert name == MODEL_IDENTIFIER
    return BIBTEX


if __name__ == '__main__':
    check_models.check_base_models(__name__)

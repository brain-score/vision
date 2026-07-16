import functools
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models


MODEL_ID = "persistence_mem"
WEIGHTS_URL = "https://huggingface.co/emirhaninan814/persistence_mem_weights/resolve/main/persistence_mem.pth"


def group_norm(channels):
    for groups in range(min(32, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class Probe(nn.Module):
    def forward(self, x):
        return x


class PersistenceMemNet(nn.Module):
    def __init__(self, num_classes=100, steps=4):
        super().__init__()
        self.steps = int(steps)
        self.gamma_max = 1.0
        base = models.resnet50(weights=None, num_classes=num_classes)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = base.fc
        self.proj_act = nn.GELU()

        def block(channels):
            return (
                nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), group_norm(channels), nn.GELU()),
                nn.Conv2d(channels * 2, channels, 1, bias=True),
                nn.Conv2d(channels, channels, 1, bias=False),
                group_norm(channels),
                nn.Parameter(torch.zeros(self.steps)),
                nn.Parameter(torch.logit(torch.tensor(0.7))),
                nn.Parameter(torch.logit(torch.tensor(0.03))),
            )

        (self.write_iconic_1, self.gate_iconic_1, self.proj_iconic_1,
         self.proj_norm_1, self.step_logits_1,
         self.alpha_logit_1, self.reentry_logit_1) = block(256)
        (self.write_iconic_2, self.gate_iconic_2, self.proj_iconic_2,
         self.proj_norm_2, self.step_logits_2,
         self.alpha_logit_2, self.reentry_logit_2) = block(512)
        (self.write_iconic_3, self.gate_iconic_3, self.proj_iconic_3,
         self.proj_norm_3, self.step_logits_3,
         self.alpha_logit_3, self.reentry_logit_3) = block(1024)
        (self.write_iconic_4, self.gate_iconic_4, self.proj_iconic_4,
         self.proj_norm_4, self.step_logits_4,
         self.alpha_logit_4, self.reentry_logit_4) = block(2048)

        self.layer1_probe = Probe()
        self.layer2_probe = Probe()
        self.v4_probe = Probe()
        self.it_probe = Probe()
        self.register_buffer("iconic_buffer", None)

    def _stem(self, x):
        return self.maxpool(self.relu(self.bn1(self.conv1(x))))

    def _mem(self, x, write, gate_layer, proj, norm, logits, alpha_logit, gamma_logit):
        memory = write(x)
        alpha = torch.sigmoid(alpha_logit)
        gamma = self.gamma_max * torch.sigmoid(gamma_logit)
        weights = F.softmax(logits, dim=0)
        acc = torch.zeros_like(x)
        for k in range(self.steps):
            gate = torch.sigmoid(gate_layer(torch.cat([x, memory], dim=1)))
            read = self.proj_act(norm(proj(memory)))
            acc = acc + weights[k] * gate * read
            memory = alpha * memory
        return x + gamma * acc

    def forward(self, x):
        x = self._stem(x)

        x = self.layer1(x)
        x = self._mem(x, self.write_iconic_1, self.gate_iconic_1, self.proj_iconic_1,
                      self.proj_norm_1, self.step_logits_1, self.alpha_logit_1,
                      self.reentry_logit_1)
        self.layer1_probe(x)

        x = self.layer2(x)
        x = self._mem(x, self.write_iconic_2, self.gate_iconic_2, self.proj_iconic_2,
                      self.proj_norm_2, self.step_logits_2, self.alpha_logit_2,
                      self.reentry_logit_2)
        self.layer2_probe(x)

        x = self.layer3(x)
        x = self._mem(x, self.write_iconic_3, self.gate_iconic_3, self.proj_iconic_3,
                      self.proj_norm_3, self.step_logits_3, self.alpha_logit_3,
                      self.reentry_logit_3)
        self.v4_probe(x)

        x = self.layer4(x)
        x = self._mem(x, self.write_iconic_4, self.gate_iconic_4, self.proj_iconic_4,
                      self.proj_norm_4, self.step_logits_4, self.alpha_logit_4,
                      self.reentry_logit_4)
        self.it_probe(x)

        return self.fc(torch.flatten(self.avgpool(x), 1))


def preprocess(image_filepaths, image_size=224):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    images = [
        transform(Image.open(path).convert("RGB")).unsqueeze(0).numpy()
        for path in image_filepaths
    ]
    return np.concatenate(images, axis=0)


def get_model(name):
    assert name == MODEL_ID
    model = PersistenceMemNet(num_classes=100)
    checkpoint_path = os.path.join(os.path.dirname(__file__), "persistence_mem.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    elif WEIGHTS_URL:
        checkpoint = torch.hub.load_state_dict_from_url(
            WEIGHTS_URL,
            map_location="cpu",
            progress=False,
            file_name="persistence_mem.pth",
        )
    else:
        raise RuntimeError(
            "persistence_mem weights are not bundled because Brain-Score website "
            "uploads are limited to 50MB. Set WEIGHTS_URL to the externally hosted "
            "persistence_mem.pth URL before submitting."
        )
    state = checkpoint.get("model", checkpoint)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    wrapper = PytorchWrapper(
        identifier=MODEL_ID,
        model=model,
        preprocessing=functools.partial(preprocess, image_size=224),
    )
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == MODEL_ID
    return ["layer1_probe", "layer2_probe", "v4_probe", "it_probe"]


def get_bibtex(model_identifier):
    return """@misc{solera2026persistence,
  author = {Solera},
  title = {Persistence memory model},
  year = {2026}
}"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)

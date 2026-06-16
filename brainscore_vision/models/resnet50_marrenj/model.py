"""ResNet50 Vanderbilt — final checkpoint of our supervised ImageNet-1k baseline
(epoch 89, 90-epoch budget). Direct counterpart to Schrimpf/Gokce 2025's
`resnet50_imagenet_full`, evaluated through the same Brain-Score pipeline.

Brain-Score sandbox runs this in a fresh Python env that only has the deps in
requirements.txt; weights are downloaded from a public HuggingFace model repo.
"""
import functools
import torch
import torchvision.models                  # NOTE: torchvision, not timm — this
                                            # matches Schrimpf's exact module-naming
                                            # convention (avgpool, layerN.M.relu) so
                                            # his fitted commitments apply verbatim.
                                            # State-dict keys are identical between
                                            # timm and torchvision resnet50, so the
                                            # checkpoint loads either way.
from huggingface_hub import hf_hub_download
from brainscore_vision.model_helpers.activations.pytorch import (
    PytorchWrapper, load_preprocess_images,
)

# === EDIT BEFORE SUBMITTING: point at YOUR HuggingFace repo holding the .ckpt ===
HF_REPO_ID = "marrenj/temporal-dynamics-baselines"     # <- change to your HF username
HF_FILENAME = "resnet50_imagenet_baseline_ep089.ckpt"

# Schrimpf/Gokce 2025's fitted region-layer commitments for resnet50_imagenet_full
# (from scaling_primate_vvs/brainscore/artifacts/commitments.json). These are
# TORCHVISION module names (not timm's — timm uses `act3`/`global_pool` where
# torchvision uses `relu`/`avgpool`). That's why we build with torchvision above.
REGION_LAYER_MAP = {
    "V1": "layer1.0.conv1",
    "V2": "layer3.5.bn3",
    "V4": "layer3.0.conv1",
    "IT": "layer4.0.relu",
}
BEHAVIORAL_READOUT_LAYER = "avgpool"
LAYERS = list(set(REGION_LAYER_MAP.values())) + [BEHAVIORAL_READOUT_LAYER]


BIBTEX = """@misc{marrenj_temporal_dynamics_2026,
  title={Temporal Dynamics of Human Behavioral Alignment in ImageNet-trained Models},
  author={Wallace Lab},
  year={2026},
  note={Supervised ResNet50, 90-epoch ImageNet-1k recipe},
}"""


def get_model():
    weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    raw = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = raw.get("state_dict", raw)
    # SupervisedLightningModule wraps the timm classifier as `self.model`; strip prefix.
    state_dict = {
        k[len("model."):]: v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    model = torchvision.models.resnet50(weights=None)
    model.load_state_dict(state_dict)
    model.eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(
        identifier="resnet50_marrenj", model=model, preprocessing=preprocessing,
    )
    wrapper.image_size = 224
    return wrapper


def get_bibtex(model_identifier):
    return BIBTEX

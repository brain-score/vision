"""CLIP-ViT-B/32 Vanderbilt — final checkpoint of our DeCLIP/YFCC15M-trained
contrastive baseline (epoch 31, 32-epoch budget). The model is our own custom
class (not OpenCLIP); we bundle a trimmed visual-only version under
`clip_arch.py` so the brain-score sandbox doesn't need our research repo.

Brain-Score coverage NOTE: behavioral benchmarks that decode from a 1000-class
ImageNet logits layer (e.g. Geirhos2021-*) will not work for this model because
contrastive CLIP doesn't have a native ImageNet classifier. To run those, we'd
need a zero-shot CLIP classifier head (cosine sim between visual features and
text embeddings of the 1000 ImageNet class names). For this first submission
we leave that out — neural V1/V2/V4/IT + Rajalingham2018-i2n behavioral all
work directly off the visual encoder's features.

We use CLIP-style preprocessing (Resize(224) → CenterCrop(224) → CLIP mean/std)
to match how the model was trained.
"""
import functools
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from huggingface_hub import hf_hub_download
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper

from .clip_arch import VisionTransformer

# === EDIT BEFORE SUBMITTING ===
HF_REPO_ID = "marrenj/temporal-dynamics-baselines"
HF_FILENAME = "clip_vitb32_baseline_ep031.ckpt"

# OpenAI CLIP normalization — what the visual encoder was trained with.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Candidate visual-encoder layers for brain-score's region commitment search.
# `ln_post` is the pre-projection 768-d CLS feature we already use as the
# behavioral readout in our own alignment metric.
LAYERS = [
    "transformer.0",
    "transformer.3",
    "transformer.6",
    "transformer.9",
    "transformer.11",
    "ln_post",
]
BEHAVIORAL_READOUT_LAYER = "ln_post"


BIBTEX = """@misc{marrenj_temporal_dynamics_2026,
  title={Temporal Dynamics of Human Behavioral Alignment in ImageNet-trained Models},
  author={Wallace Lab},
  year={2026},
  note={CLIP-ViT-B/32 (custom DeCLIP-trained), YFCC15M, 32 epochs},
}"""


def _clip_preprocessing(image_filepaths):
    """CLIP-style preprocessing pipeline that matches our model's training
    preprocessing (src/dataset.py eval branch). Returns a (B, C, 224, 224)
    numpy stack — brain-score's PytorchWrapper expects this exact shape."""
    val_transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    out = []
    for p in image_filepaths:
        img = Image.open(p).convert("RGB")
        out.append(val_transform(img).numpy())
    return np.stack(out)


def get_model():
    weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    raw = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = raw.get("state_dict", raw)
    # Lightning prefix: model.visual.<...> | model.text.* | model.logit_scale
    # We need: <...> for just the visual encoder.
    visual_sd = {
        k[len("model.visual."):]: v
        for k, v in state_dict.items()
        if k.startswith("model.visual.")
    }
    if not visual_sd:
        # In case the ckpt isn't Lightning-wrapped: try `visual.<...>` direct.
        visual_sd = {
            k[len("visual."):]: v for k, v in state_dict.items()
            if k.startswith("visual.")
        }
    if not visual_sd:
        raise RuntimeError(
            f"could not find any visual-encoder keys in checkpoint {HF_FILENAME}. "
            "Expected keys prefixed with 'model.visual.' or 'visual.'."
        )

    model = VisionTransformer()  # uses ViT-B/32 defaults
    missing, unexpected = model.load_state_dict(visual_sd, strict=False)
    if missing or unexpected:
        print(f"  [clip_vitb32_marrenj] state_dict load: "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        if missing[:3]: print(f"    sample missing: {missing[:3]}")
        if unexpected[:3]: print(f"    sample unexpected: {unexpected[:3]}")
    model.eval()

    wrapper = PytorchWrapper(
        identifier="clip_vitb32_marrenj",
        model=model,
        preprocessing=_clip_preprocessing,
    )
    wrapper.image_size = 224
    return wrapper


def get_bibtex(model_identifier):
    return BIBTEX

import functools
import logging

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import VJEPA2Model, AutoVideoProcessor

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"
MODEL_REVISION = "875c192b7b704b87d1e1d99345769632dd5f739a"
NUM_ENCODER_LAYERS = 40
LAYERS = [f"model.encoder.layer.{i}" for i in range(NUM_ENCODER_LAYERS)]

BIBTEX = """@article{bardes2025vjepa2,
  title={V-JEPA 2: Self-Supervised Video Models Enable Understanding, Generation, and Segmentation},
  author={Bardes, Adrien and Garrido, Quentin and Assran, Mahmoud and Balestriero, Randall
          and Misra, Ishan and LeCun, Yann and Rabbat, Michael and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2506.09985},
  year={2025}
}"""


class VJEPA2ImageModel(nn.Module):
    """Thin wrapper that adapts V-JEPA2 video model for single-image input.

    Accepts standard ``(B, C, H, W)`` tensors (what PytorchWrapper produces)
    and reshapes to ``(B, 1, C, H, W)`` (single-frame video) before forwarding.
    """

    def __init__(self, vjepa2_model: VJEPA2Model) -> None:
        super().__init__()
        self.model = vjepa2_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, C, H, W) -> (B, 1, C, H, W)
        return self.model(pixel_values_videos=x, skip_predictor=True)


class VJEPA2PytorchWrapper(PytorchWrapper):
    """PytorchWrapper subclass that handles tuple outputs from transformer blocks.

    Preserves full spatial token activations ``(batch, 256, 1408)`` so that
    downstream PLS regression can exploit retinotopic structure for V1/V2.
    """

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            if isinstance(output, tuple):
                output = output[0]
            target_dict[name] = output.cpu().data.numpy()

        hook = layer.register_forward_hook(hook_function)
        return hook


def _load_preprocess_images(
    image_filepaths: list[str],
    processor: AutoVideoProcessor,
    image_size: int,
) -> np.ndarray:
    """Load images from disk and preprocess with the HuggingFace video processor.

    Returns a numpy array of shape ``(N, C, H, W)`` ready for PytorchWrapper.
    The video/frame dimension is added later by :class:`VJEPA2ImageModel`.
    """
    images: list[Image.Image] = []
    for fp in image_filepaths:
        with Image.open(fp) as img:
            images.append(img.convert("RGB"))

    processed = []
    for img in images:
        inputs = processor([img], return_tensors="pt")
        # pixel_values_videos shape: (1, 1, C, H, W) -- squeeze frame dim
        pv = inputs["pixel_values_videos"].squeeze(1)  # -> (1, C, H, W)
        processed.append(pv.numpy())

    return np.concatenate(processed, axis=0)  # (N, C, H, W)


def get_model() -> VJEPA2PytorchWrapper:
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    vjepa2 = VJEPA2Model.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16)
    vjepa2.eval()

    image_model = VJEPA2ImageModel(vjepa2)
    preprocessing = functools.partial(
        _load_preprocess_images, processor=processor, image_size=256
    )

    wrapper = VJEPA2PytorchWrapper(
        identifier="vjepa2-vitg",
        model=image_model,
        preprocessing=preprocessing,
        batch_size=4,
    )
    wrapper.image_size = 256
    return wrapper

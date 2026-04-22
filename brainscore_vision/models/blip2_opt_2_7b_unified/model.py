"""
BLIP-2 OPT-2.7B registered as a BrainScoreModel — second VLM with a different
architecture from CLIP and Qwen.

Architecture: ViT-G/14 vision encoder + Q-Former + OPT-2.7B causal LM. This
is a third VLM topology (encoder + adapter + decoder), distinct from CLIP's
dual-encoder pair and Qwen's flattened-patch native VLM.

Vision input is standard (batch, C, H, W) — wraps cleanly with PytorchWrapper,
unlike Qwen which needed VLMVisionWrapper. This validates that the unified
interface scales across VLM families with very different vision input layouts:
- CLIP: dual encoder, vision_model.encoder.layers.{N}
- Qwen: flattened patches, VLMVisionWrapper
- BLIP-2: standard ViT, PytorchWrapper

Layer counts:
  vision_model.encoder.layers.{0-38}             — 39 ViT-G blocks
  language_model.model.decoder.layers.{0-31}     — 32 OPT decoder layers
"""

import functools
import re
from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from brainscore.model_helpers.text_wrapper import TextWrapper
from brainscore_core.model_interface import BrainScoreModel
from brainscore_vision.model_helpers.activations.pca import LayerPCA
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper


REGION_LAYER_MAP = {
    # Vision regions — relative to vision_model (PytorchWrapper root)
    'V1': 'encoder.layers.4',
    'V2': 'encoder.layers.10',
    'V4': 'encoder.layers.20',
    'IT': 'encoder.layers.34',
    # Language region — relative to language_model.model.decoder (TextWrapper root)
    'language_system': 'layers.28',
}


def _make_generation_fn(blip_model, blip_processor, max_new_tokens: int = 8):
    """Generation callable for BLIP-2 instruction-following.

    BLIP-2 OPT-2.7B isn't as instruction-tuned as Qwen-VL (no RLHF), so we
    prompt it in a captioning-style template that it handles better:
    "Question: {instruction} Answer:" → model completes with label.
    """
    device = next(blip_model.parameters()).device

    def generate(stimulus_row, instruction: str, label_set: List[str]) -> str:
        image = Image.open(stimulus_row['image_file_name']).convert('RGB')
        # BLIP-2 OPT responds best to "Question: ... Answer:" template
        prompt = (f"Question: {instruction} "
                  f"Choose from: {', '.join(label_set)}. Answer:")
        inputs = blip_processor(text=prompt, images=image,
                                return_tensors='pt')
        # Explicit per-tensor device move. BatchFeature.to(device) does not
        # reliably move all tensors on MPS; BLIP-2's generate() accesses
        # input_ids inside get_input_embeddings() and needs it on the same
        # device as the embedding weights.
        model_dtype = next(blip_model.parameters()).dtype
        inputs = {
            k: (v.to(device, dtype=model_dtype) if torch.is_tensor(v) and v.is_floating_point()
                else (v.to(device) if torch.is_tensor(v) else v))
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated = blip_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        response = blip_processor.tokenizer.decode(
            generated[0], skip_special_tokens=True).strip().lower()
        # BLIP-2 echoes the prompt sometimes; strip it
        if 'answer:' in response:
            response = response.split('answer:')[-1].strip()

        for lbl in label_set:
            if response == lbl.lower():
                return lbl
        for lbl in label_set:
            if re.search(rf'\b{re.escape(lbl.lower())}\b', response):
                return lbl
        return response  # unparseable — caller defaults to label_set[0]

    return generate


def _load_preprocess_images(image_filepaths, processor, image_size=224):
    """Load and preprocess images using BLIP-2's processor."""
    images = []
    for path in image_filepaths:
        with Image.open(path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img.copy())

    # BLIP-2 processor with images-only input
    processed = processor(images=images, return_tensors='pt')
    return processed['pixel_values'].numpy()


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'blip2-opt-2.7b'

    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        'Salesforce/blip2-opt-2.7b',
        torch_dtype=torch.float16,
    )
    blip_processor = AutoProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')

    preprocessing = functools.partial(
        _load_preprocess_images,
        processor=blip_processor,
        image_size=224,
    )

    # Vision: PytorchWrapper wraps vision_model. BLIP-2's vision_model can be
    # called standalone (forward signature: pixel_values -> last_hidden_state)
    # so the full model wrapper isn't needed.
    activations_model = PytorchWrapper(
        identifier=identifier,
        model=blip_model.vision_model,
        preprocessing=preprocessing,
    )
    activations_model.image_size = 224

    # ViT-G hook outputs (batch, 257, 1408) flatten to 361K features per image.
    # That's too many for sklearn PLS at scoring time. LayerPCA(1000) is the
    # convention used by LayerSelection / Layer Mapping Explorer — it fits a
    # PCA on 1000 ImageNet validation images and reduces hook output to 1000D.
    # First-time overhead is ~1-2 min per layer; result cached to disk.
    LayerPCA.hook(activations_model, n_components=1000)

    # Text: TextWrapper wraps the OPT decoder. Causal LM → last_token aggregation.
    # Wrap language_model.model.decoder so layer paths are 'layers.{N}'.
    text_wrapper = TextWrapper(
        model=blip_model.language_model.model.decoder,
        tokenizer=blip_processor.tokenizer,
        identifier=f'{identifier}-text',
        layer_aggregation='last_token',
        max_length=512,
    )

    # BLIP-2 is composed of vision_model + qformer + language_projection +
    # language_model. Each wrapper moves only its sub-module to device, so
    # qformer/language_projection can end up on a different device than
    # vision_model and language_model. Force-align to one device for
    # end-to-end generate() to work.
    if torch.cuda.is_available():
        target_device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        target_device = torch.device('mps')
    else:
        target_device = torch.device('cpu')
    blip_model = blip_model.to(target_device)

    # Generation path (BLIP-2's OPT decoder does instruction-style completion).
    generation_fn = _make_generation_fn(blip_model, blip_processor)

    return BrainScoreModel(
        identifier=identifier,
        model=blip_model,
        region_layer_map=REGION_LAYER_MAP,
        preprocessors={
            'vision': preprocessing,
            'text': text_wrapper,
        },
        activations_model=activations_model,
        visual_degrees=8,
        generation_fn=generation_fn,
        # Readout path also available — PCA-1000 features from 'IT' layer.
        behavioral_readout_layer='encoder.layers.34',
    )

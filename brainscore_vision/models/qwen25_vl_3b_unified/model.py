"""
Qwen2.5-VL-3B registered as a BrainScoreModel -- a true VLM where the vision
encoder feeds into a causal language model.

Uses callable extractors (not processor classes) for both vision and text.
Vision extracts from model.visual.blocks.*, text extracts from
model.language_model.layers.*.

Architecture:
  model.visual.blocks.{0-31}          -- 32 vision transformer blocks
  model.language_model.layers.{0-35}  -- 36 decoder layers
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from typing import Any, Optional

from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly


REGION_LAYER_MAP = {
    # Vision regions -- relative to the full Qwen2_5_VLForConditionalGeneration
    # (the vision extractor still wraps it until per-batch forward_kwargs lands
    # in PytorchWrapper to handle Qwen's per-image image_grid_thw)
    'V1': 'model.visual.blocks.2',
    'V2': 'model.visual.blocks.6',
    'V4': 'model.visual.blocks.14',
    'IT': 'model.visual.blocks.28',
    # Language region -- relative to model.language_model (TextWrapper root)
    'language_system': 'layers.28',
}


def _get_layer(model, layer_name: str):
    """Navigate to a layer by dot-separated path."""
    module = model
    for part in layer_name.split('.'):
        module = getattr(module, part, None)
        if module is None:
            raise ValueError(f"Layer '{layer_name}' not found at part '{part}'")
    return module


def _make_qwen_vision_extractor(processor, visual_degrees=8):
    """Create a vision extraction callable for Qwen2.5-VL.

    Handles Qwen's specific preprocessing (image_grid_thw, Conv3d patches)
    and extracts layer activations from the vision transformer blocks.
    """
    def extract(model, stimuli, *, recording_layer=None, **kwargs):
        if not recording_layer:
            raise ValueError("recording_layer must be specified")

        if hasattr(stimuli, 'stimulus_paths'):
            stimulus_ids = list(stimuli['stimulus_id'].values)
            image_paths = [str(stimuli.get_stimulus(sid)) for sid in stimulus_ids]
        elif hasattr(stimuli, 'columns') and 'image_file_name' in stimuli.columns:
            stimulus_ids = list(stimuli['stimulus_id'].values)
            image_paths = list(stimuli['image_file_name'].values)
        else:
            raise ValueError("Cannot extract image paths from stimuli")

        device = next(model.parameters()).device
        vision_encoder = model.model.visual
        layer = _get_layer(model, recording_layer)
        model.eval()

        all_activations = []

        for img_path in image_paths:
            image = Image.open(img_path).convert('RGB')

            messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
            text = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors='pt',
                               padding=True)

            pixel_values = inputs['pixel_values'].to(device)
            grid_thw = inputs['image_grid_thw'].to(device)

            activations = {}
            def hook_fn(_module, _input, output, name=recording_layer):
                if isinstance(output, (tuple, list)):
                    output = output[0]
                activations[name] = output.detach().cpu().float().numpy()

            hook = layer.register_forward_hook(hook_fn)

            with torch.no_grad():
                vision_encoder(pixel_values, grid_thw=grid_thw)

            hook.remove()

            act = activations[recording_layer]
            if act.ndim == 3:
                act = act.mean(axis=1)
            elif act.ndim == 2:
                act = act.mean(axis=0, keepdims=True)
            all_activations.append(act)

        raw = np.concatenate(all_activations, axis=0)
        n_stimuli, n_features = raw.shape
        neuroid_ids = [f'{recording_layer}.{i}' for i in range(n_features)]

        presentation_idx = pd.MultiIndex.from_arrays(
            [stimulus_ids], names=['stimulus_id'])
        return NeuroidAssembly(
            raw,
            coords={
                'presentation': presentation_idx,
                'neuroid_id': ('neuroid', neuroid_ids),
                'layer': ('neuroid', [recording_layer] * n_features),
            },
            dims=['presentation', 'neuroid'],
        )

    return extract


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'qwen2.5-vl-3b'

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from brainscore.model_helpers.text_wrapper import TextWrapper

    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        'Qwen/Qwen2.5-VL-3B-Instruct',
        torch_dtype=torch.float16,
    )
    qwen_processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')

    vision_extractor = _make_qwen_vision_extractor(
        processor=qwen_processor,
        visual_degrees=8,
    )

    # Text: TextWrapper wraps model.language_model (causal decoder).
    # last_token aggregation with attention_mask handles padded batches.
    text_wrapper = TextWrapper(
        model=qwen_model.model.language_model,
        tokenizer=qwen_processor.tokenizer,
        identifier=f'{identifier}-text',
        layer_aggregation='last_token',
        max_length=512,
    )

    return BrainScoreModel(
        identifier=identifier,
        model=qwen_model,
        region_layer_map=REGION_LAYER_MAP,
        preprocessors={
            'vision': vision_extractor,
            'text': text_wrapper,
        },
        visual_degrees=8,
    )

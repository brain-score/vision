"""
CLIP ViT-B/32 registered as a BrainScoreModel -- the first VLM in Brain-Score.

Uses PytorchWrapper (shared activations_model) wrapping clip_model.vision_model
for vision extraction, and a text extractor callable for language extraction.
Both share the same underlying CLIPModel.

Note: PytorchWrapper wraps vision_model (not the full CLIPModel) because
CLIPModel.forward() unconditionally invokes the text encoder, which fails
without input_ids. Vision region_layer_map paths are therefore relative to
vision_model. Text paths are relative to the full clip_model. When TextWrapper
is built to wrap the full model, all paths will unify.
"""

import functools

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper


REGION_LAYER_MAP = {
    # Vision regions -- relative to vision_model (PytorchWrapper root)
    'V1': 'encoder.layers.1',
    'V2': 'encoder.layers.3',
    'V4': 'encoder.layers.6',
    'IT': 'encoder.layers.10',
    # Language region -- relative to full CLIPModel (text extractor root)
    'language_system': 'text_model.encoder.layers.10',
}


def _load_preprocess_images(image_filepaths, processor, image_size=224):
    """Load and preprocess images using CLIPProcessor."""
    images = []
    for path in image_filepaths:
        with Image.open(path) as img:
            if img.mode not in ('RGB',):
                img = img.convert('RGB')
            images.append(img.copy())

    processed = processor(images=images, return_tensors='pt')
    return processed['pixel_values'].numpy()


def _make_encoder_text_extractor(tokenizer, text_model_attr='text_model',
                                 max_length=77):
    """Create a text extraction callable for CLIP's encoder.

    Returns a callable(model, stimuli, *, recording_layer) -> NeuroidAssembly.
    Temporary until TextWrapper provides symmetric extraction with PytorchWrapper.
    """
    def extract(model, stimuli, *, recording_layer=None, **kwargs):
        if hasattr(stimuli, 'columns'):
            if 'sentence' in stimuli.columns:
                texts = list(stimuli['sentence'].values)
            elif 'text' in stimuli.columns:
                texts = list(stimuli['text'].values)
            else:
                raise ValueError(
                    f"No text column in stimuli. Columns: {list(stimuli.columns)}")
            stimulus_ids = list(stimuli['stimulus_id'].values)
        elif isinstance(stimuli, (list, np.ndarray)):
            texts = list(stimuli)
            stimulus_ids = list(range(len(texts)))
        else:
            raise ValueError(f"Unsupported stimuli type: {type(stimuli)}")

        if not recording_layer:
            raise ValueError("recording_layer must be specified for neural recording")

        tokens = tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors='pt',
        )
        device = next(model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        # Navigate to target layer on the full model and hook it
        layer_module = model
        for part in recording_layer.split('.'):
            layer_module = getattr(layer_module, part, None)
            if layer_module is None:
                raise ValueError(
                    f"Layer '{recording_layer}' not found at part '{part}'")

        activations = {}
        def hook_fn(_module, _input, output, name=recording_layer):
            if isinstance(output, (tuple, list)):
                output = output[0]
            activations[name] = output.detach().cpu().numpy()

        hook = layer_module.register_forward_hook(hook_fn)

        # Forward through text encoder only
        text_model = model
        for attr in text_model_attr.split('.'):
            text_model = getattr(text_model, attr)

        model.eval()
        with torch.no_grad():
            text_model(**tokens)
        hook.remove()

        raw = activations[recording_layer]
        if raw.ndim == 3:
            raw = raw.mean(axis=1)

        n_stimuli, n_features = raw.shape
        neuroid_ids = [f'{recording_layer}.{i}' for i in range(n_features)]

        # Use simple range index for presentation — language benchmarks
        # need to set stimulus_id as a coordinate after construction.
        # Vision benchmarks use MultiIndex but language does not.
        return NeuroidAssembly(
            raw,
            coords={
                'stimulus_id': ('presentation', stimulus_ids),
                'neuroid_id': ('neuroid', neuroid_ids),
                'layer': ('neuroid', [recording_layer] * n_features),
            },
            dims=['presentation', 'neuroid'],
        )

    return extract


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'clip-vit-b-32'

    clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

    preprocessing = functools.partial(
        _load_preprocess_images,
        processor=clip_processor,
        image_size=224,
    )

    # activations_model wraps vision_model (not full CLIPModel, because
    # CLIPModel.forward() unconditionally calls text_model which fails
    # without input_ids). Vision layer paths in region_layer_map are
    # relative to vision_model.
    activations_model = PytorchWrapper(
        identifier=identifier,
        model=clip_model.vision_model,
        preprocessing=preprocessing,
    )
    activations_model.image_size = 224

    text_extractor = _make_encoder_text_extractor(
        tokenizer=clip_processor.tokenizer,
        text_model_attr='text_model',
        max_length=77,
    )

    return BrainScoreModel(
        identifier=identifier,
        model=clip_model,
        region_layer_map=REGION_LAYER_MAP,
        preprocessors={
            'vision': preprocessing,
            'text': text_extractor,
        },
        activations_model=activations_model,
        visual_degrees=8,
    )

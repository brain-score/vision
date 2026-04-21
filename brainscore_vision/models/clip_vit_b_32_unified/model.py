"""
CLIP ViT-B/32 registered as a BrainScoreModel -- the first VLM in Brain-Score.

Uses PytorchWrapper wrapping clip_model.vision_model for vision extraction,
and TextWrapper wrapping clip_model.text_model for text extraction. Both
share the same underlying CLIPModel.

PytorchWrapper wraps vision_model (not full CLIPModel) because CLIPModel.forward()
unconditionally invokes the text encoder, which fails without input_ids.
Similarly, TextWrapper wraps text_model directly. All region_layer_map paths
are therefore relative to the respective sub-module root.
"""

import functools

from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from brainscore.model_helpers.text_wrapper import TextWrapper
from brainscore_core.model_interface import BrainScoreModel
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper


REGION_LAYER_MAP = {
    # Vision regions -- relative to vision_model (PytorchWrapper root)
    'V1': 'encoder.layers.1',
    'V2': 'encoder.layers.3',
    'V4': 'encoder.layers.6',
    'IT': 'encoder.layers.10',
    # Language region -- relative to text_model (TextWrapper root)
    'language_system': 'encoder.layers.10',
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


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'clip-vit-b-32'

    clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

    preprocessing = functools.partial(
        _load_preprocess_images,
        processor=clip_processor,
        image_size=224,
    )

    # Vision: PytorchWrapper wraps vision_model sub-module.
    activations_model = PytorchWrapper(
        identifier=identifier,
        model=clip_model.vision_model,
        preprocessing=preprocessing,
    )
    activations_model.image_size = 224

    # Text: TextWrapper wraps text_model sub-module. CLIP's text encoder is
    # bidirectional (BERT-like), so mean_tokens aggregation is appropriate.
    text_wrapper = TextWrapper(
        model=clip_model.text_model,
        tokenizer=clip_processor.tokenizer,
        identifier=f'{identifier}-text',
        layer_aggregation='mean_tokens',
        max_length=77,
    )

    return BrainScoreModel(
        identifier=identifier,
        model=clip_model,
        region_layer_map=REGION_LAYER_MAP,
        preprocessors={
            'vision': preprocessing,
            'text': text_wrapper,
        },
        activations_model=activations_model,
        visual_degrees=8,
        # Behavioral readout uses the final vision encoder block (matches
        # the 'IT' layer convention). Behavioral benchmarks like Ferguson2024
        # will fit a logistic readout on top of this layer's features.
        behavioral_readout_layer='encoder.layers.10',
    )

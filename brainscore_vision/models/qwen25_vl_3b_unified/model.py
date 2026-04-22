"""
Qwen2.5-VL-3B registered as a BrainScoreModel -- a true VLM where the vision
encoder feeds into a causal language model.

Uses VLMVisionWrapper for the vision path (handles Qwen's flattened-patch
layout with image_grid_thw metadata) and TextWrapper for the text path
(causal mode with last_token aggregation). Both are class-based, cached via
@store_xarray, and symmetric in their calling convention.

Architecture:
  model.visual.blocks.{0-31}          -- 32 vision transformer blocks
  model.language_model.layers.{0-35}  -- 36 decoder layers

Region layer paths are relative to each wrapper's sub-model root:
  vision paths    → relative to qwen_model.model.visual
  language paths  → relative to qwen_model.model.language_model
"""

import re
from typing import List

import torch
from PIL import Image

from brainscore_core.model_interface import BrainScoreModel


REGION_LAYER_MAP = {
    # Vision regions -- relative to model.visual (VLMVisionWrapper root)
    'V1': 'blocks.2',
    'V2': 'blocks.6',
    'V4': 'blocks.14',
    'IT': 'blocks.28',
    # Language region -- relative to model.language_model (TextWrapper root)
    'language_system': 'layers.28',
}


def _make_generation_fn(qwen_model, qwen_processor, max_new_tokens: int = 8):
    """Create a generation callable for instruction-following behavioral
    tasks. Used by BrainScoreModel when TaskContext carries an
    ``instruction`` (not just fitting_stimuli).

    Signature (matches BrainScoreModel._generate_predictions contract):
        fn(stimulus_row, instruction, label_set) -> predicted_label
    """
    device = next(qwen_model.parameters()).device

    def generate(stimulus_row, instruction: str, label_set: List[str]) -> str:
        image = Image.open(stimulus_row['image_file_name']).convert('RGB')
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text',  'text': instruction +
                 f' Answer with exactly one of: {", ".join(label_set)}.'},
            ],
        }]
        prompt = qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_processor(
            text=[prompt], images=[image], return_tensors='pt', padding=True,
        ).to(device)

        with torch.no_grad():
            generated = qwen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Strip the prompt tokens, decode the tail
        input_len = inputs['input_ids'].shape[1]
        response_ids = generated[0, input_len:]
        response = qwen_processor.tokenizer.decode(
            response_ids, skip_special_tokens=True).strip().lower()

        # Robust parsing: prefer exact match, then substring match, then
        # fall back to first label (BrainScoreModel emits a warning).
        for lbl in label_set:
            if response == lbl.lower():
                return lbl
        for lbl in label_set:
            if re.search(rf'\b{re.escape(lbl.lower())}\b', response):
                return lbl
        return response  # unparseable — caller will default to label_set[0]

    return generate


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'qwen2.5-vl-3b'

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from brainscore.model_helpers.text_wrapper import TextWrapper
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper

    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        'Qwen/Qwen2.5-VL-3B-Instruct',
        torch_dtype=torch.float16,
    )
    qwen_processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')

    # Vision: VLMVisionWrapper handles Qwen's flattened-patch layout.
    # image_grid_thw tells the wrapper how to segment patches back to images.
    vision_wrapper = VLMVisionWrapper(
        model=qwen_model.model.visual,
        processor=qwen_processor,
        identifier=f'{identifier}-vision',
        image_input_key='pixel_values',
        forward_kwargs_map={'grid_thw': 'image_grid_thw'},
        patch_count_fn=lambda out: [int(t * h * w) for t, h, w in out['image_grid_thw']],
        layer_aggregation='mean_patches',
        batch_size=4,
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

    # Qwen-VL is instruction-following, so expose its generate() as the
    # behavioral-task generation path. Benchmarks that pass a TaskContext
    # with an `instruction` and `label_set` (e.g., ROAR) will use this
    # path; others fall back to the logistic readout (not configured for
    # Qwen — it doesn't have a behavioral_readout_layer set here).
    generation_fn = _make_generation_fn(qwen_model, qwen_processor)

    return BrainScoreModel(
        identifier=identifier,
        model=qwen_model,
        region_layer_map=REGION_LAYER_MAP,
        preprocessors={
            'vision': vision_wrapper,
            'text': text_wrapper,
        },
        visual_degrees=8,
        generation_fn=generation_fn,
        # Also enable the readout path so Qwen can be compared on the same
        # behavioral benchmark both ways. Uses the same vision layer as 'IT'.
        behavioral_readout_layer='blocks.28',
    )

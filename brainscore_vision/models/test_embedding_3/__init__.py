"""
Test model for workflow testing.

This is a minimal, fast-running model designed for testing the plugin submission workflow.
It uses distilgpt2 from HuggingFace, similar to the gpt models.
"""
from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# Register the test model using distilgpt2 (smallest/fastest GPT model)
model_registry['test_embedding_3'] = lambda: HuggingfaceSubject(
    model_id='distilgpt2',
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'transformer.h.5'
    }
)
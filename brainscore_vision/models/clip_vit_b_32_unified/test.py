import numpy as np
import os
import tempfile

import pandas as pd
import pytest
from PIL import Image

from brainscore_core.model_interface import BrainScoreModel, UnifiedModel
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


@pytest.fixture(scope='module')
def model():
    from brainscore_vision.models.clip_vit_b_32_unified.model import get_model
    return get_model('clip-vit-b-32')


_img_counter = 0

def _make_image_stimuli(n=3):
    """Create a fresh image stimulus set with a unique identifier."""
    global _img_counter
    _img_counter += 1
    tmpdir = tempfile.mkdtemp()
    paths = {}
    for i in range(n):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(tmpdir, f'img{i}.png')
        img.save(img_path)
        paths[f'img{i}'] = img_path

    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': list(paths.keys()),
        'image_file_name': list(paths.values()),
    }))
    stimuli.identifier = f'test_images_{_img_counter}'
    stimuli.stimulus_paths = paths
    return stimuli, tmpdir


@pytest.fixture
def image_stimuli():
    stimuli, tmpdir = _make_image_stimuli()
    yield stimuli
    import shutil
    shutil.rmtree(tmpdir)


@pytest.fixture
def text_stimuli():
    stimuli = StimulusSet(pd.DataFrame({
        'sentence': ['the quick brown fox', 'a dog runs fast', 'neurons fire together'],
        'stimulus_id': ['s0', 's1', 's2'],
    }))
    stimuli.identifier = 'test_text'
    return stimuli


class TestModelIdentity:
    def test_is_unified_model(self, model):
        assert isinstance(model, UnifiedModel)

    def test_is_brainscore_model(self, model):
        assert isinstance(model, BrainScoreModel)

    def test_identifier(self, model):
        assert model.identifier == 'clip-vit-b-32'

    def test_supported_modalities(self, model):
        assert model.supported_modalities == {'vision', 'text'}

    def test_region_layer_map(self, model):
        rlm = model.region_layer_map
        assert 'V1' in rlm
        assert 'IT' in rlm
        assert 'language_system' in rlm

    def test_visual_degrees(self, model):
        assert model.visual_degrees() == 8


class TestVisionPath:
    def test_look_at_returns_assembly(self, model, image_stimuli):
        model.start_recording('IT', time_bins=[(70, 170)])
        result = model.look_at(image_stimuli)
        assert result.dims == ('presentation', 'neuroid')
        assert result.shape[0] == 3

    def test_different_layers(self, model):
        # Use separate stimulus sets to avoid caching conflicts
        tmpdir = tempfile.mkdtemp()
        try:
            paths = {}
            for i in range(2):
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                img_path = os.path.join(tmpdir, f'layer_test_{i}.png')
                img.save(img_path)
                paths[f'lt{i}'] = img_path

            stim_v1 = StimulusSet(pd.DataFrame({
                'stimulus_id': list(paths.keys()),
                'image_file_name': list(paths.values()),
            }))
            stim_v1.identifier = 'layer_test_v1'
            stim_v1.stimulus_paths = dict(paths)

            stim_it = StimulusSet(pd.DataFrame({
                'stimulus_id': list(paths.keys()),
                'image_file_name': list(paths.values()),
            }))
            stim_it.identifier = 'layer_test_it'
            stim_it.stimulus_paths = dict(paths)

            model.start_recording('V1', time_bins=[(70, 170)])
            v1 = model.look_at(stim_v1)

            model.start_recording('IT', time_bins=[(70, 170)])
            it = model.look_at(stim_it)

            assert v1.shape[0] == it.shape[0] == 2
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_process_detects_vision(self, model):
        stimuli, tmpdir = _make_image_stimuli(2)
        try:
            model.start_recording('IT', time_bins=[(70, 170)])
            result = model.process(stimuli)
            assert result.dims == ('presentation', 'neuroid')
            assert result.shape[0] == 2
        finally:
            import shutil
            shutil.rmtree(tmpdir)


class TestTextPath:
    def test_process_detects_text(self, model, text_stimuli):
        model.start_recording('language_system', time_bins=[(70, 170)])
        result = model.process(text_stimuli)
        assert result.dims == ('presentation', 'neuroid')
        assert result.shape[0] == 3

    def test_digest_text_compat(self, model, text_stimuli):
        model.start_recording('language_system', time_bins=[(70, 170)])
        result = model.digest_text(list(text_stimuli['sentence'].values))
        assert 'neural' in result
        assert result['neural'].shape[0] == 3


class TestModalities:
    def test_vision_and_text_different_features(self, model, text_stimuli):
        stimuli, tmpdir = _make_image_stimuli(3)
        try:
            model.start_recording('IT', time_bins=[(70, 170)])
            vision_result = model.process(stimuli)

            model.start_recording('language_system', time_bins=[(70, 170)])
            text_result = model.process(text_stimuli)

            # Vision and text produce different feature dimensions
            assert vision_result.shape[1] != text_result.shape[1]
        finally:
            import shutil
            shutil.rmtree(tmpdir)

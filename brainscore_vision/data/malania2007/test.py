import numpy as np
import pytest

from brainscore_vision import load_stimulus_set, load_dataset
from brainscore_vision.benchmarks.malania2007.benchmark import DATASETS


@pytest.mark.private_access
class TestAssemblies:
    # test the number of subjects:
    @pytest.mark.parametrize('identifier, num_subjects', [
        ('short2', 6),
        ('short4', 5),
        ('short6', 5),
        ('short8', 5),
        ('short16', 6),
        ('equal2', 5),
        ('long2', 5),
        ('equal16', 5),
        ('long16', 5),
        ('vernier_only', 6)
    ])
    def test_num_subjects(self, identifier, num_subjects):
        assembly = load_dataset(f"Malania2007.{identifier}")
        assembly = assembly.dropna(dim='subject')
        assert len(np.unique(assembly['subject'].values)) == num_subjects

    # test assembly coords present in ALL 17 sets:
    @pytest.mark.parametrize('identifier', [
        'short2',
        'short4',
        'short6',
        'short8',
        'short16',
        'equal2',
        'long2',
        'equal16',
        'long16',
        'vernier_only',
    ])
    @pytest.mark.parametrize('field', [
        'subject'
    ])
    def test_fields_present(self, identifier, field):
        assembly = load_dataset(f"Malania2007.{identifier}")
        assert hasattr(assembly, field)


@pytest.mark.private_access
class TestStimulusSets:
    # test stimulus_set data:
    @pytest.mark.parametrize('identifier', [
        'short2',
        'short4',
        'short6',
        'short8',
        'short16',
        'equal2',
        'long2',
        'equal16',
        'long16',
        'short2_fit',
        'short4_fit',
        'short6_fit',
        'short8_fit',
        'short16_fit',
        'equal2_fit',
        'long2_fit',
        'equal16_fit',
        'long16_fit',
        'vernier_only'
    ])
    def test_stimulus_set_exist(self, identifier):
        full_name = f"Malania2007.{identifier}"
        stimulus_set = load_stimulus_set(full_name)
        assert stimulus_set is not None
        stripped_actual_identifier = stimulus_set.identifier.replace('.', '').replace('_', '').replace('-', '')
        stripped_expected_identifier = full_name.replace('.', '').replace('_', '').replace('-', '')
        assert stripped_actual_identifier == stripped_expected_identifier

    @pytest.mark.parametrize('identifier, num_images', [
        ('short2', 50),
        ('short4', 50),
        ('short6', 50),
        ('short8', 50),
        ('short16', 50),
        ('equal2', 50),
        ('long2', 50),
        ('equal16', 50),
        ('long16', 50),
        ('short2_fit', 500),
        ('short4_fit', 500),
        ('short6_fit', 500),
        ('short8_fit', 500),
        ('short16_fit', 500),
        ('equal2_fit', 500),
        ('long2_fit', 500),
        ('equal16_fit', 500),
        ('long16_fit', 500),
        ('vernier_only', 50)
    ])
    def test_number_of_images(self, identifier, num_images):
        stimulus_set = load_stimulus_set(f"Malania2007.{identifier}")
        assert len(np.unique(stimulus_set['stimulus_id'].values)) == num_images

    # tests stimulus_set coords for the 14 "normal" sets:
    @pytest.mark.parametrize('identifier', [
        'short2',
        'short4',
        'short6',
        'short8',
        'short16',
        'equal2',
        'long2',
        'equal16',
        'long16',
        'short2_fit',
        'short4_fit',
        'short6_fit',
        'short8_fit',
        'short16_fit',
        'equal2_fit',
        'long2_fit',
        'equal16_fit',
        'long16_fit',
        'vernier_only'
    ])
    @pytest.mark.parametrize('field', [
        'image_size_x_pix',
        'image_size_y_pix',
        'image_size_c',
        'image_size_degrees',
        'vernier_height_arcsec',
        'vernier_offset_arcsec',
        'image_label',
        'flanker_height_arcsec',
        'flanker_spacing_arcsec',
        'line_width_arcsec',
        'flanker_distance_arcsec',
        'num_flankers',
        'vernier_position_x_pix',
        'vernier_position_y_pix',
        'stimulus_id',
    ])
    def test_fields_present(self, identifier, field):
        stimulus_set = load_stimulus_set(f"Malania2007.{identifier}")
        assert hasattr(stimulus_set, field)

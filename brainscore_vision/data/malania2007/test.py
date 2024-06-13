import numpy as np
import pytest

from brainscore_vision import load_stimulus_set, load_dataset
from brainscore_vision.benchmarks.malania2007.benchmark import DATASETS


@pytest.mark.private_access
class TestAssemblies:
    # test the number of subjects:
    @pytest.mark.parametrize('identifier, num_subjects', [
        ('short-2', 6),
        ('short-4', 5),
        ('short-6', 5),
        ('short-8', 5),
        ('short-16', 6),
        ('equal-2', 5),
        ('long-2', 5),
        ('equal-16', 5),
        ('long-16', 5),
        ('vernier-only', 6)
    ])
    def test_num_subjects(self, identifier, num_subjects):
        assembly = load_dataset(f"Malania2007_{identifier}")
        assembly = assembly.where(~np.isnan(assembly.values), drop=True)
        assert len(np.unique(assembly['subject'].values)) == num_subjects

    # test assembly coords present in ALL 17 sets:
    @pytest.mark.parametrize('identifier', [
        'short-2',
        'short-4',
        'short-6',
        'short-8',
        'short-16',
        'equal-2',
        'long-2',
        'equal-16',
        'long-16',
        'vernier-only',
    ])
    @pytest.mark.parametrize('field', [
        'subject'
    ])
    def test_fields_present(self, identifier, field):
        assembly = load_dataset(f"Malania2007_{identifier}")
        assert hasattr(assembly, field)


@pytest.mark.private_access
class TestStimulusSets:
    # test stimulus_set data:
    @pytest.mark.parametrize('identifier', [
        'short-2',
        'short-4',
        'short-6',
        'short-8',
        'short-16',
        'equal-2',
        'long-2',
        'equal-16',
        'long-16',
        'short-2_fit',
        'short-4_fit',
        'short-6_fit',
        'short-8_fit',
        'short-16_fit',
        'equal-2_fit',
        'long-2_fit',
        'equal-16_fit',
        'long-16_fit',
        'vernier-only'
    ])
    def test_stimulus_set_exist(self, identifier):
        full_name = f"Malania2007_{identifier}"
        stimulus_set = load_stimulus_set(full_name)
        assert stimulus_set is not None
        assert stimulus_set.identifier == full_name

    @pytest.mark.parametrize('identifier, num_images', [
        ('short-2', 50),
        ('short-4', 50),
        ('short-6', 50),
        ('short-8', 50),
        ('short-16', 50),
        ('equal-2', 50),
        ('long-2', 50),
        ('equal-16', 50),
        ('long-16', 50),
        ('short-2_fit', 50),
        ('short-4_fit', 50),
        ('short-6_fit', 50),
        ('short-8_fit', 50),
        ('short-16_fit', 50),
        ('equal-2_fit', 50),
        ('long-2_fit', 50),
        ('equal-16_fit', 50),
        ('long-16_fit', 50),
        ('vernier-only', 50)
    ])
    def test_number_of_images(self, identifier, num_images):
        stimulus_set = load_stimulus_set(f"Malania2007_{identifier}")
        assert len(np.unique(stimulus_set['stimulus_id'].values)) == num_images

    # tests stimulus_set coords for the 14 "normal" sets:
    @pytest.mark.parametrize('identifier', [
        'short-2',
        'short-4',
        'short-6',
        'short-8',
        'short-16',
        'equal-2',
        'long-2',
        'equal-16',
        'long-16',
        'short-2_fit',
        'short-4_fit',
        'short-6_fit',
        'short-8_fit',
        'short-16_fit',
        'equal-2_fit',
        'long-2_fit',
        'equal-16_fit',
        'long-16_fit',
        'vernier-only'
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
        stimulus_set = load_stimulus_set(f"Malania2007_{identifier}")
        assert hasattr(stimulus_set, field)

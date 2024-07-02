import pytest
from brainscore_vision import load_dataset, load_stimulus_set
import numpy as np


@pytest.mark.private_access
class TestBaker2022Stimuli:

    # general tests
    @pytest.mark.parametrize('identifier', [
        'normal',
        'inverted'
    ])
    def test_stimulus_set_exist(self, identifier):
        full_name = f'Baker2022_{identifier}_distortion'
        stimulus_set = load_stimulus_set(full_name)
        assert stimulus_set is not None
        assert stimulus_set.identifier == full_name

    # tests number of images
    @pytest.mark.parametrize('identifier, num_images', [
        ('normal', 716),
        ('inverted', 360),
    ])
    def test_num_stimuli(self, identifier, num_images):
        stimulus_set = load_stimulus_set(f'Baker2022_{identifier}_distortion')
        assert len(stimulus_set) == num_images
        assert len(np.unique(stimulus_set["stimulus_id"])) == num_images

    # tests stimulus_set coords. Ensure normal/inverted only have their respective stimuli
    @pytest.mark.parametrize('field', [
        'stimulus_id',
        'animal',
        'image_type',
        'image_number',
        "orientation",
    ])
    @pytest.mark.parametrize('identifier', [
        'normal',
        'inverted',
    ])
    def test_fields_present(self, identifier, field):
        stimulus_set = load_stimulus_set(f'Baker2022_{identifier}_distortion')
        assert hasattr(stimulus_set, field)

    # make sure there are at least whole, frankenstein in each stimulus_set. Inverted does not have fragmented stimuli.
    @pytest.mark.parametrize('identifier, count', [
        ('normal', 3),
        ('inverted', 2),
    ])
    def test_distortion_counts(self, identifier, count):
        stimulus_set = load_stimulus_set(f'Baker2022_{identifier}_distortion')
        assert len(np.unique(stimulus_set["image_type"])) == count
        assert "w" in set(stimulus_set["image_type"])
        assert "f" in set(stimulus_set["image_type"])

    # make sure there are 9 possible animals in each stimulus_set -> 9 way AFC
    @pytest.mark.parametrize('identifier', [
        'normal',
        'inverted',
    ])
    def test_ground_truth_types(self, identifier):
        stimulus_set = load_stimulus_set(f'Baker2022_{identifier}_distortion')
        assert len(np.unique(stimulus_set["animal"])) == 9

    # make sure there are 40 unique image numbers
    @pytest.mark.parametrize('identifier', [
        'normal',
        'inverted',
    ])
    def test_image_types(self, identifier):
        stimulus_set = load_stimulus_set(f'Baker2022_{identifier}_distortion')
        assert len(np.unique(stimulus_set["image_number"])) == 40



@pytest.mark.private_access
class TestBaker2022Assemblies:

    # tests alignments that are the same across normal and inverted assemblies
    @pytest.mark.parametrize('identifier, length', [
        ('normal', 3702),
        ('inverted', 4320),
    ])
    def test_stimulus_set_assembly_alignment(self, identifier, length):
        full_name = f'Baker2022_{identifier}_distortion'
        assembly = load_dataset(full_name)
        assert assembly.stimulus_set is not None
        assert assembly.stimulus_set.identifier == f'Baker2022_{identifier}_distortion'
        assert set(assembly.stimulus_set["animal"]) == set(assembly["truth"].values)
        assert set(assembly.stimulus_set["stimulus_id"]) == set(assembly["stimulus_id"].values)
        assert len(assembly.presentation) == length

    # tests counts that are the same across normal and inverted assemblies
    @pytest.mark.parametrize('identifier', [
        'normal',
        'inverted',
    ])
    def test_same_counts(self, identifier):
        full_name = f'Baker2022_{identifier}_distortion'
        assembly = load_dataset(full_name)
        assert len(set((assembly["truth"]).values)) == 9
        assert set((assembly["correct"]).values) == {0, 1}

    # tests number of subjects
    @pytest.mark.parametrize('identifier, num_subjects', [
        ('normal', 32),
        ('inverted', 12),
    ])
    def test_subjects(self, identifier, num_subjects):
        full_name = f'Baker2022_{identifier}_distortion'
        assembly = load_dataset(full_name)
        assert len(set((assembly["subject"]).values)) == num_subjects

    # tests number of configurations in {whole, fragmented, frankenstein} Inverted does not have fragmented)
    @pytest.mark.parametrize('identifier, conditions', [
        ('normal', {'whole', 'fragmented', 'Frankenstein'}),
        ('inverted', {'whole', 'Frankenstein'}),
    ])
    def test_conditions(self, identifier, conditions):
        full_name = f'Baker2022_{identifier}_distortion'
        assembly = load_dataset(full_name)
        assert set((assembly["condition"]).values) == conditions

    # tests number of unique images
    @pytest.mark.parametrize('identifier, num_images', [
        ('normal', 716),
        ('inverted', 360),
    ])
    def test_subjects(self, identifier, num_images):
        full_name = f'Baker2022_{identifier}_distortion'
        assembly = load_dataset(full_name)
        assert len(set((assembly["stimulus_id"]).values)) == num_images



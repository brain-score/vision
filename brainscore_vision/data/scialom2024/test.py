import numpy as np
import pytest

from brainscore_vision import load_stimulus_set, load_dataset
from brainscore_vision.benchmarks.scialom2024 import DATASETS


def test_count():
    assert len(DATASETS) == 11 + 11 + 2  # phosphenes + segments + composites


@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('Scialom2024_rgb', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_contours', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_phosphenes-12', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_phosphenes-16', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_phosphenes-21', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_phosphenes-27', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_phosphenes-35', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_phosphenes-46', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_phosphenes-59', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_phosphenes-77', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_phosphenes-100', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_segments-12', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_segments-16', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_segments-21', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_segments-27', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_segments-35', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_segments-46', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_segments-59', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_segments-77', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_segments-100', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_phosphenes-composite', marks=[pytest.mark.private_access]),
    pytest.param('Scialom2024_segments-composite', marks=[pytest.mark.private_access])
])
def test_existence(assembly_identifier):
    assert load_dataset(assembly_identifier) is not None


class TestAssemblies:
    @pytest.mark.parametrize('identifier', [
        'Scialom2024_rgb',
        'Scialom2024_contours',
        'Scialom2024_phosphenes-12',
        'Scialom2024_phosphenes-16',
        'Scialom2024_phosphenes-21',
        'Scialom2024_phosphenes-27',
        'Scialom2024_phosphenes-35',
        'Scialom2024_phosphenes-46',
        'Scialom2024_phosphenes-59',
        'Scialom2024_phosphenes-77',
        'Scialom2024_phosphenes-100',
        'Scialom2024_segments-12',
        'Scialom2024_segments-16',
        'Scialom2024_segments-21',
        'Scialom2024_segments-27',
        'Scialom2024_segments-35',
        'Scialom2024_segments-46',
        'Scialom2024_segments-59',
        'Scialom2024_segments-77',
        'Scialom2024_segments-100',
        'Scialom2024_phosphenes-composite',
        'Scialom2024_segments-composite',
    ])
    @pytest.mark.parametrize('field', [
        'stimulus_id',
        'condition',
        'truth'
    ])
    def test_stimulus_set_assembly_alignment(self, identifier, field):
        assembly = load_dataset(identifier)
        assert assembly.stimulus_set is not None
        assert assembly.stimulus_set.identifier == identifier
        assert set(assembly.stimulus_set[field]) == set(assembly[field].values)

    # test the number of subjects
    @pytest.mark.parametrize('identifier, num_subjects', [
        ('Scialom2024_rgb', 50),
        ('Scialom2024_contours', 50),
        ('Scialom2024_phosphenes-12', 25),
        ('Scialom2024_phosphenes-16', 25),
        ('Scialom2024_phosphenes-21', 25),
        ('Scialom2024_phosphenes-27', 25),
        ('Scialom2024_phosphenes-35', 25),
        ('Scialom2024_phosphenes-46', 25),
        ('Scialom2024_phosphenes-59', 25),
        ('Scialom2024_phosphenes-77', 25),
        ('Scialom2024_phosphenes-100', 25),
        ('Scialom2024_segments-12', 25),
        ('Scialom2024_segments-16', 25),
        ('Scialom2024_segments-21', 25),
        ('Scialom2024_segments-27', 25),
        ('Scialom2024_segments-35', 25),
        ('Scialom2024_segments-46', 25),
        ('Scialom2024_segments-59', 25),
        ('Scialom2024_segments-77', 25),
        ('Scialom2024_segments-100', 25),
        ('Scialom2024_phosphenes-composite', 25),
        ('Scialom2024_segments-composite', 25),
    ])
    def test_num_subjects(self, identifier, num_subjects):
        assembly = load_dataset(identifier)
        assert len(np.unique(assembly['subject'].values)) == num_subjects

    # test number of unique images
    @pytest.mark.parametrize('identifier, num_unique_images', [
        ('Scialom2024_rgb', 48),
        ('Scialom2024_contours', 48),
        ('Scialom2024_phosphenes-12', 48),
        ('Scialom2024_phosphenes-16', 48),
        ('Scialom2024_phosphenes-21', 48),
        ('Scialom2024_phosphenes-27', 48),
        ('Scialom2024_phosphenes-35', 48),
        ('Scialom2024_phosphenes-46', 48),
        ('Scialom2024_phosphenes-59', 48),
        ('Scialom2024_phosphenes-77', 48),
        ('Scialom2024_phosphenes-100', 48),
        ('Scialom2024_segments-12', 48),
        ('Scialom2024_segments-16', 48),
        ('Scialom2024_segments-21', 48),
        ('Scialom2024_segments-27', 48),
        ('Scialom2024_segments-35', 48),
        ('Scialom2024_segments-46', 48),
        ('Scialom2024_segments-59', 48),
        ('Scialom2024_segments-77', 48),
        ('Scialom2024_segments-100', 48),
        ('Scialom2024_phosphenes-composite', 528),
        ('Scialom2024_segments-composite', 528),
    ])
    def test_num_unique_images(self, identifier, num_unique_images):
        assembly = load_dataset(identifier)
        assert len(np.unique(assembly['image_id'].values)) == num_unique_images

    # tests assembly dim for ALL datasets
    @pytest.mark.parametrize('identifier, length', [
        ('Scialom2024_rgb', 2400),
        ('Scialom2024_contours', 2400),
        ('Scialom2024_phosphenes-12', 1200),
        ('Scialom2024_phosphenes-16', 1200),
        ('Scialom2024_phosphenes-21', 1200),
        ('Scialom2024_phosphenes-27', 1200),
        ('Scialom2024_phosphenes-35', 1200),
        ('Scialom2024_phosphenes-46', 1200),
        ('Scialom2024_phosphenes-59', 1200),
        ('Scialom2024_phosphenes-77', 1200),
        ('Scialom2024_phosphenes-100', 1200),
        ('Scialom2024_segments-12', 1200),
        ('Scialom2024_segments-16', 1200),
        ('Scialom2024_segments-21', 1200),
        ('Scialom2024_segments-27', 1200),
        ('Scialom2024_segments-35', 1200),
        ('Scialom2024_segments-46', 1200),
        ('Scialom2024_segments-59', 1200),
        ('Scialom2024_segments-77', 1200),
        ('Scialom2024_segments-100', 1200),
        ('Scialom2024_phosphenes-composite', 13200),
        ('Scialom2024_segments-composite', 13200),
    ])
    def test_length(self, identifier, length):
        assembly = load_dataset(identifier)
        assert len(assembly['presentation']) == length

    # test assembly coords present in ALL 17 sets:
    @pytest.mark.parametrize('identifier', [
        'Scialom2024_rgb',
        'Scialom2024_contours',
        'Scialom2024_phosphenes-12',
        'Scialom2024_phosphenes-16',
        'Scialom2024_phosphenes-21',
        'Scialom2024_phosphenes-27',
        'Scialom2024_phosphenes-35',
        'Scialom2024_phosphenes-46',
        'Scialom2024_phosphenes-59',
        'Scialom2024_phosphenes-77',
        'Scialom2024_phosphenes-100',
        'Scialom2024_segments-12',
        'Scialom2024_segments-16',
        'Scialom2024_segments-21',
        'Scialom2024_segments-27',
        'Scialom2024_segments-35',
        'Scialom2024_segments-46',
        'Scialom2024_segments-59',
        'Scialom2024_segments-77',
        'Scialom2024_segments-100',
        'Scialom2024_phosphenes-composite',
        'Scialom2024_segments-composite',
    ])
    @pytest.mark.parametrize('field', [
        'subject',
        'subject_group',
        'visual_degrees',
        'image_duration',
        'is_correct',
        'subject_answer',
        'condition',
        'percentage_elements',
        'stimulus_id',
        'truth'
    ])
    def test_fields_present(self, identifier, field):
        assembly = load_dataset(identifier)
        assert hasattr(assembly, field)


# testing stimulus sets
@pytest.mark.slow
class TestStimulusSets:
    # test stimulus_set data:
    @pytest.mark.parametrize('identifier', [
        'Scialom2024_rgb',
        'Scialom2024_contours',
        'Scialom2024_phosphenes-12',
        'Scialom2024_phosphenes-16',
        'Scialom2024_phosphenes-21',
        'Scialom2024_phosphenes-27',
        'Scialom2024_phosphenes-35',
        'Scialom2024_phosphenes-46',
        'Scialom2024_phosphenes-59',
        'Scialom2024_phosphenes-77',
        'Scialom2024_phosphenes-100',
        'Scialom2024_segments-12',
        'Scialom2024_segments-16',
        'Scialom2024_segments-21',
        'Scialom2024_segments-27',
        'Scialom2024_segments-35',
        'Scialom2024_segments-46',
        'Scialom2024_segments-59',
        'Scialom2024_segments-77',
        'Scialom2024_segments-100',
        'Scialom2024_phosphenes-composite',
        'Scialom2024_segments-composite',
    ])
    def test_stimulus_set_exists(self, identifier):
        stimulus_set = load_stimulus_set(identifier)
        assert stimulus_set is not None
        assert stimulus_set.identifier == identifier

    # test number of images
    @pytest.mark.parametrize('identifier, num_images', [
        ('Scialom2024_rgb', 48),
        ('Scialom2024_contours', 48),
        ('Scialom2024_phosphenes-12', 48),
        ('Scialom2024_phosphenes-16', 48),
        ('Scialom2024_phosphenes-21', 48),
        ('Scialom2024_phosphenes-27', 48),
        ('Scialom2024_phosphenes-35', 48),
        ('Scialom2024_phosphenes-46', 48),
        ('Scialom2024_phosphenes-59', 48),
        ('Scialom2024_phosphenes-77', 48),
        ('Scialom2024_phosphenes-100', 48),
        ('Scialom2024_segments-12', 48),
        ('Scialom2024_segments-16', 48),
        ('Scialom2024_segments-21', 48),
        ('Scialom2024_segments-27', 48),
        ('Scialom2024_segments-35', 48),
        ('Scialom2024_segments-46', 48),
        ('Scialom2024_segments-59', 48),
        ('Scialom2024_segments-77', 48),
        ('Scialom2024_segments-100', 48),
        ('Scialom2024_phosphenes-composite', 528),
        ('Scialom2024_segments-composite', 528),
    ])
    def test_num_images(self, identifier, num_images):
        stimulus_set = load_stimulus_set(identifier)
        assert len(np.unique(stimulus_set['image_id'].values)) == num_images

    # test assembly coords present in ALL 17 sets:
    @pytest.mark.parametrize('identifier', [
        'Scialom2024_rgb',
        'Scialom2024_contours',
        'Scialom2024_phosphenes-12',
        'Scialom2024_phosphenes-16',
        'Scialom2024_phosphenes-21',
        'Scialom2024_phosphenes-27',
        'Scialom2024_phosphenes-35',
        'Scialom2024_phosphenes-46',
        'Scialom2024_phosphenes-59',
        'Scialom2024_phosphenes-77',
        'Scialom2024_phosphenes-100',
        'Scialom2024_segments-12',
        'Scialom2024_segments-16',
        'Scialom2024_segments-21',
        'Scialom2024_segments-27',
        'Scialom2024_segments-35',
        'Scialom2024_segments-46',
        'Scialom2024_segments-59',
        'Scialom2024_segments-77',
        'Scialom2024_segments-100',
        'Scialom2024_phosphenes-composite',
        'Scialom2024_segments-composite',
    ])
    @pytest.mark.parametrize('field', [
        'image_height',
        'image_width',
        'num_channels',
        'dataset',
        'object_id',
        'stimulus_id',
        'truth',
        'percentage_elements',
        'condition'
    ])
    def test_fields_present(self, identifier, field):
        stimulus_set = load_stimulus_set(identifier)
        assert hasattr(stimulus_set, field)

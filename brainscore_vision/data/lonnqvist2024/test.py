import numpy as np
import pytest

from brainscore_vision import load_stimulus_set, load_dataset


@pytest.mark.private_access
@pytest.mark.parametrize('assembly_identifier', [
    'Lonnqvist2024_inlab-instructions',
    'Lonnqvist2024_inlab-no-instructions',
    'Lonnqvist2024_online-no-instructions'
])
def test_existence(assembly_identifier):
    assert load_dataset(assembly_identifier) is not None


@pytest.mark.private_access
class TestAssemblies:
    @pytest.mark.parametrize('assembly', [
        'Lonnqvist2024_inlab-instructions',
        'Lonnqvist2024_inlab-no-instructions',
        'Lonnqvist2024_online-no-instructions'
    ])
    @pytest.mark.parametrize('identifier', [
        'Lonnqvist2024_test'
    ])
    @pytest.mark.parametrize('field', [
        'stimulus_id',
        'truth'
    ])
    def test_stimulus_set_assembly_alignment(self, assembly, identifier, field):
        assembly = load_dataset(assembly)
        assert assembly.stimulus_set is not None
        assert assembly.stimulus_set.identifier == identifier
        assert set(assembly.stimulus_set[field]) == set(assembly[field].values)

    # test the number of subjects
    @pytest.mark.parametrize('identifier, num_subjects', [
        ('Lonnqvist2024_inlab-instructions', 10),
        ('Lonnqvist2024_inlab-no-instructions', 10),
        ('Lonnqvist2024_online-no-instructions', 92),
    ])
    def test_num_subjects(self, identifier, num_subjects):
        assembly = load_dataset(identifier)
        assert len(np.unique(assembly['subject'].values)) == num_subjects

    # test number of unique images
    @pytest.mark.parametrize('identifier, num_unique_images', [
        ('Lonnqvist2024_inlab-instructions', 380),
        ('Lonnqvist2024_inlab-no-instructions', 380),
        ('Lonnqvist2024_online-no-instructions', 380),
    ])
    def test_num_unique_images(self, identifier, num_unique_images):
        assembly = load_dataset(identifier)
        assert len(np.unique(assembly['stimulus_id'].values)) == num_unique_images

    # tests assembly dim for ALL datasets
    @pytest.mark.parametrize('identifier, length', [
        ('Lonnqvist2024_inlab-instructions', 3800),
        ('Lonnqvist2024_inlab-no-instructions', 3800),
        ('Lonnqvist2024_online-no-instructions', 34960),
    ])
    def test_length(self, identifier, length):
        assembly = load_dataset(identifier)
        assert len(assembly['presentation']) == length

    # test assembly coords present in ALL 17 sets:
    @pytest.mark.parametrize('identifier', [
        'Lonnqvist2024_inlab-instructions',
        'Lonnqvist2024_inlab-no-instructions',
        'Lonnqvist2024_online-no-instructions'
    ])
    @pytest.mark.parametrize('field', [
        'subject',
        'visual_degrees',
        'image_duration',
        'is_correct',
        'subject_answer',
        'curve_length',
        'n_cross',
        'image_path',
        'stimulus_id',
        'truth',
        'image_label'
    ])
    def test_fields_present(self, identifier, field):
        assembly = load_dataset(identifier)
        assert hasattr(assembly, field)


@pytest.mark.private_access
@pytest.mark.slow
class TestStimulusSets:
    # test stimulus_set data:
    @pytest.mark.parametrize('identifier', [
        'Lonnqvist2024_train',
        'Lonnqvist2024_test',
    ])
    def test_stimulus_set_exists(self, identifier):
        stimulus_set = load_stimulus_set(identifier)
        assert stimulus_set is not None
        assert stimulus_set.identifier == identifier

    @pytest.mark.parametrize('identifier, num_images', [
        ('Lonnqvist2024_train', 185),
        ('Lonnqvist2024_test', 380),
    ])
    def test_number_of_images(self, identifier, num_images):
        stimulus_set = load_stimulus_set(identifier)
        assert len(np.unique(stimulus_set['stimulus_id'].values)) == num_images

    # test assembly coords present in ALL 17 sets:
    @pytest.mark.parametrize('identifier', [
        'Lonnqvist2024_train',
        'Lonnqvist2024_test',
    ])
    @pytest.mark.parametrize('field', [
        'curve_length',
        'n_cross',
        'image_path',
        'stimulus_id',
        'truth',
        'image_label'
    ])
    def test_fields_present(self, identifier, field):
        stimulus_set = load_stimulus_set(identifier)
        assert hasattr(stimulus_set, field)

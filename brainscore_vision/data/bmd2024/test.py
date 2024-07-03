import numpy as np
import pytest

from brainscore_vision import load_stimulus_set, load_dataset


@pytest.mark.private_access
@pytest.mark.parametrize('assembly_identifier', [
    'BMD_2024_texture_1',
    'BMD_2024_texture_2',
    'BMD_2024_dotted_1',
    'BMD_2024_dotted_2',
])
def test_existence(assembly_identifier):
    assert load_dataset(assembly_identifier) is not None
    

@pytest.mark.private_access
class TestAssemblies:
    @pytest.mark.parametrize('identifier', [
        'BMD_2024_texture_1',
        'BMD_2024_texture_2',
        'BMD_2024_dotted_1',
        'BMD_2024_dotted_2',
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
        ('BMD_2024_texture_1', 51),
        ('BMD_2024_texture_2', 52),
        ('BMD_2024_dotted_1', 54),
        ('BMD_2024_dotted_2', 54),
    ])
    def test_num_subjects(self, identifier, num_subjects):
        assembly = load_dataset(identifier)
        assert len(np.unique(assembly['subject'].values)) == num_subjects
        


    # test number of unique images
    @pytest.mark.parametrize('identifier, num_unique_images', [
        ('BMD_2024_texture_1', 100),
        ('BMD_2024_texture_2', 100),
        ('BMD_2024_dotted_1', 100),
        ('BMD_2024_dotted_2', 100),
    ])
    def test_num_unique_images(self, identifier, num_unique_images):
        assembly = load_dataset(identifier)
        assert len(np.unique(assembly['stimulus_id'].values)) == num_unique_images
        

    # tests assembly dim for ALL datasets
    @pytest.mark.parametrize('identifier, length', [
        ('BMD_2024_texture_1', 5100),
        ('BMD_2024_texture_2', 5200),
        ('BMD_2024_dotted_1', 5400),
        ('BMD_2024_dotted_2', 5400),
    ])
    def test_length(self, identifier, length):
        assembly = load_dataset(identifier)
        assert len(assembly['presentation']) == length

    # test assembly coords present in ALL 17 sets:
    @pytest.mark.parametrize('identifier', [
        'BMD_2024_texture_1',
        'BMD_2024_texture_2',
        'BMD_2024_dotted_1',
        'BMD_2024_dotted_2',
    ])
    @pytest.mark.parametrize('field', [
        'subject',
        'subject_answer',
        'condition',
        'stimulus_id',
        'truth'
    ])
    def test_fields_present(self, identifier, field):
        assembly = load_dataset(identifier)
        assert hasattr(assembly, field)


@pytest.mark.private_access
@pytest.mark.slow
class TestStimulusSets:
    # test stimulus_set data:
    @pytest.mark.parametrize('identifier', [
        'BMD_2024_texture_1',
        'BMD_2024_texture_2',
        'BMD_2024_dotted_1',
        'BMD_2024_dotted_2',
    ])
    def test_stimulus_set_exists(self, identifier):
        stimulus_set = load_stimulus_set(identifier)
        assert stimulus_set is not None
        assert stimulus_set.identifier == identifier

    @pytest.mark.parametrize('identifier, num_images', [
        ('BMD_2024_texture_1', 100),
        ('BMD_2024_texture_2', 100),
        ('BMD_2024_dotted_1', 100),
        ('BMD_2024_dotted_2', 100),
    ])
    def test_number_of_images(self, identifier, num_images):
        stimulus_set = load_stimulus_set(identifier)
        assert len(np.unique(stimulus_set['image_id'].values)) == num_images

    # test assembly coords present in ALL 17 sets:
    @pytest.mark.parametrize('identifier', [
        'BMD_2024_texture_1',
        'BMD_2024_texture_2',
        'BMD_2024_dotted_1',
        'BMD_2024_dotted_2',
    ])
    @pytest.mark.parametrize('field', [
        'stimulus_id',
        'truth',
        'condition'
    ])
    def test_fields_present(self, identifier, field):
        stimulus_set = load_stimulus_set(identifier)
        assert hasattr(stimulus_set, field)
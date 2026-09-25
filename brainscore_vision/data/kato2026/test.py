import numpy as np
import pytest
from brainscore_vision import load_stimulus_set, load_dataset

#def test_count():
#    assert len(DATASETS) == 12 + 5

@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('Kato2026.Ori', marks=[pytest.mark.private_access]),
    pytest.param('Kato2026.Fil', marks=[pytest.mark.private_access]),
])
def test_existence(assembly_identifier):
    assert load_dataset(assembly_identifier) is not None


class TestAssemblies:
    # test stimulus_set data alignment with assembly:
    @pytest.mark.parametrize('identifier', [
        'Ori',
        'Fil',
        'Lin',
        'SilTex',
        'Sil',
        'Tex',
        'LinFil',
        'Sparse',
        'Dense'
    ])
    @pytest.mark.parametrize('field', [
        'stimulus_id',
        'condition',
        'truth',
    ])
    def test_stimulus_set_assembly_alignment(self, identifier, field):
        full_name = f"Kato2026.{identifier}"
        assembly = load_dataset(full_name)
        assert assembly.stimulus_set is not None
        assert assembly.stimulus_set.identifier == full_name
        assert set(assembly.stimulus_set[field]) == set(assembly[field].values)

    # test the number of subjects:
    @pytest.mark.parametrize('identifier, num_subjects', [
        ('Ori', 60),
        ('Fil', 10),
        ('Lin', 10),
        ('SilTex', 10),
        ('Sil', 10),
        ('Tex', 10),
        ('LinFil', 10),
        ('Sparse', 10),
        ('Dense', 10),
    ])
    def test_num_subjects(self, identifier, num_subjects):
        assembly = load_dataset(f"Kato2026.{identifier}")
        assert len(np.unique(assembly['subject'].values)) == num_subjects

    # test the number of images
    @pytest.mark.parametrize('identifier, num_images', [
        ('Ori', 240),
        ('Fil', 240),
        ('Lin', 240),
        ('SilTex', 240),
        ('Sil', 240),
        ('Tex', 240),
        ('LinFil', 240),
        ('Sparse', 171),
        ('Dense', 171),
    ])
    def test_num_images(self, identifier, num_images):
        assembly = load_dataset(f"Kato2026.{identifier}")
        assert len(np.unique(assembly['stimulus_id'].values)) == num_images

    # tests assembly dim (#images x #subjects)
    @pytest.mark.parametrize('identifier, length', [
        ('Ori', 14400),
        ('Fil', 2400),
        ('Lin', 2400),
        ('SilTex', 2400),
        ('Sil', 2400),
        ('Tex', 2400),
        ('LinFil', 2400),
        ('Sparse', 1710),
        ('Dense', 1710),
    ])
    def test_length(self, identifier, length):
        assembly = load_dataset(f"Kato2026.{identifier}")
        assert len(assembly['presentation']) == length

    # test assembly coords present in ALL 17 sets:
    @pytest.mark.parametrize('identifier', [
        'Ori',
        'Fil',
        'Lin',
        'SilTex',
        'Sil',
        'Tex',
        'LinFil',
        'Sparse',
        'Dense'
    ])
    @pytest.mark.parametrize('field', [
        'stimulus_id',
        'subject',
        'correct',
        'condition',
        'response',
        'truth'
    ])
    def test_fields_present(self, identifier, field):
        assembly = load_dataset(f"Kato2026.{identifier}")
        assert hasattr(assembly, field)


# testing stimulus sets
@pytest.mark.slow
class TestStimulusSets:
    # test stimulus_set data:
    @pytest.mark.parametrize('identifier', [
        'Ori',
        'Fil',
        'Lin',
        'SilTex',
        'Sil',
        'Tex',
        'LinFil',
        'Sparse',
        'Dense'
    ])
    def test_stimulus_set_exist(self, identifier):
        full_name = f"Kato2026.{identifier}"
        stimulus_set = load_stimulus_set(full_name)
        assert stimulus_set is not None
        assert stimulus_set.identifier == f"{full_name}"

    # test the number of images
    @pytest.mark.parametrize('identifier, num_images', [
        ('Ori', 240),
        ('Fil', 240),
        ('Lin', 240),
        ('SilTex', 240),
        ('Sil', 240),
        ('Tex', 240),
        ('LinFil', 240),
        ('Sparse', 171),
        ('Dense', 171),
    ])
    def test_num_images(self, identifier, num_images):
        stimulus_set = load_stimulus_set(f"Kato2026.{identifier}")
        assert len(np.unique(stimulus_set['stimulus_id'].values)) == num_images

    @pytest.mark.parametrize('identifier', [
        'Ori',
        'Fil',
        'Lin',
        'SilTex',
        'Sil',
        'Tex',
        'LinFil',
        'Sparse',
        'Dense'
    ])
    @pytest.mark.parametrize('field', [
        'stimulus_id',
        'truth',
        'condition',
    ])
    def test_fields_present(self, identifier, field):
        stimulus_set = load_stimulus_set(f"Kato2026.{identifier}")
        assert hasattr(stimulus_set, field)


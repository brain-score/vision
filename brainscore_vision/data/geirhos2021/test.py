import numpy as np
import pytest

from brainscore_vision import load_stimulus_set, load_dataset
from brainscore_vision.benchmarks.geirhos2021.benchmark import DATASETS


def test_count():
    assert len(DATASETS) == 12 + 5


@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('brendel.Geirhos2021_colour', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_contrast', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_cue-conflict', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_edge', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_eidolonI', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_eidolonII', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_eidolonIII', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_false-colour', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_high-pass', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_low-pass', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_phase-scrambling', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_power-equalisation', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_rotation', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_silhouette', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_stylized', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_sketch', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_uniform-noise', marks=[pytest.mark.private_access]),
])
def test_existence(assembly_identifier):
    assert load_dataset(assembly_identifier) is not None


class TestAssemblies:
    # test stimulus_set data alignment with assembly:
    @pytest.mark.parametrize('identifier', [
        'colour',
        'contrast',
        'cue-conflict',
        'edge',
        'eidolonI',
        'eidolonII',
        'eidolonIII',
        'false-colour',
        'high-pass',
        'low-pass',
        'phase-scrambling',
        'power-equalisation',
        'rotation',
        'silhouette',
        'stylized',
        'sketch',
        'uniform-noise',
    ])
    @pytest.mark.parametrize('field', [
        'image_id',
        'condition',
        'truth',
    ])
    def test_stimulus_set_assembly_alignment(self, identifier, field):
        full_name = f"brendel.Geirhos2021_{identifier}"
        assembly = load_dataset(full_name)
        assert assembly.stimulus_set is not None
        assert assembly.stimulus_set.identifier == full_name
        assert set(assembly.stimulus_set[field]) == set(assembly[field].values)

    # test the number of subjects:
    @pytest.mark.parametrize('identifier, num_subjects', [
        ('colour', 4),
        ('contrast', 4),
        ('cue-conflict', 10),
        ('edge', 10),
        ('eidolonI', 4),
        ('eidolonII', 4),
        ('eidolonIII', 4),
        ('false-colour', 4),
        ('high-pass', 4),
        ('low-pass', 4),
        ('phase-scrambling', 4),
        ('power-equalisation', 4),
        ('rotation', 4),
        ('silhouette', 10),
        ('stylized', 5),
        ('sketch', 7),
        ('uniform-noise', 4),
    ])
    def test_num_subjects(self, identifier, num_subjects):
        assembly = load_dataset(f"brendel.Geirhos2021_{identifier}")
        assert len(np.unique(assembly['subject'].values)) == num_subjects

    # test the number of images
    @pytest.mark.parametrize('identifier, num_images', [
        ('colour', 1280),
        ('contrast', 1280),
        ('cue-conflict', 1280),
        ('edge', 160),
        ('eidolonI', 1280),
        ('eidolonII', 1280),
        ('eidolonIII', 1280),
        ('false-colour', 1120),
        ('high-pass', 1280),
        ('low-pass', 1280),
        ('phase-scrambling', 1120),
        ('power-equalisation', 1120),
        ('rotation', 1280),
        ('silhouette', 160),
        ('stylized', 800),
        ('sketch', 800),
        ('uniform-noise', 1280),
    ])
    def test_num_images(self, identifier, num_images):
        assembly = load_dataset(f"brendel.Geirhos2021_{identifier}")
        assert len(np.unique(assembly['image_id'].values)) == num_images

    # tests assembly dim for ALL 17 sets:
    @pytest.mark.parametrize('identifier, length', [
        ('colour', 5120),
        ('contrast', 5120),
        ('cue-conflict', 12800),
        ('edge', 1600),
        ('eidolonI', 5120),
        ('eidolonII', 5120),
        ('eidolonIII', 5120),
        ('false-colour', 4480),
        ('high-pass', 5120),
        ('low-pass', 5120),
        ('phase-scrambling', 4480),
        ('power-equalisation', 4480),
        ('rotation', 5120),
        ('silhouette', 1600),
        ('stylized', 4000),
        ('sketch', 5600),
        ('uniform-noise', 5120),
    ])
    def test_length(self, identifier, length):
        assembly = load_dataset(f"brendel.Geirhos2021_{identifier}")
        assert len(assembly['presentation']) == length

    # test assembly coords present in ALL 17 sets:
    @pytest.mark.parametrize('identifier', [
        'colour',
        'contrast',
        'cue-conflict',
        'edge',
        'eidolonI',
        'eidolonII',
        'eidolonIII',
        'false-colour',
        'high-pass',
        'low-pass',
        'phase-scrambling',
        'power-equalisation',
        'rotation',
        'silhouette',
        'stylized',
        'sketch',
        'uniform-noise',
    ])
    @pytest.mark.parametrize('field', [
        'image_id',
        'image_id_long',
        'choice',
        'truth',
        'condition',
        'response_time',
        'trial',
        'subject',
        'session',
    ])
    def test_fields_present(self, identifier, field):
        assembly = load_dataset(f"brendel.Geirhos2021_{identifier}")
        assert hasattr(assembly, field)

    # tests assembly coords for the 2 "abnormal" sets:
    @pytest.mark.parametrize('identifier', [
        'edge',
        'silhouette',
    ])
    @pytest.mark.parametrize('field', [
        'image_id',
        'image_category',
        'truth',
        'image_variation',
        'condition',
    ])
    def test_fields_present_abnormal_sets(self, identifier, field):
        assembly = load_dataset(f"brendel.Geirhos2021_{identifier}")
        assert hasattr(assembly, field)

    # tests assembly coords for the cue-conflict different set:
    @pytest.mark.parametrize('identifier', [
        'cue-conflict',
    ])
    @pytest.mark.parametrize('field', [
        'image_id',
        'original_image',
        'truth',
        'category',
        'conflict_image',
        'original_image_category',
        'original_image_variation',
        'conflict_image_category',
        'conflict_image_variation',
        'condition',
    ])
    def test_fields_present_cue_conflict(self, identifier, field):
        assembly = load_dataset(f"brendel.Geirhos2021_{identifier}")
        assert hasattr(assembly, field)


# testing stimulus sets
@pytest.mark.slow
class TestStimulusSets:
    # test stimulus_set data:
    @pytest.mark.parametrize('identifier', [
        'colour',
        'contrast',
        'cue-conflict',
        'edge',
        'eidolonI',
        'eidolonII',
        'eidolonIII',
        'false-colour',
        'high-pass',
        'low-pass',
        'phase-scrambling',
        'power-equalisation',
        'rotation',
        'silhouette',
        'stylized',
        'sketch',
        'uniform-noise',
    ])
    def test_stimulus_set_exist(self, identifier):
        full_name = f"brendel.Geirhos2021_{identifier}"
        stimulus_set = load_stimulus_set(full_name)
        assert stimulus_set is not None
        assert stimulus_set.identifier == full_name

    # test the number of images
    @pytest.mark.parametrize('identifier, num_images', [
        ('colour', 1280),
        ('contrast', 1280),
        ('cue-conflict', 1280),
        ('edge', 160),
        ('eidolonI', 1280),
        ('eidolonII', 1280),
        ('eidolonIII', 1280),
        ('false-colour', 1120),
        ('high-pass', 1280),
        ('low-pass', 1280),
        ('phase-scrambling', 1120),
        ('power-equalisation', 1120),
        ('rotation', 1280),
        ('silhouette', 160),
        ('stylized', 800),
        ('sketch', 800),
        ('uniform-noise', 1280),
    ])
    def test_num_images(self, identifier, num_images):
        stimulus_set = load_stimulus_set(f"brendel.Geirhos2021_{identifier}")
        assert len(np.unique(stimulus_set['image_id'].values)) == num_images

    # tests stimulus_set coords for the 14 "normal" sets:
    @pytest.mark.parametrize('identifier', [
        'colour',
        'contrast',
        'eidolonI',
        'eidolonII',
        'eidolonIII',
        'false-colour',
        'high-pass',
        'low-pass',
        'phase-scrambling',
        'power-equalisation',
        'rotation',
        'stylized',
        'sketch',
        'uniform-noise',
    ])
    @pytest.mark.parametrize('field', [
        'image_id',
        'image_id_long',
        'image_number',
        'experiment_code',
        'condition',
        'truth',
        'category_ground_truth',
        'random_number',
    ])
    def test_fields_present(self, identifier, field):
        stimulus_set = load_stimulus_set(f"brendel.Geirhos2021_{identifier}")
        assert hasattr(stimulus_set, field)

    # tests assembly coords for the 2 "abnormal" sets:
    @pytest.mark.parametrize('identifier', [
        'edge',
        'silhouette',
    ])
    @pytest.mark.parametrize('field', [
        'image_id',
        'image_category',
        'truth',
        'image_variation',
        'condition',
    ])
    def test_fields_present2(self, identifier, field):
        stimulus_set = load_dataset(f"brendel.Geirhos2021_{identifier}")
        assert hasattr(stimulus_set, field)

    # test assembly fields for cue-conflict's odd stimulus_set:
    @pytest.mark.parametrize('identifier', [
        'cue-conflict',
    ])
    @pytest.mark.parametrize('field', [
        'image_id',
        'original_image',
        'truth',
        'category',
        'conflict_image',
        'original_image_category',
        'original_image_variation',
        'conflict_image_category',
        'conflict_image_variation',
        'condition',
    ])
    def test_fields_present3(self, identifier, field):
        stimulus_set = load_dataset(f"brendel.Geirhos2021_{identifier}")
        assert hasattr(stimulus_set, field)

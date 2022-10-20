import os

import brainscore
import pytest
import numpy as np

import brainio


@pytest.mark.parametrize('stimulus_set', (
        'dicarlo.hvm',
        'dicarlo.hvm-public',
        'dicarlo.hvm-private',
        'gallant.David2004',
        'tolias.Cadena2017',
        'movshon.FreemanZiemba2013',
        'movshon.FreemanZiemba2013-public',
        'movshon.FreemanZiemba2013-private',
        'dicarlo.objectome.public',
        'dicarlo.objectome.private',
        'dicarlo.Kar2018cocogray',
        'klab.Zhang2018.search_obj_array',
        'dicarlo.Rajalingham2020',
        'dicarlo.Rust2012',
        'dicarlo.BOLD5000',
        'dicarlo.THINGS1',
        'dicarlo.THINGS2',
        'aru.Kuzovkin2018',
        'dietterich.Hendrycks2019.noise',
        'dietterich.Hendrycks2019.blur',
        'dietterich.Hendrycks2019.weather',
        'dietterich.Hendrycks2019.digital',
        'katz.BarbuMayo2019',
        'fei-fei.Deng2009',
        'aru.Cichy2019',
        'dicarlo.BashivanKar2019.naturalistic',
        'dicarlo.BashivanKar2019.synthetic',
        'dicarlo.Marques2020_blank',
        'dicarlo.Marques2020_receptive_field',
        'dicarlo.Marques2020_orientation',
        'dicarlo.Marques2020_spatial_frequency',
        'dicarlo.Marques2020_size',
        'movshon.FreemanZiemba2013_properties',
        'brendel.Geirhos2021_colour',
        'brendel.Geirhos2021_contrast',
        'brendel.Geirhos2021_cue-conflict',
        'brendel.Geirhos2021_edge',
        'brendel.Geirhos2021_eidolonI',
        'brendel.Geirhos2021_eidolonII',
        'brendel.Geirhos2021_eidolonIII',
        'brendel.Geirhos2021_false-colour',
        'brendel.Geirhos2021_high-pass',
        'brendel.Geirhos2021_low-pass',
        'brendel.Geirhos2021_phase-scrambling',
        'brendel.Geirhos2021_power-equalisation',
        'brendel.Geirhos2021_rotation',
        'brendel.Geirhos2021_silhouette',
        'brendel.Geirhos2021_stylized',
        'brendel.Geirhos2021_sketch',
        'brendel.Geirhos2021_uniform-noise',
        'yuille.Zhu2019_extreme_occlusion',
))
def test_list_stimulus_set(stimulus_set):
    l = brainio.list_stimulus_sets()
    assert stimulus_set in l


@pytest.mark.private_access
def test_klab_Zhang2018search():
    stimulus_set = brainio.get_stimulus_set('klab.Zhang2018.search_obj_array')
    # There are 300 presentation images in the assembly but 606 in the StimulusSet (explanation from @shashikg follows).
    # For each of the visual search task out of total 300, you need two images (one - the target image,
    # second - the search space image) plus there are 6 different mask images to mask objects
    # present at 6 different locations in a specified search image.
    # Therefore, a total of 300 * 2 + 6 images are there in the stimulus set.
    assert len(stimulus_set) == 606
    assert len(set(stimulus_set['stimulus_id'])) == 606


@pytest.mark.private_access
@pytest.mark.slow
class TestDietterichHendrycks2019:
    def test_noise(self):
        stimulus_set = brainio.get_stimulus_set('dietterich.Hendrycks2019.noise')
        assert len(stimulus_set) == 3 * 5 * 50000
        assert len(set(stimulus_set['synset'])) == 1000

    def test_blur(self):
        stimulus_set = brainio.get_stimulus_set('dietterich.Hendrycks2019.blur')
        assert len(stimulus_set) == 4 * 5 * 50000
        assert len(set(stimulus_set['synset'])) == 1000

    def test_weather(self):
        stimulus_set = brainio.get_stimulus_set('dietterich.Hendrycks2019.weather')
        assert len(stimulus_set) == 4 * 5 * 50000
        assert len(set(stimulus_set['synset'])) == 1000

    def test_digital(self):
        stimulus_set = brainio.get_stimulus_set('dietterich.Hendrycks2019.digital')
        assert len(stimulus_set) == 4 * 5 * 50000
        assert len(set(stimulus_set['synset'])) == 1000


@pytest.mark.private_access
def test_Katz_BarbuMayo2019():
    stimulus_set = brainio.get_stimulus_set('katz.BarbuMayo2019')
    assert len(stimulus_set) == 17261
    assert len(set(stimulus_set['synset'])) == 104

@pytest.mark.private_access
def test_feifei_Deng2009():
    stimulus_set = brainio.get_stimulus_set('fei-fei.Deng2009')
    assert len(stimulus_set) == 50_000
    assert len(set(stimulus_set['label'])) == 1_000


@pytest.mark.private_access
class TestMarques2020V1Properties:
    @pytest.mark.parametrize('identifier,num_stimuli', [
        ('dicarlo.Marques2020_blank', 1),
        ('dicarlo.Marques2020_receptive_field', 3528),
        ('dicarlo.Marques2020_orientation', 1152),
        ('dicarlo.Marques2020_spatial_frequency', 2112),
        ('dicarlo.Marques2020_size', 2304),
        ('movshon.FreemanZiemba2013_properties', 450),
    ])
    def test_num_stimuli(self, identifier, num_stimuli):
        stimulus_set = brainio.get_stimulus_set(identifier)
        assert len(stimulus_set) == num_stimuli


@pytest.mark.slow
class TestGeirhos2021:
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
        stimulus_set = brainio.get_stimulus_set(full_name)
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
        stimulus_set = brainscore.get_stimulus_set(f"brendel.Geirhos2021_{identifier}")
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
        stimulus_set = brainscore.get_stimulus_set(f"brendel.Geirhos2021_{identifier}")
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
        stimulus_set = brainscore.get_assembly(f"brendel.Geirhos2021_{identifier}")
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
        stimulus_set = brainscore.get_assembly(f"brendel.Geirhos2021_{identifier}")
        assert hasattr(stimulus_set, field)


class TestZhu2019:

    def test_stimulus_set_exist(self):
        full_name = 'yuille.Zhu2019_extreme_occlusion'
        stimulus_set = brainio.get_stimulus_set(full_name)
        assert stimulus_set is not None
        assert stimulus_set.identifier == full_name

    def test_num_images(self):
        stimulus_set = brainio.get_stimulus_set('yuille.Zhu2019_extreme_occlusion')
        assert len(np.unique(stimulus_set['stimulus_id'].values)) == 500

    @pytest.mark.parametrize('field', [
        'stimulus_id',
        'ground_truth',
        'occlusion_strength',
        'word_image',
        'image_number',
    ])
    def test_fields_present(self, field):
        stimulus_set = brainio.get_stimulus_set('yuille.Zhu2019_extreme_occlusion')
        assert hasattr(stimulus_set, field)
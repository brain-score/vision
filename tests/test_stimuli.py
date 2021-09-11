import pytest

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
        'dicarlo.Kar2018coco_color.public',
        'dicarlo.Kar2018coco_color.private',
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
    assert len(set(stimulus_set['image_id'])) == 606


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


@pytest.mark.parametrize('access,num_images', [
    pytest.param('public', 1_280, marks=[]),
    pytest.param('private', 320, marks=[pytest.mark.private_access]),
])
def test_kar2018coco_color(access, num_images):
    stimulus_set = brainio.get_stimulus_set(f'dicarlo.Kar2018coco_color.{access}')
    assert len(stimulus_set) == num_images
    assert len(set(stimulus_set['image_id'])) == num_images
    assert set(stimulus_set['object_name']) == {'breed_pug', 'bear', 'zebra', 'ELEPHANT_M', '_001', 'f16', 'face0001',
                                                'lo_poly_animal_CHICKDEE', 'alfa155', 'Apple_Fruit_obj'}

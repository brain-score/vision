import pytest
from brainscore_vision import load_dataset, load_stimulus_set


@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('klab.Zhang2018search_obj_array', marks=[pytest.mark.private_access]),
])
def test_existence(assembly_identifier):
    assert load_dataset(assembly_identifier) is not None

@pytest.mark.private_access
def test_klab_Zhang2018search():
    assembly = load_dataset('klab.Zhang2018search_obj_array')
    assert set(assembly.dims) == {'presentation', 'fixation', 'position'}
    assert len(assembly['presentation']) == 4500
    assert len(set(assembly['stimulus_id'].values)) == 300
    assert len(set(assembly['subjects'].values)) == 15
    assert len(assembly['fixation']) == 8
    assert len(assembly['position']) == 2
    assert assembly.stimulus_set is not None


@pytest.mark.private_access
def test_klab_Zhang2018search():
    stimulus_set = load_stimulus_set('klab.Zhang2018.search_obj_array')
    # There are 300 presentation images in the assembly but 606 in the StimulusSet (explanation from @shashikg follows).
    # For each of the visual search task out of total 300, you need two images (one - the target image,
    # second - the search space image) plus there are 6 different mask images to mask objects
    # present at 6 different locations in a specified search image.
    # Therefore, a total of 300 * 2 + 6 images are there in the stimulus set.
    assert len(stimulus_set) == 606
    assert len(set(stimulus_set['stimulus_id'])) == 606

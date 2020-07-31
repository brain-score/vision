from brainio_base.stimuli import StimulusSet
from brainscore.benchmarks.trials import repeat_trials


def _dummy_stimulus_set():
    stimulus_set = StimulusSet([
        {'image_id': 'a'},
        {'image_id': 'b'},
        {'image_id': 'c'},
    ])
    stimulus_set.image_paths = {
        'a': 'a.png',
        'b': 'b.png',
        'c': 'c.png',
    }
    stimulus_set.identifier = 'dummy'
    return stimulus_set


def test_integer_repeat():
    stimulus_set = _dummy_stimulus_set()
    repeat_stimulus_set = repeat_trials(stimulus_set, number_of_trials=5)
    assert len(repeat_stimulus_set) == len(stimulus_set) * 5
    original_image_paths = [stimulus_set.get_image(image_id) for image_id in stimulus_set['image_id']]
    repeat_image_paths = [repeat_stimulus_set.get_image(image_id) for image_id in repeat_stimulus_set['image_id']]
    assert set(repeat_image_paths) == set(original_image_paths)
    assert all(len(group) == 5 and set(group['repetition']) == {0, 1, 2, 3, 4}
               for name, group in repeat_stimulus_set.groupby('image_id'))
    assert repeat_stimulus_set.identifier == 'dummy-5trials'


def test_per_image_repeat():
    pass


def test_average_trials():
    pass
